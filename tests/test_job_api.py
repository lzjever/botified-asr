from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import anyio
import pytest
from starlette.testclient import TestClient

import botified_asr.api as api_module
import botified_asr.storage as storage_module
from botified_asr.api import Readiness, create_app
from botified_asr.config import LimitsConfig, RESERVATION_QUANTUM
from botified_asr.errors import PipelineError
from botified_asr.jobs import (
    DurableJob,
    JobDeletionOutcome,
    JobPhase,
    JobStatus,
    QueuedJobSpec,
)
from botified_asr.result_artifact import CanonicalArtifactError
from botified_asr.runtime import JobExecutor
from botified_asr.speakers import SpeakerEmbeddingPolicy
from botified_asr.storage import Storage


AUTH = {"Authorization": "Bearer test-secret"}
CREATED_AT = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
STARTED_AT = CREATED_AT + timedelta(minutes=1)
FINISHED_AT = STARTED_AT + timedelta(minutes=1)
OPTIONS = (
    '{"chunking_strategy":null,"include":[],"known_speaker_ids":[],'
    '"language":"auto","model":"sensevoice","response_format":"json"}'
)


def embedding_policy() -> SpeakerEmbeddingPolicy:
    return SpeakerEmbeddingPolicy(
        model_id="funasr/campplus",
        model_revision="1" * 40,
        embedding_dimension=2,
        sample_rate=16_000,
        downmix_policy_version="ffmpeg-first-audio-stream-ac1-v1",
        window_samples=24_000,
        window_shift_samples=12_000,
        padding_policy_version="right-zero-pad-v1",
        normalization_policy_version="int16-div-32768-l2-v1",
        enrollment_aggregation_policy_version=(
            "sample-centroid-equal-average-v1"
        ),
    )


def durable_job(
    status: JobStatus,
    *,
    error_code: str | None = None,
) -> DurableJob:
    running = status is JobStatus.RUNNING
    terminal = status in {
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    }
    succeeded = status is JobStatus.SUCCEEDED
    cancelled = status is JobStatus.CANCELLED
    return DurableJob(
        id="7K3M9Q2W",
        phase=JobPhase.VISIBLE,
        status=status,
        input_lease_id=None if terminal else "7K3M9Q2W",
        canonical_options_json=OPTIONS,
        selected_speaker_snapshot=b'{"speakers":[]}',
        snapshot_sha256="1" * 64,
        input_size_bytes=5,
        effective_max_audio_samples=32_000,
        effective_direct_max_audio_samples=16_000,
        total_samples=32_000,
        processed_samples=(
            32_000 if succeeded else 16_000 if running else 0
        ),
        request_fingerprint="2" * 64,
        processor_fingerprint="3" * 64,
        attempt_no=1,
        attempt_token="attempt-1" if running else None,
        owner_generation="generation-1" if running else None,
        crash_recoveries=0,
        cancel_requested=cancelled,
        result_lease_id="a" * 32 if succeeded else None,
        error_code=error_code,
        input_cleanup_pending=False,
        created_at=CREATED_AT,
        started_at=None if status is JobStatus.QUEUED else STARTED_AT,
        finished_at=FINISHED_AT if terminal else None,
    )


class FakeStoredResult:
    def __init__(
        self,
        chunks: tuple[bytes, ...],
        *,
        response_started: Callable[[], bool] = lambda: True,
    ) -> None:
        self.chunks = chunks
        self.response_started = response_started
        self.claimed = False
        self.closed = False
        self.private_manifest = b'{"attempt_no":1,"request_fingerprint":"secret"}'

    def iter_body(self):
        if self.claimed:
            raise RuntimeError("stored result body was already requested")
        self.claimed = True

        def body():
            assert self.response_started()
            yield from self.chunks

        return body()

    def close(self) -> None:
        self.closed = True


class FakeStorage:
    def __init__(
        self,
        job: DurableJob | None,
        *,
        malformed: bool = False,
        result: FakeStoredResult | None = None,
        open_error: Exception | None = None,
        deletion_outcome: JobDeletionOutcome = JobDeletionOutcome.NOT_FOUND,
        delete_error: Exception | None = None,
        retention_sweep: Callable[[datetime], bool] | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.job = job
        self.malformed = malformed
        self.result = result
        self.open_error = open_error
        self.deletion_outcome = deletion_outcome
        self.delete_error = delete_error
        self.retention_sweep = retention_sweep
        self.events = events
        self.get_calls = 0
        self.open_calls = 0
        self.release_calls = 0
        self.delete_calls: list[tuple[str, datetime]] = []
        self.cleanup_calls: list[str] = []
        self.retention_calls: list[datetime] = []
        self.retention_thread_ids: list[int] = []
        self.retention_had_running_loop = False
        self.delete_thread_ids: list[int] = []
        self.delete_had_running_loop = False

    def get_visible_job(self, _: str) -> DurableJob | None:
        self.get_calls += 1
        if self.malformed:
            raise ValueError("invalid job id")
        return self.job

    def open_succeeded_job_result(
        self,
        _: str,
    ) -> FakeStoredResult | None:
        self.open_calls += 1
        if self.open_error is not None:
            raise self.open_error
        return self.result

    def release_artifact(self, _: object) -> None:
        self.release_calls += 1


    def delete_or_cancel_job(
        self,
        job_id: str,
        requested_at: datetime,
    ) -> JobDeletionOutcome:
        self.delete_calls.append((job_id, requested_at))
        self.delete_thread_ids.append(threading.get_ident())
        if self.events is not None:
            self.events.append("storage")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            self.delete_had_running_loop = True
        if self.delete_error is not None:
            raise self.delete_error
        return self.deletion_outcome

    def cleanup_cancelled_job_input(self, job_id: str) -> None:
        self.cleanup_calls.append(job_id)
        if self.events is not None:
            self.events.append("cleanup")

    def delete_next_expired_terminal_job(
        self,
        sweep_at: datetime,
    ) -> bool:
        self.retention_calls.append(sweep_at)
        self.retention_thread_ids.append(threading.get_ident())
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            self.retention_had_running_loop = True
        if self.retention_sweep is None:
            return False
        return self.retention_sweep(sweep_at)


class FakeJobExecutor:
    def __init__(self, events: list[str] | None = None) -> None:
        self.ready = True
        self.events = events
        self.notifications: list[str] = []

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def notify_cancellation(self, job_id: str) -> None:
        if self.events is not None:
            self.events.append("notify")
        self.notifications.append(job_id)


class OrderedFakeJobExecutor(FakeJobExecutor):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.lifecycle_events = events

    def stop(self) -> None:
        self.lifecycle_events.append("executor_stop")


def app(
    storage: FakeStorage | Storage,
    readiness: Readiness | None = None,
    *,
    job_executor: FakeJobExecutor | JobExecutor | None = None,
    close_storage_on_shutdown: bool = False,
):
    return create_app(
        api_key="test-secret",
        readiness=readiness or Readiness(True, True, True),
        storage=storage,
        processor=object(),
        audio_prober=lambda _path, _cancellation: None,
        processor_fingerprint="3" * 64,
        speaker_embedding_policy=embedding_policy(),
        job_executor=job_executor,
        close_storage_on_shutdown=close_storage_on_shutdown,
    )


def real_storage(tmp_path: Path) -> Storage:
    return Storage(
        tmp_path,
        LimitsConfig(
            max_upload_bytes=RESERVATION_QUANTUM,
            sync_max_upload_bytes=RESERVATION_QUANTUM,
            max_active_uploads=4,
            max_queued_jobs=4,
            max_job_storage_bytes=4 * RESERVATION_QUANTUM,
            min_filesystem_free_bytes=1,
        ),
        current_processor_fingerprint="3" * 64,
        free_bytes=lambda _: 1 << 40,
    )


def queue_real_job(storage: Storage) -> DurableJob:
    upload = storage.begin_job_upload(CREATED_AT)
    storage.append_job_upload(upload, b"audio")
    input_ref = storage.seal_job_upload(upload)
    return storage.publish_job(
        input_ref,
        QueuedJobSpec(
            canonical_options_json=OPTIONS,
            effective_max_audio_samples=32_000,
            effective_direct_max_audio_samples=16_000,
            processor_fingerprint="3" * 64,
        ),
        speaker_embedding_policy=embedding_policy(),
    )


def wait_until(predicate: Callable[[], bool], timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for job state")
        time.sleep(0.005)


def install_controlled_maintenance_timeout(
    monkeypatch: pytest.MonkeyPatch,
    *,
    count: int = 1,
) -> tuple[
    threading.Semaphore,
    list[threading.Event],
    list[float],
]:
    original_wait_for = api_module.wait_for
    release_timeout = threading.Semaphore(0)
    timeout_waiting = [threading.Event() for _ in range(count)]
    observed_timeouts: list[float] = []

    async def controlled_wait_for(
        awaitable: object,
        timeout: float,
    ) -> object:
        observed_timeouts.append(timeout)
        index = len(observed_timeouts) - 1
        if index < count:
            close = getattr(awaitable, "close")
            close()
            timeout_waiting[index].set()
            await anyio.to_thread.run_sync(release_timeout.acquire)
            raise TimeoutError
        return await original_wait_for(awaitable, timeout)

    monkeypatch.setattr(api_module, "wait_for", controlled_wait_for)
    return release_timeout, timeout_waiting, observed_timeouts


def test_retention_startup_sweep_uses_fixed_utc_and_bounded_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sweep_at = datetime(
        2026,
        7,
        27,
        13,
        14,
        15,
        tzinfo=timezone.utc,
    )

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz: timezone | None = None) -> datetime:
            assert tz is timezone.utc
            return sweep_at

    monkeypatch.setattr(api_module, "datetime", FixedDateTime)
    storage = FakeStorage(None, retention_sweep=lambda _: True)
    executor = FakeJobExecutor()

    with TestClient(app(storage, job_executor=executor)):
        pass

    assert storage.retention_calls == [sweep_at] * 32
    assert storage.retention_thread_ids
    assert not storage.retention_had_running_loop


def test_retention_waits_sixty_seconds_between_empty_sweeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_timeout, timeout_waiting, observed_timeouts = (
        install_controlled_maintenance_timeout(monkeypatch)
    )
    periodic_sweep = threading.Event()

    def sweep(_: datetime) -> bool:
        if len(storage.retention_calls) == 2:
            periodic_sweep.set()
        return False

    storage = FakeStorage(None, retention_sweep=sweep)
    executor = FakeJobExecutor()
    with TestClient(app(storage, job_executor=executor)):
        assert timeout_waiting[0].wait(timeout=2)
        assert len(storage.retention_calls) == 1
        release_timeout.release()
        assert periodic_sweep.wait(timeout=2)

    assert observed_timeouts[0] == 60.0
    assert len(storage.retention_calls) == 2


def test_retention_startup_error_prevents_ready() -> None:
    readiness = Readiness(True, True, False)
    sentinel = RuntimeError("injected startup retention failure")

    def fail(_: datetime) -> bool:
        raise sentinel

    storage = FakeStorage(None, retention_sweep=fail)
    executor = FakeJobExecutor()

    with pytest.raises(RuntimeError) as raised:
        with TestClient(
            app(storage, readiness, job_executor=executor),
        ):
            pass

    assert raised.value is sentinel
    assert not readiness.database


def test_retention_periodic_failure_recovers_on_next_successful_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_timeout, timeout_waiting, _ = (
        install_controlled_maintenance_timeout(monkeypatch, count=2)
    )
    readiness = Readiness(True, True, False)
    sentinel = RuntimeError("injected periodic retention failure")
    periodic_failed = threading.Event()
    periodic_recovered = threading.Event()

    def sweep(_: datetime) -> bool:
        if len(storage.retention_calls) == 2:
            periodic_failed.set()
            raise sentinel
        if len(storage.retention_calls) == 3:
            periodic_recovered.set()
        return False

    storage = FakeStorage(None, retention_sweep=sweep)
    executor = FakeJobExecutor()

    with TestClient(
        app(storage, readiness, job_executor=executor),
    ):
        assert timeout_waiting[0].wait(timeout=2)
        release_timeout.release()
        assert periodic_failed.wait(timeout=2)
        wait_until(lambda: not readiness.database)
        assert timeout_waiting[1].wait(timeout=2)
        release_timeout.release()
        assert periodic_recovered.wait(timeout=2)
        wait_until(lambda: readiness.database)


def test_retention_continuous_failure_propagates_last_error_at_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_timeout, timeout_waiting, _ = (
        install_controlled_maintenance_timeout(monkeypatch, count=2)
    )
    readiness = Readiness(True, True, False)
    errors = (
        RuntimeError("injected first periodic retention failure"),
        RuntimeError("injected last periodic retention failure"),
    )
    failed = (threading.Event(), threading.Event())

    def sweep(_: datetime) -> bool:
        call = len(storage.retention_calls)
        if call >= 2:
            index = call - 2
            failed[index].set()
            raise errors[index]
        return False

    storage = FakeStorage(None, retention_sweep=sweep)
    executor = FakeJobExecutor()

    with pytest.raises(RuntimeError) as raised:
        with TestClient(
            app(storage, readiness, job_executor=executor),
        ):
            for index in range(2):
                assert timeout_waiting[index].wait(timeout=2)
                release_timeout.release()
                assert failed[index].wait(timeout=2)
            wait_until(lambda: not readiness.database)

    assert raised.value is errors[-1]


def test_shutdown_awaits_inflight_retention_before_executor_and_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_timeout, timeout_waiting, _ = (
        install_controlled_maintenance_timeout(monkeypatch)
    )
    sweep_entered = threading.Event()
    release_sweep = threading.Event()
    lifecycle_events: list[str] = []

    def sweep(_: datetime) -> bool:
        if len(storage.retention_calls) == 1:
            return False
        lifecycle_events.append("sweep_enter")
        sweep_entered.set()
        assert release_sweep.wait(timeout=2)
        lifecycle_events.append("sweep_exit")
        return False

    storage = FakeStorage(None, retention_sweep=sweep)
    storage.close = lambda: lifecycle_events.append("storage_close")
    executor = OrderedFakeJobExecutor(lifecycle_events)
    client = TestClient(
        app(
            storage,
            job_executor=executor,
            close_storage_on_shutdown=True,
        )
    )
    client.__enter__()
    assert timeout_waiting[0].wait(timeout=2)
    release_timeout.release()
    assert sweep_entered.wait(timeout=2)
    shutdown_complete = threading.Event()

    def shutdown() -> None:
        client.__exit__(None, None, None)
        shutdown_complete.set()

    shutdown_thread = threading.Thread(target=shutdown)
    shutdown_thread.start()
    assert not shutdown_complete.wait(timeout=0.05)
    assert lifecycle_events == ["sweep_enter"]
    release_sweep.set()
    shutdown_thread.join(timeout=2)

    assert not shutdown_thread.is_alive()
    assert lifecycle_events == [
        "sweep_enter",
        "sweep_exit",
        "executor_stop",
        "storage_close",
    ]


@pytest.mark.parametrize(
    ("headers", "readiness", "status"),
    (
        ({}, Readiness(True, True, True), 401),
        (AUTH, Readiness(True, True, False), 503),
    ),
)
def test_job_get_checks_auth_and_readiness_before_storage(
    headers: dict[str, str],
    readiness: Readiness,
    status: int,
) -> None:
    storage = FakeStorage(durable_job(JobStatus.QUEUED))
    with TestClient(app(storage, readiness)) as client:
        response = client.get(
            "/v1/audio/transcriptions/7K3M9Q2W",
            headers=headers,
        )

    assert response.status_code == status
    assert storage.get_calls == 0


@pytest.mark.parametrize(
    ("headers", "readiness", "status"),
    (
        ({}, Readiness(True, True, True), 401),
        (AUTH, Readiness(True, True, False), 503),
    ),
)
def test_job_delete_checks_auth_and_readiness_before_storage(
    headers: dict[str, str],
    readiness: Readiness,
    status: int,
) -> None:
    storage = FakeStorage(
        None,
        deletion_outcome=JobDeletionOutcome.QUEUED_CANCELLED,
    )
    with TestClient(app(storage, readiness)) as client:
        response = client.delete(
            "/v1/audio/transcriptions/7K3M9Q2W",
            headers=headers,
        )

    assert response.status_code == status
    assert storage.delete_calls == []


@pytest.mark.parametrize(
    ("outcome", "status", "body", "events"),
    (
        (
            JobDeletionOutcome.QUEUED_CANCELLED,
            202,
            {"id": "7K3M9Q2W", "status": "cancelled"},
            ["storage", "cleanup"],
        ),
        (
            JobDeletionOutcome.RUNNING_CANCEL_REQUESTED,
            202,
            {"id": "7K3M9Q2W", "status": "running"},
            ["storage", "notify"],
        ),
        (
            JobDeletionOutcome.TERMINAL_DELETED,
            204,
            None,
            ["storage"],
        ),
        (
            JobDeletionOutcome.NOT_FOUND,
            404,
            {
                "error": {
                    "message": "Transcription job not found",
                    "type": "invalid_request_error",
                    "param": "job_id",
                    "code": "job_not_found",
                }
            },
            ["storage"],
        ),
    ),
)
def test_job_delete_maps_outcome_and_only_notifies_running(
    outcome: JobDeletionOutcome,
    status: int,
    body: dict[str, object] | None,
    events: list[str],
) -> None:
    observed_events: list[str] = []
    storage = FakeStorage(
        None,
        deletion_outcome=outcome,
        events=observed_events,
    )
    executor = FakeJobExecutor(observed_events)
    with TestClient(app(storage, job_executor=executor)) as client:
        response = client.delete(
            "/v1/audio/transcriptions/7K3M9Q2W",
            headers=AUTH,
        )

    assert response.status_code == status
    if body is None:
        assert response.content == b""
    else:
        assert response.json() == body
    assert observed_events == events
    assert executor.notifications == (
        ["7K3M9Q2W"]
        if outcome is JobDeletionOutcome.RUNNING_CANCEL_REQUESTED
        else []
    )
    assert storage.cleanup_calls == (
        ["7K3M9Q2W"]
        if outcome is JobDeletionOutcome.QUEUED_CANCELLED
        else []
    )


def test_queued_job_delete_cleans_real_input_after_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        storage_module,
        "generate_job_id",
        lambda: "7K3M9Q2W",
    )
    storage = real_storage(tmp_path)
    try:
        queued = queue_real_job(storage)
        input_path = storage.staging_dir / f"{queued.id}.ready"
        with TestClient(app(storage)) as client:
            response = client.delete(
                f"/v1/audio/transcriptions/{queued.id}",
                headers=AUTH,
            )

        assert response.status_code == 202
        assert response.json() == {
            "id": queued.id,
            "status": "cancelled",
        }
        cancelled = storage.get_visible_job(queued.id)
        assert cancelled is not None
        assert cancelled.status is JobStatus.CANCELLED
        assert cancelled.input_lease_id is None
        assert not cancelled.input_cleanup_pending
        assert not input_path.exists()
        assert storage.total_reserved_bytes() == 0
    finally:
        storage.close()


def test_job_delete_malformed_id_is_not_found_without_storage_call() -> None:
    storage = FakeStorage(None)
    with TestClient(app(storage)) as client:
        response = client.delete(
            "/v1/audio/transcriptions/malformed",
            headers=AUTH,
        )

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "message": "Transcription job not found",
            "type": "invalid_request_error",
            "param": "job_id",
            "code": "job_not_found",
        }
    }
    assert storage.delete_calls == []


def test_job_delete_runs_storage_in_threadpool_with_aware_utc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_at = datetime(
        2026,
        7,
        27,
        13,
        14,
        15,
        tzinfo=timezone.utc,
    )

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz: timezone | None = None) -> datetime:
            assert tz is timezone.utc
            return requested_at

    monkeypatch.setattr(api_module, "datetime", FixedDateTime)
    storage = FakeStorage(
        None,
        deletion_outcome=JobDeletionOutcome.TERMINAL_DELETED,
    )
    caller_thread_id = threading.get_ident()
    with TestClient(app(storage)) as client:
        response = client.delete(
            "/v1/audio/transcriptions/7K3M9Q2W",
            headers=AUTH,
        )

    assert response.status_code == 204
    assert storage.delete_calls == [("7K3M9Q2W", requested_at)]
    assert storage.delete_calls[0][1].tzinfo is timezone.utc
    assert storage.delete_thread_ids != [caller_thread_id]
    assert not storage.delete_had_running_loop


@pytest.mark.parametrize(
    "error",
    (
        ValueError("injected valid-ID storage failure"),
        RuntimeError("injected delete failure"),
    ),
)
def test_job_delete_storage_error_does_not_notify(error: Exception) -> None:
    events: list[str] = []
    storage = FakeStorage(
        None,
        delete_error=error,
        events=events,
    )
    executor = FakeJobExecutor(events)
    with TestClient(
        app(storage, job_executor=executor),
        raise_server_exceptions=False,
    ) as client:
        response = client.delete(
            "/v1/audio/transcriptions/7K3M9Q2W",
            headers=AUTH,
        )

    assert response.status_code == 500
    assert events == ["storage"]
    assert executor.notifications == []


def test_job_delete_running_without_executor_still_returns_accepted() -> None:
    storage = FakeStorage(
        None,
        deletion_outcome=JobDeletionOutcome.RUNNING_CANCEL_REQUESTED,
    )
    with TestClient(app(storage)) as client:
        response = client.delete(
            "/v1/audio/transcriptions/7K3M9Q2W",
            headers=AUTH,
        )

    assert response.status_code == 202
    assert response.json() == {"id": "7K3M9Q2W", "status": "running"}


def test_running_job_delete_notifies_real_executor_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_timeout, timeout_waiting, _ = (
        install_controlled_maintenance_timeout(monkeypatch)
    )
    job_ids = iter(("7K3M9Q2W", "8K3M9Q2W"))
    monkeypatch.setattr(
        storage_module,
        "generate_job_id",
        lambda: next(job_ids),
    )
    storage = real_storage(tmp_path)
    first = queue_real_job(storage)
    second = queue_real_job(storage)
    first_started = threading.Event()
    second_started = threading.Event()
    periodic_sweep = threading.Event()
    retention_calls: list[datetime] = []

    def sweep(sweep_at: datetime) -> bool:
        retention_calls.append(sweep_at)
        if len(retention_calls) == 2:
            periodic_sweep.set()
        return False

    monkeypatch.setattr(
        storage,
        "delete_next_expired_terminal_job",
        sweep,
        raising=False,
    )

    class BlockingThenFailingProcessor:
        calls = 0

        def process(
            self,
            _input_path: Path,
            _options: object,
            cancellation: Any,
            _progress: object,
            _sink: object,
            **_kwargs: Any,
        ) -> object:
            self.calls += 1
            if self.calls == 1:
                first_started.set()
                deadline = time.monotonic() + 2
                while not cancellation.cancelled:
                    if time.monotonic() >= deadline:
                        raise AssertionError(
                            "timed out waiting for API cancellation"
                        )
                    time.sleep(0.005)
                raise PipelineError("cancelled", "job cancelled")
            second_started.set()
            raise PipelineError("invalid_audio", "second job finished")

    executor = JobExecutor(
        storage,
        BlockingThenFailingProcessor(),
        embedding_policy(),
        "generation-1",
        lambda: datetime.now(timezone.utc),
    )
    try:
        with TestClient(
            app(storage, job_executor=executor),
        ) as client:
            assert first_started.wait(timeout=2)
            assert timeout_waiting[0].wait(timeout=2)
            assert len(retention_calls) == 1
            release_timeout.release()
            assert periodic_sweep.wait(timeout=2)
            assert (
                storage.get_visible_job(first.id).status
                is JobStatus.RUNNING
            )
            response = client.delete(
                f"/v1/audio/transcriptions/{first.id}",
                headers=AUTH,
            )

            assert response.status_code == 202
            assert response.json() == {
                "id": first.id,
                "status": "running",
            }
            assert second_started.wait(timeout=2)
            wait_until(
                lambda: storage.get_visible_job(first.id).status
                is JobStatus.CANCELLED
            )
            wait_until(
                lambda: storage.get_visible_job(second.id).status
                is JobStatus.FAILED
            )
            assert executor.ready
            assert executor.failure is None
    finally:
        executor.stop()
        storage.close()


@pytest.mark.parametrize("malformed", (False, True))
def test_job_get_missing_and_malformed_ids_share_not_found(
    malformed: bool,
) -> None:
    storage = FakeStorage(None, malformed=malformed)
    job_id = "malformed" if malformed else "7K3M9Q2W"
    with TestClient(app(storage)) as client:
        response = client.get(
            f"/v1/audio/transcriptions/{job_id}",
            headers=AUTH,
        )

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "message": "Transcription job not found",
            "type": "invalid_request_error",
            "param": "job_id",
            "code": "job_not_found",
        }
    }
    assert storage.get_calls == 1


@pytest.mark.parametrize(
    ("status", "processed"),
    (
        (JobStatus.QUEUED, 0.0),
        (JobStatus.RUNNING, 1.0),
    ),
)
def test_job_get_returns_exact_active_progress(
    status: JobStatus,
    processed: float,
) -> None:
    storage = FakeStorage(durable_job(status))
    with TestClient(app(storage)) as client:
        response = client.get(
            "/v1/audio/transcriptions/7K3M9Q2W",
            headers=AUTH,
        )

    assert response.status_code == 200
    assert response.json() == {
        "id": "7K3M9Q2W",
        "status": status.value,
        "progress": {
            "processed_audio_secs": processed,
            "total_audio_secs": 2.0,
        },
    }


@pytest.mark.parametrize(
    ("status", "processed_samples"),
    (
        (JobStatus.QUEUED, 0),
        (JobStatus.RUNNING, 16_000),
    ),
)
def test_job_get_returns_null_total_until_decoder_eof(
    status: JobStatus,
    processed_samples: int,
) -> None:
    job = SimpleNamespace(
        id="7K3M9Q2W",
        status=status,
        processed_samples=processed_samples,
        total_samples=None,
    )
    storage = FakeStorage(job)
    with TestClient(app(storage), raise_server_exceptions=False) as client:
        response = client.get(
            "/v1/audio/transcriptions/7K3M9Q2W",
            headers=AUTH,
        )

    assert response.status_code == 200
    assert response.json() == {
        "id": "7K3M9Q2W",
        "status": status.value,
        "progress": {
            "processed_audio_secs": processed_samples / 16_000,
            "total_audio_secs": None,
        },
    }


@pytest.mark.parametrize(
    ("stored_code", "error"),
    (
        (
            "worker_crashed",
            {
                "message": "The transcription worker crashed",
                "type": "server_error",
                "param": None,
                "code": "worker_crashed",
            },
        ),
        (
            "private_exception_name",
            {
                "message": "Internal server error",
                "type": "server_error",
                "param": None,
                "code": "internal_error",
            },
        ),
    ),
)
def test_job_get_returns_safe_failed_error(
    stored_code: str,
    error: dict[str, object],
) -> None:
    storage = FakeStorage(
        durable_job(JobStatus.FAILED, error_code=stored_code)
    )
    with TestClient(app(storage)) as client:
        response = client.get(
            "/v1/audio/transcriptions/7K3M9Q2W",
            headers=AUTH,
        )

    assert response.json() == {
        "id": "7K3M9Q2W",
        "status": "failed",
        "error": error,
        "finished_at": "2026-07-27T12:02:00Z",
    }
    assert stored_code not in response.text or stored_code == "worker_crashed"


def test_job_get_returns_exact_cancelled_state() -> None:
    storage = FakeStorage(durable_job(JobStatus.CANCELLED))
    with TestClient(app(storage)) as client:
        response = client.get(
            "/v1/audio/transcriptions/7K3M9Q2W",
            headers=AUTH,
        )

    assert response.json() == {
        "id": "7K3M9Q2W",
        "status": "cancelled",
        "finished_at": "2026-07-27T12:02:00Z",
    }


def test_succeeded_job_open_failure_is_invalid_result_artifact() -> None:
    storage = FakeStorage(
        durable_job(JobStatus.SUCCEEDED),
        open_error=CanonicalArtifactError("injected open failure"),
    )
    with TestClient(app(storage), raise_server_exceptions=False) as client:
        response = client.get(
            "/v1/audio/transcriptions/7K3M9Q2W",
            headers=AUTH,
        )

    assert response.status_code == 500
    assert response.json()["error"] == {
        "message": "The transcription result artifact is invalid",
        "type": "server_error",
        "param": None,
        "code": "invalid_result_artifact",
    }


def test_succeeded_job_deleted_after_metadata_read_is_not_found() -> None:
    storage = FakeStorage(durable_job(JobStatus.SUCCEEDED))
    with TestClient(app(storage)) as client:
        response = client.get(
            "/v1/audio/transcriptions/7K3M9Q2W",
            headers=AUTH,
        )

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "message": "Transcription job not found",
            "type": "invalid_request_error",
            "param": "job_id",
            "code": "job_not_found",
        }
    }
    assert storage.get_calls == 1
    assert storage.open_calls == 1


def test_succeeded_job_get_streams_exact_envelope_after_response_start() -> None:
    async def run() -> tuple[
        list[dict[str, object]], FakeStoredResult, FakeStorage
    ]:
        response_started = False
        result = FakeStoredResult(
            (b'{"text":"hel', b'lo"}'),
            response_started=lambda: response_started,
        )
        storage = FakeStorage(
            durable_job(JobStatus.SUCCEEDED),
            result=result,
        )
        sent: list[dict[str, object]] = []

        async def receive() -> dict[str, object]:
            await anyio.sleep_forever()

        async def send(message: dict[str, object]) -> None:
            nonlocal response_started
            sent.append(message)
            if message["type"] == "http.response.start":
                response_started = True

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/v1/audio/transcriptions/7K3M9Q2W",
            "raw_path": b"/v1/audio/transcriptions/7K3M9Q2W",
            "query_string": b"",
            "headers": [(b"authorization", b"Bearer test-secret")],
            "client": ("test", 1),
            "server": ("testserver", 80),
        }
        await app(storage)(scope, receive, send)
        return sent, result, storage

    messages, result, storage = asyncio.run(run())
    start = next(
        message
        for message in messages
        if message["type"] == "http.response.start"
    )
    headers = dict(start["headers"])
    body_chunks = [
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
        and message.get("body", b"")
    ]

    assert start["status"] == 200
    assert headers[b"content-type"] == b"application/json"
    assert b"content-length" not in headers
    assert len(body_chunks) == 4
    assert b"".join(body_chunks) == (
        b'{"id":"7K3M9Q2W","status":"succeeded","result":'
        b'{"text":"hello"}}'
    )
    assert result.private_manifest not in b"".join(body_chunks)
    assert result.claimed
    assert result.closed
    assert storage.open_calls == 1
    assert storage.release_calls == 0
