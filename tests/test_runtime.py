from __future__ import annotations

import importlib
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pytest

from botified_asr import jobs
from botified_asr import storage as storage_module
from botified_asr.audio import Cancellation
from botified_asr.canonical_options import serialize_canonical_options
from botified_asr.composition import TranscriptionProcessorPool
from botified_asr.config import LimitsConfig, RESERVATION_QUANTUM
from botified_asr.contracts import CanonicalOptions
from botified_asr.errors import PipelineError
from botified_asr.inference import SerialInferenceLane
from botified_asr.speakers import SpeakerEmbeddingPolicy
from botified_asr.storage import Storage, StorageSchemaError


CREATED_AT = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
NOW = datetime(2026, 7, 27, 12, 1, tzinfo=timezone.utc)
PROCESSOR_FINGERPRINT = "3" * 64
GENERATION = "generation-1"


def _runtime():
    return importlib.import_module("botified_asr.runtime")


def _now() -> datetime:
    return NOW


def _limits() -> LimitsConfig:
    return LimitsConfig(
        max_upload_bytes=RESERVATION_QUANTUM,
        sync_max_upload_bytes=RESERVATION_QUANTUM,
        max_active_uploads=8,
        max_queued_jobs=8,
        max_job_storage_bytes=8 * RESERVATION_QUANTUM,
        min_filesystem_free_bytes=1,
    )


def _speaker_embedding_policy() -> SpeakerEmbeddingPolicy:
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


def _storage(tmp_path: Path) -> Storage:
    return Storage(
        tmp_path,
        _limits(),
        current_processor_fingerprint=PROCESSOR_FINGERPRINT,
        free_bytes=lambda _: 1 << 40,
    )


def _queue_job(storage: Storage) -> jobs.DurableJob:
    options = CanonicalOptions(
        model="sensevoice",
        language="auto",
        response_format="json",
        chunking_strategy=None,
        include=(),
        known_speaker_ids=(),
    )
    upload = storage.begin_job_upload(CREATED_AT)
    storage.append_job_upload(upload, b"durable audio")
    input_ref = storage.seal_job_upload(upload)
    return storage.publish_job(
        input_ref,
        jobs.QueuedJobSpec(
            canonical_options_json=serialize_canonical_options(options),
            effective_max_audio_samples=32_000,
            effective_direct_max_audio_samples=16_000,
            processor_fingerprint=PROCESSOR_FINGERPRINT,
        ),
        speaker_embedding_policy=_speaker_embedding_policy(),
    )


def _wait_until(predicate: Callable[[], bool], timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for worker state")
        time.sleep(0.005)


def test_job_executor_claims_fifo_only_when_slot_is_free(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = _runtime()
    ids = iter(("ABCDEFGH", "01234567"))
    monkeypatch.setattr(storage_module, "generate_job_id", lambda: next(ids))
    storage = _storage(tmp_path)
    first_created = _queue_job(storage)
    second_created = _queue_job(storage)
    first, second = sorted(
        (first_created, second_created),
        key=lambda job: (job.created_at, job.id),
    )
    processor = object()
    policy = _speaker_embedding_policy()
    entered = threading.Event()
    cancellation_seen = threading.Event()
    executions: list[str] = []
    claim_results: list[str | None] = []
    original_claim = storage.claim_next_job

    def claim(generation: str, claimed_at: datetime):
        claimed = original_claim(generation, claimed_at)
        claim_results.append(None if claimed is None else claimed.id)
        return claimed

    def execute(
        actual_storage: Storage,
        actual_processor: object,
        running: jobs.DurableJob,
        cancellation: Cancellation,
        *,
        speaker_embedding_policy: SpeakerEmbeddingPolicy,
        now: Callable[[], datetime],
    ) -> None:
        executions.append(running.id)
        assert actual_storage is storage
        assert actual_processor is processor
        assert speaker_embedding_policy is policy
        assert now is _now
        entered.set()
        while not cancellation.cancelled:
            time.sleep(0.005)
        cancellation_seen.set()

    monkeypatch.setattr(storage, "claim_next_job", claim)
    monkeypatch.setattr(runtime, "execute_claimed_job_attempt", execute)
    executor = runtime.JobExecutor(
        storage,
        processor,
        policy,
        GENERATION,
        _now,
    )
    try:
        executor.start()
        executor.wake()
        assert entered.wait(timeout=2)

        assert executions == [first.id]
        assert claim_results == [first.id]
        assert storage.get_visible_job(first.id).status is jobs.JobStatus.RUNNING
        assert storage.get_visible_job(second.id).status is jobs.JobStatus.QUEUED
    finally:
        executor.stop()
        storage.close()
    assert cancellation_seen.is_set()


def test_shutdown_during_claim_does_not_block_or_run_claimed_job(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = _runtime()
    ids = iter(("01234567", "ABCDEFGH"))
    monkeypatch.setattr(storage_module, "generate_job_id", lambda: next(ids))
    storage = _storage(tmp_path)
    first = _queue_job(storage)
    second = _queue_job(storage)
    claim_entered = threading.Event()
    allow_claim = threading.Event()
    marker_entered = threading.Event()
    marker_has_storage = threading.Event()
    allow_marker = threading.Event()
    executions: list[str] = []
    requeue_calls: list[str] = []
    original_claim = storage.claim_next_job
    original_marker = storage.write_shutdown_marker
    original_requeue = storage.requeue_job_at_shutdown

    def claim(generation: str, claimed_at: datetime):
        with storage._lock:
            claim_entered.set()
            if not allow_claim.wait(timeout=2):
                raise RuntimeError("timed out waiting to release claim")
            return original_claim(generation, claimed_at)

    def write_marker(generation: str, created_at: datetime) -> None:
        marker_entered.set()
        with storage._lock:
            marker_has_storage.set()
            if not allow_marker.wait(timeout=2):
                raise RuntimeError("timed out waiting to write marker")
            original_marker(generation, created_at)

    def requeue(
        job_id: str,
        attempt_token: str,
        generation: str,
    ) -> bool:
        requeue_calls.append(job_id)
        return original_requeue(job_id, attempt_token, generation)

    def execute(
        _storage: Storage,
        _processor: object,
        running: jobs.DurableJob,
        _cancellation: Cancellation,
        **_kwargs: Any,
    ) -> None:
        executions.append(running.id)

    monkeypatch.setattr(storage, "claim_next_job", claim)
    monkeypatch.setattr(storage, "write_shutdown_marker", write_marker)
    monkeypatch.setattr(storage, "requeue_job_at_shutdown", requeue)
    monkeypatch.setattr(runtime, "execute_claimed_job_attempt", execute)
    executor = runtime.JobExecutor(
        storage,
        object(),
        _speaker_embedding_policy(),
        GENERATION,
        _now,
    )
    stop_errors: list[BaseException] = []
    stopper = threading.Thread(
        target=lambda: _call_and_capture(executor.stop, stop_errors),
        daemon=True,
    )
    try:
        executor.start()
        executor.wake()
        assert claim_entered.wait(timeout=2)

        begin_started = time.monotonic()
        executor.begin_shutdown()
        assert time.monotonic() - begin_started < 0.5
        assert marker_entered.wait(timeout=2)
        assert not executor.ready
        assert executor._wake_event.is_set()
        assert executions == []

        stopper.start()
        allow_claim.set()
        assert marker_has_storage.wait(timeout=2)
        _wait_until(lambda: executor._active_cancellation is not None)
        with executor._lock:
            cancellation = executor._active_cancellation
        assert cancellation is not None
        assert cancellation.cancelled
        assert executions == []

        allow_marker.set()
        stopper.join(timeout=2)
        assert not stopper.is_alive()
        assert stop_errors == []
        assert executions == []
        assert requeue_calls == [first.id]
        requeued = storage.get_visible_job(first.id)
        assert requeued is not None
        assert requeued.status is jobs.JobStatus.QUEUED
        assert requeued.attempt_no == 1
        waiting = storage.get_visible_job(second.id)
        assert waiting is not None
        assert waiting.status is jobs.JobStatus.QUEUED
        assert waiting.attempt_no == 0
    finally:
        allow_claim.set()
        allow_marker.set()
        if stopper.is_alive():
            stopper.join(timeout=2)
        executor.stop()
        storage.close()


def test_job_executor_starts_idle_then_claims_a_later_job(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = _runtime()
    monkeypatch.setattr(storage_module, "generate_job_id", lambda: "01234567")
    storage = _storage(tmp_path)
    processor = object()
    policy = _speaker_embedding_policy()
    executed = threading.Event()
    executions: list[str] = []
    log_records: list[dict[str, object]] = []

    class RecordingLogger:
        def info(self, _message: str, *, extra: dict[str, object]) -> None:
            log_records.append(extra)

    def execute(
        actual_storage: Storage,
        actual_processor: object,
        running: jobs.DurableJob,
        _cancellation: Cancellation,
        *,
        speaker_embedding_policy: SpeakerEmbeddingPolicy,
        now: Callable[[], datetime],
    ) -> None:
        executions.append(running.id)
        actual_storage.commit_job_failure(
            running.id,
            running.attempt_token,
            "internal_error",
            now(),
        )
        executed.set()

    monkeypatch.setattr(runtime, "execute_claimed_job_attempt", execute)
    monkeypatch.setattr(runtime, "_JOB_LOGGER", RecordingLogger(), raising=False)
    executor = runtime.JobExecutor(
        storage,
        processor,
        policy,
        GENERATION,
        _now,
    )
    try:
        assert not executor.ready
        assert executor.failure is None
        executor.start()
        assert executor.ready

        queued = _queue_job(storage)
        executor.wake()
        assert executed.wait(timeout=2)

        assert executions == [queued.id]
        assert storage.get_visible_job(queued.id).status is jobs.JobStatus.FAILED
        _wait_until(lambda: len(log_records) == 2)
        assert log_records == [
            {
                "event": "job_started",
                "job_id": queued.id,
                "attempt": 1,
                "model": "sensevoice",
                "queue_wait_ms": 60_000.0,
            },
            {
                "event": "job_finished",
                "job_id": queued.id,
                "attempt": 1,
                "model": "sensevoice",
                "status": "failed",
                "error_code": "internal_error",
                "elapsed_ms": 0.0,
                "audio_duration_seconds": None,
            },
        ]
        assert executor.ready
        assert executor.failure is None
    finally:
        executor.stop()
        storage.close()
    assert not executor.ready


def test_job_finished_log_reads_one_canonical_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    monkeypatch.setattr(storage_module, "generate_job_id", lambda: "01234567")
    storage = _storage(tmp_path)
    queued = _queue_job(storage)
    running = storage.claim_next_job(GENERATION, NOW)
    assert running is not None
    succeeded = replace(
        running,
        status=jobs.JobStatus.SUCCEEDED,
        total_samples=16_000,
        processed_samples=16_000,
        attempt_token=None,
        owner_generation=None,
        result_lease_id="result-lease",
        input_cleanup_pending=True,
        finished_at=NOW,
    )
    reads: list[str] = []
    records: list[dict[str, object]] = []

    class CanonicalReader:
        def get_visible_job(self, job_id: str) -> jobs.DurableJob:
            reads.append(job_id)
            return succeeded

    class RecordingLogger:
        def info(self, _message: str, *, extra: dict[str, object]) -> None:
            records.append(extra)

    monkeypatch.setattr(runtime, "_JOB_LOGGER", RecordingLogger(), raising=False)

    runtime._log_finished_job(CanonicalReader(), running)

    assert reads == [queued.id]
    assert records == [
        {
            "event": "job_finished",
            "job_id": queued.id,
            "attempt": 1,
            "model": "sensevoice",
            "status": "succeeded",
            "error_code": None,
            "elapsed_ms": 0.0,
            "audio_duration_seconds": 1.0,
        }
    ]
    storage.close()


@pytest.mark.parametrize("failure_site", ("logger", "readback"))
def test_job_logging_failure_is_best_effort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_site: str,
) -> None:
    runtime = _runtime()
    monkeypatch.setattr(storage_module, "generate_job_id", lambda: "01234567")
    storage = _storage(tmp_path)
    _queue_job(storage)
    running = storage.claim_next_job(GENERATION, NOW)
    assert running is not None

    class FailingLogger:
        def info(self, _message: str, *, extra: dict[str, object]) -> None:
            raise RuntimeError("private logging failure")

    class FailingReader:
        def get_visible_job(self, _job_id: str) -> jobs.DurableJob:
            raise RuntimeError("private readback failure")

    if failure_site == "logger":
        monkeypatch.setattr(runtime, "_JOB_LOGGER", FailingLogger(), raising=False)
        runtime._log_started_job(running)
    else:
        runtime._log_finished_job(FailingReader(), running)

    storage.close()


def test_job_executor_notify_cancellation_only_targets_active_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    ids = iter(("01234567", "ABCDEFGH"))
    monkeypatch.setattr(storage_module, "generate_job_id", lambda: next(ids))
    storage = _storage(tmp_path)
    first = _queue_job(storage)
    second = _queue_job(storage)
    started = {
        first.id: threading.Event(),
        second.id: threading.Event(),
    }
    cancelled = {
        first.id: threading.Event(),
        second.id: threading.Event(),
    }

    def execute(
        actual_storage: Storage,
        _processor: object,
        running: jobs.DurableJob,
        cancellation: Cancellation,
        **_kwargs: Any,
    ) -> None:
        started[running.id].set()
        assert cancellation._event.wait(timeout=2)
        cancelled[running.id].set()
        actual_storage.commit_job_failure(
            running.id,
            running.attempt_token,
            "internal_error",
            NOW,
        )

    monkeypatch.setattr(runtime, "execute_claimed_job_attempt", execute)
    executor = runtime.JobExecutor(
        storage,
        object(),
        _speaker_embedding_policy(),
        GENERATION,
        _now,
    )
    try:
        executor.notify_cancellation(first.id)
        executor.start()
        executor.wake()
        assert started[first.id].wait(timeout=2)
        assert not cancelled[first.id].is_set()

        executor.notify_cancellation(second.id)
        assert not cancelled[first.id].wait(timeout=0.05)
        executor.notify_cancellation(first.id)
        assert cancelled[first.id].wait(timeout=2)
        assert started[second.id].wait(timeout=2)

        executor.notify_cancellation(first.id)
        assert not cancelled[second.id].wait(timeout=0.05)
        executor.notify_cancellation(second.id)
        assert cancelled[second.id].wait(timeout=2)
        _wait_until(
            lambda: storage.get_visible_job(second.id).status
            is jobs.JobStatus.FAILED
        )

        executor.notify_cancellation(second.id)
        assert executor.ready
        assert executor.failure is None
    finally:
        executor.stop()
        storage.close()


def test_running_job_cancellation_removes_inference_waiter_and_executor_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    ids = iter(("01234567", "ABCDEFGH"))
    monkeypatch.setattr(storage_module, "generate_job_id", lambda: next(ids))
    storage = _storage(tmp_path)
    first = _queue_job(storage)
    second = _queue_job(storage)
    lane = SerialInferenceLane()
    holder_entered = threading.Event()
    release_holder = threading.Event()
    first_started = threading.Event()
    late_entry = threading.Event()
    second_started = threading.Event()

    def hold_lane() -> None:
        lane.invoke(
            lambda: (
                holder_entered.set(),
                release_holder.wait(timeout=2),
            )
        )

    class LaneProcessor:
        calls = 0

        def process(self, *_args: Any, **_kwargs: Any) -> object:
            self.calls += 1
            if self.calls == 1:
                first_started.set()
                return lane.invoke(late_entry.set)
            second_started.set()
            raise PipelineError("invalid_audio", "second job finished")

    holder = threading.Thread(target=hold_lane, daemon=True)
    holder.start()
    assert holder_entered.wait(timeout=2)
    processor = TranscriptionProcessorPool((LaneProcessor(),)).async_processor
    executor = runtime.JobExecutor(
        storage,
        processor,
        _speaker_embedding_policy(),
        GENERATION,
        _now,
    )
    try:
        executor.start()
        executor.wake()
        assert first_started.wait(timeout=2)
        outcome = storage.delete_or_cancel_job(first.id, NOW)
        assert outcome is jobs.JobDeletionOutcome.RUNNING_CANCEL_REQUESTED

        executor.notify_cancellation(first.id)
        assert second_started.wait(timeout=2)
        _wait_until(
            lambda: storage.get_visible_job(first.id).status
            is jobs.JobStatus.CANCELLED
        )
        _wait_until(
            lambda: storage.get_visible_job(second.id).status
            is jobs.JobStatus.FAILED
        )

        release_holder.set()
        holder.join(timeout=2)
        assert not holder.is_alive()
        assert not late_entry.is_set()
        assert executor.ready
        assert executor.failure is None
    finally:
        release_holder.set()
        holder.join(timeout=2)
        executor.stop()
        storage.close()


def test_shutdown_local_cancellation_remains_stale_and_requeues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    monkeypatch.setattr(
        storage_module,
        "generate_job_id",
        lambda: "01234567",
    )
    storage = _storage(tmp_path)
    queued = _queue_job(storage)
    entered = threading.Event()

    class CancelOnSignalProcessor:
        def process(
            self,
            _input_path: Path,
            _options: CanonicalOptions,
            cancellation: Cancellation,
            _progress: object,
            _sink: object,
            **_kwargs: Any,
        ) -> object:
            entered.set()
            assert cancellation._event.wait(timeout=2)
            raise PipelineError("cancelled", "shutdown cancellation")

    executor = runtime.JobExecutor(
        storage,
        CancelOnSignalProcessor(),
        _speaker_embedding_policy(),
        GENERATION,
        _now,
    )
    try:
        executor.start()
        executor.wake()
        assert entered.wait(timeout=2)

        executor.stop()

        requeued = storage.get_visible_job(queued.id)
        assert requeued is not None
        assert requeued.status is jobs.JobStatus.QUEUED
        assert requeued.attempt_no == 1
        assert not requeued.cancel_requested
        assert requeued.attempt_token is None
        assert requeued.owner_generation is None
        assert executor.failure is None
    finally:
        executor.stop()
        storage.close()


def test_job_executor_shutdown_fences_claims_cancels_and_requeues_from_zero(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = _runtime()
    ids = iter(("01234567", "ABCDEFGH"))
    monkeypatch.setattr(storage_module, "generate_job_id", lambda: next(ids))
    storage = _storage(tmp_path)
    active = _queue_job(storage)
    waiting = _queue_job(storage)
    policy = _speaker_embedding_policy()
    entered = threading.Event()
    cancellation_seen = threading.Event()
    marker_entered = threading.Event()
    allow_marker = threading.Event()
    allow_runner_return = threading.Event()
    runner_returned = threading.Event()
    executions: list[tuple[str, str]] = []
    marker_calls: list[str] = []
    requeue_calls: list[tuple[str, str, str, bool]] = []
    original_marker = storage.write_shutdown_marker
    original_requeue = storage.requeue_job_at_shutdown

    def execute(
        actual_storage: Storage,
        _processor: object,
        running: jobs.DurableJob,
        cancellation: Cancellation,
        *,
        speaker_embedding_policy: SpeakerEmbeddingPolicy,
        now: Callable[[], datetime],
    ) -> None:
        del speaker_embedding_policy, now
        executions.append((running.id, running.attempt_token))
        actual_storage.update_job_progress(
            running.id,
            running.attempt_token,
            100,
            total_samples=100,
        )
        entered.set()
        while not cancellation.cancelled:
            time.sleep(0.005)
        cancellation_seen.set()
        allow_runner_return.wait(timeout=2)
        runner_returned.set()

    def requeue(
        job_id: str,
        attempt_token: str,
        generation: str,
    ) -> bool:
        requeue_calls.append(
            (
                job_id,
                attempt_token,
                generation,
                runner_returned.is_set(),
            )
        )
        return original_requeue(job_id, attempt_token, generation)

    def write_marker(generation: str, created_at: datetime) -> None:
        marker_calls.append(generation)
        marker_entered.set()
        if not allow_marker.wait(timeout=2):
            raise RuntimeError("timed out waiting to write shutdown marker")
        original_marker(generation, created_at)

    monkeypatch.setattr(runtime, "execute_claimed_job_attempt", execute)
    monkeypatch.setattr(storage, "write_shutdown_marker", write_marker)
    monkeypatch.setattr(storage, "requeue_job_at_shutdown", requeue)
    executor = runtime.JobExecutor(
        storage,
        object(),
        policy,
        GENERATION,
        _now,
    )
    stop_errors: list[BaseException] = []
    stopper = threading.Thread(
        target=lambda: _call_and_capture(executor.stop, stop_errors),
        daemon=True,
    )
    try:
        executor.start()
        executor.wake()
        assert entered.wait(timeout=2)
        running = storage.get_visible_job(active.id)
        assert running is not None
        assert running.attempt_token is not None

        begin_started = time.monotonic()
        executor.begin_shutdown()
        assert time.monotonic() - begin_started < 0.5
        executor.begin_shutdown()
        assert marker_entered.wait(timeout=2)
        assert cancellation_seen.wait(timeout=2)
        assert not executor.ready
        assert executor._wake_event.is_set()
        assert marker_calls == [GENERATION]
        assert storage.get_visible_job(waiting.id).status is jobs.JobStatus.QUEUED
        assert executions == [(active.id, running.attempt_token)]

        stopper.start()
        time.sleep(0.05)
        assert stopper.is_alive()

        allow_marker.set()
        time.sleep(0.05)
        assert stopper.is_alive()
        with storage._lock:
            marker = storage._connection.execute(
                "SELECT generation FROM shutdown_marker"
            ).fetchone()
        assert tuple(marker) == (GENERATION,)
        assert storage.get_visible_job(waiting.id).status is jobs.JobStatus.QUEUED

        allow_runner_return.set()
        stopper.join(timeout=2)
        assert not stopper.is_alive()
        assert stop_errors == []

        requeued = storage.get_visible_job(active.id)
        assert requeued is not None
        assert requeued.status is jobs.JobStatus.QUEUED
        assert requeued.processed_samples == 0
        assert requeued.total_samples == 100
        assert requeued.attempt_no == 1
        assert requeued.crash_recoveries == 0
        assert requeued.attempt_token is None
        assert requeued.owner_generation is None
        still_waiting = storage.get_visible_job(waiting.id)
        assert still_waiting is not None
        assert still_waiting.status is jobs.JobStatus.QUEUED
        assert still_waiting.attempt_no == 0
        assert executions == [(active.id, running.attempt_token)]
        assert requeue_calls == [
            (
                active.id,
                running.attempt_token,
                GENERATION,
                True,
            )
        ]
        with storage._lock:
            artifact_count = storage._connection.execute(
                """
                SELECT COUNT(*) FROM storage_leases
                WHERE lease_type = 'artifact'
                  AND owner_kind = 'job' AND owner_id = ?
                """,
                (active.id,),
            ).fetchone()[0]
        assert artifact_count == 0
    finally:
        allow_marker.set()
        allow_runner_return.set()
        if stopper.is_alive():
            stopper.join(timeout=2)
        executor.stop()
        storage.close()


def test_job_executor_integrity_failure_is_visible_and_stops_claiming(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = _runtime()
    ids = iter(("01234567", "ABCDEFGH", "7K3M9Q2W", "JKMNPQRT"))
    monkeypatch.setattr(storage_module, "generate_job_id", lambda: next(ids))
    log_records: list[dict[str, object]] = []

    class RecordingLogger:
        def info(self, _message: str, *, extra: dict[str, object]) -> None:
            log_records.append(extra)

    for failure_site in ("claim", "runner"):
        log_records.clear()
        storage = _storage(tmp_path / failure_site)
        first = _queue_job(storage)
        second = _queue_job(storage)
        failure = StorageSchemaError(f"{failure_site} integrity failure")
        claim_calls: list[str] = []
        original_claim = storage.claim_next_job

        def claim(
            generation: str,
            claimed_at: datetime,
            *,
            _site: str = failure_site,
        ):
            claim_calls.append(_site)
            if _site == "claim":
                raise failure
            return original_claim(generation, claimed_at)

        def execute(*_args: Any, **_kwargs: Any) -> None:
            raise failure

        monkeypatch.setattr(storage, "claim_next_job", claim)
        monkeypatch.setattr(runtime, "execute_claimed_job_attempt", execute)
        monkeypatch.setattr(
            runtime,
            "_JOB_LOGGER",
            RecordingLogger(),
            raising=False,
        )
        executor = runtime.JobExecutor(
            storage,
            object(),
            _speaker_embedding_policy(),
            GENERATION,
            _now,
        )
        try:
            executor.start()
            executor.wake()
            _wait_until(lambda: executor.failure is failure)

            assert not executor.ready
            assert executor.failure is failure
            executor.wake()
            time.sleep(0.05)
            assert claim_calls == [failure_site]
            if failure_site == "runner":
                _wait_until(lambda: len(log_records) == 2)
                assert log_records[0]["event"] == "job_started"
                assert log_records[1] == {
                    "event": "job_executor_failed",
                    "job_id": first.id,
                    "exception_type": "StorageSchemaError",
                }
                assert "integrity failure" not in repr(log_records)
                assert (
                    storage.get_visible_job(first.id).status
                    is jobs.JobStatus.RUNNING
                )
                assert (
                    storage.get_visible_job(second.id).status
                    is jobs.JobStatus.QUEUED
                )
            else:
                assert (
                    storage.get_visible_job(first.id).status
                    is jobs.JobStatus.QUEUED
                )
                assert (
                    storage.get_visible_job(second.id).status
                    is jobs.JobStatus.QUEUED
                )
        finally:
            executor.stop()
            storage.close()


def test_job_executor_marker_failure_cancels_active_and_is_idempotent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = _runtime()
    monkeypatch.setattr(storage_module, "generate_job_id", lambda: "01234567")
    storage = _storage(tmp_path)
    queued = _queue_job(storage)
    sentinel = OSError("shutdown marker failed")
    entered = threading.Event()
    cancellation_seen = threading.Event()
    marker_entered = threading.Event()
    allow_marker_failure = threading.Event()
    marker_calls: list[str] = []
    requeue_calls: list[str] = []

    def execute(
        _storage: Storage,
        _processor: object,
        _running: jobs.DurableJob,
        cancellation: Cancellation,
        **_kwargs: Any,
    ) -> None:
        entered.set()
        deadline = time.monotonic() + 0.5
        while not cancellation.cancelled and time.monotonic() < deadline:
            time.sleep(0.005)
        if cancellation.cancelled:
            cancellation_seen.set()

    def fail_marker(generation: str, _created_at: datetime) -> None:
        marker_calls.append(generation)
        marker_entered.set()
        if not allow_marker_failure.wait(timeout=2):
            raise RuntimeError("timed out waiting to fail shutdown marker")
        raise sentinel

    def observe_requeue(*_args: Any, **_kwargs: Any) -> bool:
        requeue_calls.append("called")
        return False

    monkeypatch.setattr(runtime, "execute_claimed_job_attempt", execute)
    monkeypatch.setattr(storage, "write_shutdown_marker", fail_marker)
    monkeypatch.setattr(
        storage,
        "requeue_job_at_shutdown",
        observe_requeue,
    )
    executor = runtime.JobExecutor(
        storage,
        object(),
        _speaker_embedding_policy(),
        GENERATION,
        _now,
    )
    stop_errors: list[BaseException] = []
    stoppers = tuple(
        threading.Thread(
            target=lambda: _call_and_capture(executor.stop, stop_errors),
            daemon=True,
        )
        for _ in range(2)
    )
    try:
        executor.start()
        executor.wake()
        assert entered.wait(timeout=2)

        for stopper in stoppers:
            stopper.start()
        assert marker_entered.wait(timeout=2)
        assert cancellation_seen.wait(timeout=2)
        assert all(stopper.is_alive() for stopper in stoppers)

        allow_marker_failure.set()
        for stopper in stoppers:
            stopper.join(timeout=2)
        assert all(not stopper.is_alive() for stopper in stoppers)

        assert stop_errors == [sentinel, sentinel]
        assert marker_calls == [GENERATION]
        assert requeue_calls == []
        current = storage.get_visible_job(queued.id)
        assert current is not None
        assert current.status is jobs.JobStatus.RUNNING
        with storage._lock:
            marker_count = storage._connection.execute(
                "SELECT COUNT(*) FROM shutdown_marker"
            ).fetchone()[0]
        assert marker_count == 0
    finally:
        allow_marker_failure.set()
        for stopper in stoppers:
            if stopper.is_alive():
                stopper.join(timeout=2)
        storage.close()


def test_job_executor_marker_thread_start_failure_unblocks_stop(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = _runtime()
    storage = _storage(tmp_path)
    sentinel = RuntimeError("marker thread did not start")
    executor = runtime.JobExecutor(
        storage,
        object(),
        _speaker_embedding_policy(),
        GENERATION,
        _now,
    )

    class FailingMarkerThread:
        def __init__(self, *, target: object, name: str) -> None:
            assert callable(target)
            assert name == "botified-asr-shutdown-marker"

        def start(self) -> None:
            raise sentinel

    try:
        executor.start()
        monkeypatch.setattr(runtime.threading, "Thread", FailingMarkerThread)

        executor.begin_shutdown()

        assert not executor.ready
        with pytest.raises(RuntimeError) as first:
            executor.stop()
        with pytest.raises(RuntimeError) as repeated:
            executor.stop()
        assert first.value is sentinel
        assert repeated.value is sentinel
    finally:
        storage.close()


def test_job_executor_stop_propagates_runner_integrity_error_after_requeue(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = _runtime()
    monkeypatch.setattr(storage_module, "generate_job_id", lambda: "01234567")
    storage = _storage(tmp_path)
    queued = _queue_job(storage)
    sentinel = StorageSchemaError("runner failed while stopping")
    entered = threading.Event()
    cancellation_seen = threading.Event()
    requeue_calls: list[tuple[str, str, str]] = []
    original_requeue = storage.requeue_job_at_shutdown

    def execute(
        _storage: Storage,
        _processor: object,
        _running: jobs.DurableJob,
        cancellation: Cancellation,
        **_kwargs: Any,
    ) -> None:
        entered.set()
        while not cancellation.cancelled:
            time.sleep(0.005)
        cancellation_seen.set()
        raise sentinel

    def observe_requeue(
        job_id: str,
        attempt_token: str,
        generation: str,
    ) -> bool:
        requeue_calls.append((job_id, attempt_token, generation))
        return original_requeue(job_id, attempt_token, generation)

    monkeypatch.setattr(runtime, "execute_claimed_job_attempt", execute)
    monkeypatch.setattr(
        storage,
        "requeue_job_at_shutdown",
        observe_requeue,
    )
    executor = runtime.JobExecutor(
        storage,
        object(),
        _speaker_embedding_policy(),
        GENERATION,
        _now,
    )
    try:
        executor.start()
        executor.wake()
        assert entered.wait(timeout=2)
        running = storage.get_visible_job(queued.id)
        assert running is not None
        assert running.attempt_token is not None

        with pytest.raises(StorageSchemaError) as first:
            executor.stop()
        with pytest.raises(StorageSchemaError) as repeated:
            executor.stop()

        assert first.value is sentinel
        assert repeated.value is sentinel
        assert cancellation_seen.is_set()
        assert requeue_calls == [
            (queued.id, running.attempt_token, GENERATION)
        ]
        requeued = storage.get_visible_job(queued.id)
        assert requeued is not None
        assert requeued.status is jobs.JobStatus.QUEUED
        assert requeued.processed_samples == 0
        assert requeued.attempt_no == 1
        assert requeued.crash_recoveries == 0
    finally:
        storage.close()


def _call_and_capture(
    operation: Callable[[], None],
    errors: list[BaseException],
) -> None:
    try:
        operation()
    except BaseException as error:
        errors.append(error)
