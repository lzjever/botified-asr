from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Callable

import anyio
import pytest
from starlette.testclient import TestClient

from botified_asr.api import Readiness, create_app
from botified_asr.jobs import DurableJob, JobPhase, JobStatus
from botified_asr.result_artifact import CanonicalArtifactError
from botified_asr.speakers import SpeakerEmbeddingPolicy


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
        total_samples=32_000,
        processed_samples=(
            32_000 if succeeded else 16_000 if running else 0
        ),
        request_fingerprint="2" * 64,
        processor_fingerprint="3" * 64,
        attempt_no=0 if status is JobStatus.QUEUED else 1,
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
    ) -> None:
        self.job = job
        self.malformed = malformed
        self.result = result
        self.open_error = open_error
        self.get_calls = 0
        self.open_calls = 0
        self.release_calls = 0

    def get_visible_job(self, _: str) -> DurableJob | None:
        self.get_calls += 1
        if self.malformed:
            raise ValueError("invalid job id")
        return self.job

    def open_succeeded_job_result(self, _: str) -> FakeStoredResult:
        self.open_calls += 1
        if self.open_error is not None:
            raise self.open_error
        assert self.result is not None
        return self.result

    def release_artifact(self, _: object) -> None:
        self.release_calls += 1


def app(storage: FakeStorage, readiness: Readiness | None = None):
    return create_app(
        api_key="test-secret",
        readiness=readiness or Readiness(True, True, True),
        storage=storage,
        processor=object(),
        speaker_embedding_policy=embedding_policy(),
        close_storage_on_shutdown=False,
    )


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
