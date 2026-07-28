from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from starlette.testclient import TestClient

import botified_asr.api as api_module
import botified_asr.storage as storage_module
from botified_asr.api import Readiness, create_app
from botified_asr.audio import AudioError, Cancellation, MediaProbe
from botified_asr.config import LimitsConfig, RESERVATION_QUANTUM
from botified_asr.speakers import SpeakerEmbeddingPolicy
from botified_asr.storage import Storage


AUTH = {
    "Authorization": "Bearer test-secret",
    "Prefer": "respond-async",
}
CREATED_AT = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
PROCESSOR_FINGERPRINT = "3" * 64
OPTIONS_JSON = (
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
        enrollment_aggregation_policy_version=("sample-centroid-equal-average-v1"),
    )


def make_storage(path: Path, *, max_queued_jobs: int = 4) -> Storage:
    return Storage(
        path,
        LimitsConfig(
            max_upload_bytes=1024,
            max_audio_duration_secs=90,
            direct_max_audio_duration_secs=7,
            sync_max_upload_bytes=512,
            sync_max_audio_duration_secs=60,
            max_queued_jobs=max_queued_jobs,
            max_job_storage_bytes=2 * RESERVATION_QUANTUM,
            min_filesystem_free_bytes=1,
        ),
        current_processor_fingerprint="3" * 64,
        free_bytes=lambda _: 1 << 40,
    )


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    value = make_storage(tmp_path)
    yield value
    value.close()


class RecordingProber:
    def __init__(
        self,
        *,
        duration: float = 1,
        error: AudioError | None = None,
    ) -> None:
        self.duration = duration
        self.error = error
        self.calls: list[tuple[Path, Cancellation, bytes]] = []
        self.decode_calls = 0

    def __call__(
        self,
        path: Path,
        cancellation: Cancellation,
    ) -> MediaProbe:
        self.calls.append((path, cancellation, path.read_bytes()))
        if self.error is not None:
            raise self.error
        return MediaProbe(self.duration, "wav")

    def decode(self, *_args: object, **_kwargs: object) -> None:
        self.decode_calls += 1
        raise AssertionError("async submission must not decode")


class RejectingProcessor:
    def __init__(self) -> None:
        self.calls = 0

    def process(self, *_args: object, **_kwargs: object) -> None:
        self.calls += 1
        raise AssertionError("async submission must not invoke the processor")


def app_client(
    storage: Storage,
    prober: RecordingProber,
    *,
    job_executor: object | None = None,
) -> tuple[TestClient, RejectingProcessor]:
    processor = RejectingProcessor()
    executor_options = (
        {}
        if job_executor is None
        else {"job_executor": job_executor}
    )
    app = create_app(
        api_key="test-secret",
        readiness=Readiness(True, True, True),
        storage=storage,
        processor=processor,
        audio_prober=prober,
        processor_fingerprint=PROCESSOR_FINGERPRINT,
        speaker_embedding_policy=embedding_policy(),
        close_storage_on_shutdown=False,
        **executor_options,
    )
    return TestClient(app, raise_server_exceptions=False), processor


def post_async(
    client: TestClient,
    *,
    audio: bytes = b"audio",
    model: str = "sensevoice",
    extra_fields: tuple[tuple[str, str], ...] = (),
):
    files = [
        ("file", ("audio.wav", audio)),
        ("model", (None, model)),
        *((name, (None, value)) for name, value in extra_fields),
    ]
    return client.post(
        "/v1/audio/transcriptions",
        headers=AUTH,
        files=files,
    )


def assert_cleaned(storage: Storage) -> None:
    assert (
        storage._connection.execute(
            "SELECT COUNT(*) FROM transcription_jobs"
        ).fetchone()[0]
        == 0
    )
    assert (
        storage._connection.execute("SELECT COUNT(*) FROM storage_leases").fetchone()[0]
        == 0
    )
    assert storage.total_reserved_bytes() == 0
    assert not list(storage.staging_dir.iterdir())


def assert_probe_only(
    prober: RecordingProber,
    processor: RejectingProcessor,
    *,
    payload: bytes = b"audio",
) -> None:
    assert len(prober.calls) == 1
    path, cancellation, seen = prober.calls[0]
    assert path.suffix == ".ready"
    assert seen == payload
    assert isinstance(cancellation, Cancellation)
    assert not cancellation.cancelled
    assert prober.decode_calls == 0
    assert processor.calls == 0


def test_async_post_returns_exact_202_and_retains_queued_input(
    storage: Storage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            assert tz is timezone.utc
            return CREATED_AT

    monkeypatch.setattr(api_module, "datetime", FixedDateTime)
    monkeypatch.setattr(
        storage_module,
        "generate_job_id",
        lambda: "7K3M9Q2W",
    )
    prober = RecordingProber()
    wake_calls = 0

    class FakeJobExecutor:
        ready = True
        failure = None

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def wake(self) -> None:
            nonlocal wake_calls
            wake_calls += 1

    client, processor = app_client(
        storage,
        prober,
        job_executor=FakeJobExecutor(),
    )
    with client:
        response = post_async(client)

    assert response.status_code == 202
    assert response.json() == {
        "id": "7K3M9Q2W",
        "status": "queued",
        "created_at": "2026-07-27T12:00:00Z",
    }
    assert response.headers["preference-applied"] == "respond-async"
    assert response.headers["location"] == ("/v1/audio/transcriptions/7K3M9Q2W")
    queued = storage.get_visible_job("7K3M9Q2W")
    assert queued.status.value == "queued"
    assert queued.total_samples is None
    assert queued.canonical_options_json == OPTIONS_JSON
    assert queued.effective_max_audio_samples == 90 * 16_000
    assert queued.effective_direct_max_audio_samples == 7 * 16_000
    assert queued.processor_fingerprint == PROCESSOR_FINGERPRINT
    assert (storage.staging_dir / "7K3M9Q2W.ready").read_bytes() == b"audio"
    assert storage.total_reserved_bytes() == len(b"audio")
    assert_probe_only(prober, processor)
    assert wake_calls == 1


@pytest.mark.parametrize(
    ("prober", "extra_fields", "status", "code"),
    (
        (
            RecordingProber(error=AudioError("invalid_audio", "private detail")),
            (),
            400,
            "invalid_audio",
        ),
        (
            RecordingProber(error=AudioError("audio_probe_timeout", "private detail")),
            (),
            503,
            "audio_probe_timeout",
        ),
        (RecordingProber(duration=8), (), 422, "long_audio_requires_vad"),
        (RecordingProber(duration=91), (), 413, "audio_too_long"),
        (
            RecordingProber(duration=91),
            (("chunking_strategy", "auto"),),
            413,
            "audio_too_long",
        ),
    ),
)
def test_async_preflight_errors_abort_the_sealed_receiving_job(
    storage: Storage,
    prober: RecordingProber,
    extra_fields: tuple[tuple[str, str], ...],
    status: int,
    code: str,
) -> None:
    client, processor = app_client(storage, prober)
    with client:
        response = post_async(client, extra_fields=extra_fields)

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert_probe_only(prober, processor)
    assert_cleaned(storage)


def test_async_missing_speaker_is_404_and_cleans_input(
    storage: Storage,
) -> None:
    prober = RecordingProber()
    client, processor = app_client(storage, prober)
    with client:
        response = post_async(
            client,
            model="sensevoice-diarize",
            extra_fields=(
                ("chunking_strategy", "auto"),
                ("response_format", "diarized_json"),
                ("known_speaker_ids[]", "00000001"),
            ),
        )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "speaker_not_found"
    assert_probe_only(prober, processor)
    assert_cleaned(storage)


def test_async_queue_admission_cleans_only_the_rejected_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = make_storage(tmp_path, max_queued_jobs=1)
    job_ids = iter(("7K3M9Q2W", "8D2N6P3R"))
    monkeypatch.setattr(storage_module, "generate_job_id", lambda: next(job_ids))
    prober = RecordingProber()
    client, processor = app_client(storage, prober)
    try:
        with client:
            accepted = post_async(client)
            rejected = post_async(client)

        assert accepted.status_code == 202
        assert rejected.status_code == 429
        assert rejected.json()["error"]["code"] == "too_many_queued_jobs"
        assert processor.calls == 0
        assert prober.decode_calls == 0
        assert [call[0].suffix for call in prober.calls] == [".ready", ".ready"]
        assert storage.get_visible_job("7K3M9Q2W") is not None
        assert storage.get_visible_job("8D2N6P3R") is None
        assert storage.total_reserved_bytes() == len(b"audio")
        assert sorted(path.name for path in storage.staging_dir.iterdir()) == [
            "7K3M9Q2W.ready"
        ]
    finally:
        storage.close()


def test_async_publish_fault_cleans_the_sealed_receiving_job(
    storage: Storage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prober = RecordingProber()
    client, processor = app_client(storage, prober)

    def fail_publish(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected private publish failure")

    monkeypatch.setattr(storage, "publish_job", fail_publish)
    with client:
        response = post_async(client)

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "injected" not in response.text
    assert_probe_only(prober, processor)
    assert_cleaned(storage)
