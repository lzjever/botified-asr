from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Callable, Iterable

import anyio
import pytest
from starlette.testclient import TestClient

from botified_asr import pipeline as pipeline_module, speakers
import botified_asr.api as api_module
from botified_asr.api import (
    ApiError,
    Readiness,
    canonicalize_options,
    create_app,
)
from botified_asr.config import LimitsConfig, RESERVATION_QUANTUM
from botified_asr.contracts import CanonicalOptions
from botified_asr.pipeline import RichAnnotations, SegmentRecord
from botified_asr.speaker_matching import SpeakerLabelMapping
from botified_asr.speaker_snapshot import SelectedSpeakerSnapshot
from botified_asr.storage import Storage


AUTH = {"Authorization": "Bearer test-secret"}
PROCESSOR_FINGERPRINT = "3" * 64


def _speaker_embedding_policy() -> speakers.SpeakerEmbeddingPolicy:
    return speakers.SpeakerEmbeddingPolicy(
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


class FakeProcessor:
    def __init__(
        self,
        callback: Callable[[Path, CanonicalOptions], str] | None = None,
    ) -> None:
        self.callback = callback or (lambda _path, _options: "ok")

    def process(
        self,
        input_path,
        options,
        _cancellation,
        progress,
        sink,
        *,
        selected_speaker_snapshot: SelectedSpeakerSnapshot,
        effective_max_audio_samples: int,
        effective_direct_max_audio_samples: int,
    ):
        del (
            selected_speaker_snapshot,
            effective_max_audio_samples,
            effective_direct_max_audio_samples,
        )
        text = self.callback(input_path, options)
        processed_samples = 1 if text else 0
        if text:
            sink.append(
                SegmentRecord(
                    0,
                    0,
                    1,
                    text,
                    "en",
                    RichAnnotations(),
                )
            )
        progress.update(
            processed_samples=processed_samples,
            total_samples=None,
        )
        progress.update(
            processed_samples=processed_samples,
            total_samples=processed_samples,
        )
        ref = sink.finalize()
        return pipeline_module.ProcessorResult(
            ref,
            SpeakerLabelMapping(()),
        )


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    value = Storage(
        tmp_path,
        LimitsConfig(
            max_upload_bytes=16,
            sync_max_upload_bytes=8,
            max_job_storage_bytes=2 * RESERVATION_QUANTUM,
            min_filesystem_free_bytes=1,
        ),
        current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40,
    )
    yield value
    value.close()


def app_client(
    storage: Storage,
    *,
    readiness: Readiness | None = None,
    processor=None,
) -> TestClient:
    app = create_app(
        api_key="test-secret",
        readiness=readiness or Readiness(True, True, True),
        storage=storage,
        processor=processor or FakeProcessor(),
        audio_prober=lambda _path, _cancellation: None,
        processor_fingerprint="3" * 64,
        speaker_embedding_policy=_speaker_embedding_policy(),
        close_storage_on_shutdown=False,
    )
    return TestClient(app, raise_server_exceptions=False)


def test_live_is_the_only_unauthenticated_route(storage: Storage) -> None:
    with app_client(storage) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 401
        assert client.get("/v1/models").status_code == 401
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404


@pytest.mark.parametrize(
    "authorization",
    [
        b"Bearer \xff",
        b"Bearer test-secret\x00",
        b"\xffBearer test-secret",
    ],
)
def test_malformed_authorization_is_stable_and_redacted(
    storage: Storage, authorization: bytes
) -> None:
    app = create_app(
        api_key="test-secret",
        readiness=Readiness(True, True, True),
        storage=storage,
        processor=FakeProcessor(
            lambda _path, _options: "unexpected"
        ),
        audio_prober=lambda _path, _cancellation: None,
        processor_fingerprint="3" * 64,
        speaker_embedding_policy=_speaker_embedding_policy(),
        close_storage_on_shutdown=False,
    )

    status, body, _ = invoke_raw_asgi(
        app,
        path="/v1/models",
        headers=[(b"authorization", authorization)],
        events=[],
    )

    assert status == 401
    assert body == {
        "error": {
            "message": "Invalid authentication credentials",
            "type": "authentication_error",
            "param": None,
            "code": "invalid_api_key",
        }
    }
    assert "test-secret" not in json.dumps(body)


def test_ready_is_503_until_database_models_and_executor_are_ready(
    storage: Storage,
) -> None:
    with app_client(storage, readiness=Readiness(True, False, True)) as client:
        response = client.get("/health/ready", headers=AUTH)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "service_not_ready"


def test_job_executor_lifespan_drives_dynamic_readiness_and_closes_in_order(
) -> None:
    events: list[str] = []

    class FakeStorage:
        def close(self) -> None:
            events.append("storage.close")

    class FakeJobExecutor:
        ready = False
        failure = None

        def start(self) -> None:
            events.append("executor.start")
            self.ready = True

        def stop(self) -> None:
            events.append("executor.stop")
            self.ready = False

        def wake(self) -> None:
            raise AssertionError("health checks must not wake the executor")

    executor = FakeJobExecutor()
    readiness = Readiness(True, True, False)
    app = create_app(
        api_key="test-secret",
        readiness=readiness,
        storage=FakeStorage(),  # type: ignore[arg-type]
        processor=FakeProcessor(),
        audio_prober=lambda _path, _cancellation: None,
        processor_fingerprint=PROCESSOR_FINGERPRINT,
        speaker_embedding_policy=_speaker_embedding_policy(),
        job_executor=executor,  # type: ignore[arg-type]
        close_storage_on_shutdown=True,
    )

    with TestClient(app) as client:
        assert client.get("/health/ready", headers=AUTH).status_code == 200
        readiness.executor = False
        executor.ready = True
        assert client.get("/health/ready", headers=AUTH).status_code == 503
        readiness.executor = True
        executor.ready = False
        assert client.get("/health/ready", headers=AUTH).status_code == 503

    assert events == [
        "executor.start",
        "executor.stop",
        "storage.close",
    ]


@pytest.mark.parametrize(
    ("authorization", "readiness", "expected_status"),
    [
        (None, Readiness(True, True, True), 401),
        (b"Bearer test-secret", Readiness(True, False, True), 503),
    ],
)
def test_auth_and_not_ready_do_not_receive_body_or_create_lease(
    storage: Storage,
    authorization: bytes | None,
    readiness: Readiness,
    expected_status: int,
) -> None:
    app = create_app(
        api_key="test-secret",
        readiness=readiness,
        storage=storage,
        processor=FakeProcessor(
            lambda _path, _options: "unexpected"
        ),
        audio_prober=lambda _path, _cancellation: None,
        processor_fingerprint="3" * 64,
        speaker_embedding_policy=_speaker_embedding_policy(),
        close_storage_on_shutdown=False,
    )
    headers = (
        []
        if authorization is None
        else [(b"authorization", authorization)]
    )

    status, _, receive_calls = invoke_raw_asgi(
        app,
        path="/v1/audio/transcriptions",
        headers=headers,
        events=None,
    )

    assert status == expected_status
    assert receive_calls == 0
    assert storage.active_upload_count() == 0
    assert storage.total_reserved_bytes() == 0


def test_models_are_fixed_and_created_is_build_constant(storage: Storage) -> None:
    with app_client(storage) as client:
        first = client.get("/v1/models", headers=AUTH)
        second = client.get("/v1/models", headers=AUTH)
        one = client.get("/v1/models/sensevoice", headers=AUTH)
        missing = client.get("/v1/models/arbitrary", headers=AUTH)

    assert first.status_code == 200
    assert first.json() == second.json()
    assert [item["id"] for item in first.json()["data"]] == [
        "sensevoice",
        "sensevoice-diarize",
    ]
    assert {item["created"] for item in first.json()["data"]} == {1785024000}
    assert one.json() == first.json()["data"][0]
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "model_not_found"


def test_errors_use_openai_envelope_without_internal_exception(
    storage: Storage,
) -> None:
    def explode(_path: Path, _options: CanonicalOptions) -> str:
        raise RuntimeError("host path /secret and traceback")

    with app_client(storage, processor=FakeProcessor(explode)) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            headers=AUTH,
            files={"file": ("audio.wav", b"audio"), "model": (None, "sensevoice")},
        )

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "message": "Internal server error",
            "type": "server_error",
            "param": None,
            "code": "internal_error",
        }
    }
    assert "secret" not in response.text
    assert storage.active_upload_count() == 0
    assert storage.total_reserved_bytes() == 0
    assert not list(storage.staging_dir.iterdir())


def test_diarization_fields_are_conditionally_required() -> None:
    with pytest.raises(ApiError) as missing_chunking:
        canonicalize_options(
            {"model": ["sensevoice-diarize"], "response_format": ["diarized_json"]}
        )
    assert missing_chunking.value.param == "chunking_strategy"

    with pytest.raises(ApiError) as missing_format:
        canonicalize_options(
            {"model": ["sensevoice-diarize"], "chunking_strategy": ["auto"]}
        )
    assert missing_format.value.param == "response_format"


def test_text_with_include_is_rejected() -> None:
    with pytest.raises(ApiError) as caught:
        canonicalize_options(
            {
                "model": ["sensevoice"],
                "response_format": ["text"],
                "include[]": ["funasr.emotion"],
            }
        )
    assert caught.value.code == "incompatible_response_format"


def test_known_speakers_require_diarization() -> None:
    with pytest.raises(ApiError) as caught:
        canonicalize_options(
            {"model": ["sensevoice"], "known_speaker_ids[]": ["4X7K2M9Q"]}
        )
    assert caught.value.param == "known_speaker_ids[]"


def test_arrays_have_stable_canonicalization() -> None:
    options = canonicalize_options(
        {
            "model": ["sensevoice-diarize"],
            "chunking_strategy": ["auto"],
            "response_format": ["diarized_json"],
            "include[]": [
                "funasr.audio_events",
                "funasr.emotion",
                "funasr.audio_events",
            ],
            "known_speaker_ids[]": ["9X7K2M9Q", "4X7K2M9Q"],
        }
    )
    assert options.include == ("funasr.emotion", "funasr.audio_events")
    assert options.known_speaker_ids == ("4X7K2M9Q", "9X7K2M9Q")

    with pytest.raises(ApiError) as duplicate:
        canonicalize_options(
            {
                "model": ["sensevoice-diarize"],
                "chunking_strategy": ["auto"],
                "response_format": ["diarized_json"],
                "known_speaker_ids[]": ["4X7K2M9Q", "4X7K2M9Q"],
            }
        )
    assert duplicate.value.code == "invalid_known_speaker_ids"

    with pytest.raises(ApiError) as unknown:
        canonicalize_options({"model": ["sensevoice"], "bogus": ["value"]})
    assert unknown.value.code == "invalid_multipart"


def test_basic_multipart_returns_text_and_cleans_staging(storage: Storage) -> None:
    captured: dict[str, object] = {}

    def transcribe(path: Path, options: CanonicalOptions) -> str:
        captured["bytes"] = path.read_bytes()
        captured["options"] = options
        return "hello"

    with app_client(
        storage,
        processor=FakeProcessor(transcribe),
    ) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            headers=AUTH,
            files={"file": ("audio.wav", b"12345678"), "model": (None, "sensevoice")},
        )

    assert response.status_code == 200
    assert response.json() == {"text": "hello"}
    assert captured["bytes"] == b"12345678"
    assert storage.total_reserved_bytes() == 0
    assert not list(storage.staging_dir.iterdir())


def test_text_response_is_plain_text(storage: Storage) -> None:
    with app_client(
        storage,
        processor=FakeProcessor(
            lambda _path, _options: "plain"
        ),
    ) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            headers=AUTH,
            files={
                "file": ("audio.wav", b"1234"),
                "model": (None, "sensevoice"),
                "response_format": (None, "text"),
            },
        )

    assert response.status_code == 200
    assert response.text == "plain"
    assert response.headers["content-type"] == "text/plain; charset=utf-8"


@pytest.mark.parametrize(
    "files",
    [
        [
            ("file", ("one.wav", b"1")),
            ("file", ("two.wav", b"2")),
            ("model", (None, "sensevoice")),
        ],
        [
            ("file", ("one.wav", b"1")),
            ("model", (None, "sensevoice")),
            ("model", (None, "sensevoice")),
        ],
        [
            ("file", ("one.wav", b"1")),
            ("model", (None, "sensevoice")),
            ("bogus", (None, "value")),
        ],
    ],
)
def test_invalid_multipart_is_rejected_and_cleaned(
    storage: Storage, files: list[tuple[str, tuple]]
) -> None:
    with app_client(storage) as client:
        response = client.post(
            "/v1/audio/transcriptions", headers=AUTH, files=files
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_multipart"
    assert storage.total_reserved_bytes() == 0
    assert not list(storage.staging_dir.iterdir())


def test_streaming_byte_boundaries_are_stable(storage: Storage) -> None:
    with app_client(storage) as client:
        sync = client.post(
            "/v1/audio/transcriptions",
            headers=AUTH,
            files={"file": ("audio.wav", b"123456789"), "model": (None, "sensevoice")},
        )
        async_hard = client.post(
            "/v1/audio/transcriptions",
            headers={**AUTH, "Prefer": "respond-async"},
            files={"file": ("audio.wav", b"1" * 17), "model": (None, "sensevoice")},
        )

    assert sync.status_code == 422
    assert sync.json()["error"]["code"] == "async_required"
    assert async_hard.status_code == 413
    assert async_hard.json()["error"]["code"] == "upload_too_large"
    assert storage.total_reserved_bytes() == 0


def test_equal_sync_and_hard_limit_returns_413(tmp_path: Path) -> None:
    storage = Storage(
        tmp_path,
        LimitsConfig(
            max_upload_bytes=8,
            sync_max_upload_bytes=8,
            max_job_storage_bytes=8 * 1024 * 1024,
            min_filesystem_free_bytes=1,
        ),
        current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40,
    )
    try:
        with app_client(storage) as client:
            response = client.post(
                "/v1/audio/transcriptions",
                headers=AUTH,
                files={
                    "file": ("audio.wav", b"123456789"),
                    "model": (None, "sensevoice"),
                },
            )
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "upload_too_large"
    finally:
        storage.close()


def test_part_header_limit_is_enforced(storage: Storage) -> None:
    with app_client(storage) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            headers=AUTH,
            files={
                "file": ("x" * (32 * 1024) + ".wav", b"1"),
                "model": (None, "sensevoice"),
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_multipart"
    assert storage.total_reserved_bytes() == 0


def test_part_count_limit_is_enforced(storage: Storage) -> None:
    files = [
        ("file", ("audio.wav", b"1")),
        ("model", (None, "sensevoice")),
        *[("include[]", (None, "funasr.emotion")) for _ in range(63)],
    ]
    with app_client(storage) as client:
        response = client.post(
            "/v1/audio/transcriptions", headers=AUTH, files=files
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_multipart"
    assert storage.total_reserved_bytes() == 0


def test_multipart_overhead_has_an_independent_limit(storage: Storage) -> None:
    with app_client(storage) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            headers=AUTH,
            files={
                "file": ("audio.wav", b"1"),
                "model": (None, "sensevoice"),
                "language": (None, "x" * (1024 * 1024)),
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_multipart"
    assert storage.total_reserved_bytes() == 0


def test_same_byte_stream_is_independent_of_asgi_chunking(storage: Storage) -> None:
    boundary = b"stable-boundary"
    body = multipart_body(boundary, b"123456789")
    app = create_app(
        api_key="test-secret",
        readiness=Readiness(True, True, True),
        storage=storage,
        processor=FakeProcessor(
            lambda _path, _options: "unexpected"
        ),
        audio_prober=lambda _path, _cancellation: None,
        processor_fingerprint="3" * 64,
        speaker_embedding_policy=_speaker_embedding_policy(),
    )

    one_chunk = invoke_asgi(app, body, [len(body)], boundary)
    byte_chunks = invoke_asgi(app, body, [1] * len(body), boundary)

    assert one_chunk == byte_chunks
    assert one_chunk[0] == 422
    assert one_chunk[1]["error"]["code"] == "async_required"
    assert storage.total_reserved_bytes() == 0


def test_success_is_independent_of_every_byte_boundary(storage: Storage) -> None:
    boundary = b"success-boundary"
    body = multipart_body(boundary, b"1234")
    seen: list[tuple[bytes, CanonicalOptions]] = []
    app = create_app(
        api_key="test-secret",
        readiness=Readiness(True, True, True),
        storage=storage,
        processor=FakeProcessor(
            lambda path, options: (
                seen.append((path.read_bytes(), options)) or "same"
            )
        ),
        audio_prober=lambda _path, _cancellation: None,
        processor_fingerprint="3" * 64,
        speaker_embedding_policy=_speaker_embedding_policy(),
        close_storage_on_shutdown=False,
    )

    one_chunk = invoke_asgi(app, body, [len(body)], boundary)
    byte_chunks = invoke_asgi(app, body, [1] * len(body), boundary)

    assert one_chunk == byte_chunks == (200, {"text": "same"})
    assert len(seen) == 2
    assert [item[0] for item in seen] == [b"1234", b"1234"]


def test_hard_limit_is_independent_of_asgi_chunking(storage: Storage) -> None:
    boundary = b"hard-boundary"
    body = multipart_body(boundary, b"1" * 17)
    app = create_app(
        api_key="test-secret",
        readiness=Readiness(True, True, True),
        storage=storage,
        processor=FakeProcessor(
            lambda _path, _options: "unexpected"
        ),
        audio_prober=lambda _path, _cancellation: None,
        processor_fingerprint="3" * 64,
        speaker_embedding_policy=_speaker_embedding_policy(),
    )

    one_chunk = invoke_asgi(
        app, body, [len(body)], boundary, prefer_async=True
    )
    byte_chunks = invoke_asgi(
        app, body, [1] * len(body), boundary, prefer_async=True
    )

    assert one_chunk == byte_chunks
    assert one_chunk[0] == 413
    assert one_chunk[1]["error"]["code"] == "upload_too_large"
    assert storage.total_reserved_bytes() == 0


def test_joint_raw_limit_exact_is_accepted_and_plus_one_is_chunk_stable(
    storage: Storage, monkeypatch: pytest.MonkeyPatch
) -> None:
    overhead_limit = 512
    monkeypatch.setattr(
        api_module, "MAX_MULTIPART_OVERHEAD_BYTES", overhead_limit
    )
    boundary = b"overhead-boundary"
    exact = multipart_body_with_overhead(
        boundary, overhead_limit, file_content=b"x" * 8
    )
    over = multipart_body_with_overhead(
        boundary, overhead_limit + 1, file_content=b"x" * 8
    )
    app = create_app(
        api_key="test-secret",
        readiness=Readiness(True, True, True),
        storage=storage,
        processor=FakeProcessor(
            lambda _path, _options: "unexpected"
        ),
        audio_prober=lambda _path, _cancellation: None,
        processor_fingerprint="3" * 64,
        speaker_embedding_policy=_speaker_embedding_policy(),
        close_storage_on_shutdown=False,
    )

    exact_status = invoke_asgi(app, exact, [len(exact)], boundary)
    one_chunk = invoke_asgi(app, over, [len(over)], boundary)
    split_chunks = invoke_asgi(app, over, [1] * len(over), boundary)

    assert exact_status[1]["error"]["code"] == "invalid_language"
    assert one_chunk == split_chunks
    assert one_chunk[1]["error"]["code"] == "invalid_multipart"


def test_overhead_limit_plus_one_is_rejected_by_final_exact_check(
    storage: Storage, monkeypatch: pytest.MonkeyPatch
) -> None:
    overhead_limit = 512
    monkeypatch.setattr(
        api_module, "MAX_MULTIPART_OVERHEAD_BYTES", overhead_limit
    )
    boundary = b"final-overhead-boundary"
    body = multipart_body_with_overhead(
        boundary, overhead_limit + 1, file_content=b"x"
    )
    app = create_app(
        api_key="test-secret",
        readiness=Readiness(True, True, True),
        storage=storage,
        processor=FakeProcessor(
            lambda _path, _options: "unexpected"
        ),
        audio_prober=lambda _path, _cancellation: None,
        processor_fingerprint="3" * 64,
        speaker_embedding_policy=_speaker_embedding_policy(),
        close_storage_on_shutdown=False,
    )

    response = invoke_asgi(app, body, [len(body)], boundary)

    assert (
        len(body)
        <= storage.limits.sync_max_upload_bytes + overhead_limit
    )
    assert response[1]["error"]["code"] == "invalid_multipart"


def test_joint_raw_rejection_stops_receiving_more_body(
    storage: Storage, monkeypatch: pytest.MonkeyPatch
) -> None:
    overhead_limit = 512
    monkeypatch.setattr(
        api_module, "MAX_MULTIPART_OVERHEAD_BYTES", overhead_limit
    )
    boundary = b"stop-boundary"
    over = multipart_body_with_overhead(
        boundary, overhead_limit + 1, file_content=b"x" * 8
    )
    app = create_app(
        api_key="test-secret",
        readiness=Readiness(True, True, True),
        storage=storage,
        processor=FakeProcessor(
            lambda _path, _options: "unexpected"
        ),
        audio_prober=lambda _path, _cancellation: None,
        processor_fingerprint="3" * 64,
        speaker_embedding_policy=_speaker_embedding_policy(),
        close_storage_on_shutdown=False,
    )

    status, body, receive_calls = invoke_raw_asgi(
        app,
        path="/v1/audio/transcriptions",
        headers=[
            (b"authorization", b"Bearer test-secret"),
            (
                b"content-type",
                b"multipart/form-data; boundary=" + boundary,
            ),
        ],
        events=[
            {"type": "http.request", "body": over, "more_body": True},
        ],
    )

    assert status == 400
    assert body["error"]["code"] == "invalid_multipart"
    assert receive_calls == 1


def test_joint_cap_wins_when_file_limit_crosses_after_full_overhead_budget(
    storage: Storage, monkeypatch: pytest.MonkeyPatch
) -> None:
    overhead_limit = 512
    file_limit = storage.limits.sync_max_upload_bytes
    monkeypatch.setattr(
        api_module, "MAX_MULTIPART_OVERHEAD_BYTES", overhead_limit
    )
    boundary = b"competing-boundary"
    body = multipart_body_with_prefile_overhead(
        boundary,
        overhead_limit,
        file_content=b"x" * (file_limit + 1),
    )
    app = create_app(
        api_key="test-secret",
        readiness=Readiness(True, True, True),
        storage=storage,
        processor=FakeProcessor(
            lambda _path, _options: "unexpected"
        ),
        audio_prober=lambda _path, _cancellation: None,
        processor_fingerprint="3" * 64,
        speaker_embedding_policy=_speaker_embedding_policy(),
        close_storage_on_shutdown=False,
    )
    headers = [
        (b"authorization", b"Bearer test-secret"),
        (
            b"content-type",
            b"multipart/form-data; boundary=" + boundary,
        ),
    ]

    one_status, one_body, one_receives = invoke_raw_asgi(
        app,
        path="/v1/audio/transcriptions",
        headers=headers,
        events=[
            {"type": "http.request", "body": body, "more_body": True},
        ],
    )
    byte_status, byte_body, byte_receives = invoke_raw_asgi(
        app,
        path="/v1/audio/transcriptions",
        headers=headers,
        events=[
            {"type": "http.request", "body": bytes([byte]), "more_body": True}
            for byte in body
        ],
    )

    assert (one_status, one_body) == (byte_status, byte_body)
    assert one_status == 400
    assert one_body["error"]["code"] == "invalid_multipart"
    assert one_receives == 1
    assert byte_receives == overhead_limit + file_limit + 1
    assert storage.active_upload_count() == 0
    assert storage.total_reserved_bytes() == 0
    assert not list(storage.staging_dir.iterdir())


def test_large_asgi_event_is_sliced_before_storage_append(tmp_path: Path) -> None:
    limits = LimitsConfig(
        max_upload_bytes=10 * 1024 * 1024,
        sync_max_upload_bytes=10 * 1024 * 1024,
        max_job_storage_bytes=3 * RESERVATION_QUANTUM,
        min_filesystem_free_bytes=1,
    )
    storage = Storage(tmp_path, limits, current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    appended_sizes: list[int] = []
    original_append = storage.append

    def recording_append(lease, data: bytes) -> None:
        appended_sizes.append(len(data))
        original_append(lease, data)

    storage.append = recording_append  # type: ignore[method-assign]
    boundary = b"large-event-boundary"
    body = multipart_body(boundary, b"x" * (9 * 1024 * 1024))
    app = create_app(
        api_key="test-secret",
        readiness=Readiness(True, True, True),
        storage=storage,
        processor=FakeProcessor(
            lambda _path, _options: "ok"
        ),
        audio_prober=lambda _path, _cancellation: None,
        processor_fingerprint="3" * 64,
        speaker_embedding_policy=_speaker_embedding_policy(),
        close_storage_on_shutdown=False,
    )
    try:
        response = invoke_asgi(app, body, [len(body)], boundary)
        assert response == (200, {"text": "ok"})
        assert appended_sizes
        assert max(appended_sizes) <= 8 * 1024 * 1024
    finally:
        storage.close()


def test_client_disconnect_cleans_receiving_upload(storage: Storage) -> None:
    boundary = b"disconnect-boundary"
    partial_body = (
        b"--"
        + boundary
        + b'\r\nContent-Disposition: form-data; name="file"; filename="a.wav"'
        + b"\r\n\r\nopen"
    )
    app = create_app(
        api_key="test-secret",
        readiness=Readiness(True, True, True),
        storage=storage,
        processor=FakeProcessor(
            lambda _path, _options: "unexpected"
        ),
        audio_prober=lambda _path, _cancellation: None,
        processor_fingerprint="3" * 64,
        speaker_embedding_policy=_speaker_embedding_policy(),
    )

    status, body = invoke_asgi(
        app, partial_body, [len(partial_body)], boundary, disconnect=True
    )

    assert status == 400
    assert body["error"]["code"] == "client_disconnected"
    assert storage.active_upload_count() == 0
    assert storage.total_reserved_bytes() == 0
    assert not list(storage.staging_dir.iterdir())


def multipart_body(boundary: bytes, file_content: bytes) -> bytes:
    return (
        b"--"
        + boundary
        + b'\r\nContent-Disposition: form-data; name="file"; filename="a.wav"'
        + b"\r\nContent-Type: audio/wav\r\n\r\n"
        + file_content
        + b"\r\n--"
        + boundary
        + b'\r\nContent-Disposition: form-data; name="model"\r\n\r\n'
        + b"sensevoice\r\n--"
        + boundary
        + b"--\r\n"
    )


def multipart_body_with_overhead(
    boundary: bytes, overhead: int, *, file_content: bytes
) -> bytes:
    prefix = (
        b"--"
        + boundary
        + b'\r\nContent-Disposition: form-data; name="file"; filename="a.wav"'
        + b"\r\n\r\n"
        + file_content
        + b"\r\n--"
        + boundary
        + b'\r\nContent-Disposition: form-data; name="model"\r\n\r\n'
        + b"sensevoice\r\n--"
        + boundary
        + b'\r\nContent-Disposition: form-data; name="language"\r\n\r\n'
    )
    suffix = b"\r\n--" + boundary + b"--\r\n"
    fixed_overhead = len(prefix) + len(suffix) - len(file_content)
    assert fixed_overhead <= overhead
    return prefix + b"x" * (overhead - fixed_overhead) + suffix


def multipart_body_with_prefile_overhead(
    boundary: bytes,
    prefile_overhead: int,
    *,
    file_content: bytes,
) -> bytes:
    before_padding = (
        b"--"
        + boundary
        + b'\r\nContent-Disposition: form-data; name="model"\r\n\r\n'
        + b"sensevoice\r\n--"
        + boundary
        + b'\r\nContent-Disposition: form-data; name="language"\r\n\r\n'
    )
    after_padding = (
        b"\r\n--"
        + boundary
        + b'\r\nContent-Disposition: form-data; name="file"; filename="a.wav"'
        + b"\r\n\r\n"
    )
    fixed = len(before_padding) + len(after_padding)
    assert fixed <= prefile_overhead
    prefix = (
        before_padding
        + b"x" * (prefile_overhead - fixed)
        + after_padding
    )
    assert len(prefix) == prefile_overhead
    return prefix + file_content + b"\r\n--" + boundary + b"--\r\n"


def invoke_asgi(
    app,
    body: bytes,
    chunk_sizes: Iterable[int],
    boundary: bytes,
    *,
    disconnect: bool = False,
    prefer_async: bool = False,
) -> tuple[int, dict]:
    async def run() -> tuple[int, dict]:
        chunks: list[bytes] = []
        offset = 0
        for size in chunk_sizes:
            chunks.append(body[offset : offset + size])
            offset += size
        if offset < len(body):
            chunks.append(body[offset:])

        events = [
            {"type": "http.request", "body": chunk, "more_body": True}
            for chunk in chunks
        ]
        if disconnect:
            events.append({"type": "http.disconnect"})
        else:
            events[-1]["more_body"] = False

        sent: list[dict] = []

        async def receive() -> dict:
            if not events:
                await anyio.sleep_forever()
            return events.pop(0)

        async def send(message: dict) -> None:
            sent.append(message)

        headers = [
            (b"authorization", b"Bearer test-secret"),
            (
                b"content-type",
                b"multipart/form-data; boundary=" + boundary,
            ),
        ]
        if prefer_async:
            headers.append((b"prefer", b"respond-async"))
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/audio/transcriptions",
            "raw_path": b"/v1/audio/transcriptions",
            "query_string": b"",
            "headers": headers,
            "client": ("test", 1),
            "server": ("testserver", 80),
        }
        await app(scope, receive, send)
        status = next(
            message["status"]
            for message in sent
            if message["type"] == "http.response.start"
        )
        response_body = b"".join(
            message.get("body", b"")
            for message in sent
            if message["type"] == "http.response.body"
        )
        return status, json.loads(response_body)

    return asyncio.run(run())


def invoke_raw_asgi(
    app,
    *,
    path: str,
    headers: list[tuple[bytes, bytes]],
    events: list[dict] | None,
) -> tuple[int, dict, int]:
    async def run() -> tuple[int, dict, int]:
        sent: list[dict] = []
        receive_calls = 0
        remaining = [] if events is None else list(events)

        async def receive() -> dict:
            nonlocal receive_calls
            receive_calls += 1
            if not remaining:
                raise AssertionError("application consumed unexpected request body")
            return remaining.pop(0)

        async def send(message: dict) -> None:
            sent.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST" if path.endswith("transcriptions") else "GET",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": b"",
            "headers": headers,
            "client": ("test", 1),
            "server": ("testserver", 80),
        }
        await app(scope, receive, send)
        status = next(
            message["status"]
            for message in sent
            if message["type"] == "http.response.start"
        )
        response_body = b"".join(
            message.get("body", b"")
            for message in sent
            if message["type"] == "http.response.body"
        )
        return status, json.loads(response_body), receive_calls

    return asyncio.run(run())


def test_processor_runs_once_outside_event_loop(storage: Storage) -> None:
    calls = 0

    def process(_path: Path, _options: CanonicalOptions) -> str:
        nonlocal calls
        calls += 1
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        return threading.current_thread().name

    with app_client(
        storage,
        processor=FakeProcessor(process),
    ) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            headers=AUTH,
            files={"file": ("audio.wav", b"1234"), "model": (None, "sensevoice")},
        )

    assert response.status_code == 200
    assert calls == 1


def test_app_composition_closes_owned_storage(tmp_path: Path) -> None:
    storage = Storage(
        tmp_path,
        LimitsConfig(
            max_upload_bytes=8,
            sync_max_upload_bytes=8,
            max_job_storage_bytes=8 * 1024 * 1024,
            min_filesystem_free_bytes=1,
        ),
        current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40,
    )
    app = create_app(
        api_key="test-secret",
        readiness=Readiness(True, True, True),
        storage=storage,
        processor=FakeProcessor(
            lambda _path, _options: "ok"
        ),
        audio_prober=lambda _path, _cancellation: None,
        processor_fingerprint="3" * 64,
        speaker_embedding_policy=_speaker_embedding_policy(),
    )

    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200

    assert storage.closed
