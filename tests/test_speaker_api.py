from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from starlette.testclient import TestClient

from botified_asr import speaker_profiles, speakers
from botified_asr.api import Readiness, create_app
from botified_asr.config import LimitsConfig, RESERVATION_QUANTUM
from botified_asr.storage import Storage


AUTH = {"Authorization": "Bearer test-secret"}
MODEL_REVISION = "1" * 40
PROCESSOR_FINGERPRINT = "3" * 64


class BombProcessor:
    def process(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("speaker management must not call the processor")


def _speaker_policy() -> speakers.SpeakerEmbeddingPolicy:
    return speakers.SpeakerEmbeddingPolicy(
        model_id="funasr/campplus",
        model_revision=MODEL_REVISION,
        embedding_dimension=2,
        sample_rate=16_000,
        downmix_policy_version="ffmpeg-first-audio-stream-ac1-v1",
        window_samples=24_000,
        window_shift_samples=12_000,
        padding_policy_version="right-zero-pad-v1",
        normalization_policy_version="int16-div-32768-l2-v1",
        enrollment_aggregation_policy_version="sample-centroid-equal-average-v1",
    )


def _profile(
    profile_id: str,
    name: str,
    *,
    description: str | None,
    created_at: datetime,
    updated_at: datetime,
) -> speaker_profiles.SpeakerProfile:
    policy = _speaker_policy()
    embedding = speaker_profiles.SpeakerEmbedding.from_numpy(
        np.array([1.0, 0.0], dtype=np.float32),
        dimension=policy.embedding_dimension,
    )
    return speaker_profiles.SpeakerProfile(
        id=profile_id,
        name=name,
        description=description,
        embedding=embedding,
        embedding_model_id=policy.model_id,
        embedding_model_revision=policy.model_revision,
        embedding_dimension=policy.embedding_dimension,
        embedding_policy_fingerprint=policy.fingerprint,
        sample_count=2,
        created_at=created_at,
        updated_at=updated_at,
    )


def _public_timestamp(value: datetime) -> str:
    fractional = f".{value.microsecond:06d}" if value.microsecond else ""
    return value.strftime("%Y-%m-%dT%H:%M:%S") + fractional + "Z"


def _speaker_resource(
    profile: speaker_profiles.SpeakerProfile,
) -> dict[str, object]:
    return {
        "id": profile.id,
        "object": "speaker",
        "name": profile.name,
        "description": profile.description,
        "sample_count": profile.sample_count,
        "embedding_model": {
            "id": "cam++",
            "revision": profile.embedding_model_revision,
            "dimension": profile.embedding_dimension,
            "policy_fingerprint": profile.embedding_policy_fingerprint,
        },
        "created_at": _public_timestamp(profile.created_at),
        "updated_at": _public_timestamp(profile.updated_at),
    }


def _client(
    storage: Storage,
    *,
    readiness: Readiness | None = None,
) -> TestClient:
    app = create_app(
        api_key="test-secret",
        readiness=readiness or Readiness(True, True, True),
        storage=storage,
        processor=BombProcessor(),
        audio_prober=lambda _path, _cancellation: None,
        processor_fingerprint="3" * 64,
        speaker_embedding_policy=_speaker_policy(),
        close_storage_on_shutdown=False,
    )
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def storage(tmp_path: Path) -> Iterator[Storage]:
    value = Storage(
        tmp_path,
        LimitsConfig(
            max_upload_bytes=RESERVATION_QUANTUM,
            sync_max_upload_bytes=RESERVATION_QUANTUM,
            max_job_storage_bytes=RESERVATION_QUANTUM,
            min_filesystem_free_bytes=1,
        ),
        current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _path: 1 << 40,
    )
    yield value
    value.close()


def test_get_speaker_returns_exact_public_resource(storage: Storage) -> None:
    profile = _profile(
        "00000001",
        "Alice",
        description=None,
        created_at=datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc),
        updated_at=datetime(
            2026,
            7,
            27,
            12,
            0,
            0,
            123,
            tzinfo=timezone.utc,
        ),
    )
    storage.create_speaker_profile(profile)

    with _client(storage) as client:
        response = client.get(f"/v1/speakers/{profile.id}", headers=AUTH)

    assert response.status_code == 200
    assert response.json() == _speaker_resource(profile)
    assert response.json()["created_at"] == "2026-07-27T12:00:00Z"
    assert response.json()["updated_at"] == "2026-07-27T12:00:00.000123Z"
    assert "embedding" not in response.json()
    assert _speaker_policy().model_id not in response.text


def test_list_speakers_is_exact_empty_and_stably_ordered(
    storage: Storage,
) -> None:
    early = _profile(
        "00000003",
        "Carol",
        description=None,
        created_at=datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc),
    )
    tied_at = datetime(
        2026,
        7,
        27,
        12,
        0,
        1,
        123,
        tzinfo=timezone.utc,
    )
    low = _profile(
        "00000001",
        "Alice",
        description="first tied ID",
        created_at=tied_at,
        updated_at=tied_at,
    )
    high = _profile(
        "00000002",
        "Bob",
        description="second tied ID",
        created_at=tied_at,
        updated_at=tied_at,
    )

    with _client(storage) as client:
        empty = client.get("/v1/speakers", headers=AUTH)
        for profile in (high, early, low):
            storage.create_speaker_profile(profile)
        first = client.get("/v1/speakers", headers=AUTH)
        second = client.get("/v1/speakers", headers=AUTH)

    expected = {
        "object": "list",
        "data": [
            _speaker_resource(early),
            _speaker_resource(low),
            _speaker_resource(high),
        ],
    }
    assert empty.status_code == 200
    assert empty.json() == {"object": "list", "data": []}
    assert first.status_code == 200
    assert first.json() == expected
    assert second.status_code == 200
    assert second.json() == expected
    assert first.json()["data"][1]["created_at"].endswith(".000123Z")


@pytest.mark.parametrize(
    ("method", "path"),
    (
        ("GET", "/v1/speakers"),
        ("GET", "/v1/speakers/00000001"),
        ("PUT", "/v1/speakers/00000001"),
        ("DELETE", "/v1/speakers/00000001"),
    ),
    ids=("list", "get", "put", "delete"),
)
@pytest.mark.parametrize(
    ("headers", "expected_status", "expected_error"),
    (
        (
            {},
            401,
            {
                "message": "Invalid authentication credentials",
                "type": "authentication_error",
                "param": None,
                "code": "invalid_api_key",
            },
        ),
        (
            AUTH,
            503,
            {
                "message": "Service is not ready",
                "type": "server_error",
                "param": None,
                "code": "service_not_ready",
            },
        ),
    ),
    ids=("auth_before_readiness", "readiness_before_storage"),
)
def test_speaker_routes_gate_before_storage(
    storage: Storage,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    path: str,
    headers: dict[str, str],
    expected_status: int,
    expected_error: dict[str, object],
) -> None:
    profile = _profile(
        "00000001",
        "Alice",
        description=None,
        created_at=datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc),
    )
    storage.create_speaker_profile(profile)
    calls: list[str] = []

    def bomb(*_args: object, **_kwargs: object) -> object:
        calls.append("storage")
        raise AssertionError("storage must not be called before request gates")

    with monkeypatch.context() as patch:
        patch.setattr(storage, "list_speaker_profiles", bomb)
        patch.setattr(storage, "get_speaker_profile", bomb)
        patch.setattr(storage, "update_speaker_profile", bomb)
        patch.setattr(storage, "delete_speaker_profile", bomb)
        with _client(
            storage,
            readiness=Readiness(False, False, False),
        ) as client:
            response = client.request(method, path, headers=headers)

    assert calls == []
    assert response.status_code == expected_status
    assert response.json() == {"error": expected_error}
    assert storage.get_speaker_profile(profile.id) == profile


@pytest.mark.parametrize("method", ("GET", "PUT", "DELETE"))
@pytest.mark.parametrize(
    "profile_id",
    ("not-valid", "ZZZZZZZZ"),
    ids=("malformed", "missing"),
)
def test_speaker_item_invalid_or_missing_id_is_the_same_404(
    storage: Storage,
    method: str,
    profile_id: str,
) -> None:
    with _client(storage) as client:
        request_kwargs = (
            {"files": {"name": (None, "Alice")}}
            if method == "PUT"
            else {}
        )
        response = client.request(
            method,
            f"/v1/speakers/{profile_id}",
            headers=AUTH,
            **request_kwargs,
        )

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "message": "Speaker not found",
            "type": "invalid_request_error",
            "param": "speaker_id",
            "code": "speaker_not_found",
        }
    }


def test_delete_speaker_returns_empty_204_and_deletes_once(
    storage: Storage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile(
        "00000001",
        "Alice",
        description=None,
        created_at=datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc),
    )
    storage.create_speaker_profile(profile)
    original_delete = storage.delete_speaker_profile
    calls: list[str] = []

    def recording_delete(profile_id: str) -> bool:
        calls.append(profile_id)
        return original_delete(profile_id)

    def bomb_get(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("DELETE must not read the speaker before deleting")

    with monkeypatch.context() as patch:
        patch.setattr(storage, "get_speaker_profile", bomb_get)
        patch.setattr(storage, "delete_speaker_profile", recording_delete)
        with _client(storage) as client:
            response = client.delete(f"/v1/speakers/{profile.id}", headers=AUTH)

    assert response.status_code == 204
    assert response.content == b""
    assert calls == [profile.id]
    assert storage.get_speaker_profile(profile.id) is None


def test_put_speaker_description_tristate_preserves_embedding_and_timestamps(
    storage: Storage,
) -> None:
    original = _profile(
        "00000001",
        "Alice",
        description="original",
        created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2020, 1, 2, tzinfo=timezone.utc),
    )
    storage.create_speaker_profile(original)
    embedding_state = (
        original.embedding.to_bytes(),
        original.embedding_model_id,
        original.embedding_model_revision,
        original.embedding_dimension,
        original.embedding_policy_fingerprint,
        original.sample_count,
    )

    with _client(storage) as client:
        preserved_response = client.put(
            f"/v1/speakers/{original.id}",
            headers=AUTH,
            files={"name": (None, "  Alice Renamed  ")},
        )
        preserved = storage.get_speaker_profile(original.id)
        cleared_response = client.put(
            f"/v1/speakers/{original.id}",
            headers=AUTH,
            files={
                "name": (None, "Alice Renamed"),
                "description": (None, ""),
            },
        )
        cleared = storage.get_speaker_profile(original.id)
        replaced_response = client.put(
            f"/v1/speakers/{original.id}",
            headers=AUTH,
            files={
                "name": (None, "Alice Renamed"),
                "description": (None, "replacement"),
            },
        )
        replaced = storage.get_speaker_profile(original.id)
        whitespace_response = client.put(
            f"/v1/speakers/{original.id}",
            headers=AUTH,
            files={
                "name": (None, "Alice Renamed"),
                "description": (None, "   "),
            },
        )
        whitespace = storage.get_speaker_profile(original.id)
        null_text_response = client.put(
            f"/v1/speakers/{original.id}",
            headers=AUTH,
            files={
                "name": (None, "Alice Renamed"),
                "description": (None, "null"),
            },
        )
        null_text = storage.get_speaker_profile(original.id)

    assert preserved is not None
    assert preserved_response.status_code == 200
    assert preserved_response.json() == _speaker_resource(preserved)
    assert preserved.name == "Alice Renamed"
    assert preserved.description == "original"
    assert preserved.updated_at > original.updated_at

    assert cleared is not None
    assert cleared_response.status_code == 200
    assert cleared_response.json() == _speaker_resource(cleared)
    assert cleared.description is None
    assert cleared.updated_at >= preserved.updated_at

    assert replaced is not None
    assert replaced_response.status_code == 200
    assert replaced_response.json() == _speaker_resource(replaced)
    assert replaced.description == "replacement"
    assert replaced.updated_at >= cleared.updated_at
    assert whitespace is not None
    assert whitespace_response.status_code == 200
    assert whitespace_response.json() == _speaker_resource(whitespace)
    assert whitespace.description == "   "
    assert null_text is not None
    assert null_text_response.status_code == 200
    assert null_text_response.json() == _speaker_resource(null_text)
    assert null_text.description == "null"
    for profile in (preserved, cleared, replaced, whitespace, null_text):
        assert profile.created_at == original.created_at
        assert (
            profile.embedding.to_bytes(),
            profile.embedding_model_id,
            profile.embedding_model_revision,
            profile.embedding_dimension,
            profile.embedding_policy_fingerprint,
            profile.sample_count,
        ) == embedding_state


def test_put_speaker_clock_rollback_is_clamped(storage: Storage) -> None:
    future = datetime(2099, 1, 1, tzinfo=timezone.utc)
    original = _profile(
        "00000001",
        "Alice",
        description=None,
        created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        updated_at=future,
    )
    storage.create_speaker_profile(original)

    with _client(storage) as client:
        response = client.put(
            f"/v1/speakers/{original.id}",
            headers=AUTH,
            files={"name": (None, "Renamed")},
        )

    updated = storage.get_speaker_profile(original.id)
    assert updated is not None
    assert response.status_code == 200
    assert response.json() == _speaker_resource(updated)
    assert updated.name == "Renamed"
    assert updated.updated_at == future


@pytest.mark.parametrize(
    "files",
    (
        [
            ("name", (None, "Alice")),
            ("name", (None, "Again")),
        ],
        {"unknown": (None, "value")},
        {"file": ("voice.wav", b"audio", "audio/wav")},
        {"samples[]": ("voice.wav", b"audio", "audio/wav")},
        {"name": ("name.txt", "Alice", "text/plain")},
    ),
    ids=(
        "duplicate-scalar",
        "unknown",
        "file",
        "samples",
        "filename-scalar",
    ),
)
def test_put_speaker_rejects_non_metadata_multipart(
    storage: Storage,
    files: object,
) -> None:
    profile = _profile(
        "00000001",
        "Alice",
        description=None,
        created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    storage.create_speaker_profile(profile)

    with _client(storage) as client:
        response = client.put(
            f"/v1/speakers/{profile.id}",
            headers=AUTH,
            files=files,
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_multipart"
    assert storage.get_speaker_profile(profile.id) == profile


@pytest.mark.parametrize("chunk_size", (1024 * 1024, 8191))
def test_put_speaker_metadata_body_limit_is_chunk_invariant(
    storage: Storage,
    chunk_size: int,
) -> None:
    profile = _profile(
        "00000001",
        "Alice",
        description=None,
        created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    storage.create_speaker_profile(profile)
    boundary = "metadata-limit"
    multipart = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="name"\r\n'
        "\r\n"
        "Alice\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    limit = 1024 * 1024
    exact = multipart + b"x" * (limit - len(multipart))
    overflow = exact + b"x"

    def chunks(body: bytes):
        for offset in range(0, len(body), chunk_size):
            yield body[offset : offset + chunk_size]

    headers = {
        **AUTH,
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }
    with _client(storage) as client:
        accepted = client.put(
            f"/v1/speakers/{profile.id}",
            headers=headers,
            content=chunks(exact),
        )
        rejected = client.put(
            f"/v1/speakers/{profile.id}",
            headers=headers,
            content=chunks(overflow),
        )

    assert accepted.status_code == 200
    assert rejected.status_code == 400
    assert rejected.json()["error"]["code"] == "invalid_multipart"


def test_put_speaker_rejects_empty_filename_attribute(
    storage: Storage,
) -> None:
    profile = _profile(
        "00000001",
        "Alice",
        description=None,
        created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    storage.create_speaker_profile(profile)
    boundary = "empty-filename"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="name"; filename=""\r\n'
        "Content-Type: text/plain\r\n"
        "\r\n"
        "Alice\r\n"
        f"--{boundary}--\r\n"
    ).encode()

    with _client(storage) as client:
        response = client.put(
            f"/v1/speakers/{profile.id}",
            headers={
                **AUTH,
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            content=body,
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_multipart"
    assert storage.get_speaker_profile(profile.id) == profile


@pytest.mark.parametrize(
    ("files", "code", "param"),
    (
        (
            {"description": (None, "missing name")},
            "invalid_speaker_name",
            "name",
        ),
        ({"name": (None, "A")}, "reserved_speaker_name", "name"),
        (
            {"name": (None, "Unknown A")},
            "reserved_speaker_name",
            "name",
        ),
        ({"name": (None, "")}, "invalid_speaker_name", "name"),
        ({"name": (None, "x" * 81)}, "invalid_speaker_name", "name"),
        (
            {
                "name": (None, "Alice"),
                "description": (None, "x" * 501),
            },
            "invalid_speaker_description",
            "description",
        ),
    ),
    ids=(
        "required-name",
        "reserved-label",
        "reserved-unknown",
        "empty-name",
        "long-name",
        "long-description",
    ),
)
def test_put_speaker_validation_errors_are_stable(
    storage: Storage,
    files: object,
    code: str,
    param: str,
) -> None:
    profile = _profile(
        "00000001",
        "Alice",
        description=None,
        created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    storage.create_speaker_profile(profile)

    with _client(storage) as client:
        response = client.put(
            f"/v1/speakers/{profile.id}",
            headers=AUTH,
            files=files,
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == code
    assert response.json()["error"]["param"] == param
    assert "speaker profile" not in response.text
    assert storage.get_speaker_profile(profile.id) == profile


def test_put_speaker_name_conflict_is_stable_and_atomic(
    storage: Storage,
) -> None:
    created_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
    first = _profile(
        "00000001",
        "Alice",
        description="unchanged",
        created_at=created_at,
        updated_at=created_at,
    )
    second = _profile(
        "00000002",
        "Bob",
        description=None,
        created_at=created_at,
        updated_at=created_at,
    )
    storage.create_speaker_profile(first)
    storage.create_speaker_profile(second)

    with _client(storage) as client:
        response = client.put(
            f"/v1/speakers/{first.id}",
            headers=AUTH,
            files={
                "name": (None, "bOB"),
                "description": (None, "must roll back"),
            },
        )

    assert response.status_code == 409
    assert response.json() == {
        "error": {
            "message": "Speaker name already exists",
            "type": "invalid_request_error",
            "param": "name",
            "code": "speaker_name_conflict",
        }
    }
    assert storage.get_speaker_profile(first.id) == first
