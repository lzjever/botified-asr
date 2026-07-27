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
        ("DELETE", "/v1/speakers/00000001"),
    ),
    ids=("list", "get", "delete"),
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


@pytest.mark.parametrize("method", ("GET", "DELETE"))
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
        response = client.request(
            method,
            f"/v1/speakers/{profile_id}",
            headers=AUTH,
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
