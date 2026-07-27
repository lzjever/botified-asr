from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from datetime import datetime, timedelta, timezone
from importlib import import_module
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from botified_asr import speaker_profiles, speakers
from botified_asr import storage as storage_module
from botified_asr.config import LimitsConfig


CREATED_AT = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
MODEL_ID = "funasr/campplus"
MODEL_REVISION = "1" * 40


def _snapshot_module() -> ModuleType:
    return import_module("botified_asr.speaker_snapshot")


def _policy() -> speakers.SpeakerEmbeddingPolicy:
    return speakers.SpeakerEmbeddingPolicy(
        model_id=MODEL_ID,
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


def _embedding(
    dimension: int = 2,
    *,
    axis: int = 0,
) -> speaker_profiles.SpeakerEmbedding:
    values = np.zeros(dimension, dtype=np.float32)
    values[axis] = 1.0
    return speaker_profiles.SpeakerEmbedding.from_numpy(
        values,
        dimension=dimension,
    )


def _profile(
    *,
    profile_id: str = "00000001",
    name: str = "Alice",
    embedding: speaker_profiles.SpeakerEmbedding | None = None,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
    policy_fingerprint: str | None = None,
) -> speaker_profiles.SpeakerProfile:
    policy = _policy()
    vector = embedding or _embedding()
    return speaker_profiles.SpeakerProfile(
        id=profile_id,
        name=name,
        description="metadata",
        embedding=vector,
        embedding_model_id=model_id,
        embedding_model_revision=model_revision,
        embedding_dimension=vector.dimension,
        embedding_policy_fingerprint=policy_fingerprint or policy.fingerprint,
        sample_count=2,
        created_at=CREATED_AT,
        updated_at=CREATED_AT,
    )


_AXIS_0_BASE64 = "AACAPwAAAAA="
_AXIS_1_BASE64 = "AAAAAAAAgD8="


def _wire_entry(
    profile_id: str = "00000001",
    *,
    name: str = "Alice",
    embedding: str = _AXIS_0_BASE64,
) -> str:
    return (
        f'{{"embedding":"{embedding}","id":"{profile_id}",'
        f'"name":"{name}"}}'
    )


def _snapshot_wire(*entries: str) -> bytes:
    return (
        f'{{"speakers":[{",".join(entries)}],"version":1}}'
    ).encode("utf-8")


class RecordingReader:
    def __init__(
        self,
        profiles: tuple[speaker_profiles.SpeakerProfile, ...],
    ) -> None:
        self.profiles = profiles
        self.calls: list[tuple[str, ...]] = []

    def get_speaker_profiles_by_ids(
        self,
        profile_ids: tuple[str, ...],
    ) -> tuple[speaker_profiles.SpeakerProfile, ...]:
        self.calls.append(profile_ids)
        return self.profiles


def _storage(path: Path) -> storage_module.Storage:
    quantum = storage_module.RESERVATION_QUANTUM
    limits = LimitsConfig(
        max_upload_bytes=quantum,
        sync_max_upload_bytes=quantum,
        max_active_uploads=2,
        max_job_storage_bytes=quantum,
        min_filesystem_free_bytes=1,
    )
    return storage_module.Storage(path, limits, free_bytes=lambda _path: 1 << 40)


def test_snapshot_dtos_are_exact_frozen_slots_and_errors_have_no_code() -> None:
    snapshot_module = _snapshot_module()
    selected_type = snapshot_module.SelectedSpeaker
    snapshot_type = snapshot_module.SelectedSpeakerSnapshot
    assert tuple(item.name for item in fields(selected_type)) == (
        "id",
        "name",
        "embedding",
    )
    assert tuple(item.name for item in fields(snapshot_type)) == ("speakers",)
    reader_methods = tuple(
        name
        for name, value in vars(snapshot_module.SpeakerProfileReader).items()
        if not name.startswith("_") and callable(value)
    )
    assert reader_methods == ("get_speaker_profiles_by_ids",)

    selected = selected_type("00000001", "Alice", _embedding())
    snapshot = snapshot_type((selected,))
    assert not hasattr(selected, "__dict__")
    assert not hasattr(snapshot, "__dict__")
    with pytest.raises(FrozenInstanceError):
        selected.name = "Bob"
    with pytest.raises(FrozenInstanceError):
        snapshot.speakers = ()

    for error_type in (
        snapshot_module.SelectedSpeakerNotFoundError,
        snapshot_module.SelectedSpeakerIncompatibleError,
    ):
        assert not hasattr(error_type(), "code")


def test_snapshot_codec_uses_its_public_wire_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_module = _snapshot_module()
    assert snapshot_module.SPEAKER_SNAPSHOT_WIRE_VERSION == 1
    monkeypatch.setattr(
        snapshot_module,
        "SPEAKER_SNAPSHOT_WIRE_VERSION",
        2,
    )
    snapshot = snapshot_module.SelectedSpeakerSnapshot(())
    wire = snapshot_module.serialize_selected_speaker_snapshot(
        snapshot,
        _policy(),
    )

    assert wire == b'{"speakers":[],"version":2}'
    assert (
        snapshot_module.parse_selected_speaker_snapshot(
            wire,
            _policy(),
            expected_ids=(),
        )
        == snapshot
    )


def test_empty_and_happy_snapshot_use_one_read_and_sorted_projection() -> None:
    snapshot_module = _snapshot_module()
    policy = _policy()
    low = _profile(profile_id="00000001", name="Alice")
    high = _profile(profile_id="00000002", name="Bob", embedding=_embedding(axis=1))
    reader = RecordingReader((low, high))

    empty = snapshot_module.resolve_selected_speaker_snapshot(
        reader,
        (),
        policy,
    )
    assert empty == snapshot_module.SelectedSpeakerSnapshot(())
    assert reader.calls == []

    snapshot = snapshot_module.resolve_selected_speaker_snapshot(
        reader,
        (low.id, high.id),
        policy,
    )
    assert reader.calls == [(low.id, high.id)]
    assert snapshot == snapshot_module.SelectedSpeakerSnapshot(
        (
            snapshot_module.SelectedSpeaker(low.id, low.name, low.embedding),
            snapshot_module.SelectedSpeaker(high.id, high.name, high.embedding),
        )
    )


def test_missing_profile_wins_before_incompatibility_in_the_same_read() -> None:
    snapshot_module = _snapshot_module()
    incompatible = _profile(
        profile_id="00000001",
        model_revision="2" * 40,
    )
    reader = RecordingReader((incompatible,))

    with pytest.raises(snapshot_module.SelectedSpeakerNotFoundError):
        snapshot_module.resolve_selected_speaker_snapshot(
            reader,
            ("00000001", "00000002"),
            _policy(),
        )
    assert reader.calls == [("00000001", "00000002")]


def test_incompatible_profile_is_rejected() -> None:
    snapshot_module = _snapshot_module()
    profile = _profile(policy_fingerprint="b" * 64)
    reader = RecordingReader((profile,))

    with pytest.raises(snapshot_module.SelectedSpeakerIncompatibleError):
        snapshot_module.resolve_selected_speaker_snapshot(
            reader,
            (profile.id,),
            _policy(),
        )
    assert reader.calls == [(profile.id,)]


def test_snapshot_is_detached_from_real_storage_update_and_delete(
    tmp_path: Path,
) -> None:
    snapshot_module = _snapshot_module()
    policy = _policy()
    original = _profile()
    storage = _storage(tmp_path)
    storage.create_speaker_profile(original)
    try:
        snapshot = snapshot_module.resolve_selected_speaker_snapshot(
            storage,
            (original.id,),
            policy,
        )
        replacement = speaker_profiles.SpeakerEmbeddingReplacement(
            embedding=_embedding(axis=1),
            embedding_model_id=MODEL_ID,
            embedding_model_revision=MODEL_REVISION,
            embedding_dimension=2,
            embedding_policy_fingerprint=policy.fingerprint,
            sample_count=3,
        )
        storage.update_speaker_profile(
            original.id,
            speaker_profiles.SpeakerProfileUpdate(
                name="Updated Alice",
                description=speaker_profiles.KEEP_EXISTING,
                embedding=replacement,
                updated_at=CREATED_AT + timedelta(minutes=1),
            ),
        )
        assert storage.delete_speaker_profile(original.id) is True

        assert snapshot.speakers == (
            snapshot_module.SelectedSpeaker(
                original.id,
                original.name,
                original.embedding,
            ),
        )
        view = snapshot.speakers[0].embedding.as_numpy()
        assert view.flags.writeable is False
    finally:
        storage.close()


def test_reader_storage_schema_error_is_propagated_unchanged() -> None:
    snapshot_module = _snapshot_module()
    expected = storage_module.StorageSchemaError("corrupt row")

    class FailingReader:
        def get_speaker_profiles_by_ids(
            self,
            profile_ids: tuple[str, ...],
        ) -> tuple[speaker_profiles.SpeakerProfile, ...]:
            del profile_ids
            raise expected

    with pytest.raises(storage_module.StorageSchemaError) as caught:
        snapshot_module.resolve_selected_speaker_snapshot(
            FailingReader(),
            ("00000001",),
            _policy(),
        )
    assert caught.value is expected


def test_snapshot_v1_empty_wire_is_exact() -> None:
    snapshot_module = _snapshot_module()

    wire = snapshot_module.serialize_selected_speaker_snapshot(
        snapshot_module.SelectedSpeakerSnapshot(()),
        _policy(),
    )

    assert wire == b'{"speakers":[],"version":1}'


def test_snapshot_v1_sorted_utf8_wire_and_roundtrip_are_exact() -> None:
    snapshot_module = _snapshot_module()
    low = snapshot_module.SelectedSpeaker(
        "00000001",
        "艾丽丝",
        _embedding(),
    )
    high = snapshot_module.SelectedSpeaker(
        "00000002",
        "Bob",
        _embedding(axis=1),
    )
    snapshot = snapshot_module.SelectedSpeakerSnapshot((low, high))

    wire = snapshot_module.serialize_selected_speaker_snapshot(
        snapshot_module.SelectedSpeakerSnapshot((high, low)),
        _policy(),
    )

    assert wire == (
        '{"speakers":['
        '{"embedding":"AACAPwAAAAA=","id":"00000001","name":"艾丽丝"},'
        '{"embedding":"AAAAAAAAgD8=","id":"00000002","name":"Bob"}'
        '],"version":1}'
    ).encode("utf-8")
    assert snapshot_module.parse_selected_speaker_snapshot(
        wire,
        _policy(),
        expected_ids=("00000001", "00000002"),
    ) == snapshot
    assert snapshot_module.parse_selected_speaker_snapshot(
        snapshot_module.serialize_selected_speaker_snapshot(
            snapshot,
            _policy(),
        ),
        _policy(),
        expected_ids=("00000001", "00000002"),
    ) == snapshot


_VALID_ENTRY = _wire_entry()
_VALID_WIRE = _snapshot_wire(_VALID_ENTRY)
_TWO_IDS = ("00000001", "00000002")
_THIRTY_THREE_IDS = tuple(f"{index:08d}" for index in range(1, 34))


@pytest.mark.parametrize(
    ("wire", "expected_ids"),
    (
        (
            b'{"speakers":[],"speakers":[],"version":1}',
            (),
        ),
        (
            b'{"extra":0,"speakers":[],"version":1}',
            (),
        ),
        (b'{"speakers":[]}', ()),
        (b'{"version":1}', ()),
        (
            _snapshot_wire(
                '{"embedding":"AACAPwAAAAA=","embedding":"AACAPwAAAAA=",'
                '"id":"00000001","name":"Alice"}'
            ),
            ("00000001",),
        ),
        (
            _snapshot_wire(
                '{"embedding":"AACAPwAAAAA=","extra":0,'
                '"id":"00000001","name":"Alice"}'
            ),
            ("00000001",),
        ),
        (
            _snapshot_wire('{"id":"00000001","name":"Alice"}'),
            ("00000001",),
        ),
        (
            _snapshot_wire(
                '{"embedding":"AACAPwAAAAA=","name":"Alice"}'
            ),
            ("00000001",),
        ),
        (
            _snapshot_wire(
                '{"embedding":"AACAPwAAAAA=","id":"00000001"}'
            ),
            ("00000001",),
        ),
        (b'{"speakers":[],"version":true}', ()),
        (b'{"speakers":[],"version":2}', ()),
        (b'{"speakers": [],"version":1}', ()),
        (b'{"version":1,"speakers":[]}', ()),
        (
            _snapshot_wire(
                _wire_entry(name=r"\u827e\u4e3d\u4e1d")
            ),
            ("00000001",),
        ),
        (
            _snapshot_wire(_wire_entry(embedding="18c-v2WzKj8=")),
            ("00000001",),
        ),
        (
            _snapshot_wire(_wire_entry(embedding="AACAPwAAAAA")),
            ("00000001",),
        ),
        (
            _snapshot_wire(_wire_entry(embedding="AACA PwAAAAA=")),
            ("00000001",),
        ),
        (
            _snapshot_wire(_wire_entry(embedding="AACAPw==")),
            ("00000001",),
        ),
        (
            _snapshot_wire(_wire_entry(embedding="AACAfwAAAAA=")),
            ("00000001",),
        ),
        (
            _snapshot_wire(_wire_entry(embedding="AAAAPwAAAAA=")),
            ("00000001",),
        ),
        (
            _snapshot_wire(_wire_entry(name=" Alice ")),
            ("00000001",),
        ),
        (
            _snapshot_wire(
                _wire_entry("00000002", embedding=_AXIS_1_BASE64),
                _VALID_ENTRY,
            ),
            _TWO_IDS,
        ),
        (
            _snapshot_wire(_VALID_ENTRY, _VALID_ENTRY),
            _TWO_IDS,
        ),
        (_VALID_WIRE, ("00000002",)),
        (
            _snapshot_wire(
                *(
                    _wire_entry(profile_id)
                    for profile_id in _THIRTY_THREE_IDS
                )
            ),
            _THIRTY_THREE_IDS,
        ),
        (
            _VALID_WIRE + b" " * (64 * 1024 + 1 - len(_VALID_WIRE)),
            ("00000001",),
        ),
    ),
    ids=(
        "duplicate_top_key",
        "unknown_top_key",
        "missing_top_version",
        "missing_top_speakers",
        "duplicate_entry_key",
        "unknown_entry_key",
        "missing_entry_embedding",
        "missing_entry_id",
        "missing_entry_name",
        "version_bool",
        "version_non_one",
        "json_whitespace",
        "json_key_order",
        "unicode_escape",
        "urlsafe_base64",
        "unpadded_base64",
        "whitespace_base64",
        "bad_embedding_length",
        "nonfinite_embedding",
        "nonunit_embedding",
        "noncanonical_name",
        "ids_out_of_order",
        "duplicate_ids",
        "expected_ids_mismatch",
        "too_many_speakers",
        "wire_over_64_kib",
    ),
)
def test_snapshot_v1_parser_rejects_noncanonical_or_invalid_wire(
    wire: bytes,
    expected_ids: tuple[str, ...],
) -> None:
    snapshot_module = _snapshot_module()

    with pytest.raises((TypeError, ValueError)):
        snapshot_module.parse_selected_speaker_snapshot(
            wire,
            _policy(),
            expected_ids=expected_ids,
        )
