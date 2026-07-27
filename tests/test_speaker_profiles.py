from __future__ import annotations

import struct
from dataclasses import MISSING, FrozenInstanceError, fields, replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from botified_asr import speaker_profiles, speakers


MODEL_ID = "funasr/campplus"
MODEL_REVISION = "1" * 40
POLICY_FINGERPRINT = "a" * 64
CREATED_AT = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
UPDATED_AT = datetime(2026, 7, 27, 12, 1, tzinfo=timezone.utc)


def _embedding(
    dimension: int = 2,
    *,
    values: np.ndarray | None = None,
) -> speaker_profiles.SpeakerEmbedding:
    if values is None:
        values = np.zeros(dimension, dtype=np.float32)
        values[0] = 1.0
    return speaker_profiles.SpeakerEmbedding.from_numpy(
        values,
        dimension=dimension,
    )


def _profile(**changes: object) -> speaker_profiles.SpeakerProfile:
    values: dict[str, object] = {
        "id": "01234567",
        "name": "Alice",
        "description": "Project lead",
        "embedding": _embedding(),
        "embedding_model_id": MODEL_ID,
        "embedding_model_revision": MODEL_REVISION,
        "embedding_dimension": 2,
        "embedding_policy_fingerprint": POLICY_FINGERPRINT,
        "sample_count": 2,
        "created_at": CREATED_AT,
        "updated_at": UPDATED_AT,
    }
    values.update(changes)
    return speaker_profiles.SpeakerProfile(**values)  # type: ignore[arg-type]


def _policy(
    *,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
    embedding_dimension: int = 2,
) -> speakers.SpeakerEmbeddingPolicy:
    return speakers.SpeakerEmbeddingPolicy(
        model_id=model_id,
        model_revision=model_revision,
        embedding_dimension=embedding_dimension,
        sample_rate=16_000,
        downmix_policy_version="ffmpeg-first-audio-stream-ac1-v1",
        window_samples=24_000,
        window_shift_samples=12_000,
        padding_policy_version="right-zero-pad-v1",
        normalization_policy_version="int16-div-32768-l2-v1",
        enrollment_aggregation_policy_version="sample-centroid-equal-average-v1",
    )


def test_embedding_uses_exact_canonical_little_endian_raw_bytes() -> None:
    raw = struct.pack("<ff", 0.6, 0.8)
    embedding = speaker_profiles.SpeakerEmbedding.from_bytes(raw, dimension=2)

    assert embedding.dimension == 2
    assert embedding.to_bytes() == raw
    assert len(embedding.to_bytes()) == 8
    assert speaker_profiles.SpeakerEmbedding.from_numpy(
        np.array([1.0, 0.0], dtype=np.float32),
        dimension=2,
    ).to_bytes() == struct.pack("<ff", 1.0, 0.0)
    assert (
        speaker_profiles.SpeakerEmbedding.from_bytes(
            embedding.to_bytes(),
            dimension=2,
        )
        == embedding
    )
    assert len(_embedding(192).to_bytes()) == 768


def test_embedding_canonicalizes_noncontiguous_input_and_owns_immutable_data() -> None:
    source = np.array([1.0, 9.0, 0.0, 9.0], dtype=np.float32)[::2]
    assert not source.flags.c_contiguous

    embedding = speaker_profiles.SpeakerEmbedding.from_numpy(source, dimension=2)
    source[:] = np.array([0.0, 1.0], dtype=np.float32)

    first = embedding.as_numpy()
    second = embedding.as_numpy()
    assert first is not second
    assert isinstance(first.base, bytes)
    assert first.dtype == np.float32
    assert first.shape == (2,)
    assert first.flags.c_contiguous
    assert not first.flags.writeable
    assert np.array_equal(first, np.array([1.0, 0.0], dtype=np.float32))
    with pytest.raises(ValueError):
        first[0] = np.float32(0.0)
    with pytest.raises(ValueError):
        first.setflags(write=True)


def test_embedding_dimension_is_an_explicit_positive_exact_integer() -> None:
    values = np.array([1.0, 0.0], dtype=np.float32)
    raw = struct.pack("<ff", 1.0, 0.0)
    for invalid in (True, 2.0, "2"):
        with pytest.raises(TypeError):
            speaker_profiles.SpeakerEmbedding.from_numpy(
                values,
                dimension=invalid,  # type: ignore[arg-type]
            )
        with pytest.raises(TypeError):
            speaker_profiles.SpeakerEmbedding.from_bytes(
                raw,
                dimension=invalid,  # type: ignore[arg-type]
            )
    for invalid in (0, -1):
        with pytest.raises(ValueError):
            speaker_profiles.SpeakerEmbedding.from_numpy(
                values,
                dimension=invalid,
            )
        with pytest.raises(ValueError):
            speaker_profiles.SpeakerEmbedding.from_bytes(
                b"",
                dimension=invalid,
            )

    with pytest.raises(TypeError):
        speaker_profiles.SpeakerEmbedding.from_numpy(values)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        speaker_profiles.SpeakerEmbedding.from_bytes(raw)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "values, dimension, error_type",
    [
        ([1.0, 0.0], 2, TypeError),
        (np.array([1.0, 0.0], dtype=np.float64), 2, TypeError),
        (np.array([[1.0, 0.0]], dtype=np.float32), 2, ValueError),
        (np.array([1.0], dtype=np.float32), 2, ValueError),
        (np.array([np.nan, 0.0], dtype=np.float32), 2, ValueError),
        (np.array([np.inf, 0.0], dtype=np.float32), 2, ValueError),
        (np.array([0.0, 0.0], dtype=np.float32), 2, ValueError),
    ],
)
def test_embedding_rejects_invalid_numpy_values(
    values: object,
    dimension: int,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        speaker_profiles.SpeakerEmbedding.from_numpy(  # type: ignore[arg-type]
            values,
            dimension=dimension,
        )


def test_embedding_enforces_unit_norm_tolerance_without_rewriting_values() -> None:
    tolerance = speakers.SPEAKER_EMBEDDING_NORM_TOLERANCE
    accepted = np.array([1.0 + tolerance / 2.0, 0.0], dtype=np.float32)
    embedding = speaker_profiles.SpeakerEmbedding.from_numpy(
        accepted,
        dimension=2,
    )
    assert embedding.to_bytes() == accepted.astype("<f4", copy=False).tobytes()

    for invalid in (
        np.array([1.0 + tolerance * 2.0, 0.0], dtype=np.float32),
        np.array([1.0 - tolerance * 2.0, 0.0], dtype=np.float32),
    ):
        with pytest.raises(ValueError):
            speaker_profiles.SpeakerEmbedding.from_numpy(invalid, dimension=2)


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        struct.pack("<f", 1.0),
        struct.pack("<ff", 1.0, 0.0) + b"\x00",
        struct.pack(">ff", 1.0, 0.0),
        struct.pack("<ff", float("nan"), 0.0),
        struct.pack("<ff", float("inf"), 0.0),
        struct.pack("<ff", 0.0, 0.0),
        struct.pack("<ff", 2.0, 0.0),
    ],
)
def test_embedding_bytes_are_strict_length_little_endian_unit_values(
    raw: bytes,
) -> None:
    with pytest.raises(ValueError):
        speaker_profiles.SpeakerEmbedding.from_bytes(raw, dimension=2)

    with pytest.raises(TypeError):
        speaker_profiles.SpeakerEmbedding.from_bytes(
            bytearray(struct.pack("<ff", 1.0, 0.0)),  # type: ignore[arg-type]
            dimension=2,
        )

    embedding = _embedding()
    with pytest.raises(FrozenInstanceError):
        embedding.dimension = 3  # type: ignore[misc]


def test_profile_has_only_the_eleven_explicit_required_frozen_fields() -> None:
    expected_names = (
        "id",
        "name",
        "description",
        "embedding",
        "embedding_model_id",
        "embedding_model_revision",
        "embedding_dimension",
        "embedding_policy_fingerprint",
        "sample_count",
        "created_at",
        "updated_at",
    )
    profile_fields = fields(speaker_profiles.SpeakerProfile)

    assert tuple(item.name for item in profile_fields) == expected_names
    assert all(
        item.default is MISSING and item.default_factory is MISSING
        for item in profile_fields
    )

    profile = _profile()
    with pytest.raises(FrozenInstanceError):
        profile.name = "Bob"  # type: ignore[misc]


def test_profile_id_is_exactly_eight_uppercase_crockford_characters() -> None:
    for valid in ("01234567", "ABCDEFGH", "JKMNPQRT", "VWXYZ234"):
        assert _profile(id=valid).id == valid

    for invalid in (
        "",
        "0123456",
        "012345678",
        "abcdefgh",
        "0123456I",
        "0123456L",
        "0123456O",
        "0123456U",
        "0123456-",
    ):
        with pytest.raises(ValueError):
            _profile(id=invalid)
    with pytest.raises(TypeError):
        _profile(id=12345678)


def test_profile_name_is_trimmed_bounded_and_casefolded_without_normalization() -> None:
    profile = _profile(name="\u2003Straße\u00a0")
    assert profile.name == "Straße"
    assert profile.name_key == "strasse"
    assert _profile(name=f" {'a' * 80} ").name == "a" * 80

    decomposed = _profile(name="e\u0301")
    assert decomposed.name_key == "e\u0301"
    assert decomposed.name_key != "é".casefold()

    for invalid in ("", " \u2003 ", "a" * 81):
        with pytest.raises(ValueError):
            _profile(name=invalid)
    with pytest.raises(TypeError):
        _profile(name=1)


def test_profile_reserves_only_ascii_labels_and_exact_unknown_label_syntax() -> None:
    for reserved in (
        "A",
        "Z",
        "AA",
        "AF",
        "ALICE",
        "Unknown A",
        "unknown a",
        "uNkNoWn aLiCe",
    ):
        with pytest.raises(ValueError):
            _profile(name=reserved)

    for allowed in (
        "Alice",
        "alice",
        "Unknown  A",
        "Unknown\tA",
        "Unknown Ａ",
        "Ａ",
    ):
        assert _profile(name=allowed).name == allowed


def test_profile_description_is_optional_bounded_and_otherwise_preserved() -> None:
    assert _profile(description=None).description is None
    assert _profile(description="").description is None
    assert _profile(description="  metadata  ").description == "  metadata  "
    assert _profile(description="x" * 500).description == "x" * 500

    with pytest.raises(ValueError):
        _profile(description="x" * 501)
    with pytest.raises(TypeError):
        _profile(description=1)


def test_profile_validates_embedding_identity_dimension_and_fingerprint() -> None:
    for invalid in (None, True, 1):
        with pytest.raises(TypeError):
            _profile(embedding_model_id=invalid)
        with pytest.raises(TypeError):
            _profile(embedding_model_revision=invalid)
        with pytest.raises(TypeError):
            _profile(embedding_policy_fingerprint=invalid)

    for invalid_model_id in (
        "",
        "campplus",
        "../campplus",
        "funasr/campplus/extra",
    ):
        with pytest.raises(ValueError):
            _profile(embedding_model_id=invalid_model_id)
    for invalid_revision in ("", "1" * 39, "1" * 41, "A" * 40):
        with pytest.raises(ValueError):
            _profile(embedding_model_revision=invalid_revision)
    for invalid_fingerprint in (
        "",
        "a" * 63,
        "a" * 65,
        "A" * 64,
        "g" * 64,
    ):
        with pytest.raises(ValueError):
            _profile(embedding_policy_fingerprint=invalid_fingerprint)

    for invalid in (True, 2.0, "2"):
        with pytest.raises(TypeError):
            _profile(embedding_dimension=invalid)
    for invalid in (0, -1, 3):
        with pytest.raises(ValueError):
            _profile(embedding_dimension=invalid)
    with pytest.raises(TypeError):
        _profile(embedding=np.array([1.0, 0.0], dtype=np.float32))


def test_profile_sample_count_is_a_strict_two_to_five() -> None:
    assert _profile(sample_count=2).sample_count == 2
    assert _profile(sample_count=5).sample_count == 5
    for invalid in (True, 2.0, "2"):
        with pytest.raises(TypeError):
            _profile(sample_count=invalid)
    for invalid in (1, 6):
        with pytest.raises(ValueError):
            _profile(sample_count=invalid)


def test_profile_timestamps_are_aware_ordered_and_canonical_utc() -> None:
    plus_eight = timezone(timedelta(hours=8))
    profile = _profile(
        created_at=datetime(2026, 7, 27, 20, 0, tzinfo=plus_eight),
        updated_at=datetime(2026, 7, 27, 20, 1, tzinfo=plus_eight),
    )
    assert profile.created_at == CREATED_AT
    assert profile.updated_at == UPDATED_AT
    assert profile.created_at.tzinfo is timezone.utc
    assert profile.updated_at.tzinfo is timezone.utc
    assert _profile(updated_at=CREATED_AT).updated_at == CREATED_AT

    for name in ("created_at", "updated_at"):
        with pytest.raises(TypeError):
            _profile(**{name: "2026-07-27T12:00:00Z"})
        with pytest.raises(ValueError):
            _profile(**{name: datetime(2026, 7, 27, 12, 0)})
    with pytest.raises(ValueError):
        _profile(created_at=UPDATED_AT, updated_at=CREATED_AT)


def test_profile_compatibility_compares_only_the_four_embedding_fields() -> None:
    policy = _policy()
    compatible = _profile(embedding_policy_fingerprint=policy.fingerprint)
    assert speaker_profiles.is_speaker_profile_compatible(compatible, policy)

    mismatches = (
        _profile(
            embedding_model_id="funasr/other",
            embedding_policy_fingerprint=policy.fingerprint,
        ),
        _profile(
            embedding_model_revision="2" * 40,
            embedding_policy_fingerprint=policy.fingerprint,
        ),
        _profile(
            embedding=_embedding(3),
            embedding_dimension=3,
            embedding_policy_fingerprint=policy.fingerprint,
        ),
        _profile(embedding_policy_fingerprint="b" * 64),
    )
    assert all(
        not speaker_profiles.is_speaker_profile_compatible(profile, policy)
        for profile in mismatches
    )

    unrelated_changes = replace(
        compatible,
        id="ABCDEFGH",
        name="Bob",
        description=None,
        embedding=_embedding(
            values=np.array([0.0, 1.0], dtype=np.float32),
        ),
        sample_count=5,
        updated_at=UPDATED_AT + timedelta(minutes=1),
    )
    assert speaker_profiles.is_speaker_profile_compatible(
        unrelated_changes,
        policy,
    )
