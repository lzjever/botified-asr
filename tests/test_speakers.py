from __future__ import annotations

from dataclasses import MISSING, FrozenInstanceError, fields, replace

import pytest

from botified_asr import speakers


EMBEDDING_DIMENSION = 192


def _synthetic_nonproduction_embedding_policy(
    **changes: object,
) -> speakers.SpeakerEmbeddingPolicy:
    values: dict[str, object] = {
        "model_id": "synthetic/test-campplus",
        "model_revision": "1" * 40,
        "embedding_dimension": 192,
        "sample_rate": 16_000,
        "downmix_policy_version": "mono-average-v1",
        "window_samples": 24_000,
        "window_shift_samples": 12_000,
        "padding_policy_version": "right-zero-pad-v1",
        "normalization_policy_version": "int16-div-32768-l2-v1",
        "enrollment_aggregation_policy_version": ("sample-centroid-equal-average-v1"),
    }
    values.update(changes)
    return speakers.SpeakerEmbeddingPolicy(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("ordinal", "label"),
    (
        (0, "A"),
        (25, "Z"),
        (26, "AA"),
        (31, "AF"),
        (32, "AG"),
        (39, "AN"),
    ),
)
def test_anonymous_speaker_labels_are_algorithmic_and_strict(
    ordinal: int,
    label: str,
) -> None:
    assert speakers.anonymous_speaker_label(ordinal) == label
    assert speakers.is_anonymous_speaker_label(label)


@pytest.mark.parametrize(
    "value",
    (None, True, 1, "", "a", "A0", "Unknown A", " A", "A ", "Ａ"),
)
def test_anonymous_speaker_label_validation_rejects_noncanonical_values(
    value: object,
) -> None:
    assert not speakers.is_anonymous_speaker_label(value)


def test_embedding_policy_has_only_explicit_required_frozen_fields() -> None:
    expected_names = (
        "model_id",
        "model_revision",
        "embedding_dimension",
        "sample_rate",
        "downmix_policy_version",
        "window_samples",
        "window_shift_samples",
        "padding_policy_version",
        "normalization_policy_version",
        "enrollment_aggregation_policy_version",
    )
    policy_fields = fields(speakers.SpeakerEmbeddingPolicy)

    assert tuple(item.name for item in policy_fields) == expected_names
    assert all(
        item.default is MISSING and item.default_factory is MISSING
        for item in policy_fields
    )

    policy = _synthetic_nonproduction_embedding_policy()
    with pytest.raises(FrozenInstanceError):
        policy.sample_rate = 8_000  # type: ignore[misc]


def test_embedding_policy_canonical_bytes_and_fingerprint_are_exact() -> None:
    policy = _synthetic_nonproduction_embedding_policy()

    assert policy.canonical_bytes == (
        b'{"downmix_policy_version":"mono-average-v1",'
        b'"embedding_dimension":192,'
        b'"enrollment_aggregation_policy_version":'
        b'"sample-centroid-equal-average-v1",'
        b'"model_id":"synthetic/test-campplus",'
        b'"model_revision":"1111111111111111111111111111111111111111",'
        b'"normalization_policy_version":"int16-div-32768-l2-v1",'
        b'"padding_policy_version":"right-zero-pad-v1",'
        b'"sample_rate":16000,"window_samples":24000,'
        b'"window_shift_samples":12000}'
    )
    assert policy.canonical_bytes.decode("utf-8").encode("utf-8") == (
        policy.canonical_bytes
    )
    assert policy.fingerprint == (
        "7887ccc2255b7894bc2eddd01d9bd7a17fb80e7587864f4ee3d7020a64a8b197"
    )
    assert len(policy.fingerprint) == 64
    assert policy.fingerprint == policy.fingerprint.lower()


def test_embedding_policy_fingerprint_is_stable_and_every_field_is_bound() -> None:
    policy = _synthetic_nonproduction_embedding_policy()
    assert _synthetic_nonproduction_embedding_policy() == policy
    assert _synthetic_nonproduction_embedding_policy().fingerprint == (
        policy.fingerprint
    )

    valid_changes: dict[str, object] = {
        "model_id": "synthetic/other-campplus",
        "model_revision": "2" * 40,
        "embedding_dimension": 256,
        "sample_rate": 8_000,
        "downmix_policy_version": "mono-left-v1",
        "window_samples": 23_000,
        "window_shift_samples": 11_000,
        "padding_policy_version": "right-zero-pad-v2",
        "normalization_policy_version": "int16-div-32768-l2-v2",
        "enrollment_aggregation_policy_version": ("sample-centroid-equal-average-v2"),
    }

    changed_fingerprints = {
        replace(policy, **{name: value}).fingerprint
        for name, value in valid_changes.items()
    }
    assert policy.fingerprint not in changed_fingerprints
    assert len(changed_fingerprints) == len(valid_changes)


def test_embedding_policy_integer_fields_are_strict_positive_and_ordered() -> None:
    integer_fields = (
        "embedding_dimension",
        "sample_rate",
        "window_samples",
        "window_shift_samples",
    )
    for name in integer_fields:
        for invalid in (True, 1.0, "1"):
            with pytest.raises(TypeError):
                _synthetic_nonproduction_embedding_policy(**{name: invalid})
        for invalid in (0, -1):
            with pytest.raises(ValueError):
                _synthetic_nonproduction_embedding_policy(**{name: invalid})

    with pytest.raises(ValueError):
        _synthetic_nonproduction_embedding_policy(
            window_samples=12_000,
            window_shift_samples=12_001,
        )


def test_embedding_policy_rejects_invalid_identity_and_processing_values() -> None:
    for invalid in (None, True, 1):
        with pytest.raises(TypeError):
            _synthetic_nonproduction_embedding_policy(model_id=invalid)
        with pytest.raises(TypeError):
            _synthetic_nonproduction_embedding_policy(model_revision=invalid)

    for invalid_model_id in (
        "",
        "campplus",
        "/campplus",
        "synthetic/",
        "synthetic//campplus",
        "synthetic/campplus/extra",
        "../campplus",
        "synthetic/..",
    ):
        with pytest.raises(ValueError):
            _synthetic_nonproduction_embedding_policy(model_id=invalid_model_id)

    for invalid_revision in ("", "1" * 39, "1" * 41, "G" * 40, "A" * 40):
        with pytest.raises(ValueError):
            _synthetic_nonproduction_embedding_policy(model_revision=invalid_revision)

    version_fields = (
        "downmix_policy_version",
        "padding_policy_version",
        "normalization_policy_version",
        "enrollment_aggregation_policy_version",
    )
    for invalid in (None, True, 1):
        for name in version_fields:
            with pytest.raises(TypeError):
                _synthetic_nonproduction_embedding_policy(**{name: invalid})

    for invalid_token in (
        "",
        "mono-average",
        "mono-average-v0",
        "Mono-average-v1",
        "mono_average_v1",
        "mono-平均-v1",
        " mono-average-v1",
        "mono-average-v1 ",
    ):
        for name in version_fields:
            with pytest.raises(ValueError):
                _synthetic_nonproduction_embedding_policy(**{name: invalid_token})


def test_finalized_cluster_dto_is_exact_and_frozen() -> None:
    cluster_fields = fields(speakers.AnonymousSpeakerCluster)
    assert tuple(item.name for item in cluster_fields) == ("label", "centroid")
    assert all(
        item.default is MISSING and item.default_factory is MISSING
        for item in cluster_fields
    )
    cluster = speakers.AnonymousSpeakerCluster(
        label="AG",
        centroid=(1.0,) + (0.0,) * (EMBEDDING_DIMENSION - 1),
    )
    assert not hasattr(cluster, "__dict__")
    with pytest.raises(FrozenInstanceError):
        cluster.label = "AH"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        cluster.centroid = ()  # type: ignore[misc]


@pytest.mark.parametrize("ordinal", (True, 1.0, "1"))
def test_anonymous_speaker_label_requires_an_exact_integer(ordinal: object) -> None:
    with pytest.raises(TypeError):
        speakers.anonymous_speaker_label(ordinal)  # type: ignore[arg-type]


def test_anonymous_speaker_label_rejects_negative_ordinals() -> None:
    with pytest.raises(ValueError):
        speakers.anonymous_speaker_label(-1)
