from __future__ import annotations

from dataclasses import MISSING, FrozenInstanceError, fields, replace

import numpy as np
import pytest

from botified_asr import pipeline, speakers


EMBEDDING_DIMENSION = 192


def _unit(*components: float) -> np.ndarray:
    embedding = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
    embedding[: len(components)] = components
    embedding /= np.linalg.norm(embedding)
    return embedding


def _window(
    embedding: np.ndarray,
    *,
    start_sample: int = 0,
    end_sample: int = 1,
) -> speakers.SpeakerEmbeddingWindow:
    return speakers.SpeakerEmbeddingWindow(
        start_sample=start_sample,
        end_sample=end_sample,
        embedding=embedding,
    )


def _state(
    *,
    threshold: float,
    max_speakers: int = 32,
) -> speakers.AnonymousSpeakerState:
    return speakers.AnonymousSpeakerState(
        speakers.AnonymousSpeakerPolicy(
            threshold=threshold,
            max_speakers=max_speakers,
        )
    )


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

def test_policy_requires_a_finite_threshold_and_exact_bounded_max_speakers() -> None:
    with pytest.raises(TypeError):
        speakers.AnonymousSpeakerPolicy(max_speakers=32)  # type: ignore[call-arg]

    for threshold in (float("-inf"), float("inf"), float("nan"), -1.01, 1.01):
        with pytest.raises(ValueError):
            speakers.AnonymousSpeakerPolicy(
                threshold=threshold,
                max_speakers=32,
            )

    for invalid_max_speakers in (True, 1.0, 0, 33):
        expected_error = (
            TypeError if type(invalid_max_speakers) is not int else ValueError
        )
        with pytest.raises(expected_error):
            speakers.AnonymousSpeakerPolicy(
                threshold=0.0,
                max_speakers=invalid_max_speakers,  # type: ignore[arg-type]
            )

    assert (
        speakers.AnonymousSpeakerPolicy(
            threshold=-1.0,
            max_speakers=1,
        ).threshold
        == -1.0
    )
    assert (
        speakers.AnonymousSpeakerPolicy(
            threshold=1.0,
            max_speakers=32,
        ).threshold
        == 1.0
    )


def test_stable_labels_use_count_weighted_centroids_and_allow_revisits() -> None:
    state = _state(threshold=0.0)
    x = _unit(1.0, 0.0)
    y = _unit(0.0, 1.0)
    weighted_centroid_probe = _unit(1.0, -2.0)

    labels = [
        state.assign_segment((_window(x),)),
        state.assign_segment((_window(x),)),
        state.assign_segment((_window(x),)),
        state.assign_segment((_window(y),)),
        state.assign_segment((_window(weighted_centroid_probe),)),
        state.assign_segment((_window(_unit(-1.0, 0.0)),)),
        state.assign_segment((_window(x),)),
    ]

    assert labels == ["A", "A", "A", "A", "A", "B", "A"]
    assert state.speaker_count == 2


def test_finalized_cluster_dto_is_exact_and_empty_finalize_is_terminal() -> None:
    cluster_fields = fields(speakers.AnonymousSpeakerCluster)
    assert tuple(item.name for item in cluster_fields) == ("label", "centroid")
    assert all(
        item.default is MISSING and item.default_factory is MISSING
        for item in cluster_fields
    )
    cluster = speakers.AnonymousSpeakerCluster(
        label="A",
        centroid=(1.0,) + (0.0,) * (EMBEDDING_DIMENSION - 1),
    )
    assert not hasattr(cluster, "__dict__")
    with pytest.raises(FrozenInstanceError):
        cluster.label = "B"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        cluster.centroid = ()  # type: ignore[misc]

    state = _state(threshold=0.5)
    assert state.finalize_clusters() == ()
    assert state.speaker_count == 0
    with pytest.raises(RuntimeError):
        state.finalize_clusters()
    with pytest.raises(RuntimeError):
        state.assign_segment((_window(_unit(1.0, 0.0)),))
    assert state.speaker_count == 0


def test_finalized_clusters_preserve_creation_order_and_cumulative_centroids() -> None:
    state = _state(threshold=0.0)
    probe_state = _state(threshold=0.0)
    source_a = _unit(1.0, 0.0)
    update_a_1 = _unit(0.0, 1.0)
    update_a_2 = _unit(-0.6, 0.8)
    source_b = _unit(-1.0, 0.0)
    discriminator = _unit(-0.96, 0.28)
    validated_a = tuple(
        embedding.astype(np.float64) / np.linalg.norm(embedding.astype(np.float64))
        for embedding in (source_a, update_a_1, update_a_2)
    )
    expected_a = sum(validated_a)
    expected_a /= np.linalg.norm(expected_a)

    assert state.assign_segment((_window(source_a),)) == "A"
    assert state.assign_segment((_window(update_a_1),)) == "A"
    assert state.assign_segment((_window(update_a_2),)) == "A"
    assert probe_state.assign_segment((_window(source_a),)) == "A"
    assert probe_state.assign_segment((_window(update_a_1),)) == "A"
    assert probe_state.assign_segment((_window(update_a_2),)) == "A"
    assert probe_state.assign_segment((_window(discriminator),)) == "A"
    assert probe_state.speaker_count == 1
    assert state.assign_segment((_window(source_b),)) == "B"
    source_a[:] = np.nan
    update_a_1[:] = np.nan
    update_a_2[:] = np.nan
    source_b[:] = np.nan
    discriminator[:] = np.nan

    clusters = state.finalize_clusters()
    assert type(clusters) is tuple
    assert tuple(cluster.label for cluster in clusters) == ("A", "B")
    assert state.speaker_count == 2
    assert all(type(cluster.centroid) is tuple for cluster in clusters)
    assert all(
        len(cluster.centroid) == EMBEDDING_DIMENSION
        and all(type(value) is float for value in cluster.centroid)
        and np.isfinite(cluster.centroid).all()
        and np.linalg.norm(cluster.centroid) == pytest.approx(1.0)
        for cluster in clusters
    )
    np.testing.assert_allclose(
        clusters[0].centroid,
        expected_a,
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        clusters[1].centroid,
        _unit(-1.0, 0.0).astype(np.float64),
        rtol=0.0,
        atol=1e-15,
    )
    with pytest.raises(RuntimeError):
        state.finalize_clusters()
    with pytest.raises(RuntimeError):
        state.assign_segment((_window(_unit(1.0, 0.0)),))
    assert state.speaker_count == 2


def test_similarity_threshold_is_inclusive_and_below_creates_a_speaker() -> None:
    at_threshold = _state(threshold=0.0)
    assert at_threshold.assign_segment((_window(_unit(1.0, 0.0)),)) == "A"
    assert at_threshold.assign_segment((_window(_unit(0.0, 1.0)),)) == "A"
    assert at_threshold.speaker_count == 1

    below_threshold = _state(threshold=0.0)
    assert below_threshold.assign_segment((_window(_unit(1.0, 0.0)),)) == "A"
    assert below_threshold.assign_segment((_window(_unit(-1.0, 0.0)),)) == "B"
    assert below_threshold.speaker_count == 2


def test_nearest_similarity_tie_uses_the_smallest_speaker_ordinal() -> None:
    state = _state(threshold=0.5)
    assert state.assign_segment((_window(_unit(1.0, 0.0)),)) == "A"
    assert state.assign_segment((_window(_unit(0.0, 1.0)),)) == "B"

    assert state.assign_segment((_window(_unit(1.0, 1.0)),)) == "A"
    assert state.speaker_count == 2


def test_segment_vote_uses_real_window_duration_and_ties_choose_smallest() -> None:
    state = _state(threshold=0.5)
    speaker_a = _unit(1.0, 0.0)
    speaker_b = _unit(0.0, 1.0)
    assert state.assign_segment((_window(speaker_a),)) == "A"
    assert state.assign_segment((_window(speaker_b),)) == "B"

    assert (
        state.assign_segment(
            (
                _window(speaker_a, start_sample=0, end_sample=3),
                _window(speaker_b, start_sample=3, end_sample=7),
            )
        )
        == "B"
    )
    assert (
        state.assign_segment(
            (
                _window(speaker_b, start_sample=0, end_sample=4),
                _window(speaker_a, start_sample=4, end_sample=8),
            )
        )
        == "A"
    )
    assert state.speaker_count == 2


def test_end_anchored_overlapping_windows_are_accepted_in_order() -> None:
    state = _state(threshold=0.5)
    speaker = _unit(1.0, 0.0)

    assert (
        state.assign_segment(
            (
                _window(speaker, start_sample=0, end_sample=24_000),
                _window(speaker, start_sample=1, end_sample=24_001),
            )
        )
        == "A"
    )
    assert state.speaker_count == 1


def test_thirty_third_unmatched_speaker_fails_without_polluting_state() -> None:
    state = _state(threshold=0.5, max_speakers=32)
    basis = tuple(
        _unit(*(1.0 if component == index else 0.0 for component in range(33)))
        for index in range(33)
    )
    expected_labels = tuple(chr(ord("A") + index) for index in range(26)) + (
        "AA",
        "AB",
        "AC",
        "AD",
        "AE",
        "AF",
    )

    assert (
        tuple(state.assign_segment((_window(embedding),)) for embedding in basis[:32])
        == expected_labels
    )
    assert state.speaker_count == 32

    with pytest.raises(pipeline.PipelineError) as caught:
        state.assign_segment((_window(basis[32]),))

    assert caught.value.code == "too_many_speakers"
    assert state.speaker_count == 32
    assert state.assign_segment((_window(basis[31]),)) == "AF"
    assert state.speaker_count == 32


def _invalid_segments() -> tuple[object, ...]:
    x = _unit(1.0, 0.0)
    zero = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
    nonfinite = x.copy()
    nonfinite[1] = np.nan
    return (
        [],
        (),
        (_window(x, start_sample=1, end_sample=1),),
        (_window(x, start_sample=0, end_sample=24_001),),
        (
            _window(x, start_sample=1, end_sample=2),
            _window(x, start_sample=0, end_sample=1),
        ),
        (_window(x[:-1]),),
        (_window(x.astype(np.float64)),),
        (_window(nonfinite),),
        (_window(x * np.float32(2.0)),),
        (
            _window(_unit(0.8, 0.6), start_sample=0, end_sample=1),
            _window(zero, start_sample=1, end_sample=2),
        ),
    )


@pytest.mark.parametrize(
    "invalid_windows",
    _invalid_segments(),
    ids=(
        "not-tuple",
        "empty",
        "zero-duration",
        "window-too-long",
        "out-of-order",
        "wrong-shape",
        "wrong-dtype",
        "non-finite",
        "non-unit",
        "zero-norm-embedding",
    ),
)
def test_invalid_segment_fails_closed_without_polluting_existing_state(
    invalid_windows: object,
) -> None:
    state = _state(threshold=0.5)
    speaker_a = _unit(1.0, 0.0)
    assert state.assign_segment((_window(speaker_a),)) == "A"

    with pytest.raises(pipeline.PipelineError) as caught:
        state.assign_segment(invalid_windows)  # type: ignore[arg-type]

    assert caught.value.code == "invalid_model_output"
    assert state.speaker_count == 1
    assert state.assign_segment((_window(speaker_a),)) == "A"
    assert state.speaker_count == 1


def test_late_invalid_window_rolls_back_centroid_and_count() -> None:
    state = _state(threshold=0.8)
    speaker_a = _unit(1.0, 0.0)
    centroid_rotator = _unit(0.8, 0.6)
    discriminator = _unit(0.6, 0.8)
    zero = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
    assert state.assign_segment((_window(speaker_a),)) == "A"

    with pytest.raises(pipeline.PipelineError) as caught:
        state.assign_segment(
            (
                _window(centroid_rotator, start_sample=0, end_sample=1),
                _window(zero, start_sample=1, end_sample=2),
            )
        )

    assert caught.value.code == "invalid_model_output"
    assert state.assign_segment((_window(discriminator),)) == "B"
    assert state.speaker_count == 2


def test_late_speaker_overflow_rolls_back_earlier_centroid_update() -> None:
    state = _state(threshold=0.8, max_speakers=1)
    speaker_a = _unit(1.0, 0.0)
    centroid_rotator = _unit(0.8, 0.6)
    discriminator = _unit(0.6, 0.8)
    unmatched = _unit(-1.0, 0.0)
    assert state.assign_segment((_window(speaker_a),)) == "A"

    with pytest.raises(pipeline.PipelineError) as caught:
        state.assign_segment(
            (
                _window(centroid_rotator, start_sample=0, end_sample=1),
                _window(unmatched, start_sample=1, end_sample=2),
            )
        )

    assert caught.value.code == "too_many_speakers"
    assert state.speaker_count == 1
    with pytest.raises(pipeline.PipelineError) as probe:
        state.assign_segment((_window(discriminator),))
    assert probe.value.code == "too_many_speakers"
    assert state.speaker_count == 1
