from __future__ import annotations

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
