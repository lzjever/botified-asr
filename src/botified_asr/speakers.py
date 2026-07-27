from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np

from botified_asr.errors import PipelineError

SPEAKER_EMBEDDING_DIMENSION = 192
SPEAKER_WINDOW_MAX_SAMPLES = 24_000
SPEAKER_EMBEDDING_NORM_TOLERANCE = 1e-5
ANONYMOUS_SPEAKER_LABELS = tuple(chr(ord("A") + ordinal) for ordinal in range(26)) + (
    "AA",
    "AB",
    "AC",
    "AD",
    "AE",
    "AF",
)
_ANONYMOUS_SPEAKER_LABEL_SET = frozenset(ANONYMOUS_SPEAKER_LABELS)


@dataclass(frozen=True)
class SpeakerEmbeddingWindow:
    start_sample: int
    end_sample: int
    embedding: np.ndarray


@dataclass(frozen=True)
class AnonymousSpeakerPolicy:
    threshold: float
    max_speakers: int

    def __post_init__(self) -> None:
        if isinstance(self.threshold, bool) or not isinstance(self.threshold, Real):
            raise TypeError("anonymous speaker threshold must be a real number")
        try:
            threshold = float(self.threshold)
        except (OverflowError, ValueError) as error:
            raise ValueError("anonymous speaker threshold must be finite") from error
        if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
            raise ValueError("anonymous speaker threshold must be between -1 and 1")
        if type(self.max_speakers) is not int:
            raise TypeError("maximum anonymous speakers must be an integer")
        if not 1 <= self.max_speakers <= 32:
            raise ValueError("maximum anonymous speakers must be between 1 and 32")
        object.__setattr__(
            self,
            "threshold",
            0.0 if threshold == 0.0 else threshold,
        )


@dataclass(frozen=True)
class _ValidatedWindow:
    duration_samples: int
    embedding: np.ndarray


class AnonymousSpeakerState:
    def __init__(self, policy: AnonymousSpeakerPolicy) -> None:
        if not isinstance(policy, AnonymousSpeakerPolicy):
            raise TypeError("anonymous speaker policy is invalid")
        self._policy = policy
        self._centroids: list[np.ndarray] = []
        self._window_counts: list[int] = []

    @property
    def speaker_count(self) -> int:
        return len(self._centroids)

    def assign_segment(
        self,
        windows: tuple[SpeakerEmbeddingWindow, ...],
    ) -> str:
        validated = _validate_windows(windows)
        staged_centroids = [centroid.copy() for centroid in self._centroids]
        staged_counts = list(self._window_counts)
        votes = [0] * len(staged_centroids)

        for window in validated:
            ordinal = _assign_window(
                window.embedding,
                staged_centroids,
                staged_counts,
                policy=self._policy,
            )
            if ordinal == len(votes):
                votes.append(0)
            votes[ordinal] += window.duration_samples

        winning_ordinal = max(
            range(len(votes)),
            key=lambda ordinal: (votes[ordinal], -ordinal),
        )
        label = _speaker_label(winning_ordinal)
        self._centroids = staged_centroids
        self._window_counts = staged_counts
        return label


def _validate_windows(
    windows: object,
) -> tuple[_ValidatedWindow, ...]:
    if type(windows) is not tuple or not windows:
        _raise_invalid_output()

    validated: list[_ValidatedWindow] = []
    previous_start: int | None = None
    previous_end: int | None = None
    for window in windows:
        if not isinstance(window, SpeakerEmbeddingWindow):
            _raise_invalid_output()
        start_sample = window.start_sample
        end_sample = window.end_sample
        if (
            type(start_sample) is not int
            or type(end_sample) is not int
            or start_sample < 0
            or end_sample <= start_sample
            or end_sample - start_sample > SPEAKER_WINDOW_MAX_SAMPLES
            or previous_start is not None
            and start_sample <= previous_start
            or previous_end is not None
            and end_sample <= previous_end
        ):
            _raise_invalid_output()
        embedding = window.embedding
        if (
            not isinstance(embedding, np.ndarray)
            or embedding.dtype != np.float32
            or embedding.shape != (SPEAKER_EMBEDDING_DIMENSION,)
            or not embedding.flags.c_contiguous
            or not np.isfinite(embedding).all()
        ):
            _raise_invalid_output()
        normalized = embedding.astype(np.float64, copy=True)
        norm = float(np.linalg.norm(normalized))
        if (
            not math.isfinite(norm)
            or norm <= 0.0
            or not math.isclose(
                norm,
                1.0,
                rel_tol=0.0,
                abs_tol=SPEAKER_EMBEDDING_NORM_TOLERANCE,
            )
        ):
            _raise_invalid_output()
        normalized /= norm
        if not np.isfinite(normalized).all():
            _raise_invalid_output()
        validated.append(
            _ValidatedWindow(
                duration_samples=end_sample - start_sample,
                embedding=normalized,
            )
        )
        previous_start = start_sample
        previous_end = end_sample
    return tuple(validated)


def _assign_window(
    embedding: np.ndarray,
    centroids: list[np.ndarray],
    window_counts: list[int],
    *,
    policy: AnonymousSpeakerPolicy,
) -> int:
    nearest_ordinal: int | None = None
    nearest_similarity = -math.inf
    for ordinal, centroid in enumerate(centroids):
        similarity = float(np.dot(centroid, embedding))
        if not math.isfinite(similarity):
            _raise_invalid_output()
        if similarity > nearest_similarity:
            nearest_ordinal = ordinal
            nearest_similarity = similarity

    if nearest_ordinal is None or nearest_similarity < policy.threshold:
        if len(centroids) >= policy.max_speakers:
            raise PipelineError(
                "too_many_speakers",
                "Audio contains too many anonymous speakers",
            )
        centroids.append(embedding.copy())
        window_counts.append(1)
        return len(centroids) - 1

    count = window_counts[nearest_ordinal]
    combined = centroids[nearest_ordinal] * count + embedding
    combined_norm = float(np.linalg.norm(combined))
    if not math.isfinite(combined_norm) or combined_norm <= 0.0:
        _raise_invalid_output()
    combined /= combined_norm
    if not np.isfinite(combined).all():
        _raise_invalid_output()
    centroids[nearest_ordinal] = combined
    window_counts[nearest_ordinal] = count + 1
    return nearest_ordinal


def _speaker_label(ordinal: int) -> str:
    return ANONYMOUS_SPEAKER_LABELS[ordinal]


def is_anonymous_speaker_label(value: object) -> bool:
    return type(value) is str and value in _ANONYMOUS_SPEAKER_LABEL_SET


def _raise_invalid_output() -> None:
    raise PipelineError(
        "invalid_model_output",
        "Speaker model returned an invalid result",
    )
