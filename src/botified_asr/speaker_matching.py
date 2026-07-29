from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np

from botified_asr.speaker_profiles import SpeakerEmbedding
from botified_asr.speaker_snapshot import (
    SelectedSpeaker,
    SelectedSpeakerSnapshot,
)
from botified_asr.speakers import AnonymousSpeakerCluster


@dataclass(frozen=True, slots=True)
class KnownSpeakerMatchPolicy:
    match_threshold: float

    def __post_init__(self) -> None:
        threshold = _finite_real(
            self.match_threshold,
            name="known speaker match threshold",
        )
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("known speaker match threshold must be between -1 and 1")
        object.__setattr__(
            self,
            "match_threshold",
            0.0 if threshold == 0.0 else threshold,
        )


@dataclass(frozen=True, slots=True)
class KnownSpeakerMatch:
    speaker_id: str
    speaker_name: str
    similarity: float


@dataclass(frozen=True, slots=True)
class SpeakerLabelResolution:
    anonymous_speaker: str
    match: KnownSpeakerMatch | None


@dataclass(frozen=True, slots=True)
class SpeakerLabelMapping:
    resolutions: tuple[SpeakerLabelResolution, ...]


class SpeakerMatchInputError(ValueError):
    def __init__(self) -> None:
        super().__init__("speaker match input is invalid")


def match_selected_speakers(
    clusters: tuple[AnonymousSpeakerCluster, ...],
    selected_snapshot: SelectedSpeakerSnapshot,
    policy: KnownSpeakerMatchPolicy,
) -> SpeakerLabelMapping:
    if type(clusters) is not tuple:
        raise TypeError("anonymous speaker clusters must be a tuple")
    if not isinstance(selected_snapshot, SelectedSpeakerSnapshot):
        raise TypeError("selected speaker snapshot is invalid")
    if not isinstance(policy, KnownSpeakerMatchPolicy):
        raise TypeError("known speaker match policy is invalid")
    if type(selected_snapshot.speakers) is not tuple:
        raise SpeakerMatchInputError
    if not clusters or not selected_snapshot.speakers:
        return SpeakerLabelMapping(())

    cluster_vectors = _validated_cluster_vectors(clusters)
    selected_vectors = _validated_selected_vectors(
        selected_snapshot,
        dimension=cluster_vectors[0][1].shape[0],
    )
    if any(
        vector.shape != cluster_vectors[0][1].shape for _, vector, _ in cluster_vectors
    ):
        raise SpeakerMatchInputError

    resolutions: list[SpeakerLabelResolution] = []
    for cluster, cluster_vector, cluster_norm in cluster_vectors:
        similarities = tuple(
            _cosine(
                cluster_vector,
                cluster_norm,
                selected_vector,
                selected_norm,
            )
            for _, selected_vector, selected_norm in selected_vectors
        )
        match = _select_match(
            selected_snapshot.speakers,
            similarities,
            policy=policy,
        )
        resolutions.append(
            SpeakerLabelResolution(
                anonymous_speaker=cluster.label,
                match=match,
            )
        )
    return SpeakerLabelMapping(tuple(resolutions))


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _validated_cluster_vectors(
    clusters: tuple[AnonymousSpeakerCluster, ...],
) -> tuple[tuple[AnonymousSpeakerCluster, np.ndarray, float], ...]:
    validated: list[tuple[AnonymousSpeakerCluster, np.ndarray, float]] = []
    for cluster in clusters:
        if not isinstance(cluster, AnonymousSpeakerCluster):
            raise SpeakerMatchInputError
        vector, norm = _validated_vector(cluster.centroid)
        validated.append((cluster, vector, norm))
    return tuple(validated)


def _validated_selected_vectors(
    snapshot: SelectedSpeakerSnapshot,
    *,
    dimension: int,
) -> tuple[tuple[SelectedSpeaker, np.ndarray, float], ...]:
    if type(snapshot.speakers) is not tuple:
        raise SpeakerMatchInputError
    validated: list[tuple[SelectedSpeaker, np.ndarray, float]] = []
    for selected in snapshot.speakers:
        if (
            not isinstance(selected, SelectedSpeaker)
            or not isinstance(selected.id, str)
            or not isinstance(selected.name, str)
            or not isinstance(selected.embedding, SpeakerEmbedding)
            or selected.embedding.dimension != dimension
        ):
            raise SpeakerMatchInputError
        vector, norm = _validated_vector(selected.embedding.as_numpy())
        validated.append((selected, vector, norm))
    return tuple(validated)


def _validated_vector(values: object) -> tuple[np.ndarray, float]:
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as error:
        raise SpeakerMatchInputError from error
    if (
        vector.ndim != 1
        or not len(vector)
        or not np.isfinite(vector).all()
        or not np.any(vector != 0.0)
    ):
        raise SpeakerMatchInputError
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise SpeakerMatchInputError
    return vector, norm


def _cosine(
    left: np.ndarray,
    left_norm: float,
    right: np.ndarray,
    right_norm: float,
) -> float:
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        denominator = left_norm * right_norm
        similarity = float(np.dot(left, right) / denominator)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise SpeakerMatchInputError
    if not math.isfinite(similarity):
        raise SpeakerMatchInputError
    return min(1.0, max(-1.0, similarity))


def _select_match(
    selected_speakers: tuple[SelectedSpeaker, ...],
    similarities: tuple[float, ...],
    *,
    policy: KnownSpeakerMatchPolicy,
) -> KnownSpeakerMatch | None:
    best_index = max(
        range(len(similarities)),
        key=similarities.__getitem__,
    )
    best_similarity = similarities[best_index]
    if best_similarity < policy.match_threshold:
        return None
    if len(similarities) > 1:
        second_similarity = max(
            similarity
            for index, similarity in enumerate(similarities)
            if index != best_index
        )
        if best_similarity == second_similarity:
            return None
    selected = selected_speakers[best_index]
    return KnownSpeakerMatch(
        speaker_id=selected.id,
        speaker_name=selected.name,
        similarity=best_similarity,
    )
