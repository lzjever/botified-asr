from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Protocol, runtime_checkable

import numpy as np

from botified_asr import audio
from botified_asr.errors import PipelineError

SPEAKER_SAMPLE_RATE = audio.SAMPLE_RATE
SPEAKER_EMBEDDING_DIMENSION = 192
SPEAKER_WINDOW_MAX_SAMPLES = 24_000
SPEAKER_WINDOW_SHIFT_SAMPLES = 12_000
SPEAKER_EMBEDDING_NORM_TOLERANCE = 1e-5
SPEAKER_DOWNMIX_POLICY_VERSION = "ffmpeg-first-audio-stream-ac1-v1"
SPEAKER_PADDING_POLICY_VERSION = "right-zero-pad-v1"
SPEAKER_NORMALIZATION_POLICY_VERSION = "int16-div-32768-l2-v1"
SPEAKER_ENROLLMENT_AGGREGATION_POLICY_VERSION = "sample-centroid-equal-average-v1"
_ANONYMOUS_SPEAKER_LABEL = re.compile(r"\A[A-Z]+\Z", flags=re.ASCII)
_MODEL_ID_PART = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_MODEL_REVISION = re.compile(r"\A[0-9a-f]{40}\Z")
_POLICY_VERSION_TOKEN = re.compile(r"\A[a-z0-9]+(?:-[a-z0-9]+)*-v[1-9][0-9]*\Z")


def validate_speaker_model_id(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("speaker embedding model ID must be a string")
    model_id_parts = value.split("/")
    if len(model_id_parts) != 2 or any(
        part in {"", ".", ".."} or _MODEL_ID_PART.fullmatch(part) is None
        for part in model_id_parts
    ):
        raise ValueError("speaker embedding model ID is invalid")
    return value


def validate_speaker_model_revision(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("speaker embedding model revision must be a string")
    if _MODEL_REVISION.fullmatch(value) is None:
        raise ValueError("speaker embedding model revision is invalid")
    return value


@dataclass(frozen=True)
class SpeakerEmbeddingPolicy:
    model_id: str
    model_revision: str
    embedding_dimension: int
    sample_rate: int
    downmix_policy_version: str
    window_samples: int
    window_shift_samples: int
    padding_policy_version: str
    normalization_policy_version: str
    enrollment_aggregation_policy_version: str

    def __post_init__(self) -> None:
        validate_speaker_model_id(self.model_id)
        validate_speaker_model_revision(self.model_revision)

        for name in (
            "embedding_dimension",
            "sample_rate",
            "window_samples",
            "window_shift_samples",
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.window_shift_samples > self.window_samples:
            raise ValueError("speaker window shift must not exceed window size")

        for name in (
            "downmix_policy_version",
            "padding_policy_version",
            "normalization_policy_version",
            "enrollment_aggregation_policy_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            if _POLICY_VERSION_TOKEN.fullmatch(value) is None:
                raise ValueError(f"{name} is invalid")

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            asdict(self),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


@dataclass(frozen=True)
class SpeakerEmbeddingWindow:
    start_sample: int
    end_sample: int
    embedding: np.ndarray


@runtime_checkable
class SpeakerEmbeddingAdapter(Protocol):
    def embed_windows(
        self,
        pcm: np.ndarray,
    ) -> tuple[SpeakerEmbeddingWindow, ...]: ...


@dataclass(frozen=True, slots=True)
class AnonymousSpeakerCluster:
    label: str
    centroid: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class AnonymousSpeakerClusteringPolicy:
    pruning_p: float
    low_frequency_beta: float
    normalized_gap_gamma: float

    def __post_init__(self) -> None:
        values = tuple(
            _finite_clustering_policy_value(getattr(self, name), name=name)
            for name in (
                "pruning_p",
                "low_frequency_beta",
                "normalized_gap_gamma",
            )
        )
        pruning_p, low_frequency_beta, normalized_gap_gamma = values
        if not 0.0 <= pruning_p <= 1.0:
            raise ValueError("speaker clustering pruning p must be between 0 and 1")
        if low_frequency_beta <= 0.0:
            raise ValueError("speaker clustering low-frequency beta must be positive")
        if not 0.0 < normalized_gap_gamma <= 1.0:
            raise ValueError(
                "speaker clustering normalized-gap gamma must be between 0 and 1"
            )
        for name, value in zip(
            (
                "pruning_p",
                "low_frequency_beta",
                "normalized_gap_gamma",
            ),
            values,
            strict=True,
        ):
            object.__setattr__(self, name, 0.0 if value == 0.0 else value)


@dataclass(frozen=True, slots=True)
class AnonymousSpeakerClusteringResult:
    window_cluster_ordinals: tuple[int, ...]
    clusters: tuple[AnonymousSpeakerCluster, ...]


def cluster_anonymous_speakers(
    embeddings: np.ndarray,
    *,
    policy: AnonymousSpeakerClusteringPolicy,
) -> AnonymousSpeakerClusteringResult:
    if not isinstance(policy, AnonymousSpeakerClusteringPolicy):
        raise TypeError("speaker clustering policy is invalid")
    if (
        type(embeddings) is not np.ndarray
        or embeddings.dtype != np.float32
        or embeddings.ndim != 2
        or embeddings.shape[1:] != (SPEAKER_EMBEDDING_DIMENSION,)
        or not embeddings.flags.c_contiguous
        or not np.isfinite(embeddings).all()
    ):
        _raise_invalid_clustering_output()
    vectors = np.ascontiguousarray(embeddings, dtype=np.float64)
    if len(vectors):
        norms = np.linalg.norm(vectors, axis=1)
        if (
            not np.isfinite(norms).all()
            or np.any(norms <= 0.0)
            or not np.allclose(
                norms,
                1.0,
                rtol=0.0,
                atol=SPEAKER_EMBEDDING_NORM_TOLERANCE,
            )
        ):
            _raise_invalid_clustering_output()

    count = len(vectors)
    if count == 0:
        return AnonymousSpeakerClusteringResult((), ())
    if count <= 2:
        return _clustering_result(vectors, np.zeros(count, dtype=np.int64), 1)

    affinity = vectors @ vectors.T
    if not np.isfinite(affinity).all():
        _raise_invalid_clustering_output()
    remove_count = max(
        0,
        min(
            int((1.0 - policy.pruning_p) * count),
            count - 6,
        ),
    )
    if remove_count:
        order = np.argsort(affinity, axis=1, kind="stable")
        rows = np.arange(count)[:, None]
        affinity[rows, order[:, :remove_count]] = 0.0
        del order, rows

    symmetric = affinity + affinity.T
    symmetric *= 0.5
    del affinity
    np.fill_diagonal(symmetric, 0.0)
    degree = np.sum(np.abs(symmetric), axis=1)
    laplacian = -symmetric
    np.fill_diagonal(laplacian, degree)
    if (
        not np.isfinite(symmetric).all()
        or not np.isfinite(degree).all()
        or not np.isfinite(laplacian).all()
    ):
        _raise_invalid_clustering_output()
    laplacian_norm = np.linalg.norm(laplacian, ord=np.inf)
    del symmetric

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    except np.linalg.LinAlgError:
        _raise_invalid_clustering_output()
    del laplacian
    if (
        type(eigenvalues) is not np.ndarray
        or type(eigenvectors) is not np.ndarray
        or eigenvalues.dtype != np.float64
        or eigenvectors.dtype != np.float64
        or eigenvalues.shape != (count,)
        or eigenvectors.shape != (count, count)
        or not np.isfinite(eigenvalues).all()
        or not np.isfinite(eigenvectors).all()
        or np.any(eigenvalues[1:] < eigenvalues[:-1])
    ):
        _raise_invalid_clustering_output()
    eigen_tolerance = (
        np.finfo(np.float64).eps
        * count
        * laplacian_norm
    )
    if not math.isfinite(float(eigen_tolerance)) or np.any(
        eigenvalues < -eigen_tolerance
    ):
        _raise_invalid_clustering_output()
    eigenvalues = np.where(eigenvalues < 0.0, 0.0, eigenvalues)

    cluster_count = _select_cluster_count(
        eigenvalues,
        degree,
        policy=policy,
    )
    del eigenvalues, degree
    if cluster_count == 1:
        del eigenvectors
        assignments = np.zeros(count, dtype=np.int64)
    else:
        spectral = np.array(
            eigenvectors[:, :cluster_count],
            dtype=np.float64,
            order="C",
            copy=True,
        )
        del eigenvectors
        assignments = _best_kmeans_assignments(spectral, cluster_count)
        del spectral
    return _clustering_result(vectors, assignments, cluster_count)


def _finite_clustering_policy_value(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _select_cluster_count(
    eigenvalues: np.ndarray,
    degree: np.ndarray,
    *,
    policy: AnonymousSpeakerClusteringPolicy,
) -> int:
    with np.errstate(over="ignore"):
        low_frequency_limit = policy.low_frequency_beta * float(np.median(degree))
    if not math.isfinite(low_frequency_limit):
        _raise_invalid_clustering_output()
    best_cluster_count = 1
    best_gap = -math.inf
    for candidate in range(2, len(eigenvalues)):
        gap = float(eigenvalues[candidate] - eigenvalues[candidate - 1])
        normalized_gap = gap / max(abs(float(eigenvalues[candidate])), 1e-12)
        if (
            eigenvalues[candidate] <= low_frequency_limit
            and normalized_gap >= policy.normalized_gap_gamma
            and normalized_gap > best_gap
        ):
            best_cluster_count = candidate
            best_gap = normalized_gap
    return best_cluster_count


def _best_kmeans_assignments(
    values: np.ndarray,
    cluster_count: int,
) -> np.ndarray:
    best_assignments: np.ndarray | None = None
    best_inertia = math.inf
    for seed in range(10):
        candidate = _kmeans_restart(values, cluster_count, seed=seed)
        if candidate is None:
            continue
        assignments, inertia = candidate
        if inertia < best_inertia:
            best_assignments = assignments
            best_inertia = inertia
    if best_assignments is None:
        _raise_invalid_clustering_output()
    return best_assignments


def _kmeans_restart(
    values: np.ndarray,
    cluster_count: int,
    *,
    seed: int,
) -> tuple[np.ndarray, float] | None:
    generator = np.random.Generator(np.random.PCG64(seed))
    point_count = len(values)
    first = int(generator.integers(point_count))
    selected_mask = np.zeros(point_count, dtype=np.bool_)
    selected_mask[first] = True
    centers = [values[first].copy()]
    closest_distances = _squared_distances(values, values[first : first + 1])[:, 0]
    closest_distances[selected_mask] = 0.0
    for _ in range(1, cluster_count):
        closest_distances[selected_mask] = 0.0
        total = float(np.sum(closest_distances))
        if not math.isfinite(total) or total <= 0.0:
            return None
        cumulative = np.cumsum(closest_distances)
        selected = int(
            np.searchsorted(
                cumulative,
                generator.random() * total,
                side="right",
            )
        )
        del cumulative
        if selected >= point_count or selected_mask[selected]:
            return None
        selected_mask[selected] = True
        centers.append(values[selected].copy())
        selected_distances = _squared_distances(
            values,
            values[selected : selected + 1],
        )[:, 0]
        closest_distances = np.minimum(
            closest_distances,
            selected_distances,
        )
        closest_distances[selected_mask] = 0.0
        del selected_distances

    center_values = np.ascontiguousarray(np.stack(centers), dtype=np.float64)
    del centers, closest_distances, selected_mask
    previous_assignments: np.ndarray | None = None
    for _ in range(300):
        distances = _squared_distances(values, center_values)
        assignments = np.argmin(distances, axis=1)
        del distances
        counts = np.bincount(assignments, minlength=cluster_count)
        if np.any(counts == 0):
            return None
        del counts
        next_centers = np.stack(
            [
                np.mean(values[assignments == ordinal], axis=0)
                for ordinal in range(cluster_count)
            ]
        )
        if not np.isfinite(next_centers).all():
            return None
        if previous_assignments is not None and np.array_equal(
            assignments,
            previous_assignments,
        ):
            final_distances = _squared_distances(values, next_centers)
            inertia = float(
                np.sum(final_distances[np.arange(point_count), assignments])
            )
            del final_distances
            if not math.isfinite(inertia):
                return None
            return assignments, inertia
        previous_assignments = assignments
        center_values = np.ascontiguousarray(next_centers, dtype=np.float64)
    return None


def _squared_distances(
    values: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    distances = (
        np.sum(values * values, axis=1)[:, None]
        + np.sum(centers * centers, axis=1)[None, :]
        - 2.0 * (values @ centers.T)
    )
    np.maximum(distances, 0.0, out=distances)
    if not np.isfinite(distances).all():
        _raise_invalid_clustering_output()
    return distances


def _clustering_result(
    vectors: np.ndarray,
    assignments: np.ndarray,
    cluster_count: int,
) -> AnonymousSpeakerClusteringResult:
    if (
        assignments.shape != (len(vectors),)
        or np.any(assignments < 0)
        or np.any(assignments >= cluster_count)
    ):
        _raise_invalid_clustering_output()
    earliest = tuple(
        int(np.flatnonzero(assignments == ordinal)[0])
        for ordinal in range(cluster_count)
    )
    canonical_order = sorted(range(cluster_count), key=earliest.__getitem__)
    canonical_ordinals = np.empty(cluster_count, dtype=np.int64)
    for canonical, original in enumerate(canonical_order):
        canonical_ordinals[original] = canonical
    canonical_assignments = canonical_ordinals[assignments]

    clusters: list[AnonymousSpeakerCluster] = []
    for canonical in range(cluster_count):
        centroid = np.mean(vectors[canonical_assignments == canonical], axis=0)
        norm = float(np.linalg.norm(centroid))
        if not math.isfinite(norm) or norm <= 0.0:
            _raise_invalid_clustering_output()
        centroid /= norm
        if not np.isfinite(centroid).all():
            _raise_invalid_clustering_output()
        clusters.append(
            AnonymousSpeakerCluster(
                label=anonymous_speaker_label(canonical),
                centroid=tuple(float(component) for component in centroid),
            )
        )
    return AnonymousSpeakerClusteringResult(
        tuple(int(ordinal) for ordinal in canonical_assignments),
        tuple(clusters),
    )


def _raise_invalid_clustering_output() -> None:
    raise PipelineError(
        "invalid_model_output",
        "Speaker clustering input or output is invalid",
    )


def anonymous_speaker_label(ordinal: int) -> str:
    if type(ordinal) is not int:
        raise TypeError("anonymous speaker ordinal must be an integer")
    if ordinal < 0:
        raise ValueError("anonymous speaker ordinal must not be negative")
    value = ordinal + 1
    characters: list[str] = []
    while value:
        value, remainder = divmod(value - 1, 26)
        characters.append(chr(ord("A") + remainder))
    return "".join(reversed(characters))


def is_anonymous_speaker_label(value: object) -> bool:
    return type(value) is str and _ANONYMOUS_SPEAKER_LABEL.fullmatch(value) is not None
