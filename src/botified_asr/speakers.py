from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from botified_asr import audio

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
