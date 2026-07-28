from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import numpy as np

from botified_asr.speakers import (
    SPEAKER_EMBEDDING_NORM_TOLERANCE,
    SpeakerEmbeddingPolicy,
    validate_speaker_model_id,
    validate_speaker_model_revision,
)

_SPEAKER_ID = re.compile(r"\A[0-9A-HJKMNP-TV-Z]{8}\Z")
_LOWERCASE_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_ANONYMOUS_LABEL_NAME = re.compile(r"\A[A-Z]+\Z")
_UNKNOWN_LABEL_NAME = re.compile(
    r"\AUnknown [A-Z]+\Z",
    flags=re.ASCII | re.IGNORECASE,
)
_LITTLE_ENDIAN_FLOAT32 = np.dtype("<f4")
MIN_SPEAKER_SAMPLES = 2
MAX_SPEAKER_SAMPLES = 5


class ReservedSpeakerProfileNameError(ValueError):
    pass


def validate_speaker_profile_id(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("speaker profile ID must be a string")
    if _SPEAKER_ID.fullmatch(value) is None:
        raise ValueError("speaker profile ID is invalid")
    return value


def canonicalize_speaker_profile_name(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("speaker profile name must be a string")
    name = value.strip()
    if not 1 <= len(name) <= 80:
        raise ValueError("speaker profile name must contain 1 to 80 characters")
    if (
        _ANONYMOUS_LABEL_NAME.fullmatch(name) is not None
        or _UNKNOWN_LABEL_NAME.fullmatch(name) is not None
    ):
        raise ReservedSpeakerProfileNameError(
            "speaker profile name is reserved"
        )
    return name


@dataclass(frozen=True, slots=True)
class SpeakerEmbedding:
    dimension: int
    _canonical_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _validate_dimension(self.dimension)
        _validate_embedding_bytes(
            self._canonical_bytes,
            dimension=self.dimension,
        )

    @classmethod
    def from_numpy(
        cls,
        values: object,
        *,
        dimension: int,
    ) -> SpeakerEmbedding:
        _validate_dimension(dimension)
        if not isinstance(values, np.ndarray) or values.dtype != np.float32:
            raise TypeError("speaker embedding must be a float32 NumPy array")
        if values.shape != (dimension,):
            raise ValueError("speaker embedding has an invalid shape")
        _validate_unit_values(values)
        canonical = np.ascontiguousarray(
            values,
            dtype=_LITTLE_ENDIAN_FLOAT32,
        ).tobytes(order="C")
        return cls(dimension=dimension, _canonical_bytes=canonical)

    @classmethod
    def from_bytes(
        cls,
        raw: object,
        *,
        dimension: int,
    ) -> SpeakerEmbedding:
        _validate_dimension(dimension)
        _validate_embedding_bytes(raw, dimension=dimension)
        return cls(dimension=dimension, _canonical_bytes=raw)

    def to_bytes(self) -> bytes:
        return self._canonical_bytes

    def as_numpy(self) -> np.ndarray:
        return np.frombuffer(
            self._canonical_bytes,
            dtype=_LITTLE_ENDIAN_FLOAT32,
            count=self.dimension,
        )


@dataclass(frozen=True, slots=True)
class SpeakerProfile:
    id: str
    name: str
    description: str | None
    embedding: SpeakerEmbedding
    embedding_model_id: str
    embedding_model_revision: str
    embedding_dimension: int
    embedding_policy_fingerprint: str
    sample_count: int
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        validate_speaker_profile_id(self.id)
        name = canonicalize_speaker_profile_name(self.name)

        description = self.description
        if description is not None and not isinstance(description, str):
            raise TypeError("speaker profile description must be a string or None")
        if description is not None and len(description) > 500:
            raise ValueError(
                "speaker profile description must not exceed 500 characters"
            )

        _validate_profile_embedding_fields(
            embedding=self.embedding,
            embedding_model_id=self.embedding_model_id,
            embedding_model_revision=self.embedding_model_revision,
            embedding_dimension=self.embedding_dimension,
            embedding_policy_fingerprint=self.embedding_policy_fingerprint,
            sample_count=self.sample_count,
        )

        created_at = _canonical_utc(self.created_at, name="created_at")
        updated_at = _canonical_utc(self.updated_at, name="updated_at")
        if updated_at < created_at:
            raise ValueError("speaker profile timestamps are out of order")

        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "description",
            None if description == "" else description,
        )
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)

    @property
    def name_key(self) -> str:
        return self.name.casefold()


class KeepExisting(Enum):
    VALUE = "keep_existing"


KEEP_EXISTING = KeepExisting.VALUE


@dataclass(frozen=True, slots=True)
class SpeakerEmbeddingReplacement:
    embedding: SpeakerEmbedding
    embedding_model_id: str
    embedding_model_revision: str
    embedding_dimension: int
    embedding_policy_fingerprint: str
    sample_count: int

    def __post_init__(self) -> None:
        _validate_profile_embedding_fields(
            embedding=self.embedding,
            embedding_model_id=self.embedding_model_id,
            embedding_model_revision=self.embedding_model_revision,
            embedding_dimension=self.embedding_dimension,
            embedding_policy_fingerprint=self.embedding_policy_fingerprint,
            sample_count=self.sample_count,
        )


@dataclass(frozen=True, slots=True)
class SpeakerProfileUpdate:
    name: str
    description: KeepExisting | str | None
    embedding: KeepExisting | SpeakerEmbeddingReplacement
    updated_at: datetime


def is_speaker_profile_compatible(
    profile: SpeakerProfile,
    policy: SpeakerEmbeddingPolicy,
) -> bool:
    if not isinstance(profile, SpeakerProfile):
        raise TypeError("speaker profile is invalid")
    if not isinstance(policy, SpeakerEmbeddingPolicy):
        raise TypeError("speaker embedding policy is invalid")
    return (
        profile.embedding_model_id == policy.model_id
        and profile.embedding_model_revision == policy.model_revision
        and profile.embedding_dimension == policy.embedding_dimension
        and profile.embedding_policy_fingerprint == policy.fingerprint
    )


def _validate_profile_embedding_fields(
    *,
    embedding: object,
    embedding_model_id: object,
    embedding_model_revision: object,
    embedding_dimension: object,
    embedding_policy_fingerprint: object,
    sample_count: object,
) -> None:
    if not isinstance(embedding, SpeakerEmbedding):
        raise TypeError("speaker profile embedding is invalid")
    validate_speaker_model_id(embedding_model_id)
    validate_speaker_model_revision(embedding_model_revision)
    _validate_dimension(embedding_dimension)
    if embedding.dimension != embedding_dimension:
        raise ValueError("speaker profile embedding dimension is inconsistent")
    if not isinstance(embedding_policy_fingerprint, str):
        raise TypeError("speaker embedding policy fingerprint must be a string")
    if _LOWERCASE_SHA256.fullmatch(embedding_policy_fingerprint) is None:
        raise ValueError("speaker embedding policy fingerprint is invalid")
    if type(sample_count) is not int:
        raise TypeError("speaker profile sample count must be an integer")
    if not MIN_SPEAKER_SAMPLES <= sample_count <= MAX_SPEAKER_SAMPLES:
        raise ValueError("speaker profile sample count must be between 2 and 5")


def _validate_dimension(value: object) -> None:
    if type(value) is not int:
        raise TypeError("speaker embedding dimension must be an integer")
    if value <= 0:
        raise ValueError("speaker embedding dimension must be positive")


def _validate_embedding_bytes(raw: object, *, dimension: int) -> None:
    if type(raw) is not bytes:
        raise TypeError("speaker embedding bytes must be bytes")
    if len(raw) != dimension * _LITTLE_ENDIAN_FLOAT32.itemsize:
        raise ValueError("speaker embedding byte length is invalid")
    values = np.frombuffer(
        raw,
        dtype=_LITTLE_ENDIAN_FLOAT32,
        count=dimension,
    )
    _validate_unit_values(values)


def _validate_unit_values(values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        raise ValueError("speaker embedding must contain finite values")
    norm = float(np.linalg.norm(values.astype(np.float64, copy=False)))
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
        raise ValueError("speaker embedding must have unit norm")


def _canonical_utc(value: object, *, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"speaker profile {name} must be a datetime")
    try:
        if value.utcoffset() is None:
            raise ValueError(f"speaker profile {name} must be timezone-aware")
        return value.astimezone(timezone.utc)
    except (OverflowError, OSError) as error:
        raise ValueError(f"speaker profile {name} is invalid") from error
