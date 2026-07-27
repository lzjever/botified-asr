from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from botified_asr.contracts import MAX_SPEAKER_SNAPSHOT_BYTES
from botified_asr.speaker_profiles import (
    SpeakerEmbedding,
    SpeakerProfile,
    canonicalize_speaker_profile_name,
    is_speaker_profile_compatible,
    validate_speaker_profile_id,
)
from botified_asr.speakers import SpeakerEmbeddingPolicy

_SNAPSHOT_VERSION = 1
_MAX_SELECTED_SPEAKERS = 32
_TOP_LEVEL_KEYS = {"speakers", "version"}
_SPEAKER_KEYS = {"embedding", "id", "name"}


class _InvalidSnapshot(ValueError):
    pass


def _reject_constant(_value: str) -> None:
    raise _InvalidSnapshot


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _InvalidSnapshot
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class SelectedSpeaker:
    id: str
    name: str
    embedding: SpeakerEmbedding


@dataclass(frozen=True, slots=True)
class SelectedSpeakerSnapshot:
    speakers: tuple[SelectedSpeaker, ...]


@runtime_checkable
class SpeakerProfileReader(Protocol):
    def get_speaker_profiles_by_ids(
        self,
        profile_ids: tuple[str, ...],
    ) -> tuple[SpeakerProfile, ...]: ...


class SelectedSpeakerNotFoundError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("one or more selected speaker profiles were not found")


class SelectedSpeakerIncompatibleError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("one or more selected speaker profiles are incompatible")


def serialize_selected_speaker_snapshot(
    snapshot: SelectedSpeakerSnapshot,
    policy: SpeakerEmbeddingPolicy,
) -> bytes:
    if type(snapshot) is not SelectedSpeakerSnapshot:
        raise TypeError("selected speaker snapshot is invalid")
    if type(policy) is not SpeakerEmbeddingPolicy:
        raise TypeError("speaker embedding policy is invalid")
    if type(snapshot.speakers) is not tuple:
        raise TypeError("selected speakers must be a tuple")
    if len(snapshot.speakers) > _MAX_SELECTED_SPEAKERS:
        raise ValueError("selected speaker snapshot has too many speakers")

    entries: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for speaker in sorted(snapshot.speakers, key=_selected_speaker_id):
        if type(speaker) is not SelectedSpeaker:
            raise TypeError("selected speaker is invalid")
        speaker_id = validate_speaker_profile_id(speaker.id)
        if speaker_id in seen_ids:
            raise ValueError("selected speaker IDs must be unique")
        seen_ids.add(speaker_id)
        name = canonicalize_speaker_profile_name(speaker.name)
        if name != speaker.name:
            raise ValueError("selected speaker name is not canonical")
        if type(speaker.embedding) is not SpeakerEmbedding:
            raise TypeError("selected speaker embedding is invalid")
        if speaker.embedding.dimension != policy.embedding_dimension:
            raise ValueError("selected speaker embedding dimension is invalid")
        encoded_embedding = base64.b64encode(
            speaker.embedding.to_bytes()
        ).decode("ascii")
        entries.append(
            {
                "embedding": encoded_embedding,
                "id": speaker_id,
                "name": name,
            }
        )

    try:
        wire = json.dumps(
            {
                "speakers": entries,
                "version": _SNAPSHOT_VERSION,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("selected speaker snapshot contains invalid text") from error
    if len(wire) > MAX_SPEAKER_SNAPSHOT_BYTES:
        raise ValueError("selected speaker snapshot exceeds the byte limit")
    return wire


def parse_selected_speaker_snapshot(
    wire: bytes,
    policy: SpeakerEmbeddingPolicy,
    *,
    expected_ids: tuple[str, ...],
) -> SelectedSpeakerSnapshot:
    if type(wire) is not bytes:
        raise TypeError("selected speaker snapshot wire must be bytes")
    if type(policy) is not SpeakerEmbeddingPolicy:
        raise TypeError("speaker embedding policy is invalid")
    _validate_expected_ids(expected_ids)
    if len(wire) > MAX_SPEAKER_SNAPSHOT_BYTES:
        raise ValueError("selected speaker snapshot exceeds the byte limit")

    try:
        value = json.loads(
            wire,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        if type(value) is not dict or set(value) != _TOP_LEVEL_KEYS:
            raise _InvalidSnapshot
        if type(value["version"]) is not int or value["version"] != _SNAPSHOT_VERSION:
            raise _InvalidSnapshot
        raw_speakers = value["speakers"]
        if (
            type(raw_speakers) is not list
            or len(raw_speakers) > _MAX_SELECTED_SPEAKERS
        ):
            raise _InvalidSnapshot

        speakers: list[SelectedSpeaker] = []
        for raw_speaker in raw_speakers:
            if type(raw_speaker) is not dict or set(raw_speaker) != _SPEAKER_KEYS:
                raise _InvalidSnapshot
            speaker_id = validate_speaker_profile_id(raw_speaker["id"])
            name = canonicalize_speaker_profile_name(raw_speaker["name"])
            if name != raw_speaker["name"]:
                raise _InvalidSnapshot
            encoded_embedding = raw_speaker["embedding"]
            if type(encoded_embedding) is not str:
                raise _InvalidSnapshot
            embedding_bytes = base64.b64decode(
                encoded_embedding,
                validate=True,
            )
            if base64.b64encode(embedding_bytes).decode("ascii") != encoded_embedding:
                raise _InvalidSnapshot
            embedding = SpeakerEmbedding.from_bytes(
                embedding_bytes,
                dimension=policy.embedding_dimension,
            )
            speakers.append(
                SelectedSpeaker(
                    id=speaker_id,
                    name=name,
                    embedding=embedding,
                )
            )
    except (
        TypeError,
        ValueError,
        UnicodeDecodeError,
        binascii.Error,
        json.JSONDecodeError,
        RecursionError,
    ) as error:
        raise ValueError("selected speaker snapshot wire is invalid") from error

    snapshot = SelectedSpeakerSnapshot(tuple(speakers))
    if (
        tuple(speaker.id for speaker in speakers) != expected_ids
        or serialize_selected_speaker_snapshot(snapshot, policy) != wire
    ):
        raise ValueError("selected speaker snapshot wire is not canonical")
    return snapshot


def _selected_speaker_id(speaker: object) -> str:
    if type(speaker) is not SelectedSpeaker:
        raise TypeError("selected speaker is invalid")
    return validate_speaker_profile_id(speaker.id)


def _validate_expected_ids(expected_ids: object) -> None:
    if type(expected_ids) is not tuple or any(
        type(speaker_id) is not str for speaker_id in expected_ids
    ):
        raise TypeError("expected speaker IDs must be a tuple of strings")
    if len(expected_ids) > _MAX_SELECTED_SPEAKERS:
        raise ValueError("expected speaker IDs exceed the limit")
    for speaker_id in expected_ids:
        validate_speaker_profile_id(speaker_id)
    if tuple(sorted(expected_ids)) != expected_ids or len(set(expected_ids)) != len(
        expected_ids
    ):
        raise ValueError("expected speaker IDs must be sorted and unique")


def resolve_selected_speaker_snapshot(
    reader: SpeakerProfileReader,
    profile_ids: tuple[str, ...],
    policy: SpeakerEmbeddingPolicy,
) -> SelectedSpeakerSnapshot:
    if not isinstance(reader, SpeakerProfileReader):
        raise TypeError("speaker profile reader is invalid")
    if not isinstance(profile_ids, tuple) or any(
        not isinstance(profile_id, str) for profile_id in profile_ids
    ):
        raise TypeError("speaker profile IDs must be a tuple of strings")
    if not isinstance(policy, SpeakerEmbeddingPolicy):
        raise TypeError("speaker embedding policy is invalid")
    if not profile_ids:
        return SelectedSpeakerSnapshot(())

    profiles = reader.get_speaker_profiles_by_ids(profile_ids)
    if tuple(profile.id for profile in profiles) != profile_ids:
        raise SelectedSpeakerNotFoundError
    if any(not is_speaker_profile_compatible(profile, policy) for profile in profiles):
        raise SelectedSpeakerIncompatibleError
    return SelectedSpeakerSnapshot(
        tuple(
            SelectedSpeaker(
                id=profile.id,
                name=profile.name,
                embedding=profile.embedding,
            )
            for profile in profiles
        )
    )
