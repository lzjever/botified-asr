from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from botified_asr.speaker_profiles import (
    SpeakerEmbedding,
    SpeakerProfile,
    is_speaker_profile_compatible,
)
from botified_asr.speakers import SpeakerEmbeddingPolicy


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
