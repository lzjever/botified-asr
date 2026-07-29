from __future__ import annotations

from dataclasses import dataclass

CANONICAL_JSONL_MAX_RECORD_BYTES = 1024 * 1024
MAX_AUDIO_SAMPLES = 691_200_000
DIRECT_MAX_SAMPLES = 480_000
DIARIZATION_MAX_DURATION_SECONDS = 1800
DIARIZATION_MAX_AUDIO_SAMPLES = 28_800_000
MAX_SPEAKER_SNAPSHOT_BYTES = 64 * 1024
PUBLIC_ID_PATTERN = r"[0-9A-HJKMNP-TV-Z]{8}"


@dataclass(frozen=True)
class CanonicalOptions:
    model: str
    language: str
    response_format: str
    chunking_strategy: str | None
    include: tuple[str, ...]
    known_speaker_ids: tuple[str, ...]
