from __future__ import annotations

from dataclasses import dataclass

CANONICAL_JSONL_MAX_RECORD_BYTES = 1024 * 1024
MAX_AUDIO_SAMPLES = 691_200_000


@dataclass(frozen=True)
class CanonicalOptions:
    model: str
    language: str
    response_format: str
    chunking_strategy: str | None
    include: tuple[str, ...]
    known_speaker_ids: tuple[str, ...]
