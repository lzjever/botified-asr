from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CanonicalOptions:
    model: str
    language: str
    response_format: str
    chunking_strategy: str | None
    include: tuple[str, ...]
    known_speaker_ids: tuple[str, ...]
