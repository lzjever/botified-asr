from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from botified_asr.canonical_options import parse_canonical_options_json
from botified_asr.contracts import MAX_SPEAKER_SNAPSHOT_BYTES

_LOWERCASE_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class RequestFingerprints:
    snapshot_sha256: str
    request_fingerprint: str


def build_request_fingerprints(
    canonical_options_json: str,
    input_sha256: str,
    speaker_snapshot_bytes: bytes,
) -> RequestFingerprints:
    if type(canonical_options_json) is not str:
        raise TypeError("canonical options JSON must be a string")
    if type(input_sha256) is not str:
        raise TypeError("input SHA-256 must be a string")
    if type(speaker_snapshot_bytes) is not bytes:
        raise TypeError("speaker snapshot must be bytes")

    options = parse_canonical_options_json(canonical_options_json)
    if _LOWERCASE_SHA256.fullmatch(input_sha256) is None:
        raise ValueError("input SHA-256 is invalid")
    if len(speaker_snapshot_bytes) > MAX_SPEAKER_SNAPSHOT_BYTES:
        raise ValueError("speaker snapshot exceeds the byte limit")

    snapshot_sha256 = hashlib.sha256(speaker_snapshot_bytes).hexdigest()
    payload = json.dumps(
        {
            "canonical_options": {
                "chunking_strategy": options.chunking_strategy,
                "include": options.include,
                "known_speaker_ids": options.known_speaker_ids,
                "language": options.language,
                "model": options.model,
                "response_format": options.response_format,
            },
            "input_sha256": input_sha256,
            "speaker_snapshot_sha256": snapshot_sha256,
            "version": 1,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return RequestFingerprints(
        snapshot_sha256=snapshot_sha256,
        request_fingerprint=hashlib.sha256(payload).hexdigest(),
    )
