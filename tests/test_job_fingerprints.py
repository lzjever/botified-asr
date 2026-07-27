from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from botified_asr.job_fingerprints import build_request_fingerprints


CANONICAL_OPTIONS_JSON = (
    '{"chunking_strategy":null,"include":[],"known_speaker_ids":[],'
    '"language":"auto","model":"sensevoice","response_format":"json"}'
)
INPUT_SHA256 = "a" * 64
EMPTY_SNAPSHOT = b'{"speakers":[],"version":1}'


def test_request_fingerprint_version_1_golden_and_immutable_result() -> None:
    fingerprints = build_request_fingerprints(
        CANONICAL_OPTIONS_JSON,
        INPUT_SHA256,
        EMPTY_SNAPSHOT,
    )

    assert fingerprints.snapshot_sha256 == (
        "37e2de7a783aa3aa11e0b56dbf8faa5ac19217e3ae9c2e2ae228592823009e3f"
    )
    assert fingerprints.request_fingerprint == (
        "b143205c1ee7e1c7fb65461d7c306315de5733572a51c824b9f8229aa15707f3"
    )
    with pytest.raises((FrozenInstanceError, AttributeError)):
        fingerprints.request_fingerprint = "0" * 64


def test_request_fingerprint_binds_every_exact_input_field() -> None:
    baseline = build_request_fingerprints(
        CANONICAL_OPTIONS_JSON,
        INPUT_SHA256,
        EMPTY_SNAPSHOT,
    )
    changed_options = build_request_fingerprints(
        CANONICAL_OPTIONS_JSON.replace('"language":"auto"', '"language":"en"'),
        INPUT_SHA256,
        EMPTY_SNAPSHOT,
    )
    changed_input = build_request_fingerprints(
        CANONICAL_OPTIONS_JSON,
        "b" * 64,
        EMPTY_SNAPSHOT,
    )
    changed_snapshot = build_request_fingerprints(
        CANONICAL_OPTIONS_JSON,
        INPUT_SHA256,
        EMPTY_SNAPSHOT + b"\n",
    )

    assert len(
        {
            baseline.request_fingerprint,
            changed_options.request_fingerprint,
            changed_input.request_fingerprint,
            changed_snapshot.request_fingerprint,
        }
    ) == 4
    assert changed_options.snapshot_sha256 == baseline.snapshot_sha256
    assert changed_input.snapshot_sha256 == baseline.snapshot_sha256
    assert changed_snapshot.snapshot_sha256 != baseline.snapshot_sha256


@pytest.mark.parametrize(
    ("canonical_options_json", "input_sha256", "speaker_snapshot_bytes", "error"),
    [
        (
            CANONICAL_OPTIONS_JSON.encode(),
            INPUT_SHA256,
            EMPTY_SNAPSHOT,
            TypeError,
        ),
        (
            CANONICAL_OPTIONS_JSON,
            INPUT_SHA256.encode(),
            EMPTY_SNAPSHOT,
            TypeError,
        ),
        (
            CANONICAL_OPTIONS_JSON,
            INPUT_SHA256,
            bytearray(EMPTY_SNAPSHOT),
            TypeError,
        ),
        (
            '{"model":"sensevoice"}',
            INPUT_SHA256,
            EMPTY_SNAPSHOT,
            ValueError,
        ),
        (
            CANONICAL_OPTIONS_JSON.replace(":null", ": null"),
            INPUT_SHA256,
            EMPTY_SNAPSHOT,
            ValueError,
        ),
        (
            CANONICAL_OPTIONS_JSON.replace('"auto"', "NaN"),
            INPUT_SHA256,
            EMPTY_SNAPSHOT,
            ValueError,
        ),
        (
            CANONICAL_OPTIONS_JSON,
            "A" * 64,
            EMPTY_SNAPSHOT,
            ValueError,
        ),
        (
            CANONICAL_OPTIONS_JSON,
            "a" * 63,
            EMPTY_SNAPSHOT,
            ValueError,
        ),
        (
            CANONICAL_OPTIONS_JSON,
            "g" * 64,
            EMPTY_SNAPSHOT,
            ValueError,
        ),
        (
            CANONICAL_OPTIONS_JSON,
            INPUT_SHA256,
            b"x" * (64 * 1024 + 1),
            ValueError,
        ),
    ],
)
def test_request_fingerprint_rejects_invalid_inputs(
    canonical_options_json: object,
    input_sha256: object,
    speaker_snapshot_bytes: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        build_request_fingerprints(
            canonical_options_json,  # type: ignore[arg-type]
            input_sha256,  # type: ignore[arg-type]
            speaker_snapshot_bytes,  # type: ignore[arg-type]
        )


def test_request_fingerprint_accepts_snapshot_at_exact_size_limit() -> None:
    fingerprints = build_request_fingerprints(
        CANONICAL_OPTIONS_JSON,
        INPUT_SHA256,
        b"x" * (64 * 1024),
    )

    assert len(fingerprints.snapshot_sha256) == 64
    assert len(fingerprints.request_fingerprint) == 64
