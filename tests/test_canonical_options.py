from __future__ import annotations

import json
from dataclasses import replace
from importlib import import_module
from types import ModuleType

import pytest

from botified_asr.contracts import CanonicalOptions


CANONICAL_OPTIONS_JSON = (
    '{"chunking_strategy":null,"include":[],"known_speaker_ids":[],'
    '"language":"auto","model":"sensevoice","response_format":"json"}'
)
DIARIZED_OPTIONS_JSON = (
    '{"chunking_strategy":"auto",'
    '"include":["funasr.emotion","funasr.audio_events"],'
    '"known_speaker_ids":["4X7K2M9Q","9X7K2M9Q"],"language":"zh",'
    '"model":"sensevoice-diarize","response_format":"diarized_json"}'
)
KNOWN_SPEAKER_IDS = tuple(f"{index:08d}" for index in range(32))


def codec() -> ModuleType:
    return import_module("botified_asr.canonical_options")


def test_defaults_serialize_to_the_exact_six_key_wire_and_round_trip() -> None:
    canonical_options = codec()

    assert canonical_options.MAX_CANONICAL_OPTIONS_JSON_BYTES == 4096
    options = canonical_options.canonicalize_option_values(model="sensevoice")

    assert options == CanonicalOptions(
        model="sensevoice",
        language="auto",
        response_format="json",
        chunking_strategy=None,
        include=(),
        known_speaker_ids=(),
    )
    wire = canonical_options.serialize_canonical_options(options)
    assert wire == CANONICAL_OPTIONS_JSON
    assert tuple(json.loads(wire)) == (
        "chunking_strategy",
        "include",
        "known_speaker_ids",
        "language",
        "model",
        "response_format",
    )
    assert canonical_options.parse_canonical_options_json(wire) == options
    assert (
        canonical_options.serialize_canonical_options(
            canonical_options.parse_canonical_options_json(wire)
        )
        == wire
    )


def test_domain_canonicalization_applies_defaults_sorting_and_deduplication() -> None:
    canonical_options = codec()

    options = canonical_options.canonicalize_option_values(
        model="sensevoice-diarize",
        language="zh",
        response_format="diarized_json",
        chunking_strategy="auto",
        include=(
            "funasr.audio_events",
            "funasr.emotion",
            "funasr.audio_events",
        ),
        known_speaker_ids=("9X7K2M9Q", "4X7K2M9Q"),
    )

    assert options == CanonicalOptions(
        model="sensevoice-diarize",
        language="zh",
        response_format="diarized_json",
        chunking_strategy="auto",
        include=("funasr.emotion", "funasr.audio_events"),
        known_speaker_ids=("4X7K2M9Q", "9X7K2M9Q"),
    )
    assert (
        canonical_options.serialize_canonical_options(options) == DIARIZED_OPTIONS_JSON
    )
    assert (
        canonical_options.parse_canonical_options_json(DIARIZED_OPTIONS_JSON) == options
    )
    max_known_speakers = canonical_options.canonicalize_option_values(
        model="sensevoice-diarize",
        response_format="diarized_json",
        chunking_strategy="auto",
        known_speaker_ids=tuple(reversed(KNOWN_SPEAKER_IDS)),
    )
    assert max_known_speakers.known_speaker_ids == KNOWN_SPEAKER_IDS


@pytest.mark.parametrize(
    "changes",
    (
        *(
            {"language": language}
            for language in ("auto", "zh", "en", "yue", "ja", "ko")
        ),
        *(
            {"response_format": response_format}
            for response_format in ("json", "text", "verbose_json")
        ),
        {"chunking_strategy": "auto"},
    ),
)
def test_domain_canonicalization_accepts_representative_normal_options(
    changes: dict[str, object],
) -> None:
    canonical_options = codec()

    options = canonical_options.canonicalize_option_values(
        model="sensevoice",
        **changes,
    )

    for name, expected in changes.items():
        assert getattr(options, name) == expected


@pytest.mark.parametrize(
    "changes",
    (
        {"model": None},
        {"model": "other"},
        {"language": "fr"},
        {"response_format": "xml"},
        {"chunking_strategy": "none"},
        {"model": "sensevoice-diarize"},
        {"response_format": "diarized_json"},
        {"response_format": "text", "include": ("funasr.emotion",)},
        {"include": ("unknown",)},
        {
            "model": "sensevoice-diarize",
            "response_format": "diarized_json",
            "chunking_strategy": "auto",
            "known_speaker_ids": ("4X7K2M9Q", "4X7K2M9Q"),
        },
        {"known_speaker_ids": ("4X7K2M9Q",)},
        {"include": "funasr.emotion"},
        {"include": None},
        {"known_speaker_ids": "4X7K2M9Q"},
        {"known_speaker_ids": None},
        {
            "model": "sensevoice-diarize",
            "response_format": "diarized_json",
            "chunking_strategy": "auto",
            "known_speaker_ids": ("ABCDEFGI",),
        },
        {
            "model": "sensevoice-diarize",
            "response_format": "diarized_json",
            "chunking_strategy": "auto",
            "known_speaker_ids": (*KNOWN_SPEAKER_IDS, "00000032"),
        },
    ),
)
def test_domain_canonicalization_rejects_invalid_values(
    changes: dict[str, object],
) -> None:
    canonical_options = codec()
    values: dict[str, object] = {"model": "sensevoice"}
    values.update(changes)

    with pytest.raises(canonical_options.CanonicalOptionsValidationError):
        canonical_options.canonicalize_option_values(**values)


def test_domain_error_carries_the_existing_api_mapping_contract() -> None:
    canonical_options = codec()

    with pytest.raises(canonical_options.CanonicalOptionsValidationError) as exc_info:
        canonical_options.canonicalize_option_values(
            model="sensevoice",
            response_format="text",
            include=("funasr.emotion",),
        )

    error = exc_info.value
    assert (
        error.code,
        error.message,
        error.param,
    ) == (
        "incompatible_response_format",
        "response_format=text cannot be combined with include[]",
        "response_format",
    )


@pytest.mark.parametrize(
    "wire",
    (
        "[]",
        '{"model":"sensevoice"}',
        (
            '{"chunking_strategy":null,"include":[],"known_speaker_ids":[],'
            '"language":"auto","model":"sensevoice","model":"sensevoice",'
            '"response_format":"json"}'
        ),
        (
            '{"chunking_strategy":null,"extra":null,"include":[],'
            '"known_speaker_ids":[],"language":"auto","model":"sensevoice",'
            '"response_format":"json"}'
        ),
        (
            '{"chunking_strategy":null,"include":[],"known_speaker_ids":[],'
            '"language":"auto","model":"sensevoice"}'
        ),
        (
            '{"chunking_strategy":null,"include":"funasr.emotion",'
            '"known_speaker_ids":[],"language":"auto","model":"sensevoice",'
            '"response_format":"json"}'
        ),
        CANONICAL_OPTIONS_JSON.replace(
            '"model":"sensevoice"',
            '"model":null',
        ),
        CANONICAL_OPTIONS_JSON.replace(
            '"language":"auto"',
            '"language":true',
        ),
        CANONICAL_OPTIONS_JSON.replace(
            '"response_format":"json"',
            '"response_format":1',
        ),
        CANONICAL_OPTIONS_JSON.replace(
            '"chunking_strategy":null',
            '"chunking_strategy":false',
        ),
        CANONICAL_OPTIONS_JSON.replace('"include":[]', '"include":[1]'),
        CANONICAL_OPTIONS_JSON.replace(
            '"known_speaker_ids":[]',
            '"known_speaker_ids":"4X7K2M9Q"',
        ),
        CANONICAL_OPTIONS_JSON.replace(
            '"known_speaker_ids":[]',
            '"known_speaker_ids":[false]',
        ),
        CANONICAL_OPTIONS_JSON.replace(
            '"include":[]',
            '"include":["funasr.emotion","funasr.emotion"]',
        ),
        CANONICAL_OPTIONS_JSON.replace(":null", ": null"),
        (
            '{"model":"sensevoice","chunking_strategy":null,"include":[],'
            '"known_speaker_ids":[],"language":"auto","response_format":"json"}'
        ),
        DIARIZED_OPTIONS_JSON.replace(
            '["funasr.emotion","funasr.audio_events"]',
            '["funasr.audio_events","funasr.emotion"]',
        ),
        DIARIZED_OPTIONS_JSON.replace(
            '["4X7K2M9Q","9X7K2M9Q"]',
            '["9X7K2M9Q","4X7K2M9Q"]',
        ),
        (
            '{"chunking_strategy":null,"include":[],"known_speaker_ids":[],'
            '"language":"auto","model":"sensevoice",'
            '"response_format":"diarized_json"}'
        ),
    ),
)
def test_parser_rejects_noncanonical_or_invalid_wire_values(wire: str) -> None:
    canonical_options = codec()

    with pytest.raises(canonical_options.CanonicalOptionsValidationError):
        canonical_options.parse_canonical_options_json(wire)


@pytest.mark.parametrize("unit", (" ", "界"))
def test_parser_enforces_public_utf8_byte_cap(unit: str) -> None:
    canonical_options = codec()
    maximum = canonical_options.MAX_CANONICAL_OPTIONS_JSON_BYTES
    wire = unit * (maximum // len(unit.encode("utf-8")) + 1)

    assert len(wire.encode("utf-8")) > maximum
    with pytest.raises(canonical_options.CanonicalOptionsValidationError):
        canonical_options.parse_canonical_options_json(wire)


def test_parser_rejects_non_string_input() -> None:
    canonical_options = codec()

    with pytest.raises(TypeError):
        canonical_options.parse_canonical_options_json(CANONICAL_OPTIONS_JSON.encode())


def test_serializer_rejects_invalid_directly_constructed_options() -> None:
    canonical_options = codec()
    normal = CanonicalOptions(
        model="sensevoice",
        language="auto",
        response_format="json",
        chunking_strategy=None,
        include=(),
        known_speaker_ids=(),
    )
    diarized = CanonicalOptions(
        model="sensevoice-diarize",
        language="auto",
        response_format="diarized_json",
        chunking_strategy="auto",
        include=("funasr.emotion", "funasr.audio_events"),
        known_speaker_ids=("4X7K2M9Q", "9X7K2M9Q"),
    )
    invalid_options = (
        replace(normal, model="other"),
        replace(normal, response_format="diarized_json"),
        replace(normal, include=["funasr.emotion"]),
        replace(normal, include=("funasr.emotion", "funasr.emotion")),
        replace(
            diarized,
            include=("funasr.audio_events", "funasr.emotion"),
        ),
        replace(
            diarized,
            known_speaker_ids=("4X7K2M9Q", "4X7K2M9Q"),
        ),
        replace(
            diarized,
            known_speaker_ids=("9X7K2M9Q", "4X7K2M9Q"),
        ),
    )

    for options in invalid_options:
        with pytest.raises(canonical_options.CanonicalOptionsValidationError):
            canonical_options.serialize_canonical_options(options)


def test_http_bridge_produces_exact_wire_and_preserves_api_errors() -> None:
    from botified_asr.api import ApiError, canonicalize_options

    canonical_options = codec()

    assert (
        canonical_options.serialize_canonical_options(
            canonicalize_options({"model": ["sensevoice"]})
        )
        == CANONICAL_OPTIONS_JSON
    )
    fields = {
        "model": ["sensevoice-diarize"],
        "language": ["zh"],
        "response_format": ["diarized_json"],
        "chunking_strategy": ["auto"],
        "include[]": [
            "funasr.audio_events",
            "funasr.emotion",
            "funasr.audio_events",
        ],
        "known_speaker_ids[]": ["9X7K2M9Q", "4X7K2M9Q"],
    }
    assert (
        canonical_options.serialize_canonical_options(canonicalize_options(fields))
        == DIARIZED_OPTIONS_JSON
    )

    with pytest.raises(ApiError) as exc_info:
        canonicalize_options(
            {
                "model": ["sensevoice"],
                "response_format": ["text"],
                "include[]": ["funasr.emotion"],
            }
        )
    error = exc_info.value
    assert (
        error.status_code,
        error.code,
        error.message,
        error.param,
        error.error_type,
    ) == (
        400,
        "incompatible_response_format",
        "response_format=text cannot be combined with include[]",
        "response_format",
        "invalid_request_error",
    )
