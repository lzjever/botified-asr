from __future__ import annotations

import json
import re
from collections.abc import Sequence

from botified_asr.contracts import CanonicalOptions, PUBLIC_ID_PATTERN

MAX_CANONICAL_OPTIONS_JSON_BYTES = 4096
MODEL_VALUES = ("sensevoice", "sensevoice-diarize")
LANGUAGE_VALUES = ("auto", "zh", "en", "yue", "ja", "ko")
RESPONSE_FORMAT_VALUES = ("json", "text", "verbose_json", "diarized_json")
CHUNKING_STRATEGY_VALUES = ("auto",)
INCLUDE_VALUES = ("funasr.emotion", "funasr.audio_events")

_CANONICAL_KEYS = {
    "chunking_strategy",
    "include",
    "known_speaker_ids",
    "language",
    "model",
    "response_format",
}


class CanonicalOptionsValidationError(ValueError):
    def __init__(self, code: str, message: str, param: str | None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.param = param


class _InvalidJson(ValueError):
    pass


def _reject_constant(_value: str) -> None:
    raise _InvalidJson


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _InvalidJson
        result[key] = value
    return result


def _error(code: str, message: str, param: str | None) -> None:
    raise CanonicalOptionsValidationError(code, message, param)


def _string_tuple(
    value: object,
    *,
    code: str,
    message: str,
    param: str,
) -> tuple[str, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or any(type(item) is not str for item in value)
    ):
        _error(code, message, param)
    return tuple(value)


def canonicalize_option_values(
    *,
    model: object,
    language: object = "auto",
    response_format: object = None,
    chunking_strategy: object = None,
    include: object = (),
    known_speaker_ids: object = (),
) -> CanonicalOptions:
    if type(model) is not str or model not in MODEL_VALUES:
        _error(
            "invalid_model",
            "model must be sensevoice or sensevoice-diarize",
            "model",
        )
    if type(language) is not str or language not in LANGUAGE_VALUES:
        _error("invalid_language", "unsupported language", "language")
    if chunking_strategy is not None and (
        type(chunking_strategy) is not str
        or chunking_strategy not in CHUNKING_STRATEGY_VALUES
    ):
        _error(
            "invalid_chunking_strategy",
            "chunking_strategy must be auto when provided",
            "chunking_strategy",
        )

    if model == "sensevoice-diarize":
        if chunking_strategy != "auto":
            _error(
                "diarization_requires_vad",
                "sensevoice-diarize requires explicit chunking_strategy=auto",
                "chunking_strategy",
            )
        if response_format != "diarized_json":
            _error(
                "diarization_requires_format",
                "sensevoice-diarize requires explicit response_format=diarized_json",
                "response_format",
            )
    else:
        if response_format is None or response_format == "":
            response_format = "json"
        if response_format == "diarized_json":
            _error(
                "diarized_format_requires_model",
                "diarized_json requires model=sensevoice-diarize",
                "response_format",
            )

    if (
        type(response_format) is not str
        or response_format not in RESPONSE_FORMAT_VALUES
    ):
        _error(
            "invalid_response_format",
            "unsupported response_format",
            "response_format",
        )

    include_values = _string_tuple(
        include,
        code="invalid_include",
        message="unsupported include value",
        param="include[]",
    )
    invalid_include = next(
        (value for value in include_values if value not in INCLUDE_VALUES),
        None,
    )
    if invalid_include is not None:
        _error("invalid_include", "unsupported include value", "include[]")
    includes = tuple(value for value in INCLUDE_VALUES if value in include_values)
    if response_format == "text" and includes:
        _error(
            "incompatible_response_format",
            "response_format=text cannot be combined with include[]",
            "response_format",
        )

    known_ids = _string_tuple(
        known_speaker_ids,
        code="invalid_known_speaker_ids",
        message="known_speaker_ids[] contains an invalid speaker ID",
        param="known_speaker_ids[]",
    )
    if len(known_ids) != len(set(known_ids)):
        _error(
            "invalid_known_speaker_ids",
            "known_speaker_ids[] must not contain duplicates",
            "known_speaker_ids[]",
        )
    if len(known_ids) > 32 or any(
        re.fullmatch(PUBLIC_ID_PATTERN, value) is None
        for value in known_ids
    ):
        _error(
            "invalid_known_speaker_ids",
            "known_speaker_ids[] contains an invalid speaker ID",
            "known_speaker_ids[]",
        )
    if known_ids and model != "sensevoice-diarize":
        _error(
            "known_speakers_require_diarization",
            "known_speaker_ids[] requires model=sensevoice-diarize",
            "known_speaker_ids[]",
        )

    return CanonicalOptions(
        model=model,
        language=language,
        response_format=response_format,
        chunking_strategy=chunking_strategy,
        include=includes,
        known_speaker_ids=tuple(sorted(known_ids)),
    )


def serialize_canonical_options(options: CanonicalOptions) -> str:
    if type(options) is not CanonicalOptions:
        raise TypeError("canonical options must be CanonicalOptions")
    canonical = canonicalize_option_values(
        model=options.model,
        language=options.language,
        response_format=options.response_format,
        chunking_strategy=options.chunking_strategy,
        include=options.include,
        known_speaker_ids=options.known_speaker_ids,
    )
    if canonical != options:
        _error(
            "invalid_canonical_options",
            "canonical options are not normalized",
            None,
        )
    wire = json.dumps(
        {
            "chunking_strategy": options.chunking_strategy,
            "include": options.include,
            "known_speaker_ids": options.known_speaker_ids,
            "language": options.language,
            "model": options.model,
            "response_format": options.response_format,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if len(wire.encode("utf-8")) > MAX_CANONICAL_OPTIONS_JSON_BYTES:
        _error(
            "invalid_canonical_options",
            "canonical options exceed byte limit",
            None,
        )
    return wire


def parse_canonical_options_json(wire: str) -> CanonicalOptions:
    if type(wire) is not str:
        raise TypeError("canonical options JSON must be a string")
    try:
        wire_size = len(wire.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise CanonicalOptionsValidationError(
            "invalid_canonical_options",
            "canonical options JSON is invalid",
            None,
        ) from error
    if wire_size > MAX_CANONICAL_OPTIONS_JSON_BYTES:
        _error(
            "invalid_canonical_options",
            "canonical options exceed byte limit",
            None,
        )
    try:
        value = json.loads(
            wire,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        if type(value) is not dict or set(value) != _CANONICAL_KEYS:
            raise _InvalidJson
        if (
            type(value["model"]) is not str
            or type(value["language"]) is not str
            or type(value["response_format"]) is not str
            or (
                value["chunking_strategy"] is not None
                and type(value["chunking_strategy"]) is not str
            )
            or type(value["include"]) is not list
            or any(type(item) is not str for item in value["include"])
            or type(value["known_speaker_ids"]) is not list
            or any(type(item) is not str for item in value["known_speaker_ids"])
        ):
            raise _InvalidJson
        raw_options = CanonicalOptions(
            model=value["model"],
            language=value["language"],
            response_format=value["response_format"],
            chunking_strategy=value["chunking_strategy"],
            include=tuple(value["include"]),
            known_speaker_ids=tuple(value["known_speaker_ids"]),
        )
    except (
        CanonicalOptionsValidationError,
        json.JSONDecodeError,
        _InvalidJson,
        RecursionError,
    ) as error:
        raise CanonicalOptionsValidationError(
            "invalid_canonical_options",
            "canonical options JSON is invalid",
            None,
        ) from error

    canonical = canonicalize_option_values(
        model=raw_options.model,
        language=raw_options.language,
        response_format=raw_options.response_format,
        chunking_strategy=raw_options.chunking_strategy,
        include=raw_options.include,
        known_speaker_ids=raw_options.known_speaker_ids,
    )
    if canonical != raw_options or serialize_canonical_options(canonical) != wire:
        _error(
            "invalid_canonical_options",
            "canonical options JSON is not canonical",
            None,
        )
    return canonical
