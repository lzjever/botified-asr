from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from botified_asr.audio import SAMPLE_RATE
from botified_asr.contracts import (
    CANONICAL_JSONL_MAX_RECORD_BYTES,
    MAX_AUDIO_SAMPLES,
    CanonicalOptions,
)
from botified_asr.pipeline import (
    RichAnnotations,
    SegmentRecord,
    iter_canonical_join,
    serialize_canonical_record,
)
from botified_asr.speaker_matching import SpeakerLabelMapping
from botified_asr.speakers import is_anonymous_speaker_label

_TOP_LEVEL_KEYS = {
    "annotations",
    "anonymous_speaker",
    "end_sample",
    "index",
    "language",
    "start_sample",
    "text",
}
_ANNOTATION_KEYS = {"audio_event", "emotion"}
_RICH_FIELDS = {
    "funasr.emotion": ("emotion", "emotion"),
    "funasr.audio_events": ("audio_events", "audio_event"),
}


class CanonicalArtifactError(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.code = "invalid_result_artifact"
        self.reason = reason


@dataclass(frozen=True)
class CanonicalSummary:
    record_count: int
    nonempty_record_count: int
    labeled_record_count: int
    last_end_sample: int
    first_nonempty_language: str | None


@dataclass(frozen=True)
class Projection:
    content_type: str
    body_factory: Callable[[], Iterator[bytes]]


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


class CanonicalJsonlReader:
    def __init__(
        self,
        path: str | Path,
        *,
        opener: Callable[[], BinaryIO] | None = None,
    ) -> None:
        self._path = Path(path)
        self._opener = opener

    def _open(self) -> BinaryIO:
        if self._opener is not None:
            return self._opener()
        return self._path.open("rb")

    def iter_records(self) -> Iterator[SegmentRecord]:
        expected_index = 0
        last_end_sample = 0
        try:
            with self._open() as source:
                while True:
                    line = source.readline(CANONICAL_JSONL_MAX_RECORD_BYTES + 2)
                    if line == b"":
                        return
                    if not isinstance(line, bytes):
                        raise CanonicalArtifactError("result artifact is not binary")
                    if not line.endswith(b"\n"):
                        raise CanonicalArtifactError(
                            "result artifact line is not terminated"
                        )
                    payload = line[:-1]
                    if len(payload) > CANONICAL_JSONL_MAX_RECORD_BYTES:
                        raise CanonicalArtifactError(
                            "result artifact record exceeds byte limit"
                        )
                    record = _decode_record(
                        payload,
                        expected_index=expected_index,
                        last_end_sample=last_end_sample,
                    )
                    yield record
                    expected_index += 1
                    last_end_sample = record.end_sample
        except CanonicalArtifactError:
            raise
        except OSError as exc:
            raise CanonicalArtifactError("result artifact could not be read") from exc

    def scan(self) -> CanonicalSummary:
        record_count = 0
        nonempty_record_count = 0
        labeled_record_count = 0
        last_end_sample = 0
        first_nonempty_language: str | None = None
        saw_nonempty = False
        for record in self.iter_records():
            record_count += 1
            last_end_sample = record.end_sample
            if record.anonymous_speaker is not None:
                labeled_record_count += 1
            if record.text.strip():
                nonempty_record_count += 1
                if not saw_nonempty:
                    saw_nonempty = True
                    first_nonempty_language = record.language
        return CanonicalSummary(
            record_count=record_count,
            nonempty_record_count=nonempty_record_count,
            labeled_record_count=labeled_record_count,
            last_end_sample=last_end_sample,
            first_nonempty_language=first_nonempty_language,
        )


def _decode_record(
    payload: bytes,
    *,
    expected_index: int,
    last_end_sample: int,
) -> SegmentRecord:
    if not payload:
        raise CanonicalArtifactError("result artifact contains a blank line")
    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (ValueError, RecursionError) as exc:
        raise CanonicalArtifactError("result artifact contains invalid JSON") from exc
    if type(value) is not dict or set(value) != _TOP_LEVEL_KEYS:
        raise CanonicalArtifactError("result artifact record has invalid fields")
    annotations = value["annotations"]
    if type(annotations) is not dict or set(annotations) != _ANNOTATION_KEYS:
        raise CanonicalArtifactError("result artifact annotations have invalid fields")
    index = value["index"]
    start_sample = value["start_sample"]
    end_sample = value["end_sample"]
    record_text = value["text"]
    language = value["language"]
    anonymous_speaker = value["anonymous_speaker"]
    emotion = annotations["emotion"]
    audio_event = annotations["audio_event"]
    if any(type(item) is not int for item in (index, start_sample, end_sample)):
        raise CanonicalArtifactError(
            "result artifact index and bounds have invalid types"
        )
    if (
        type(record_text) is not str
        or language is not None
        and type(language) is not str
        or emotion is not None
        and type(emotion) is not str
        or audio_event is not None
        and type(audio_event) is not str
        or anonymous_speaker is not None
        and type(anonymous_speaker) is not str
    ):
        raise CanonicalArtifactError("result artifact values have invalid types")
    if anonymous_speaker is not None and not is_anonymous_speaker_label(
        anonymous_speaker
    ):
        raise CanonicalArtifactError("result artifact speaker label is invalid")
    if index != expected_index:
        raise CanonicalArtifactError("result artifact indices are not contiguous")
    if (
        start_sample < 0
        or end_sample <= start_sample
        or start_sample < last_end_sample
        or end_sample > MAX_AUDIO_SAMPLES
    ):
        raise CanonicalArtifactError("result artifact sample bounds are invalid")
    record = SegmentRecord(
        index=index,
        start_sample=start_sample,
        end_sample=end_sample,
        text=record_text,
        language=language,
        annotations=RichAnnotations(
            emotion=emotion,
            audio_event=audio_event,
        ),
        anonymous_speaker=anonymous_speaker,
    )
    try:
        canonical_payload = serialize_canonical_record(record)
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise CanonicalArtifactError("result artifact contains invalid text") from exc
    if canonical_payload != payload:
        raise CanonicalArtifactError("result artifact is not canonical")
    return record


class ResultProjector:
    def prepare(
        self,
        reader: CanonicalJsonlReader,
        options: CanonicalOptions,
        total_samples: int,
        *,
        speaker_mapping: SpeakerLabelMapping,
    ) -> Projection:
        if (
            type(speaker_mapping) is not SpeakerLabelMapping
            or type(speaker_mapping.resolutions) is not tuple
            or speaker_mapping.resolutions != ()
        ):
            raise CanonicalArtifactError("result artifact speaker mapping is invalid")
        summary = reader.scan()
        if (
            type(total_samples) is not int
            or not 0 <= total_samples <= MAX_AUDIO_SAMPLES
            or summary.last_end_sample > total_samples
        ):
            raise CanonicalArtifactError(
                "result artifact total sample count is invalid"
            )
        if options.response_format not in {
            "json",
            "text",
            "verbose_json",
            "diarized_json",
        }:
            raise ValueError("unsupported response format")
        if options.response_format == "diarized_json":
            if summary.labeled_record_count != summary.record_count:
                raise CanonicalArtifactError(
                    "diarized result artifact contains an unlabeled record"
                )
        elif summary.labeled_record_count != 0:
            raise CanonicalArtifactError(
                "non-diarized result artifact contains a labeled record"
            )
        rich_fields = _rich_fields(options.include)
        if options.response_format == "text":
            if rich_fields:
                raise ValueError("text response format does not support rich fields")
            return Projection(
                content_type="text/plain; charset=utf-8",
                body_factory=lambda: _iter_text_body(reader),
            )
        if options.response_format == "json":
            return Projection(
                content_type="application/json",
                body_factory=lambda: _iter_json_body(reader, rich_fields),
            )
        if options.response_format == "diarized_json":
            return Projection(
                content_type="application/json",
                body_factory=lambda: _iter_diarized_body(
                    reader,
                    rich_fields,
                    total_samples=total_samples,
                ),
            )
        language = (
            options.language
            if options.language != "auto"
            else summary.first_nonempty_language or "unknown"
        )
        return Projection(
            content_type="application/json",
            body_factory=lambda: _iter_verbose_body(
                reader,
                rich_fields,
                language=language,
                total_samples=total_samples,
            ),
        )


def _rich_fields(
    requested: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    try:
        return tuple(_RICH_FIELDS[item] for item in requested)
    except KeyError as exc:
        raise ValueError("unsupported rich include") from exc


def _iter_joined_text(reader: CanonicalJsonlReader) -> Iterator[str]:
    return iter_canonical_join(record.text for record in reader.iter_records())


def _iter_text_body(reader: CanonicalJsonlReader) -> Iterator[bytes]:
    for chunk in _iter_joined_text(reader):
        yield chunk.encode("utf-8")


def _json_string_fragment(value: str) -> bytes:
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    return encoded[1:-1].encode("utf-8")


def _json_scalar(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _iter_json_body(
    reader: CanonicalJsonlReader,
    rich_fields: tuple[tuple[str, str], ...],
) -> Iterator[bytes]:
    yield b'{"text":"'
    for chunk in _iter_joined_text(reader):
        yield _json_string_fragment(chunk)
    yield b'"'
    if rich_fields:
        yield b',"funasr":'
        yield from _iter_rich(reader, rich_fields)
    yield b"}"


def _iter_verbose_body(
    reader: CanonicalJsonlReader,
    rich_fields: tuple[tuple[str, str], ...],
    *,
    language: str,
    total_samples: int,
) -> Iterator[bytes]:
    duration: int | float = 0 if total_samples == 0 else total_samples / SAMPLE_RATE
    yield b'{"task":"transcribe","language":'
    yield _json_scalar(language)
    yield b',"duration":'
    yield _json_scalar(duration)
    yield b',"text":"'
    for chunk in _iter_joined_text(reader):
        yield _json_string_fragment(chunk)
    yield b'","segments":['
    first = True
    for record in reader.iter_records():
        text = record.text.strip()
        if not text:
            continue
        if not first:
            yield b","
        first = False
        yield _json_scalar(
            {
                "id": str(record.index),
                "start": record.start_sample / SAMPLE_RATE,
                "end": record.end_sample / SAMPLE_RATE,
                "text": text,
            }
        )
    yield b"]"
    if rich_fields:
        yield b',"funasr":'
        yield from _iter_rich(reader, rich_fields)
    yield b"}"


def _iter_diarized_body(
    reader: CanonicalJsonlReader,
    rich_fields: tuple[tuple[str, str], ...],
    *,
    total_samples: int,
) -> Iterator[bytes]:
    duration: int | float = 0 if total_samples == 0 else total_samples / SAMPLE_RATE
    yield b'{"task":"transcribe","duration":'
    yield _json_scalar(duration)
    yield b',"text":"'
    for chunk in _iter_joined_text(reader):
        yield _json_string_fragment(chunk)
    yield b'","segments":['
    first = True
    for record in reader.iter_records():
        text = record.text.strip()
        if not text:
            continue
        if record.anonymous_speaker is None:
            raise CanonicalArtifactError(
                "diarized result artifact contains an unlabeled record"
            )
        if not first:
            yield b","
        first = False
        yield _json_scalar(
            {
                "id": str(record.index),
                "type": "transcript.text.segment",
                "start": record.start_sample / SAMPLE_RATE,
                "end": record.end_sample / SAMPLE_RATE,
                "speaker": record.anonymous_speaker,
                "text": text,
            }
        )
    yield b"]"
    if rich_fields:
        yield b',"funasr":'
        yield from _iter_rich(reader, rich_fields)
    yield b"}"


def _iter_rich(
    reader: CanonicalJsonlReader,
    rich_fields: tuple[tuple[str, str], ...],
) -> Iterator[bytes]:
    yield b"{"
    for field_index, (public_key, record_field) in enumerate(rich_fields):
        if field_index:
            yield b","
        yield _json_scalar(public_key)
        yield b":["
        first = True
        for record in reader.iter_records():
            if not record.text.strip():
                continue
            label = getattr(record.annotations, record_field)
            if label is None:
                continue
            if not first:
                yield b","
            first = False
            yield _json_scalar(
                {
                    "label": label,
                    "start": record.start_sample / SAMPLE_RATE,
                    "end": record.end_sample / SAMPLE_RATE,
                }
            )
        yield b"]"
    yield b"}"
