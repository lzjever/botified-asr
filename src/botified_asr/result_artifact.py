from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from botified_asr.audio import SAMPLE_RATE
from botified_asr.contracts import (
    CANONICAL_JSONL_MAX_RECORD_BYTES,
    MAX_AUDIO_SAMPLES,
    CanonicalOptions,
)
from botified_asr.jobs import validate_job_id
from botified_asr.pipeline import (
    RichAnnotations,
    SegmentRecord,
    iter_canonical_join,
    serialize_canonical_record,
)
from botified_asr.speaker_matching import (
    KnownSpeakerMatch,
    SpeakerLabelMapping,
    SpeakerLabelResolution,
)
from botified_asr.speaker_profiles import (
    canonicalize_speaker_profile_name,
    validate_speaker_profile_id,
)
from botified_asr.speakers import (
    ANONYMOUS_SPEAKER_LABELS,
    is_anonymous_speaker_label,
)

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
_LOWERCASE_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_RESULT_ENVELOPE_VERSION = 1
_RESULT_ENVELOPE_CHUNK_BYTES = 64 * 1024
_RESULT_MANIFEST_MAX_JSON_BYTES = 4 * 1024
_RESULT_MANIFEST_KEYS = {
    "attempt_no",
    "finished_at",
    "job_id",
    "processor_fingerprint",
    "request_fingerprint",
    "version",
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
    visible_speaker_labels: tuple[str, ...]


@dataclass(frozen=True)
class Projection:
    content_type: str
    body_factory: Callable[[], Iterator[bytes]]


@dataclass(frozen=True, slots=True)
class ResultEnvelopeManifest:
    version: int
    job_id: str
    attempt_no: int
    request_fingerprint: str
    processor_fingerprint: str
    finished_at: datetime

    def __post_init__(self) -> None:
        if type(self.version) is not int:
            raise TypeError("result envelope version must be an integer")
        if self.version != _RESULT_ENVELOPE_VERSION:
            raise ValueError("result envelope version is unsupported")
        validate_job_id(self.job_id)
        if type(self.attempt_no) is not int:
            raise TypeError("result envelope attempt number must be an integer")
        if self.attempt_no < 1:
            raise ValueError("result envelope attempt number must be positive")
        for name in ("request_fingerprint", "processor_fingerprint"):
            value = getattr(self, name)
            if type(value) is not str:
                raise TypeError(f"result envelope {name} must be a string")
            if _LOWERCASE_SHA256.fullmatch(value) is None:
                raise ValueError(f"result envelope {name} is invalid")
        _encode_envelope_timestamp(self.finished_at)


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


def serialize_result_manifest(manifest: ResultEnvelopeManifest) -> bytes:
    if type(manifest) is not ResultEnvelopeManifest:
        raise TypeError("serialize_result_manifest requires a ResultEnvelopeManifest")
    return _json_scalar(
        {
            "attempt_no": manifest.attempt_no,
            "finished_at": _encode_envelope_timestamp(manifest.finished_at),
            "job_id": manifest.job_id,
            "processor_fingerprint": manifest.processor_fingerprint,
            "request_fingerprint": manifest.request_fingerprint,
            "version": manifest.version,
        }
    )


def finalize_result_envelope(
    reader: CanonicalJsonlReader,
    options: CanonicalOptions,
    total_samples: int,
    *,
    writer: object,
    manifest: ResultEnvelopeManifest,
    speaker_mapping: SpeakerLabelMapping,
) -> object:
    if type(reader) is not CanonicalJsonlReader:
        raise TypeError("finalize_result_envelope requires a CanonicalJsonlReader")
    if type(options) is not CanonicalOptions:
        raise TypeError("finalize_result_envelope requires CanonicalOptions")
    if type(manifest) is not ResultEnvelopeManifest:
        raise TypeError("finalize_result_envelope requires a ResultEnvelopeManifest")
    for method_name in ("write", "seal", "abort"):
        if not callable(getattr(writer, method_name, None)):
            raise TypeError("result envelope writer is invalid")

    try:
        projection = ResultProjector().prepare(
            reader,
            options,
            total_samples,
            speaker_mapping=speaker_mapping,
        )
        writer.write(serialize_result_manifest(manifest) + b"\n")
        if options.response_format == "text":
            writer.write(b'{"text":"')
            for chunk in projection.body_factory():
                writer.write(_json_string_fragment(chunk.decode("utf-8")))
            writer.write(b'"}')
        else:
            for chunk in projection.body_factory():
                if type(chunk) is not bytes:
                    raise TypeError("result projection yielded a non-bytes chunk")
                writer.write(chunk)
        return writer.seal()
    except BaseException:
        writer.abort()
        raise


class ResultEnvelopeReader:
    def __init__(
        self,
        path: str | Path,
        *,
        expected_size_bytes: int,
        expected_sha256: str,
        expected_job_id: str,
        expected_attempt_no: int,
        expected_request_fingerprint: str,
        expected_processor_fingerprint: str,
        canonical_options: CanonicalOptions,
        expected_total_samples: int,
        opener: Callable[[], BinaryIO] | None = None,
    ) -> None:
        if type(expected_size_bytes) is not int:
            raise TypeError("result envelope size must be an integer")
        if expected_size_bytes < 0:
            raise ValueError("result envelope size must be nonnegative")
        if (
            type(expected_sha256) is not str
            or _LOWERCASE_SHA256.fullmatch(expected_sha256) is None
        ):
            raise ValueError("result envelope SHA-256 is invalid")
        validate_job_id(expected_job_id)
        if type(expected_attempt_no) is not int:
            raise TypeError("result envelope attempt number must be an integer")
        if expected_attempt_no < 1:
            raise ValueError("result envelope attempt number must be positive")
        for name, value in (
            ("request fingerprint", expected_request_fingerprint),
            ("processor fingerprint", expected_processor_fingerprint),
        ):
            if type(value) is not str or _LOWERCASE_SHA256.fullmatch(value) is None:
                raise ValueError(f"result envelope {name} is invalid")
        if type(canonical_options) is not CanonicalOptions:
            raise TypeError("result envelope options must be CanonicalOptions")
        if (
            type(expected_total_samples) is not int
            or not 0 <= expected_total_samples <= MAX_AUDIO_SAMPLES
        ):
            raise ValueError("result envelope total samples are invalid")
        if opener is not None and not callable(opener):
            raise TypeError("result envelope opener must be callable")

        self._path = Path(path)
        self._expected_size_bytes = expected_size_bytes
        self._expected_sha256 = expected_sha256
        self._expected_job_id = expected_job_id
        self._expected_attempt_no = expected_attempt_no
        self._expected_request_fingerprint = expected_request_fingerprint
        self._expected_processor_fingerprint = expected_processor_fingerprint
        self._canonical_options = canonical_options
        self._expected_total_samples = expected_total_samples
        self._opener = opener

    def _open(self) -> BinaryIO:
        if self._opener is not None:
            return self._opener()
        return self._path.open("rb")

    def validate(self) -> ResultEnvelopeManifest:
        try:
            with self._open() as source:
                manifest_line = source.readline(CANONICAL_JSONL_MAX_RECORD_BYTES + 2)
                manifest = _decode_result_manifest_line(manifest_line)
                self._validate_identity(manifest)
                digest = hashlib.sha256()
                digest.update(manifest_line)
                cursor = _EnvelopeCursor(
                    source,
                    digest=digest,
                    initial_size=len(manifest_line),
                )
                _validate_envelope_body(
                    cursor,
                    self._canonical_options,
                    self._expected_total_samples,
                )
                if cursor.total_size != self._expected_size_bytes:
                    raise CanonicalArtifactError(
                        "result envelope size does not match storage"
                    )
                if digest.hexdigest() != self._expected_sha256:
                    raise CanonicalArtifactError(
                        "result envelope SHA-256 does not match storage"
                    )
                return manifest
        except CanonicalArtifactError:
            raise
        except OSError as error:
            raise CanonicalArtifactError("result envelope could not be read") from error

    def iter_body(self) -> Iterator[bytes]:
        def chunks() -> Iterator[bytes]:
            try:
                with self._open() as source:
                    manifest_line = source.readline(
                        CANONICAL_JSONL_MAX_RECORD_BYTES + 2
                    )
                    manifest = _decode_result_manifest_line(manifest_line)
                    self._validate_identity(manifest)
                    while True:
                        chunk = source.read(_RESULT_ENVELOPE_CHUNK_BYTES)
                        if chunk == b"":
                            return
                        if type(chunk) is not bytes:
                            raise CanonicalArtifactError(
                                "result envelope is not binary"
                            )
                        yield chunk
            except CanonicalArtifactError:
                raise
            except OSError as error:
                raise CanonicalArtifactError(
                    "result envelope could not be read"
                ) from error

        return chunks()

    def _validate_identity(
        self,
        manifest: ResultEnvelopeManifest,
    ) -> None:
        if (
            manifest.job_id != self._expected_job_id
            or manifest.attempt_no != self._expected_attempt_no
            or manifest.request_fingerprint != self._expected_request_fingerprint
            or manifest.processor_fingerprint != self._expected_processor_fingerprint
        ):
            raise CanonicalArtifactError("result envelope identity does not match job")


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
        visible_speaker_labels: list[str] = []
        seen_visible_speaker_labels: set[str] = set()
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
                if (
                    record.anonymous_speaker is not None
                    and record.anonymous_speaker not in seen_visible_speaker_labels
                ):
                    seen_visible_speaker_labels.add(record.anonymous_speaker)
                    visible_speaker_labels.append(record.anonymous_speaker)
        return CanonicalSummary(
            record_count=record_count,
            nonempty_record_count=nonempty_record_count,
            labeled_record_count=labeled_record_count,
            last_end_sample=last_end_sample,
            first_nonempty_language=first_nonempty_language,
            visible_speaker_labels=tuple(visible_speaker_labels),
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
        speaker_lookup = _validated_speaker_lookup(
            options,
            speaker_mapping,
        )
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
            if speaker_lookup is not None and not set(
                summary.visible_speaker_labels
            ).issubset(speaker_lookup):
                raise CanonicalArtifactError(
                    "result artifact speaker mapping is incomplete"
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
                    speaker_lookup=speaker_lookup,
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


def _validated_speaker_lookup(
    options: CanonicalOptions,
    speaker_mapping: object,
) -> dict[str, SpeakerLabelResolution] | None:
    if (
        type(speaker_mapping) is not SpeakerLabelMapping
        or type(speaker_mapping.resolutions) is not tuple
    ):
        raise CanonicalArtifactError("result artifact speaker mapping is invalid")
    resolutions = speaker_mapping.resolutions
    known_speaker_ids = options.known_speaker_ids
    if not known_speaker_ids:
        if resolutions:
            raise CanonicalArtifactError("result artifact speaker mapping is invalid")
        return None
    if options.response_format != "diarized_json" and resolutions:
        raise CanonicalArtifactError("result artifact speaker mapping is invalid")
    if any(
        type(resolution) is not SpeakerLabelResolution for resolution in resolutions
    ):
        raise CanonicalArtifactError("result artifact speaker mapping is invalid")
    if any(
        not is_anonymous_speaker_label(resolution.anonymous_speaker)
        for resolution in resolutions
    ):
        raise CanonicalArtifactError("result artifact speaker mapping is invalid")
    if (
        len(resolutions) > len(ANONYMOUS_SPEAKER_LABELS)
        or tuple(resolution.anonymous_speaker for resolution in resolutions)
        != ANONYMOUS_SPEAKER_LABELS[: len(resolutions)]
    ):
        raise CanonicalArtifactError("result artifact speaker mapping is invalid")

    names_by_id: dict[str, str] = {}
    lookup: dict[str, SpeakerLabelResolution] = {}
    for resolution in resolutions:
        match = resolution.match
        if match is not None:
            _validate_known_speaker_match(
                match,
                known_speaker_ids=known_speaker_ids,
                names_by_id=names_by_id,
            )
        lookup[resolution.anonymous_speaker] = resolution
    return lookup


def _validate_known_speaker_match(
    match: object,
    *,
    known_speaker_ids: tuple[str, ...],
    names_by_id: dict[str, str],
) -> None:
    if type(match) is not KnownSpeakerMatch:
        raise CanonicalArtifactError("result artifact speaker mapping is invalid")
    try:
        speaker_id = validate_speaker_profile_id(match.speaker_id)
        speaker_name = canonicalize_speaker_profile_name(match.speaker_name)
        _json_scalar(match.speaker_name)
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise CanonicalArtifactError(
            "result artifact speaker mapping is invalid"
        ) from exc
    if (
        speaker_id not in known_speaker_ids
        or speaker_name != match.speaker_name
        or type(match.similarity) is not float
        or not math.isfinite(match.similarity)
        or not -1.0 <= match.similarity <= 1.0
    ):
        raise CanonicalArtifactError("result artifact speaker mapping is invalid")
    existing_name = names_by_id.setdefault(speaker_id, speaker_name)
    if existing_name != speaker_name:
        raise CanonicalArtifactError("result artifact speaker mapping is invalid")


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


def _encode_envelope_timestamp(value: datetime) -> str:
    if type(value) is not datetime:
        raise TypeError("result envelope timestamp must be a datetime")
    if value.tzinfo is not UTC:
        raise ValueError("result envelope timestamp must use canonical UTC")
    fraction = f".{value.microsecond:06d}" if value.microsecond else ""
    return (
        f"{value.year:04d}-{value.month:02d}-{value.day:02d}"
        f"T{value.hour:02d}:{value.minute:02d}:{value.second:02d}"
        f"{fraction}Z"
    )


def _decode_envelope_timestamp(value: object) -> datetime:
    if type(value) is not str:
        raise TypeError("result envelope timestamp must be text")
    parsed = datetime.strptime(
        value,
        ("%Y-%m-%dT%H:%M:%S.%fZ" if "." in value else "%Y-%m-%dT%H:%M:%SZ"),
    ).replace(tzinfo=UTC)
    if _encode_envelope_timestamp(parsed) != value:
        raise ValueError("result envelope timestamp is not canonical")
    return parsed


def _decode_result_manifest_line(
    line: object,
) -> ResultEnvelopeManifest:
    if type(line) is not bytes:
        raise CanonicalArtifactError("result envelope manifest is not binary")
    if not line.endswith(b"\n"):
        raise CanonicalArtifactError("result envelope manifest is not terminated")
    payload = line[:-1]
    if (
        len(payload) > CANONICAL_JSONL_MAX_RECORD_BYTES
        or len(payload) > _RESULT_MANIFEST_MAX_JSON_BYTES
    ):
        raise CanonicalArtifactError("result envelope manifest exceeds byte limit")
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        if type(value) is not dict or set(value) != _RESULT_MANIFEST_KEYS:
            raise ValueError
        manifest = ResultEnvelopeManifest(
            version=value["version"],
            job_id=value["job_id"],
            attempt_no=value["attempt_no"],
            request_fingerprint=value["request_fingerprint"],
            processor_fingerprint=value["processor_fingerprint"],
            finished_at=_decode_envelope_timestamp(value["finished_at"]),
        )
    except (TypeError, ValueError, RecursionError) as error:
        raise CanonicalArtifactError("result envelope manifest is invalid") from error
    if serialize_result_manifest(manifest) != payload:
        raise CanonicalArtifactError("result envelope manifest is not canonical")
    return manifest


class _EnvelopeCursor:
    def __init__(
        self,
        source: BinaryIO,
        *,
        digest: object,
        initial_size: int,
    ) -> None:
        self._source = source
        self._digest = digest
        self._buffer = b""
        self._offset = 0
        self._eof = False
        self.total_size = initial_size

    def peek(self) -> int | None:
        if not self._ensure():
            return None
        return self._buffer[self._offset]

    def take(self) -> int:
        if not self._ensure():
            raise CanonicalArtifactError("result envelope body is truncated")
        value = self._buffer[self._offset]
        self._offset += 1
        return value

    def expect(self, expected: bytes) -> None:
        for value in expected:
            if self.take() != value:
                raise CanonicalArtifactError("result envelope body is not canonical")

    def finish(self) -> None:
        if self.peek() is not None:
            raise CanonicalArtifactError("result envelope body has trailing data")

    def read_json_string(
        self,
        *,
        capture: bool = False,
    ) -> str | None:
        raw = bytearray() if capture else None

        def retain(value: int) -> None:
            if raw is not None:
                if len(raw) >= _RESULT_ENVELOPE_CHUNK_BYTES:
                    raise CanonicalArtifactError(
                        "result envelope scalar exceeds byte limit"
                    )
                raw.append(value)

        first = self.take()
        if first != 0x22:
            raise CanonicalArtifactError("result envelope string is invalid")
        retain(first)
        while True:
            value = self.take()
            retain(value)
            if value == 0x22:
                break
            if value < 0x20:
                raise CanonicalArtifactError("result envelope string is invalid")
            if value == 0x5C:
                escaped = self.take()
                retain(escaped)
                if escaped in b'"\\bfnrt':
                    continue
                if escaped != ord("u"):
                    raise CanonicalArtifactError(
                        "result envelope string escape is not canonical"
                    )
                digits = bytes(self.take() for _ in range(4))
                for digit in digits:
                    retain(digit)
                if any(digit not in b"0123456789abcdef" for digit in digits):
                    raise CanonicalArtifactError(
                        "result envelope string escape is not canonical"
                    )
                codepoint = int(digits, 16)
                if codepoint >= 0x20 or codepoint in {8, 9, 10, 12, 13}:
                    raise CanonicalArtifactError(
                        "result envelope string escape is not canonical"
                    )
                continue
            if value < 0x80:
                continue
            width = (
                2
                if 0xC2 <= value <= 0xDF
                else 3
                if 0xE0 <= value <= 0xEF
                else 4
                if 0xF0 <= value <= 0xF4
                else 0
            )
            if width == 0:
                raise CanonicalArtifactError(
                    "result envelope string contains invalid UTF-8"
                )
            encoded = bytes([value] + [self.take() for _ in range(width - 1)])
            for byte in encoded[1:]:
                retain(byte)
            try:
                encoded.decode("utf-8", errors="strict")
            except UnicodeDecodeError as error:
                raise CanonicalArtifactError(
                    "result envelope string contains invalid UTF-8"
                ) from error
        if raw is None:
            return None
        try:
            decoded = json.loads(bytes(raw))
        except (TypeError, ValueError) as error:
            raise CanonicalArtifactError("result envelope string is invalid") from error
        if type(decoded) is not str or _json_scalar(decoded) != bytes(raw):
            raise CanonicalArtifactError("result envelope string is not canonical")
        return decoded

    def read_number(self) -> int | float:
        raw = bytearray()
        while True:
            value = self.peek()
            if value is None or value not in b"-+0123456789.eE":
                break
            if len(raw) >= 128:
                raise CanonicalArtifactError(
                    "result envelope number exceeds byte limit"
                )
            raw.append(self.take())
        if not raw:
            raise CanonicalArtifactError("result envelope number is missing")
        try:
            number = json.loads(bytes(raw), parse_constant=_reject_constant)
        except (TypeError, ValueError) as error:
            raise CanonicalArtifactError("result envelope number is invalid") from error
        if type(number) not in {int, float} or _json_scalar(number) != bytes(raw):
            raise CanonicalArtifactError("result envelope number is not canonical")
        return number

    def _ensure(self) -> bool:
        if self._offset < len(self._buffer):
            return True
        if self._eof:
            return False
        try:
            chunk = self._source.read(_RESULT_ENVELOPE_CHUNK_BYTES)
        except OSError as error:
            raise CanonicalArtifactError("result envelope could not be read") from error
        if type(chunk) is not bytes:
            raise CanonicalArtifactError("result envelope is not binary")
        if chunk == b"":
            self._eof = True
            self._buffer = b""
            self._offset = 0
            return False
        self._digest.update(chunk)
        self.total_size += len(chunk)
        self._buffer = chunk
        self._offset = 0
        return True


def _validate_envelope_body(
    cursor: _EnvelopeCursor,
    options: CanonicalOptions,
    total_samples: int,
) -> None:
    try:
        response_format = options.response_format
        if response_format in {"text", "json"}:
            _validate_simple_body(cursor, options, total_samples)
        elif response_format == "verbose_json":
            _validate_verbose_body(cursor, options, total_samples)
        elif response_format == "diarized_json":
            _validate_diarized_body(cursor, options, total_samples)
        else:
            raise CanonicalArtifactError(
                "result envelope response format is unsupported"
            )
        cursor.finish()
    except CanonicalArtifactError:
        raise
    except (TypeError, ValueError, UnicodeError) as error:
        raise CanonicalArtifactError("result envelope body is invalid") from error


def _validate_simple_body(
    cursor: _EnvelopeCursor,
    options: CanonicalOptions,
    total_samples: int,
) -> None:
    cursor.expect(b'{"text":')
    cursor.read_json_string()
    rich_fields = _rich_fields(options.include)
    if rich_fields:
        if options.response_format == "text":
            raise CanonicalArtifactError("text result envelope contains rich fields")
        cursor.expect(b',"funasr":')
        _validate_rich_body(cursor, rich_fields, total_samples)
    cursor.expect(b"}")


def _validate_verbose_body(
    cursor: _EnvelopeCursor,
    options: CanonicalOptions,
    total_samples: int,
) -> None:
    cursor.expect(b'{"task":"transcribe","language":')
    explicit_language = options.language != "auto"
    language = cursor.read_json_string(capture=explicit_language)
    if explicit_language and language != options.language:
        raise CanonicalArtifactError("result envelope language does not match request")
    cursor.expect(b',"duration":')
    cursor.expect(_json_scalar(_duration(total_samples)))
    cursor.expect(b',"text":')
    cursor.read_json_string()
    cursor.expect(b',"segments":[')
    _validate_segments(
        cursor,
        diarized=False,
        options=options,
        total_samples=total_samples,
    )
    cursor.expect(b"]")
    rich_fields = _rich_fields(options.include)
    if rich_fields:
        cursor.expect(b',"funasr":')
        _validate_rich_body(cursor, rich_fields, total_samples)
    cursor.expect(b"}")


def _validate_diarized_body(
    cursor: _EnvelopeCursor,
    options: CanonicalOptions,
    total_samples: int,
) -> None:
    cursor.expect(b'{"task":"transcribe","duration":')
    cursor.expect(_json_scalar(_duration(total_samples)))
    cursor.expect(b',"text":')
    cursor.read_json_string()
    cursor.expect(b',"segments":[')
    _validate_segments(
        cursor,
        diarized=True,
        options=options,
        total_samples=total_samples,
    )
    cursor.expect(b"]")
    rich_fields = _rich_fields(options.include)
    if rich_fields:
        cursor.expect(b',"funasr":')
        _validate_rich_body(cursor, rich_fields, total_samples)
    cursor.expect(b"}")


def _validate_segments(
    cursor: _EnvelopeCursor,
    *,
    diarized: bool,
    options: CanonicalOptions,
    total_samples: int,
) -> None:
    first = True
    last_end = 0.0
    last_identifier = -1
    duration = _duration(total_samples)
    while cursor.peek() != ord("]"):
        if not first:
            cursor.expect(b",")
        first = False
        cursor.expect(b'{"id":')
        identifier = cursor.read_json_string(capture=True)
        if (
            identifier is None
            or not identifier.isascii()
            or not identifier.isdecimal()
            or int(identifier) <= last_identifier
        ):
            raise CanonicalArtifactError("result envelope segment ID is invalid")
        last_identifier = int(identifier)
        if diarized:
            cursor.expect(b',"type":"transcript.text.segment"')
        cursor.expect(b',"start":')
        start = cursor.read_number()
        cursor.expect(b',"end":')
        end = cursor.read_number()
        if start < 0 or end <= start or start < last_end or end > duration:
            raise CanonicalArtifactError("result envelope segment bounds are invalid")
        last_end = float(end)
        if diarized:
            cursor.expect(b',"speaker":')
            cursor.read_json_string()
        cursor.expect(b',"text":')
        cursor.read_json_string()
        if diarized and cursor.peek() == ord(","):
            cursor.expect(b',"funasr":{"speaker_id":')
            speaker_id = cursor.read_json_string(capture=True)
            if speaker_id not in options.known_speaker_ids:
                raise CanonicalArtifactError("result envelope speaker ID is invalid")
            cursor.expect(b',"anonymous_speaker":')
            anonymous = cursor.read_json_string(capture=True)
            if anonymous is None or not is_anonymous_speaker_label(anonymous):
                raise CanonicalArtifactError("result envelope speaker label is invalid")
            cursor.expect(b',"similarity":')
            similarity = cursor.read_number()
            if not -1.0 <= similarity <= 1.0:
                raise CanonicalArtifactError(
                    "result envelope speaker similarity is invalid"
                )
            cursor.expect(b"}")
        cursor.expect(b"}")


def _validate_rich_body(
    cursor: _EnvelopeCursor,
    rich_fields: tuple[tuple[str, str], ...],
    total_samples: int,
) -> None:
    duration = _duration(total_samples)
    cursor.expect(b"{")
    for field_index, (public_key, _) in enumerate(rich_fields):
        if field_index:
            cursor.expect(b",")
        cursor.expect(_json_scalar(public_key))
        cursor.expect(b":[")
        first = True
        while cursor.peek() != ord("]"):
            if not first:
                cursor.expect(b",")
            first = False
            cursor.expect(b'{"label":')
            cursor.read_json_string()
            cursor.expect(b',"start":')
            start = cursor.read_number()
            cursor.expect(b',"end":')
            end = cursor.read_number()
            if start < 0 or end <= start or end > duration:
                raise CanonicalArtifactError("result envelope rich bounds are invalid")
            cursor.expect(b"}")
        cursor.expect(b"]")
    cursor.expect(b"}")


def _duration(total_samples: int) -> int | float:
    return 0 if total_samples == 0 else total_samples / SAMPLE_RATE


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
    speaker_lookup: dict[str, SpeakerLabelResolution] | None,
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
        resolution = (
            None
            if speaker_lookup is None
            else speaker_lookup.get(record.anonymous_speaker)
        )
        if speaker_lookup is not None and resolution is None:
            raise CanonicalArtifactError(
                "result artifact speaker mapping is incomplete"
            )
        match = None if resolution is None else resolution.match
        speaker = (
            record.anonymous_speaker
            if speaker_lookup is None
            else (
                f"Unknown {record.anonymous_speaker}"
                if match is None
                else match.speaker_name
            )
        )
        segment: dict[str, object] = {
            "id": str(record.index),
            "type": "transcript.text.segment",
            "start": record.start_sample / SAMPLE_RATE,
            "end": record.end_sample / SAMPLE_RATE,
            "speaker": speaker,
            "text": text,
        }
        if match is not None:
            segment["funasr"] = {
                "speaker_id": match.speaker_id,
                "anonymous_speaker": record.anonymous_speaker,
                "similarity": match.similarity,
            }
        yield _json_scalar(segment)
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
