from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from botified_asr.contracts import (
    CANONICAL_JSONL_MAX_RECORD_BYTES,
    CanonicalOptions,
)
from botified_asr.pipeline import (
    RichAnnotations,
    SegmentRecord,
    serialize_canonical_record,
)
from botified_asr.speaker_matching import (
    KnownSpeakerMatch,
    SpeakerLabelMapping,
    SpeakerLabelResolution,
)


FINISHED_AT = datetime(2026, 7, 27, 12, 5, tzinfo=timezone.utc)
JOB_ID = "7K3M9Q2W"
KNOWN_SPEAKER_ID = "4X7K2M9Q"
REQUEST_FINGERPRINT = "2" * 64
PROCESSOR_FINGERPRINT = "3" * 64
MANIFEST_FIELDS = (
    "version",
    "job_id",
    "attempt_no",
    "request_fingerprint",
    "processor_fingerprint",
    "finished_at",
)


def result_envelope_api() -> tuple[Any, ...]:
    from botified_asr.result_artifact import (
        ResultEnvelopeManifest,
        ResultEnvelopeReader,
        finalize_result_envelope,
        serialize_result_manifest,
    )

    return (
        ResultEnvelopeManifest,
        ResultEnvelopeReader,
        finalize_result_envelope,
        serialize_result_manifest,
    )


def manifest(manifest_type: Any, **changes: object) -> object:
    values = {
        "version": 1,
        "job_id": JOB_ID,
        "attempt_no": 2,
        "request_fingerprint": REQUEST_FINGERPRINT,
        "processor_fingerprint": PROCESSOR_FINGERPRINT,
        "finished_at": FINISHED_AT,
    }
    values.update(changes)
    return manifest_type(**values)


def options(
    response_format: str,
    *,
    language: str = "auto",
    include: tuple[str, ...] = (),
    known_speaker_ids: tuple[str, ...] = (),
) -> CanonicalOptions:
    diarized = response_format == "diarized_json"
    return CanonicalOptions(
        model="sensevoice-diarize" if diarized else "sensevoice",
        language=language,
        response_format=response_format,
        chunking_strategy="auto" if diarized else None,
        include=include,
        known_speaker_ids=known_speaker_ids,
    )


def canonical_source(
    *,
    text: str = "hello",
    language: str = "en",
    emotion: str | None = None,
    anonymous_speaker: str | None = None,
) -> bytes:
    record = SegmentRecord(
        index=0,
        start_sample=0,
        end_sample=8_000,
        text=text,
        language=language,
        annotations=RichAnnotations(emotion=emotion),
        anonymous_speaker=anonymous_speaker,
    )
    return serialize_canonical_record(record) + b"\n"


class MemoryWriter:
    def __init__(self, fault: str | None = None) -> None:
        self.chunks: list[bytes] = []
        self.reference = object()
        self.fault = fault
        self.write_count = 0
        self.seal_count = 0
        self.abort_count = 0

    def write(self, payload: bytes) -> None:
        assert type(payload) is bytes
        self.write_count += 1
        if self.fault == "write":
            raise OSError("injected envelope write failure")
        self.chunks.append(payload)

    def seal(self) -> object:
        self.seal_count += 1
        if self.fault == "seal":
            raise OSError("injected envelope seal failure")
        return self.reference

    def abort(self) -> None:
        self.abort_count += 1


class CappedStream(io.BytesIO):
    def __init__(
        self,
        payload: bytes,
        cap: int,
        *,
        line_cap: int | None = None,
    ) -> None:
        super().__init__(payload)
        self.cap = cap
        self.line_cap = cap if line_cap is None else line_cap
        self.read_sizes: list[int] = []
        self.readline_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        assert 0 < size <= self.cap, "result envelope reads must be bounded"
        self.read_sizes.append(size)
        return super().read(size)

    def readline(self, size: int = -1) -> bytes:
        assert 0 < size <= self.line_cap, "result envelope line reads must be bounded"
        self.readline_sizes.append(size)
        return super().readline(size)


class CappedOpener:
    def __init__(
        self,
        payload: bytes,
        cap: int,
        *,
        line_cap: int | None = None,
    ) -> None:
        self.payload = payload
        self.cap = cap
        self.line_cap = line_cap
        self.streams: list[CappedStream] = []

    def __call__(self) -> CappedStream:
        stream = CappedStream(
            self.payload,
            self.cap,
            line_cap=self.line_cap,
        )
        self.streams.append(stream)
        return stream


def finalize(
    finalize_result_envelope: Any,
    manifest_value: object,
    source: bytes,
    response_format: str,
    *,
    include: tuple[str, ...] = (),
    speaker_mapping: SpeakerLabelMapping = SpeakerLabelMapping(()),
) -> tuple[MemoryWriter, bytes, CanonicalOptions]:
    from botified_asr.result_artifact import CanonicalJsonlReader

    writer = MemoryWriter()
    source_opener = CappedOpener(
        source,
        CANONICAL_JSONL_MAX_RECORD_BYTES + 2,
    )
    canonical_options = options(
        response_format,
        include=include,
        known_speaker_ids=((KNOWN_SPEAKER_ID,) if speaker_mapping.resolutions else ()),
    )
    result = finalize_result_envelope(
        CanonicalJsonlReader(
            Path("source-is-injected.jsonl"),
            opener=source_opener,
        ),
        canonical_options,
        16_000,
        writer=writer,
        manifest=manifest_value,
        speaker_mapping=speaker_mapping,
    )
    assert result is writer.reference
    assert writer.seal_count == 1
    assert writer.abort_count == 0
    assert source_opener.streams
    return writer, b"".join(writer.chunks), canonical_options


def result_reader(
    reader_type: Any,
    payload: bytes,
    *,
    expected_job_id: str = JOB_ID,
    expected_attempt_no: int = 2,
    expected_request_fingerprint: str = REQUEST_FINGERPRINT,
    expected_processor_fingerprint: str = PROCESSOR_FINGERPRINT,
    canonical_options: CanonicalOptions | None = None,
    expected_total_samples: int = 16_000,
    opener: CappedOpener | None = None,
    expected_size_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> object:
    return reader_type(
        Path("complete-is-injected.json"),
        expected_size_bytes=(
            len(payload) if expected_size_bytes is None else expected_size_bytes
        ),
        expected_sha256=(
            hashlib.sha256(payload).hexdigest()
            if expected_sha256 is None
            else expected_sha256
        ),
        expected_job_id=expected_job_id,
        expected_attempt_no=expected_attempt_no,
        expected_request_fingerprint=expected_request_fingerprint,
        expected_processor_fingerprint=expected_processor_fingerprint,
        canonical_options=canonical_options or options("text"),
        expected_total_samples=expected_total_samples,
        opener=opener
        or CappedOpener(
            payload,
            64 * 1024,
            line_cap=CANONICAL_JSONL_MAX_RECORD_BYTES + 2,
        ),
    )


def test_result_manifest_validates_and_has_exact_canonical_wire() -> None:
    (
        manifest_type,
        _,
        _,
        serialize_result_manifest,
    ) = result_envelope_api()

    value = manifest(manifest_type)
    serialized = serialize_result_manifest(value)
    assert serialized == (
        b'{"attempt_no":2,"finished_at":"2026-07-27T12:05:00Z",'
        b'"job_id":"7K3M9Q2W","processor_fingerprint":"'
        + PROCESSOR_FINGERPRINT.encode()
        + b'","request_fingerprint":"'
        + REQUEST_FINGERPRINT.encode()
        + b'","version":1}'
    )
    assert set(json.loads(serialized)) == set(MANIFEST_FIELDS)
    assert b"speaker" not in serialized
    assert b"attempt_token" not in serialized

    invalid = (
        {"version": 2},
        {"job_id": "lower-id"},
        {"attempt_no": True},
        {"attempt_no": 0},
        {"request_fingerprint": "A" * 64},
        {"processor_fingerprint": "3" * 63},
        {"finished_at": FINISHED_AT.replace(tzinfo=None)},
        {"finished_at": FINISHED_AT.astimezone(timezone(timedelta(hours=1)))},
    )
    for changes in invalid:
        with pytest.raises((TypeError, ValueError)):
            manifest(manifest_type, **changes)


@pytest.mark.parametrize(
    ("response_format", "include", "emotion", "expected_body"),
    (
        ("text", (), None, b'{"text":"hello"}'),
        (
            "json",
            ("funasr.emotion",),
            "happy",
            b'{"text":"hello","funasr":{"emotion":['
            b'{"label":"happy","start":0.0,"end":0.5}]}}',
        ),
        (
            "verbose_json",
            (),
            None,
            b'{"task":"transcribe","language":"en","duration":1.0,'
            b'"text":"hello","segments":[{"id":"0","start":0.0,'
            b'"end":0.5,"text":"hello"}]}',
        ),
        (
            "diarized_json",
            (),
            None,
            b'{"task":"transcribe","duration":1.0,"text":"hello",'
            b'"segments":[{"id":"0","type":"transcript.text.segment",'
            b'"start":0.0,"end":0.5,"speaker":"Percy","text":"hello",'
            b'"funasr":{"speaker_id":"4X7K2M9Q",'
            b'"anonymous_speaker":"A","similarity":0.75}}]}',
        ),
    ),
)
def test_async_finalize_writes_manifest_and_one_canonical_object(
    response_format: str,
    include: tuple[str, ...],
    emotion: str | None,
    expected_body: bytes,
) -> None:
    (
        manifest_type,
        reader_type,
        finalize_result_envelope,
        serialize_result_manifest,
    ) = result_envelope_api()
    manifest_value = manifest(manifest_type)
    mapping = (
        SpeakerLabelMapping(
            (
                SpeakerLabelResolution(
                    "A",
                    KnownSpeakerMatch(
                        KNOWN_SPEAKER_ID,
                        "Percy",
                        0.75,
                    ),
                ),
            )
        )
        if response_format == "diarized_json"
        else SpeakerLabelMapping(())
    )
    writer, payload, canonical_options = finalize(
        finalize_result_envelope,
        manifest_value,
        canonical_source(
            emotion=emotion,
            anonymous_speaker=("A" if response_format == "diarized_json" else None),
        ),
        response_format,
        include=include,
        speaker_mapping=mapping,
    )
    reader = result_reader(
        reader_type,
        payload,
        canonical_options=canonical_options,
    )
    assert reader.validate() == manifest_value
    assert b"".join(reader.iter_body()) == expected_body
    if response_format in {"verbose_json", "diarized_json"}:
        wrong_total = result_reader(
            reader_type,
            payload,
            canonical_options=canonical_options,
            expected_total_samples=32_000,
        )
        with pytest.raises(ValueError):
            wrong_total.validate()
    if response_format == "verbose_json":
        from botified_asr.result_artifact import CanonicalArtifactError

        wrong_language_payload = payload.replace(
            b'"language":"en"',
            b'"language":"de"',
        )
        wrong_language = result_reader(
            reader_type,
            wrong_language_payload,
            canonical_options=options(
                "verbose_json",
                language="fr",
            ),
        )
        with pytest.raises(CanonicalArtifactError):
            wrong_language.validate()

    assert payload == (
        serialize_result_manifest(manifest_value) + b"\n" + expected_body
    )
    assert payload.count(b"\n") == 1
    assert json.loads(payload.split(b"\n", 1)[1]) == json.loads(expected_body)
    assert all(b"attempt_token" not in chunk for chunk in writer.chunks)
    manifest_line = payload.split(b"\n", 1)[0]
    assert b"Percy" not in manifest_line
    assert b"speaker" not in manifest_line


def test_complete_survives_source_deletion_and_validates_identity_size_hash(
    tmp_path: Path,
) -> None:
    (
        manifest_type,
        reader_type,
        finalize_result_envelope,
        _,
    ) = result_envelope_api()
    manifest_value = manifest(manifest_type)
    source_path = tmp_path / "attempt.jsonl"
    source_path.write_bytes(canonical_source())

    from botified_asr.result_artifact import CanonicalJsonlReader

    writer = MemoryWriter()
    finalize_result_envelope(
        CanonicalJsonlReader(source_path),
        options("text"),
        16_000,
        writer=writer,
        manifest=manifest_value,
        speaker_mapping=SpeakerLabelMapping(()),
    )
    complete = b"".join(writer.chunks)
    source_path.unlink()

    reader = result_reader(reader_type, complete)
    assert reader.validate() == manifest_value
    body_chunks = list(reader.iter_body())
    assert b"".join(body_chunks) == b'{"text":"hello"}'

    wrong_storage_metadata = (
        {"expected_size_bytes": len(complete) + 1},
        {"expected_sha256": "0" * 64},
    )
    for changes in wrong_storage_metadata:
        invalid_reader = result_reader(
            reader_type,
            complete,
            **changes,
        )
        with pytest.raises(ValueError):
            invalid_reader.validate()

    identity_mismatches = (
        {"expected_job_id": "01234567"},
        {"expected_attempt_no": 1},
        {"expected_request_fingerprint": "4" * 64},
        {"expected_processor_fingerprint": "5" * 64},
    )
    for mismatch in identity_mismatches:
        invalid_reader = result_reader(reader_type, complete, **mismatch)
        with pytest.raises(ValueError):
            invalid_reader.validate()

    wrong_format = result_reader(
        reader_type,
        complete,
        canonical_options=options("verbose_json"),
    )
    with pytest.raises(ValueError):
        wrong_format.validate()


@pytest.mark.parametrize(
    "corruption",
    (
        "manifest-duplicate",
        "manifest-unknown",
        "manifest-noncanonical",
        "manifest-oversized",
        "manifest-no-newline",
        "truncated",
        "trailing-whitespace",
        "body-duplicate",
        "body-unknown",
        "body-noncanonical",
    ),
)
def test_envelope_rejects_malformed_manifest_body_and_eof(
    corruption: str,
) -> None:
    (
        manifest_type,
        reader_type,
        _,
        serialize_result_manifest,
    ) = result_envelope_api()
    manifest_value = manifest(manifest_type)
    manifest_line = serialize_result_manifest(manifest_value)
    body = b'{"text":"ok"}'
    if corruption == "manifest-duplicate":
        manifest_line = manifest_line.replace(
            b'{"attempt_no":2,',
            b'{"attempt_no":2,"attempt_no":2,',
        )
    elif corruption == "manifest-unknown":
        manifest_line = manifest_line.replace(
            b'{"attempt_no":2,',
            b'{"attempt_no":2,"unknown":null,',
        )
    elif corruption == "manifest-noncanonical":
        manifest_line = manifest_line.replace(b'"version":1', b'"version": 1')
    elif corruption == "manifest-oversized":
        manifest_line = b"x" * (CANONICAL_JSONL_MAX_RECORD_BYTES + 1)
    elif corruption == "manifest-no-newline":
        manifest_line += body
        body = b""
    elif corruption == "truncated":
        body = body[:-1]
    elif corruption == "trailing-whitespace":
        body += b"\n"
    elif corruption == "body-duplicate":
        body = b'{"text":"ok","text":"ok"}'
    elif corruption == "body-unknown":
        body = b'{"text":"ok","unknown":null}'
    else:
        body = b'{"text": "ok"}'
    payload = (
        manifest_line
        if corruption == "manifest-no-newline"
        else manifest_line + b"\n" + body
    )
    opener = CappedOpener(
        payload,
        64 * 1024,
        line_cap=CANONICAL_JSONL_MAX_RECORD_BYTES + 2,
    )
    reader = result_reader(reader_type, payload, opener=opener)

    from botified_asr.result_artifact import CanonicalArtifactError

    with pytest.raises(CanonicalArtifactError) as caught:
        reader.validate()
    assert caught.value.code == "invalid_result_artifact"
    if corruption == "manifest-oversized":
        manifest_reads = [
            size
            for stream in opener.streams
            for size in stream.readline_sizes
            if size == CANONICAL_JSONL_MAX_RECORD_BYTES + 2
        ]
        assert manifest_reads == [CANONICAL_JSONL_MAX_RECORD_BYTES + 2]


def test_validator_never_uses_unbounded_read_or_json_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        manifest_type,
        reader_type,
        _,
        serialize_result_manifest,
    ) = result_envelope_api()
    manifest_value = manifest(manifest_type)
    payload = serialize_result_manifest(manifest_value) + b'\n{"text":"hello"}'
    opener = CappedOpener(
        payload,
        64 * 1024,
        line_cap=CANONICAL_JSONL_MAX_RECORD_BYTES + 2,
    )

    import botified_asr.result_artifact as result_module

    def reject_json_load(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("result envelope must not use json.load")

    monkeypatch.setattr(result_module.json, "load", reject_json_load)
    reader = result_reader(reader_type, payload, opener=opener)

    reader.validate()
    assert b"".join(reader.iter_body()) == b'{"text":"hello"}'
    assert opener.streams
    assert all(stream.read_sizes or stream.readline_sizes for stream in opener.streams)


@pytest.mark.parametrize("fault", ("source", "write", "seal"))
def test_finalize_aborts_attempt_once_for_each_failure_window(
    fault: str,
) -> None:
    (
        manifest_type,
        _,
        finalize_result_envelope,
        _,
    ) = result_envelope_api()
    manifest_value = manifest(manifest_type)

    from botified_asr.result_artifact import (
        CanonicalArtifactError,
        CanonicalJsonlReader,
    )

    writer = MemoryWriter(None if fault == "source" else fault)
    source = (
        b'{"not":"a canonical segment"}\n' if fault == "source" else canonical_source()
    )
    expected_error = CanonicalArtifactError if fault == "source" else OSError
    with pytest.raises(expected_error):
        finalize_result_envelope(
            CanonicalJsonlReader(
                Path("failing-source-is-injected.jsonl"),
                opener=CappedOpener(
                    source,
                    CANONICAL_JSONL_MAX_RECORD_BYTES + 2,
                ),
            ),
            options("text"),
            16_000,
            writer=writer,
            manifest=manifest_value,
            speaker_mapping=SpeakerLabelMapping(()),
        )

    assert writer.abort_count == 1


@pytest.mark.parametrize(
    "shape",
    ("large-text", "large-auto-language", "many-segments"),
)
def test_finalize_and_validate_large_envelopes_with_bounded_reads(
    shape: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        manifest_type,
        reader_type,
        finalize_result_envelope,
        _,
    ) = result_envelope_api()
    manifest_value = manifest(manifest_type)
    if shape == "large-text":
        source = canonical_source(text="x" * 500_000)
        response_format = "text"
        total_samples = 16_000
    elif shape == "large-auto-language":
        source = canonical_source(language="l" * 70_000)
        response_format = "verbose_json"
        total_samples = 16_000
    else:
        record_count = 20_000
        source = b"".join(
            serialize_canonical_record(
                SegmentRecord(
                    index=index,
                    start_sample=index,
                    end_sample=index + 1,
                    text=f"s{index}",
                    language="en",
                    annotations=RichAnnotations(),
                )
            )
            + b"\n"
            for index in range(record_count)
        )
        response_format = "verbose_json"
        total_samples = 32_000

    from botified_asr.result_artifact import CanonicalJsonlReader

    writer = MemoryWriter()
    canonical_options = options(response_format)
    source_opener = CappedOpener(
        source,
        CANONICAL_JSONL_MAX_RECORD_BYTES + 2,
    )
    finalize_result_envelope(
        CanonicalJsonlReader(
            Path("large-source-is-injected.jsonl"),
            opener=source_opener,
        ),
        canonical_options,
        total_samples,
        writer=writer,
        manifest=manifest_value,
        speaker_mapping=SpeakerLabelMapping(()),
    )
    payload = b"".join(writer.chunks)
    assert writer.write_count > 3
    assert max(map(len, writer.chunks)) <= CANONICAL_JSONL_MAX_RECORD_BYTES
    if shape == "many-segments":
        assert len(payload) > 1024 * 1024
    elif shape == "large-auto-language":
        assert len(payload) > 64 * 1024
    else:
        assert len(payload) > 100_000

    import botified_asr.result_artifact as result_module

    original_loads = result_module.json.loads

    def capped_json_loads(
        value: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        raw = value if isinstance(value, bytes) else str(value).encode()
        assert len(raw) <= 64 * 1024, "json.loads input must stay bounded"
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr(result_module.json, "loads", capped_json_loads)

    opener = CappedOpener(
        payload,
        64 * 1024,
        line_cap=CANONICAL_JSONL_MAX_RECORD_BYTES + 2,
    )
    reader = result_reader(
        reader_type,
        payload,
        canonical_options=canonical_options,
        expected_total_samples=total_samples,
        opener=opener,
    )
    assert reader.validate() == manifest_value
    body_chunks = list(reader.iter_body())
    assert len(body_chunks) > 1
    assert max(map(len, body_chunks)) <= 64 * 1024
    assert opener.streams
    assert all(
        all(0 < size <= 64 * 1024 for size in stream.read_sizes)
        and all(
            0 < size <= CANONICAL_JSONL_MAX_RECORD_BYTES + 2
            for size in stream.readline_sizes
        )
        for stream in opener.streams
    )
