from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from botified_asr.contracts import CanonicalOptions
from botified_asr.pipeline import (
    CanonicalJsonlSegmentSink,
    RichAnnotations,
    SegmentRecord,
)


class SpyStream(io.BytesIO):
    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.readline_sizes: list[int] = []

    def read(self, _size: int = -1) -> bytes:
        raise AssertionError("reader must use bounded readline")

    def readline(self, size: int = -1) -> bytes:
        self.readline_sizes.append(size)
        return super().readline(size)


class FreshOpener:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.streams: list[SpyStream] = []

    def __call__(self) -> SpyStream:
        stream = SpyStream(self.payload)
        self.streams.append(stream)
        return stream


class RecordingWriter:
    def __init__(self) -> None:
        self.payloads: list[bytes] = []
        self.aborted = 0

    def write(self, payload: bytes) -> None:
        self.payloads.append(payload)

    def seal(self) -> object:
        return object()

    def abort(self) -> None:
        self.aborted += 1


def _options(
    response_format: str,
    *,
    language: str = "auto",
    include: tuple[str, ...] = (),
) -> CanonicalOptions:
    diarized = response_format == "diarized_json"
    return CanonicalOptions(
        model="sensevoice-diarize" if diarized else "sensevoice",
        language=language,
        response_format=response_format,
        chunking_strategy="auto" if diarized else None,
        include=include,
        known_speaker_ids=(),
    )


def _mapping(
    *,
    index: object = 0,
    start_sample: object = 0,
    end_sample: object = 1,
    text: object = "ok",
    language: object = "en",
    emotion: object = None,
    audio_event: object = None,
    anonymous_speaker: object = None,
) -> dict[str, object]:
    return {
        "annotations": {
            "audio_event": audio_event,
            "emotion": emotion,
        },
        "anonymous_speaker": anonymous_speaker,
        "end_sample": end_sample,
        "index": index,
        "language": language,
        "start_sample": start_sample,
        "text": text,
    }


def _jsonl(*mappings: dict[str, object]) -> bytes:
    return b"".join(
        json.dumps(
            mapping,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        for mapping in mappings
    )


def _body(projection: Any) -> tuple[list[bytes], bytes]:
    chunks = list(projection.body_factory())
    return chunks, b"".join(chunks)


def test_reader_is_lazy_bounded_reopens_and_scan_stays_summary_only(
    tmp_path: Path,
) -> None:
    from botified_asr.contracts import CANONICAL_JSONL_MAX_RECORD_BYTES
    from botified_asr.result_artifact import CanonicalJsonlReader

    opener = FreshOpener(_jsonl(_mapping()))
    reader = CanonicalJsonlReader(
        tmp_path / "never-opened-directly.jsonl",
        opener=opener,
    )

    records = reader.iter_records()
    assert opener.streams == []
    assert next(records) == SegmentRecord(
        0,
        0,
        1,
        "ok",
        "en",
        RichAnnotations(),
    )
    with pytest.raises(StopIteration):
        next(records)
    assert len(opener.streams) == 1
    assert opener.streams[0].readline_sizes
    assert set(opener.streams[0].readline_sizes) == {
        CANONICAL_JSONL_MAX_RECORD_BYTES + 2
    }

    assert next(reader.iter_records()).text == "ok"
    summary = reader.scan()
    assert len(opener.streams) == 3
    assert summary.record_count == 1
    assert summary.nonempty_record_count == 1
    assert summary.last_end_sample == 1
    assert summary.first_nonempty_language == "en"
    assert not hasattr(summary, "text")


def test_corruption_cap_lf_and_writer_multibyte_cap_are_fail_closed(
    tmp_path: Path,
) -> None:
    from botified_asr.contracts import CANONICAL_JSONL_MAX_RECORD_BYTES
    from botified_asr.pipeline import serialize_canonical_record
    from botified_asr.result_artifact import (
        CanonicalArtifactError,
        CanonicalJsonlReader,
    )

    canonical = _jsonl(_mapping())
    duplicate_top = canonical.replace(b'"index":0,', b'"index":0,"index":0,')
    duplicate_nested = canonical.replace(
        b'"emotion":null', b'"emotion":null,"emotion":null'
    )
    unknown = canonical.replace(b'"text":"ok"', b'"text":"ok","unknown":1')
    missing = canonical.replace(b',"text":"ok"', b"")
    missing_speaker = canonical.replace(b',"anonymous_speaker":null', b"")
    corruptions = (
        b"\n",
        canonical + b"\n",
        canonical[:-1],
        canonical[:-1] + b"\r\n",
        b"\xff\n",
        b"[]\n",
        b"[" * 2_000 + b"]" * 2_000 + b"\n",
        canonical.replace(
            b'"index":0',
            b'"index":' + b"9" * 10_000,
        ),
        duplicate_top,
        duplicate_nested,
        unknown,
        missing,
        missing_speaker,
        canonical.replace(
            b'"emotion":null',
            b'"emotion":null,"unknown":null',
        ),
        canonical.replace(b',"emotion":null', b""),
        canonical.replace(b'"index":0', b'"index":NaN'),
        canonical.replace(b'"index":0', b'"index":Infinity'),
        canonical.replace(b'"text":"ok"', b'"text":"\\ud800"'),
        b"x" * (CANONICAL_JSONL_MAX_RECORD_BYTES + 1) + b"\n",
        (
            b'{"annotations":{"audio_event":null,"emotion":null},'
            + b'"anonymous_speaker":null,"end_sample":1,"index":0,'
            + b'"language":"en","start_sample":0,"text":"'
            + ("界" * (CANONICAL_JSONL_MAX_RECORD_BYTES // 3)).encode()
            + b'"}\n'
        ),
    )
    for payload in corruptions:
        path = tmp_path / "private-name.jsonl"
        with pytest.raises(CanonicalArtifactError) as caught:
            list(CanonicalJsonlReader(path, opener=FreshOpener(payload)).iter_records())
        assert caught.value.code == "invalid_result_artifact"
        assert str(path) not in str(caught.value)
        assert "private-name" not in str(caught.value)

    writer = RecordingWriter()
    sink = CanonicalJsonlSegmentSink(writer)
    empty_record = SegmentRecord(
        0,
        0,
        1,
        "",
        "zh",
        RichAnnotations(),
    )
    base_bytes = len(serialize_canonical_record(empty_record))
    exact_text_bytes = CANONICAL_JSONL_MAX_RECORD_BYTES - base_bytes
    exact_text = "界" * (exact_text_bytes // 3) + "x" * (exact_text_bytes % 3)
    exact_record = SegmentRecord(
        0,
        0,
        1,
        exact_text,
        "zh",
        RichAnnotations(),
    )
    sink.append(exact_record)
    assert len(writer.payloads[0]) == CANONICAL_JSONL_MAX_RECORD_BYTES + 1
    assert list(
        CanonicalJsonlReader(
            tmp_path / "exact-cap.jsonl",
            opener=FreshOpener(writer.payloads[0]),
        ).iter_records()
    ) == [exact_record]

    with pytest.raises(ValueError, match="byte limit"):
        sink.append(
            SegmentRecord(
                1,
                1,
                2,
                "界" * (CANONICAL_JSONL_MAX_RECORD_BYTES // 3),
                "zh",
                RichAnnotations(),
            )
        )
    assert len(writer.payloads) == 1
    sink.abort()
    assert writer.aborted == 1


@pytest.mark.parametrize(
    "payload",
    [
        _jsonl(_mapping(index=True)),
        _jsonl(_mapping(start_sample=0.0)),
        _jsonl(_mapping(end_sample="1")),
        _jsonl(_mapping(text=1)),
        _jsonl(_mapping(language=False)),
        _jsonl(_mapping(emotion=True)),
        _jsonl(_mapping(audio_event=1)),
        _jsonl(_mapping(anonymous_speaker=True)),
        _jsonl(_mapping(anonymous_speaker=1)),
        _jsonl(_mapping(anonymous_speaker="Unknown A")),
        _jsonl(_mapping(anonymous_speaker="AG")),
        _jsonl(_mapping(index=1)),
        _jsonl(_mapping(start_sample=-1)),
        _jsonl(_mapping(end_sample=0)),
        _jsonl(
            _mapping(index=0, start_sample=0, end_sample=2),
            _mapping(index=1, start_sample=1, end_sample=3),
        ),
        _jsonl(_mapping(end_sample=691_200_001)),
    ],
)
def test_read_side_rejects_strict_types_indices_bounds_and_overlap(
    payload: bytes,
    tmp_path: Path,
) -> None:
    from botified_asr.result_artifact import (
        CanonicalArtifactError,
        CanonicalJsonlReader,
    )

    with pytest.raises(CanonicalArtifactError) as caught:
        list(
            CanonicalJsonlReader(
                tmp_path / "artifact.jsonl",
                opener=FreshOpener(payload),
            ).iter_records()
        )
    assert caught.value.code == "invalid_result_artifact"


@pytest.mark.parametrize("anonymous_speaker", [None, "A", "Z", "AA", "AF"])
def test_canonical_speaker_field_round_trips_exactly(
    anonymous_speaker: str | None,
    tmp_path: Path,
) -> None:
    from botified_asr.pipeline import serialize_canonical_record
    from botified_asr.result_artifact import CanonicalJsonlReader

    payload = _jsonl(_mapping(anonymous_speaker=anonymous_speaker))
    records = list(
        CanonicalJsonlReader(
            tmp_path / "artifact.jsonl",
            opener=FreshOpener(payload),
        ).iter_records()
    )

    assert records == [
        SegmentRecord(
            0,
            0,
            1,
            "ok",
            "en",
            RichAnnotations(),
            anonymous_speaker=anonymous_speaker,
        )
    ]
    assert serialize_canonical_record(records[0]) + b"\n" == payload


def test_three_projections_share_reader_sample_clock_and_streaming_join(
    tmp_path: Path,
) -> None:
    from botified_asr.result_artifact import (
        CanonicalJsonlReader,
        ResultProjector,
    )

    payload = _jsonl(
        _mapping(
            index=0,
            start_sample=0,
            end_sample=8_000,
            text='  你"\n\\  ',
            language="zh",
            emotion="happy",
        ),
        _mapping(
            index=1,
            start_sample=8_000,
            end_sample=16_000,
            text=" \t ",
            language="ja",
            emotion="ignored",
            audio_event="ignored",
        ),
        _mapping(
            index=2,
            start_sample=24_000,
            end_sample=32_000,
            text="hello",
            language="en",
            audio_event="speech",
        ),
    )
    opener = FreshOpener(payload)
    reader = CanonicalJsonlReader(tmp_path / "artifact.jsonl", opener=opener)
    projector = ResultProjector()

    text_projection = projector.prepare(
        reader,
        _options("text"),
        total_samples=48_000,
    )
    text_chunks, text_body = _body(text_projection)
    assert text_projection.content_type == "text/plain; charset=utf-8"
    assert text_body == '你"\n\\ hello'.encode()
    assert len(text_chunks) > 1

    json_projection = projector.prepare(
        reader,
        _options(
            "json",
            include=("funasr.emotion", "funasr.audio_events"),
        ),
        total_samples=48_000,
    )
    json_chunks, json_body = _body(json_projection)
    assert json_projection.content_type == "application/json"
    assert len(json_chunks) > 4
    assert json.loads(json_body) == {
        "text": '你"\n\\ hello',
        "funasr": {
            "emotion": [
                {
                    "label": "happy",
                    "start": 0.0,
                    "end": 0.5,
                }
            ],
            "audio_events": [
                {
                    "label": "speech",
                    "start": 1.5,
                    "end": 2.0,
                }
            ],
        },
    }
    assert b"\\u4f60" not in json_body
    assert b'\\"' in json_body
    assert b"\\n" in json_body
    assert b"\\\\" in json_body
    assert json_body == (
        b'{"text":"\xe4\xbd\xa0\\"\\n\\\\ hello","funasr":'
        b'{"emotion":[{"label":"happy","start":0.0,"end":0.5}],'
        b'"audio_events":[{"label":"speech","start":1.5,"end":2.0}]}}'
    )

    verbose_projection = projector.prepare(
        reader,
        _options(
            "verbose_json",
            include=("funasr.emotion", "funasr.audio_events"),
        ),
        total_samples=48_000,
    )
    _, verbose_body = _body(verbose_projection)
    assert json.loads(verbose_body) == {
        "task": "transcribe",
        "language": "zh",
        "duration": 3.0,
        "text": '你"\n\\ hello',
        "segments": [
            {
                "id": "0",
                "start": 0.0,
                "end": 0.5,
                "text": '你"\n\\',
            },
            {
                "id": "2",
                "start": 1.5,
                "end": 2.0,
                "text": "hello",
            },
        ],
        "funasr": {
            "emotion": [
                {
                    "label": "happy",
                    "start": 0.0,
                    "end": 0.5,
                }
            ],
            "audio_events": [
                {
                    "label": "speech",
                    "start": 1.5,
                    "end": 2.0,
                }
            ],
        },
    }
    assert all(
        "funasr" not in segment for segment in json.loads(verbose_body)["segments"]
    )
    assert verbose_body == (
        b'{"task":"transcribe","language":"zh","duration":3.0,'
        b'"text":"\xe4\xbd\xa0\\"\\n\\\\ hello","segments":['
        b'{"id":"0","start":0.0,"end":0.5,"text":"\xe4\xbd\xa0\\"\\n\\\\"},'
        b'{"id":"2","start":1.5,"end":2.0,"text":"hello"}],'
        b'"funasr":{"emotion":[{"label":"happy","start":0.0,"end":0.5}],'
        b'"audio_events":[{"label":"speech","start":1.5,"end":2.0}]}}'
    )
    assert all(
        set(stream.readline_sizes) == {1_048_576 + 2} for stream in opener.streams
    )


def test_diarized_projection_has_exact_wire_and_reuses_top_level_rich(
    tmp_path: Path,
) -> None:
    from botified_asr.result_artifact import (
        CanonicalJsonlReader,
        ResultProjector,
    )

    reader = CanonicalJsonlReader(
        tmp_path / "diarized.jsonl",
        opener=FreshOpener(
            _jsonl(
                _mapping(
                    index=0,
                    start_sample=0,
                    end_sample=8_000,
                    text=" 你 ",
                    language="zh",
                    emotion="happy",
                    anonymous_speaker="A",
                ),
                _mapping(
                    index=1,
                    start_sample=16_000,
                    end_sample=32_000,
                    text="hello",
                    language="en",
                    audio_event="speech",
                    anonymous_speaker="AA",
                ),
            )
        ),
    )

    projection = ResultProjector().prepare(
        reader,
        _options(
            "diarized_json",
            include=("funasr.emotion", "funasr.audio_events"),
        ),
        total_samples=48_000,
    )
    _, body = _body(projection)

    assert projection.content_type == "application/json"
    assert body == (
        b'{"task":"transcribe","duration":3.0,"text":"\xe4\xbd\xa0hello",'
        b'"segments":[{"id":"0","type":"transcript.text.segment",'
        b'"start":0.0,"end":0.5,"speaker":"A","text":"\xe4\xbd\xa0"},'
        b'{"id":"1","type":"transcript.text.segment","start":1.0,'
        b'"end":2.0,"speaker":"AA","text":"hello"}],'
        b'"funasr":{"emotion":[{"label":"happy","start":0.0,"end":0.5}],'
        b'"audio_events":[{"label":"speech","start":1.0,"end":2.0}]}}'
    )
    decoded = json.loads(body)
    assert "language" not in decoded
    assert "embedding" not in decoded
    assert all("funasr" not in segment for segment in decoded["segments"])


def test_empty_projection_language_duration_and_requested_rich_shape(
    tmp_path: Path,
) -> None:
    from botified_asr.result_artifact import (
        CanonicalJsonlReader,
        ResultProjector,
    )

    opener = FreshOpener(b"")
    reader = CanonicalJsonlReader(tmp_path / "empty.jsonl", opener=opener)
    projector = ResultProjector()

    _, text_body = _body(projector.prepare(reader, _options("text"), total_samples=0))
    assert text_body == b""
    _, json_body = _body(
        projector.prepare(
            reader,
            _options("json", include=("funasr.emotion",)),
            total_samples=0,
        )
    )
    assert json_body == b'{"text":"","funasr":{"emotion":[]}}'
    _, verbose_auto = _body(
        projector.prepare(
            reader,
            _options(
                "verbose_json",
                include=("funasr.audio_events",),
            ),
            total_samples=0,
        )
    )
    assert verbose_auto == (
        b'{"task":"transcribe","language":"unknown","duration":0,'
        b'"text":"","segments":[],"funasr":{"audio_events":[]}}'
    )
    _, diarized_empty = _body(
        projector.prepare(
            reader,
            _options("diarized_json"),
            total_samples=0,
        )
    )
    assert (
        diarized_empty == b'{"task":"transcribe","duration":0,"text":"","segments":[]}'
    )
    _, verbose_explicit = _body(
        projector.prepare(
            reader,
            _options("verbose_json", language="fr"),
            total_samples=0,
        )
    )
    assert json.loads(verbose_explicit)["language"] == "fr"

    language_reader = CanonicalJsonlReader(
        tmp_path / "language.jsonl",
        opener=FreshOpener(
            _jsonl(
                _mapping(index=0, text="first", language=None),
                _mapping(
                    index=1,
                    start_sample=1,
                    end_sample=2,
                    text="second",
                    language="zh",
                ),
            )
        ),
    )
    _, verbose_unknown = _body(
        projector.prepare(
            language_reader,
            _options("verbose_json"),
            total_samples=2,
        )
    )
    assert json.loads(verbose_unknown)["language"] == "unknown"


@pytest.mark.parametrize("response_format", ["json", "text", "verbose_json"])
def test_normal_projection_rejects_any_labeled_artifact_before_return(
    response_format: str,
    tmp_path: Path,
) -> None:
    from botified_asr.result_artifact import (
        CanonicalArtifactError,
        CanonicalJsonlReader,
        ResultProjector,
    )

    opener = FreshOpener(
        _jsonl(
            _mapping(index=0, end_sample=10),
            _mapping(
                index=1,
                start_sample=10,
                end_sample=20,
                anonymous_speaker="A",
            ),
        )
    )
    reader = CanonicalJsonlReader(tmp_path / "labeled.jsonl", opener=opener)

    with pytest.raises(CanonicalArtifactError) as caught:
        ResultProjector().prepare(
            reader,
            _options(response_format),
            total_samples=20,
        )

    assert caught.value.code == "invalid_result_artifact"
    assert len(opener.streams) == 1
    assert len(opener.streams[0].readline_sizes) >= 2


def test_diarized_projection_rejects_any_unlabeled_record_before_return(
    tmp_path: Path,
) -> None:
    from botified_asr.result_artifact import (
        CanonicalArtifactError,
        CanonicalJsonlReader,
        ResultProjector,
    )

    opener = FreshOpener(
        _jsonl(
            _mapping(index=0, end_sample=10, anonymous_speaker="A"),
            _mapping(index=1, start_sample=10, end_sample=20),
        )
    )
    reader = CanonicalJsonlReader(tmp_path / "unlabeled.jsonl", opener=opener)

    with pytest.raises(CanonicalArtifactError) as caught:
        ResultProjector().prepare(
            reader,
            _options("diarized_json"),
            total_samples=20,
        )

    assert caught.value.code == "invalid_result_artifact"
    assert len(opener.streams) == 1
    assert len(opener.streams[0].readline_sizes) >= 2


def test_prepare_fully_scans_before_return_and_rejects_corrupt_or_bad_total(
    tmp_path: Path,
) -> None:
    from botified_asr.result_artifact import (
        CanonicalArtifactError,
        CanonicalJsonlReader,
        ResultProjector,
    )

    valid_first = _jsonl(_mapping(index=0, end_sample=10))
    corrupt_last = _jsonl(_mapping(index=1, start_sample=10, end_sample=20))[:-1]
    opener = FreshOpener(valid_first + corrupt_last)
    reader = CanonicalJsonlReader(tmp_path / "corrupt.jsonl", opener=opener)
    with pytest.raises(CanonicalArtifactError):
        ResultProjector().prepare(
            reader,
            _options("json", include=("funasr.emotion",)),
            total_samples=20,
        )
    assert len(opener.streams) == 1
    assert len(opener.streams[0].readline_sizes) >= 2

    valid_reader = CanonicalJsonlReader(
        tmp_path / "valid.jsonl",
        opener=FreshOpener(valid_first),
    )
    for total_samples in (True, 1.0, -1, 691_200_001, 9):
        with pytest.raises(CanonicalArtifactError) as caught:
            ResultProjector().prepare(
                valid_reader,
                _options("json"),
                total_samples=total_samples,  # type: ignore[arg-type]
            )
        assert caught.value.code == "invalid_result_artifact"
