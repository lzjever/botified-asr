from __future__ import annotations

import json
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from botified_asr.audio import (
    BLOCK_SAMPLES,
    Cancellation,
    DecodedBlock,
    MediaProbe,
)
from botified_asr.contracts import (
    CANONICAL_JSONL_MAX_RECORD_BYTES,
    MAX_AUDIO_SAMPLES,
    CanonicalOptions,
)

DIRECT_MAX_SAMPLES = 480_000
HAN_HIRAGANA_KATAKANA_RANGES = (
    (0x2E80, 0x2E99),
    (0x2E9B, 0x2EF3),
    (0x2F00, 0x2FD5),
    (0x3005, 0x3005),
    (0x3007, 0x3007),
    (0x3021, 0x3029),
    (0x3038, 0x303A),
    (0x303B, 0x303B),
    (0x3041, 0x3096),
    (0x309D, 0x309E),
    (0x309F, 0x309F),
    (0x30A1, 0x30FA),
    (0x30FD, 0x30FE),
    (0x30FF, 0x30FF),
    (0x31F0, 0x31FF),
    (0x32D0, 0x32FE),
    (0x3300, 0x3357),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFA6D),
    (0xFA70, 0xFAD9),
    (0xFF66, 0xFF6F),
    (0xFF71, 0xFF9D),
    (0x16FE2, 0x16FE2),
    (0x16FE3, 0x16FE3),
    (0x16FF0, 0x16FF1),
    (0x16FF2, 0x16FF3),
    (0x16FF4, 0x16FF6),
    (0x1AFF0, 0x1AFF3),
    (0x1AFF5, 0x1AFFB),
    (0x1AFFD, 0x1AFFE),
    (0x1B000, 0x1B000),
    (0x1B001, 0x1B11F),
    (0x1B120, 0x1B122),
    (0x1B132, 0x1B132),
    (0x1B150, 0x1B152),
    (0x1B155, 0x1B155),
    (0x1B164, 0x1B167),
    (0x1F200, 0x1F200),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B81D),
    (0x2B820, 0x2CEAD),
    (0x2CEB0, 0x2EBE0),
    (0x2EBF0, 0x2EE5D),
    (0x2F800, 0x2FA1D),
    (0x30000, 0x3134A),
    (0x31350, 0x33479),
)


class PipelineError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class PipelineNotReady(PipelineError):
    def __init__(self) -> None:
        super().__init__(
            "pipeline_not_ready",
            "The requested audio pipeline is not ready",
        )


@dataclass(frozen=True)
class RichAnnotations:
    emotion: str | None = None
    audio_event: str | None = None


@dataclass(frozen=True)
class AsrResult:
    text: str
    language: str | None
    annotations: RichAnnotations


@dataclass(frozen=True)
class SegmentRecord:
    index: int
    start_sample: int
    end_sample: int
    text: str
    language: str | None
    annotations: RichAnnotations


@dataclass(frozen=True)
class VadMarker:
    start_ms: int | None
    end_ms: int | None


@dataclass(frozen=True)
class SpeechSpan:
    start_sample: int
    end_sample: int


@runtime_checkable
class AsrAdapter(Protocol):
    def transcribe(self, pcm: np.ndarray) -> AsrResult: ...


@runtime_checkable
class StreamingVadAdapter(Protocol):
    def generate(
        self,
        pcm: np.ndarray,
        *,
        cache: dict[str, object],
        is_final: bool,
    ) -> tuple[VadMarker, ...]: ...


@runtime_checkable
class Float32AsrModel(Protocol):
    def infer(self, pcm: np.ndarray) -> AsrResult: ...


class NormalizingAsrAdapter:
    def __init__(self, model: Float32AsrModel) -> None:
        self._model = model

    def transcribe(self, pcm: np.ndarray) -> AsrResult:
        if (
            pcm.dtype != np.int16
            or pcm.ndim != 1
            or not pcm.flags.c_contiguous
            or not 1 <= len(pcm) <= DIRECT_MAX_SAMPLES
        ):
            raise PipelineError("invalid_audio", "ASR input segment is invalid")
        normalized = pcm.astype(np.float32)
        normalized /= np.float32(32768.0)
        if not np.isfinite(normalized).all():
            raise PipelineError("invalid_audio", "ASR input segment is invalid")
        return self._model.infer(normalized)


class StreamingVadSession:
    def __init__(self, adapter: StreamingVadAdapter) -> None:
        self._adapter = adapter
        self._cache: dict[str, object] = {}
        self._state = "idle"
        self._pending_start_ms: int | None = None
        self._last_end_ms = 0

    @property
    def open_start_sample(self) -> int | None:
        if self._state != "pending" or self._pending_start_ms is None:
            return None
        return self._pending_start_ms * 16

    def process(
        self,
        block: DecodedBlock,
        *,
        is_final: bool,
    ) -> tuple[SpeechSpan, ...]:
        if self._state == "closed":
            self._raise_invalid_output()

        try:
            markers = self._adapter.generate(
                block.pcm,
                cache=self._cache,
                is_final=is_final,
            )
            next_state = self._state
            pending_start_ms = self._pending_start_ms
            last_end_ms = self._last_end_ms
            spans: list[SpeechSpan] = []
            for marker in markers:
                start_ms = marker.start_ms
                end_ms = marker.end_ms
                if (
                    start_ms is not None
                    and (type(start_ms) is not int or start_ms < 0)
                    or end_ms is not None
                    and (type(end_ms) is not int or end_ms < 0)
                ):
                    self._raise_invalid_output()

                if start_ms is not None and end_ms is None:
                    if next_state != "idle" or start_ms < last_end_ms:
                        self._raise_invalid_output()
                    next_state = "pending"
                    pending_start_ms = start_ms
                    continue

                if start_ms is None and end_ms is not None:
                    if (
                        next_state != "pending"
                        or pending_start_ms is None
                        or end_ms <= pending_start_ms
                    ):
                        self._raise_invalid_output()
                    spans.append(
                        SpeechSpan(
                            start_sample=pending_start_ms * 16,
                            end_sample=end_ms * 16,
                        )
                    )
                    next_state = "idle"
                    pending_start_ms = None
                    last_end_ms = end_ms
                    continue

                if start_ms is not None and end_ms is not None:
                    if (
                        next_state != "idle"
                        or start_ms < last_end_ms
                        or end_ms <= start_ms
                    ):
                        self._raise_invalid_output()
                    spans.append(
                        SpeechSpan(
                            start_sample=start_ms * 16,
                            end_sample=end_ms * 16,
                        )
                    )
                    last_end_ms = end_ms
                    continue

                self._raise_invalid_output()

            if is_final:
                if next_state != "idle":
                    self._raise_invalid_output()
                next_state = "closed"

            self._state = next_state
            self._pending_start_ms = pending_start_ms
            self._last_end_ms = last_end_ms
            return tuple(spans)
        except Exception:
            self._state = "closed"
            self._pending_start_ms = None
            raise

    def _raise_invalid_output(self) -> None:
        self._state = "closed"
        self._pending_start_ms = None
        raise PipelineError(
            "invalid_model_output",
            "VAD model returned an invalid marker transition",
        )


@runtime_checkable
class ByteWriter(Protocol):
    def write(self, payload: bytes) -> None: ...

    def seal(self) -> object: ...

    def abort(self) -> None: ...


@runtime_checkable
class SegmentSink(Protocol):
    def append(self, record: SegmentRecord) -> None: ...

    def finalize(self) -> object: ...

    def abort(self) -> None: ...


@runtime_checkable
class DecodedBlocks(Protocol):
    def __iter__(self) -> Iterator[DecodedBlock]: ...

    def close(self) -> None: ...


@runtime_checkable
class AudioFrontend(Protocol):
    def probe(self, input_path: Path, cancellation: Cancellation) -> MediaProbe: ...

    def decode(
        self,
        input_path: Path,
        probe: MediaProbe,
        cancellation: Cancellation,
    ) -> DecodedBlocks: ...


@runtime_checkable
class ProgressSink(Protocol):
    def update(self, *, processed_samples: int, total_samples: int | None) -> None: ...


class CanonicalJsonlSegmentSink:
    def __init__(self, writer: ByteWriter) -> None:
        self._writer = writer
        self._next_index = 0
        self._last_end_sample = 0
        self._state = "open"

    def append(self, record: SegmentRecord) -> None:
        if self._state != "open":
            raise RuntimeError("segment sink is already finalized")
        if any(
            type(value) is not int
            for value in (
                record.index,
                record.start_sample,
                record.end_sample,
            )
        ):
            raise TypeError("segment index and bounds must be integers")
        if (
            record.index < 0
            or record.start_sample < 0
            or record.end_sample < 0
            or record.end_sample > MAX_AUDIO_SAMPLES
        ):
            raise ValueError("segment sample bounds exceed the allowed range")
        if record.index != self._next_index:
            raise ValueError("segment index must be contiguous")
        if (
            record.start_sample < self._last_end_sample
            or record.end_sample <= record.start_sample
        ):
            raise ValueError("segment sample bounds must be ordered")
        if (
            not isinstance(record.text, str)
            or record.language is not None
            and not isinstance(record.language, str)
            or not isinstance(record.annotations, RichAnnotations)
        ):
            raise TypeError("segment record has invalid field types")
        if any(
            value is not None and not isinstance(value, str)
            for value in (
                record.annotations.emotion,
                record.annotations.audio_event,
            )
        ):
            raise TypeError("rich annotation values must be strings or None")
        payload = serialize_canonical_record(record)
        if len(payload) > CANONICAL_JSONL_MAX_RECORD_BYTES:
            raise ValueError("canonical result record exceeds byte limit")
        self._writer.write(payload + b"\n")
        self._next_index += 1
        self._last_end_sample = record.end_sample

    def finalize(self) -> object:
        if self._state != "open":
            raise RuntimeError("segment sink is already finalized")
        artifact_ref = self._writer.seal()
        self._state = "sealed"
        return artifact_ref

    def abort(self) -> None:
        if self._state == "aborted":
            return
        if self._state == "sealed":
            raise RuntimeError("sealed segment sink cannot be aborted")
        self._writer.abort()
        self._state = "aborted"


class Processor:
    def __init__(
        self,
        frontend: AudioFrontend,
        adapter: AsrAdapter,
    ) -> None:
        self._frontend = frontend
        self._adapter = adapter

    def process(
        self,
        input_path: Path,
        canonical_options: CanonicalOptions,
        cancellation: Cancellation,
        progress_sink: ProgressSink,
        segment_sink: SegmentSink,
    ) -> object:
        failed = True
        blocks: DecodedBlocks | None = None
        decoder_close_attempted = False

        def close_decoder() -> None:
            nonlocal decoder_close_attempted
            if decoder_close_attempted or blocks is None:
                return
            decoder_close_attempted = True
            blocks.close()

        try:
            if (
                canonical_options.chunking_strategy == "auto"
                or canonical_options.model == "sensevoice-diarize"
            ):
                raise PipelineNotReady()
            if not (
                canonical_options.model == "sensevoice"
                and canonical_options.chunking_strategy is None
            ):
                raise PipelineError(
                    "invalid_pipeline_mode",
                    "Unsupported audio pipeline mode",
                )
            if cancellation.cancelled:
                raise PipelineError("cancelled", "Audio processing was cancelled")
            probe = self._frontend.probe(input_path, cancellation)
            blocks = self._frontend.decode(input_path, probe, cancellation)
            self._process_direct(
                blocks,
                cancellation,
                progress_sink,
                segment_sink,
            )
            close_decoder()
            artifact_ref = segment_sink.finalize()
            failed = False
            return artifact_ref
        finally:
            try:
                close_decoder()
            finally:
                if failed:
                    segment_sink.abort()

    def _process_direct(
        self,
        blocks: DecodedBlocks,
        cancellation: Cancellation,
        progress_sink: ProgressSink,
        segment_sink: SegmentSink,
    ) -> None:
        chunks: list[np.ndarray] = []
        sample_count = 0
        saw_short_block = False
        for block in blocks:
            if cancellation.cancelled:
                raise PipelineError("cancelled", "Audio processing was cancelled")
            if saw_short_block:
                raise PipelineError(
                    "invalid_audio",
                    "Decoded audio has a non-final short block",
                )
            if block.start_sample != sample_count:
                raise PipelineError(
                    "invalid_audio",
                    "Decoded audio blocks are not contiguous",
                )
            next_count = sample_count + len(block.pcm)
            if next_count > DIRECT_MAX_SAMPLES:
                raise PipelineError(
                    "long_audio_requires_vad",
                    "chunking_strategy=auto is required for long audio",
                )
            chunks.append(block.pcm)
            sample_count = next_count
            saw_short_block = len(block.pcm) < BLOCK_SAMPLES
            progress_sink.update(
                processed_samples=sample_count,
                total_samples=None,
            )

        if not sample_count:
            progress_sink.update(
                processed_samples=0,
                total_samples=None,
            )
            return
        int16_pcm = np.concatenate(chunks)
        if (
            int16_pcm.dtype != np.int16
            or int16_pcm.ndim != 1
            or not int16_pcm.flags.c_contiguous
            or len(int16_pcm) > DIRECT_MAX_SAMPLES
        ):
            raise PipelineError("invalid_audio", "Decoded audio is invalid")
        result = self._adapter.transcribe(int16_pcm)
        if not isinstance(result, AsrResult):
            raise PipelineError(
                "invalid_model_output",
                "ASR model returned an invalid result",
            )
        segment_sink.append(
            SegmentRecord(
                index=0,
                start_sample=0,
                end_sample=sample_count,
                text=result.text,
                language=result.language,
                annotations=result.annotations,
            )
        )


def serialize_canonical_record(record: SegmentRecord) -> bytes:
    return json.dumps(
        asdict(record),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def iter_canonical_join(parts: Iterable[str]) -> Iterator[str]:
    previous_last: str | None = None
    for raw_part in parts:
        part = raw_part.strip()
        if not part:
            continue
        if previous_last is not None and _needs_join_space(previous_last, part[0]):
            yield " "
        yield part
        previous_last = part[-1]


def canonical_join(parts: Iterable[str]) -> str:
    return "".join(iter_canonical_join(parts))


def _needs_join_space(left: str, right: str) -> bool:
    if unicodedata.category(left) in {"Ps", "Pi"}:
        return False
    if unicodedata.category(right) in {"Pe", "Pf", "Po"}:
        return False
    if _is_han_hiragana_or_katakana(left):
        return False
    return not _is_han_hiragana_or_katakana(right)


def _is_han_hiragana_or_katakana(character: str) -> bool:
    codepoint = ord(character)
    return any(start <= codepoint <= end for start, end in HAN_HIRAGANA_KATAKANA_RANGES)
