from __future__ import annotations

import json
import unicodedata
from collections import deque
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
    DIRECT_MAX_SAMPLES,
    MAX_AUDIO_SAMPLES,
    CanonicalOptions,
)
from botified_asr.errors import PipelineError, PipelineNotReady
from botified_asr.speaker_matching import SpeakerLabelMapping
from botified_asr.speaker_snapshot import (
    SelectedSpeaker,
    SelectedSpeakerSnapshot,
)
from botified_asr.speakers import (
    AnonymousSpeakerClusteringResult,
    SPEAKER_EMBEDDING_BATCH_MAX_WINDOWS,
    SPEAKER_EMBEDDING_DIMENSION,
    SPEAKER_EMBEDDING_NORM_TOLERANCE,
    SPEAKER_WINDOW_MAX_SAMPLES,
    SPEAKER_WINDOW_SHIFT_SAMPLES,
    SpeakerEmbeddingAdapter,
    SpeakerEmbeddingWindow,
    is_anonymous_speaker_label,
)

ASR_BATCH_MAX_SEGMENTS = 32
ASR_BATCH_MAX_PCM_SAMPLES = 960_000
ASR_BATCH_MAX_WALL_SAMPLES = 4_800_000
VAD_FRONTEND_CONTEXT_SAMPLES = 640
VAD_BACKTRACK_SAMPLES = 6_400
VAD_PREPADDING_SAMPLES = 3_200
VAD_IDLE_RING_SAMPLES = (
    BLOCK_SAMPLES
    + VAD_FRONTEND_CONTEXT_SAMPLES
    + VAD_BACKTRACK_SAMPLES
    + VAD_PREPADDING_SAMPLES
)
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


@dataclass(frozen=True)
class RichAnnotations:
    emotion: str | None = None
    audio_event: str | None = None


@dataclass(frozen=True)
class AsrResult:
    text: str
    language: str | None
    annotations: RichAnnotations


@dataclass(frozen=True, slots=True)
class ProcessorResult:
    artifact_ref: object
    speaker_mapping: SpeakerLabelMapping


@dataclass(frozen=True)
class SegmentRecord:
    index: int
    start_sample: int
    end_sample: int
    text: str
    language: str | None
    annotations: RichAnnotations
    anonymous_speaker: str | None = None


@dataclass(frozen=True)
class VadMarker:
    start_ms: int | None
    end_ms: int | None


@dataclass(frozen=True)
class SpeechSpan:
    start_sample: int
    end_sample: int


@dataclass(frozen=True)
class BufferedSpeechSegment:
    span: SpeechSpan
    pcm_start_sample: int
    pcm: np.ndarray

    def __post_init__(self) -> None:
        if (
            self.pcm_start_sample < 0
            or self.span.start_sample < self.pcm_start_sample
            or self.span.end_sample <= self.span.start_sample
            or len(self.pcm) != self.span.end_sample - self.pcm_start_sample
            or self.pcm.dtype != np.int16
            or self.pcm.ndim != 1
            or not self.pcm.flags.c_contiguous
            or len(self.pcm) > DIRECT_MAX_SAMPLES
        ):
            raise ValueError("buffered speech segment is invalid")


def canonical_speech_pcm(segment: object) -> np.ndarray:
    if not isinstance(segment, BufferedSpeechSegment):
        raise PipelineError(
            "invalid_model_output",
            "Speech segmenter returned an invalid segment",
        )
    canonical_start = segment.span.start_sample - segment.pcm_start_sample
    canonical_end = segment.span.end_sample - segment.pcm_start_sample
    canonical_pcm = np.ascontiguousarray(
        segment.pcm[canonical_start:canonical_end],
        dtype=np.int16,
    )
    if (
        canonical_pcm.ndim != 1
        or not canonical_pcm.flags.c_contiguous
        or len(canonical_pcm) != segment.span.end_sample - segment.span.start_sample
        or not len(canonical_pcm)
    ):
        raise PipelineError(
            "invalid_model_output",
            "Speech segmenter returned an invalid segment",
        )
    return canonical_pcm


def _embed_global_speaker_windows(
    segments: tuple[BufferedSpeechSegment, ...],
    *,
    total_samples: int,
    speaker_adapter: SpeakerEmbeddingAdapter,
) -> tuple[SpeakerEmbeddingWindow, ...]:
    def invalid_output() -> PipelineError:
        return PipelineError(
            "invalid_model_output",
            "Speaker embedding input or output is invalid",
        )

    if type(segments) is not tuple:
        raise invalid_output()
    if type(total_samples) is not int or total_samples < 0:
        raise invalid_output()

    canonical_segments: list[tuple[SpeechSpan, np.ndarray]] = []
    previous_end = 0
    for segment in segments:
        if (
            not isinstance(segment, BufferedSpeechSegment)
            or type(segment.span.start_sample) is not int
            or type(segment.span.end_sample) is not int
            or type(segment.pcm_start_sample) is not int
            or not isinstance(segment.pcm, np.ndarray)
            or segment.pcm.dtype != np.int16
            or segment.pcm.ndim != 1
            or not segment.pcm.flags.c_contiguous
            or segment.pcm_start_sample < 0
            or segment.span.start_sample < segment.pcm_start_sample
            or segment.span.end_sample <= segment.span.start_sample
            or len(segment.pcm)
            != segment.span.end_sample - segment.pcm_start_sample
            or len(segment.pcm) > DIRECT_MAX_SAMPLES
            or segment.span.start_sample < previous_end
            or segment.span.end_sample > total_samples
        ):
            raise invalid_output()
        try:
            pcm = canonical_speech_pcm(segment)
        except PipelineError as error:
            raise invalid_output() from error
        canonical_segments.append((segment.span, pcm))
        previous_end = segment.span.end_sample

    if not canonical_segments:
        return ()

    windows: list[SpeakerEmbeddingWindow] = []
    pending_ranges: list[tuple[int, int]] = []
    pending_pcms: list[np.ndarray] = []

    def flush() -> None:
        if not pending_pcms:
            return
        embeddings = speaker_adapter.embed_exact_windows(tuple(pending_pcms))
        if type(embeddings) is not tuple or len(embeddings) != len(pending_ranges):
            raise invalid_output()
        for (start_sample, end_sample), embedding in zip(
            pending_ranges,
            embeddings,
            strict=True,
        ):
            if (
                not isinstance(embedding, np.ndarray)
                or embedding.dtype != np.float32
                or embedding.shape != (SPEAKER_EMBEDDING_DIMENSION,)
                or not embedding.flags.c_contiguous
                or not np.isfinite(embedding).all()
            ):
                raise invalid_output()
            norm = float(
                np.linalg.norm(embedding.astype(np.float64, copy=False))
            )
            if (
                not np.isfinite(norm)
                or not np.isclose(
                    norm,
                    1.0,
                    rtol=0.0,
                    atol=SPEAKER_EMBEDDING_NORM_TOLERANCE,
                )
            ):
                raise invalid_output()
            windows.append(
                SpeakerEmbeddingWindow(
                    start_sample=start_sample,
                    end_sample=end_sample,
                    embedding=embedding,
                )
            )
        pending_ranges.clear()
        pending_pcms.clear()

    segment_cursor = 0
    for start_sample in range(
        0,
        total_samples,
        SPEAKER_WINDOW_SHIFT_SAMPLES,
    ):
        logical_end_sample = start_sample + SPEAKER_WINDOW_MAX_SAMPLES
        physical_end_sample = min(logical_end_sample, total_samples)
        while (
            segment_cursor < len(canonical_segments)
            and canonical_segments[segment_cursor][0].end_sample <= start_sample
        ):
            segment_cursor += 1

        overlaps: list[tuple[SpeechSpan, np.ndarray, int, int]] = []
        scan_index = segment_cursor
        while (
            scan_index < len(canonical_segments)
            and canonical_segments[scan_index][0].start_sample
            < physical_end_sample
        ):
            span, speech_pcm = canonical_segments[scan_index]
            overlap_start = max(start_sample, span.start_sample)
            overlap_end = min(physical_end_sample, span.end_sample)
            if overlap_start < overlap_end:
                overlaps.append(
                    (span, speech_pcm, overlap_start, overlap_end)
                )
            scan_index += 1
        if not overlaps:
            continue

        pcm = np.zeros(
            physical_end_sample - start_sample,
            dtype=np.int16,
        )
        for span, speech_pcm, overlap_start, overlap_end in overlaps:
            pcm[
                overlap_start - start_sample : overlap_end - start_sample
            ] = speech_pcm[
                overlap_start - span.start_sample : overlap_end - span.start_sample
            ]
        pending_ranges.append((start_sample, logical_end_sample))
        pending_pcms.append(pcm)
        if len(pending_pcms) == SPEAKER_EMBEDDING_BATCH_MAX_WINDOWS:
            flush()

    flush()
    return tuple(windows)


def _project_speaker_regions(
    speech_islands: tuple[SpeechSpan, ...],
    windows: tuple[SpeakerEmbeddingWindow, ...],
    clustering_result: AnonymousSpeakerClusteringResult,
    *,
    total_samples: int,
) -> tuple[tuple[SpeechSpan, int], ...]:
    def invalid_output() -> PipelineError:
        return PipelineError(
            "invalid_model_output",
            "Speaker region projection input is invalid",
        )

    if (
        type(speech_islands) is not tuple
        or type(windows) is not tuple
        or type(clustering_result) is not AnonymousSpeakerClusteringResult
        or type(total_samples) is not int
        or total_samples < 0
    ):
        raise invalid_output()
    window_cluster_ordinals = clustering_result.window_cluster_ordinals
    clusters = clustering_result.clusters
    if (
        type(window_cluster_ordinals) is not tuple
        or type(clusters) is not tuple
        or len(window_cluster_ordinals) != len(windows)
    ):
        raise invalid_output()

    previous_end: int | None = None
    for island in speech_islands:
        if (
            type(island) is not SpeechSpan
            or type(island.start_sample) is not int
            or type(island.end_sample) is not int
            or island.start_sample < 0
            or island.end_sample <= island.start_sample
            or island.end_sample > total_samples
            or previous_end is not None
            and island.start_sample < previous_end
        ):
            raise invalid_output()
        previous_end = island.end_sample

    previous_start: int | None = None
    for window in windows:
        if (
            type(window) is not SpeakerEmbeddingWindow
            or type(window.start_sample) is not int
            or type(window.end_sample) is not int
            or window.start_sample < 0
            or window.start_sample >= total_samples
            or window.start_sample % SPEAKER_WINDOW_SHIFT_SAMPLES != 0
            or window.end_sample
            != window.start_sample + SPEAKER_WINDOW_MAX_SAMPLES
            or previous_start is not None
            and window.start_sample <= previous_start
        ):
            raise invalid_output()
        previous_start = window.start_sample

    if not speech_islands and not windows:
        if window_cluster_ordinals or clusters:
            raise invalid_output()
        return ()
    if not speech_islands or not windows:
        raise invalid_output()
    if not clusters:
        raise invalid_output()

    used_clusters = [False] * len(clusters)
    for ordinal in window_cluster_ordinals:
        if (
            type(ordinal) is not int
            or ordinal < 0
            or ordinal >= len(clusters)
        ):
            raise invalid_output()
        used_clusters[ordinal] = True
    if not all(used_clusters):
        raise invalid_output()

    island_cursor = 0
    for window in windows:
        physical_end = min(window.end_sample, total_samples)
        while (
            island_cursor < len(speech_islands)
            and speech_islands[island_cursor].end_sample
            <= window.start_sample
        ):
            island_cursor += 1
        if (
            island_cursor == len(speech_islands)
            or speech_islands[island_cursor].start_sample >= physical_end
        ):
            raise invalid_output()

    window_cursor = 0
    for island in speech_islands:
        while (
            window_cursor < len(windows)
            and min(windows[window_cursor].end_sample, total_samples)
            <= island.start_sample
        ):
            window_cursor += 1
        if (
            window_cursor == len(windows)
            or windows[window_cursor].start_sample >= island.end_sample
        ):
            raise invalid_output()

    boundaries = [0]
    for left, right in zip(windows, windows[1:], strict=False):
        left_center = left.start_sample + SPEAKER_WINDOW_MAX_SAMPLES // 2
        right_center = right.start_sample + SPEAKER_WINDOW_MAX_SAMPLES // 2
        midpoint = (left_center + right_center + 1) // 2
        clipped = min(max(midpoint, 0), total_samples)
        boundaries.append(max(boundaries[-1], clipped))
    boundaries.append(total_samples)

    projected: list[tuple[SpeechSpan, int]] = []
    cell_cursor = 0
    for island in speech_islands:
        while (
            cell_cursor < len(window_cluster_ordinals)
            and boundaries[cell_cursor + 1] <= island.start_sample
        ):
            cell_cursor += 1
        merged: list[tuple[int, int, int]] = []
        scan_index = cell_cursor
        while (
            scan_index < len(window_cluster_ordinals)
            and boundaries[scan_index] < island.end_sample
        ):
            ordinal = window_cluster_ordinals[scan_index]
            start_sample = max(
                island.start_sample,
                boundaries[scan_index],
            )
            end_sample = min(
                island.end_sample,
                boundaries[scan_index + 1],
            )
            if start_sample >= end_sample:
                scan_index += 1
                continue
            if (
                merged
                and merged[-1][1] == start_sample
                and merged[-1][2] == ordinal
            ):
                merged[-1] = (
                    merged[-1][0],
                    end_sample,
                    ordinal,
                )
            else:
                merged.append((start_sample, end_sample, ordinal))
            scan_index += 1
        if not merged:
            raise invalid_output()
        for start_sample, end_sample, ordinal in merged:
            while end_sample - start_sample > DIRECT_MAX_SAMPLES:
                split_end = start_sample + DIRECT_MAX_SAMPLES
                projected.append(
                    (SpeechSpan(start_sample, split_end), ordinal)
                )
                start_sample = split_end
            projected.append(
                (SpeechSpan(start_sample, end_sample), ordinal)
            )
    return tuple(projected)


@dataclass(frozen=True)
class _BufferedPcmChunk:
    start_sample: int
    pcm: np.ndarray

    @property
    def end_sample(self) -> int:
        return self.start_sample + len(self.pcm)


class BoundedSpeechPcmBuffer:
    def __init__(self) -> None:
        self._chunks: deque[_BufferedPcmChunk] = deque()
        self._processed_end_sample = 0
        self._open_origin_sample: int | None = None
        self._canonical_cursor_sample: int | None = None
        self._pcm_cursor_sample: int | None = None
        self._canonical_watermark_sample = 0
        self._forced_floor_sample: int | None = None
        self._terminal = False

    @property
    def retained_start_sample(self) -> int:
        if not self._chunks:
            return self._processed_end_sample
        return self._chunks[0].start_sample

    @property
    def retained_sample_count(self) -> int:
        return self._processed_end_sample - self.retained_start_sample

    def consume(
        self,
        block: DecodedBlock,
        *,
        completed_spans: tuple[SpeechSpan, ...] = (),
        open_start_sample: int | None = None,
    ) -> tuple[BufferedSpeechSegment, ...]:
        if self._terminal:
            raise PipelineError(
                "invalid_model_output",
                "Speech PCM buffer is no longer usable",
            )
        try:
            return self._consume(
                block,
                completed_spans=completed_spans,
                open_start_sample=open_start_sample,
            )
        except Exception:
            self._terminal = True
            raise

    def _consume(
        self,
        block: DecodedBlock,
        *,
        completed_spans: tuple[SpeechSpan, ...],
        open_start_sample: int | None,
    ) -> tuple[BufferedSpeechSegment, ...]:
        if block.start_sample != self._processed_end_sample:
            raise PipelineError(
                "invalid_audio",
                "Decoded audio blocks are not contiguous",
            )
        if open_start_sample is not None and (
            type(open_start_sample) is not int or open_start_sample < 0
        ):
            self._raise_invalid_output()

        self._append(block)
        if self._open_origin_sample is None:
            self._trim_idle_history()

        for span in completed_spans:
            self._validate_completed_span(span)

        emitted: list[BufferedSpeechSegment] = []
        for span in completed_spans:
            if self._open_origin_sample is None:
                self._begin_speech(span.start_sample)
            elif span.start_sample != self._open_origin_sample:
                self._raise_invalid_output()
            emitted.extend(self._complete_speech(span.end_sample))

        if open_start_sample is None:
            if self._open_origin_sample is not None:
                self._raise_invalid_output()
        elif self._open_origin_sample is None:
            self._begin_speech(open_start_sample)
        elif open_start_sample != self._open_origin_sample:
            self._raise_invalid_output()

        if self._open_origin_sample is not None:
            emitted.extend(self._release_full_segments(self._processed_end_sample))
        else:
            self._trim_idle_history()

        return tuple(emitted)

    def _append(self, block: DecodedBlock) -> None:
        if len(block.pcm):
            self._chunks.append(
                _BufferedPcmChunk(
                    start_sample=block.start_sample,
                    pcm=block.pcm,
                )
            )
        self._processed_end_sample += len(block.pcm)

    def _validate_completed_span(self, span: SpeechSpan) -> None:
        if (
            not isinstance(span, SpeechSpan)
            or type(span.start_sample) is not int
            or type(span.end_sample) is not int
            or span.start_sample < 0
            or span.end_sample <= span.start_sample
            or span.end_sample > self._processed_end_sample
        ):
            self._raise_invalid_output()

    def _begin_speech(self, origin_sample: int) -> None:
        if origin_sample > self._processed_end_sample:
            self._raise_invalid_output()
        if origin_sample == self._canonical_watermark_sample:
            canonical_start_sample = origin_sample
            pcm_start_sample = origin_sample
            if self._forced_floor_sample != self._canonical_watermark_sample:
                self._forced_floor_sample = None
        elif (
            origin_sample <= self._canonical_watermark_sample
            and self._forced_floor_sample == self._canonical_watermark_sample
        ):
            if (
                self._canonical_watermark_sample - origin_sample
                > VAD_PREPADDING_SAMPLES
            ):
                self._raise_invalid_output()
            canonical_start_sample = self._canonical_watermark_sample
            pcm_start_sample = canonical_start_sample
        else:
            if origin_sample < self._canonical_watermark_sample:
                self._raise_invalid_output()
            canonical_start_sample = origin_sample
            pcm_start_sample = max(
                self._canonical_watermark_sample,
                origin_sample - VAD_PREPADDING_SAMPLES,
            )
            self._forced_floor_sample = None
        if pcm_start_sample < self.retained_start_sample:
            self._raise_invalid_output()
        self._open_origin_sample = origin_sample
        self._canonical_cursor_sample = canonical_start_sample
        self._pcm_cursor_sample = pcm_start_sample

    def _complete_speech(
        self,
        end_sample: int,
    ) -> tuple[BufferedSpeechSegment, ...]:
        canonical_cursor = self._require_canonical_cursor()
        had_forced_continuation = self._forced_floor_sample == canonical_cursor
        if end_sample < canonical_cursor:
            if (
                not had_forced_continuation
                or canonical_cursor - end_sample > VAD_PREPADDING_SAMPLES
            ):
                self._raise_invalid_output()
            effective_end_sample = canonical_cursor
        else:
            effective_end_sample = end_sample

        emitted = list(self._release_full_segments(effective_end_sample))
        canonical_cursor = self._require_canonical_cursor()
        pcm_cursor = self._require_pcm_cursor()
        if effective_end_sample > pcm_cursor:
            emitted.append(
                self._make_segment(
                    canonical_start=canonical_cursor,
                    pcm_start=pcm_cursor,
                    end=effective_end_sample,
                )
            )

        self._canonical_watermark_sample = effective_end_sample
        if had_forced_continuation:
            self._forced_floor_sample = effective_end_sample
        else:
            self._forced_floor_sample = None
        self._open_origin_sample = None
        self._canonical_cursor_sample = None
        self._pcm_cursor_sample = None
        return tuple(emitted)

    def _release_full_segments(
        self,
        available_end_sample: int,
    ) -> tuple[BufferedSpeechSegment, ...]:
        emitted: list[BufferedSpeechSegment] = []
        pcm_cursor = self._require_pcm_cursor()
        while available_end_sample - pcm_cursor >= DIRECT_MAX_SAMPLES:
            canonical_cursor = self._require_canonical_cursor()
            segment_end = pcm_cursor + DIRECT_MAX_SAMPLES
            emitted.append(
                self._make_segment(
                    canonical_start=canonical_cursor,
                    pcm_start=pcm_cursor,
                    end=segment_end,
                )
            )
            self._canonical_cursor_sample = segment_end
            self._pcm_cursor_sample = segment_end
            self._canonical_watermark_sample = segment_end
            self._forced_floor_sample = segment_end
            self._trim_before(segment_end)
            pcm_cursor = segment_end
        return tuple(emitted)

    def _make_segment(
        self,
        *,
        canonical_start: int,
        pcm_start: int,
        end: int,
    ) -> BufferedSpeechSegment:
        return BufferedSpeechSegment(
            span=SpeechSpan(
                start_sample=canonical_start,
                end_sample=end,
            ),
            pcm_start_sample=pcm_start,
            pcm=self._copy_range(pcm_start, end),
        )

    def _copy_range(self, start_sample: int, end_sample: int) -> np.ndarray:
        if (
            start_sample < self.retained_start_sample
            or end_sample > self._processed_end_sample
            or end_sample <= start_sample
            or end_sample - start_sample > DIRECT_MAX_SAMPLES
        ):
            self._raise_invalid_output()

        parts: list[np.ndarray] = []
        copied_samples = 0
        for chunk in self._chunks:
            overlap_start = max(start_sample, chunk.start_sample)
            overlap_end = min(end_sample, chunk.end_sample)
            if overlap_start >= overlap_end:
                continue
            local_start = overlap_start - chunk.start_sample
            local_end = overlap_end - chunk.start_sample
            part = chunk.pcm[local_start:local_end]
            parts.append(part)
            copied_samples += len(part)

        if copied_samples != end_sample - start_sample:
            self._raise_invalid_output()
        if len(parts) == 1:
            return parts[0].copy()
        return np.concatenate(parts)

    def _trim_idle_history(self) -> None:
        self._trim_before(max(0, self._processed_end_sample - VAD_IDLE_RING_SAMPLES))

    def _trim_before(self, start_sample: int) -> None:
        if start_sample < 0 or start_sample > self._processed_end_sample:
            self._raise_invalid_output()
        if start_sample <= self.retained_start_sample:
            return
        while self._chunks and self._chunks[0].end_sample <= start_sample:
            self._chunks.popleft()
        if self._chunks and self._chunks[0].start_sample < start_sample:
            chunk = self._chunks.popleft()
            local_start = start_sample - chunk.start_sample
            self._chunks.appendleft(
                _BufferedPcmChunk(
                    start_sample=start_sample,
                    pcm=chunk.pcm[local_start:],
                )
            )

    def _require_canonical_cursor(self) -> int:
        if self._canonical_cursor_sample is None:
            self._raise_invalid_output()
        return self._canonical_cursor_sample

    def _require_pcm_cursor(self) -> int:
        if self._pcm_cursor_sample is None:
            self._raise_invalid_output()
        return self._pcm_cursor_sample

    def _raise_invalid_output(self) -> None:
        raise PipelineError(
            "invalid_model_output",
            "VAD model returned a span outside retained PCM history",
        )


@runtime_checkable
class AsrAdapter(Protocol):
    def transcribe(self, pcm: np.ndarray) -> AsrResult: ...

    def transcribe_batch(
        self,
        pcms: tuple[np.ndarray, ...],
        *,
        language: str,
    ) -> tuple[AsrResult, ...]: ...


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

    def transcribe_batch(
        self,
        pcms: tuple[np.ndarray, ...],
        *,
        language: str,
    ) -> tuple[AsrResult, ...]:
        del language
        return tuple(self.transcribe(pcm) for pcm in pcms)


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


class StreamingSpeechSegmenter:
    def __init__(self, adapter: StreamingVadAdapter) -> None:
        self._vad_session = StreamingVadSession(adapter)
        self._pcm_buffer = BoundedSpeechPcmBuffer()
        self._terminal = False

    def process(
        self,
        block: DecodedBlock,
        *,
        is_final: bool,
    ) -> tuple[BufferedSpeechSegment, ...]:
        if self._terminal:
            raise PipelineError(
                "invalid_model_output",
                "Streaming speech segmenter is no longer usable",
            )
        try:
            completed_spans = self._vad_session.process(
                block,
                is_final=is_final,
            )
            return self._pcm_buffer.consume(
                block,
                completed_spans=completed_spans,
                open_start_sample=self._vad_session.open_start_sample,
            )
        except Exception:
            self._terminal = True
            raise


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
        if record.anonymous_speaker is not None:
            if type(record.anonymous_speaker) is not str:
                raise TypeError("anonymous speaker must be a string or None")
            if not is_anonymous_speaker_label(record.anonymous_speaker):
                raise ValueError("anonymous speaker label is invalid")
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
        *,
        vad_adapter: StreamingVadAdapter | None = None,
    ) -> None:
        self._frontend = frontend
        self._adapter = adapter
        self._vad_adapter = vad_adapter

    def process(
        self,
        input_path: Path,
        canonical_options: CanonicalOptions,
        cancellation: Cancellation,
        progress_sink: ProgressSink,
        segment_sink: SegmentSink,
        *,
        selected_speaker_snapshot: SelectedSpeakerSnapshot,
        effective_max_audio_samples: int,
        effective_direct_max_audio_samples: int,
        media_probe: MediaProbe | None = None,
    ) -> ProcessorResult:
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
            _validate_effective_sample_caps(
                effective_max_audio_samples,
                effective_direct_max_audio_samples,
            )
            if (
                type(selected_speaker_snapshot) is not SelectedSpeakerSnapshot
                or type(selected_speaker_snapshot.speakers) is not tuple
                or any(
                    type(speaker) is not SelectedSpeaker
                    for speaker in selected_speaker_snapshot.speakers
                )
                or tuple(speaker.id for speaker in selected_speaker_snapshot.speakers)
                != canonical_options.known_speaker_ids
            ):
                raise RuntimeError("selected speaker snapshot does not match options")
            is_direct = (
                canonical_options.model == "sensevoice"
                and canonical_options.chunking_strategy is None
            )
            is_vad = (
                canonical_options.model in {"sensevoice", "sensevoice-diarize"}
                and canonical_options.chunking_strategy == "auto"
            )
            is_diarize = (
                canonical_options.model == "sensevoice-diarize"
                and canonical_options.chunking_strategy == "auto"
            )
            if is_diarize:
                raise PipelineNotReady()
            if is_vad and self._vad_adapter is None:
                raise PipelineNotReady()
            if not (is_direct or is_vad):
                raise PipelineError(
                    "invalid_pipeline_mode",
                    "Unsupported audio pipeline mode",
                )
            if cancellation.cancelled:
                raise PipelineError("cancelled", "Audio processing was cancelled")
            probe = (
                self._frontend.probe(input_path, cancellation)
                if media_probe is None
                else media_probe
            )
            blocks = self._frontend.decode(input_path, probe, cancellation)
            if is_vad:
                vad_adapter = self._vad_adapter
                if vad_adapter is None:
                    raise PipelineNotReady()
                actual_samples = self._process_vad(
                    blocks,
                    cancellation,
                    progress_sink,
                    segment_sink,
                    vad_adapter=vad_adapter,
                    language=canonical_options.language,
                    effective_max_audio_samples=effective_max_audio_samples,
                )
            else:
                actual_samples = self._process_direct(
                    blocks,
                    cancellation,
                    progress_sink,
                    segment_sink,
                    language=canonical_options.language,
                    effective_direct_max_audio_samples=(
                        effective_direct_max_audio_samples
                    ),
                )
            close_decoder()
            speaker_mapping = SpeakerLabelMapping(())
            if cancellation.cancelled:
                raise PipelineError("cancelled", "Audio processing was cancelled")
            progress_sink.update(
                processed_samples=actual_samples,
                total_samples=actual_samples,
            )
            artifact_ref = segment_sink.finalize()
            failed = False
            return ProcessorResult(
                artifact_ref,
                speaker_mapping,
            )
        finally:
            try:
                close_decoder()
            finally:
                if failed:
                    segment_sink.abort()

    def _process_vad(
        self,
        blocks: DecodedBlocks,
        cancellation: Cancellation,
        progress_sink: ProgressSink,
        segment_sink: SegmentSink,
        *,
        vad_adapter: StreamingVadAdapter,
        language: str,
        effective_max_audio_samples: int,
    ) -> int:
        segmenter = StreamingSpeechSegmenter(vad_adapter)
        pending: list[BufferedSpeechSegment] = []
        pending_pcm_samples = 0
        next_record_index = 0
        processed_end_sample = 0

        def check_cancellation() -> None:
            if cancellation.cancelled:
                raise PipelineError("cancelled", "Audio processing was cancelled")

        def durable_fence() -> None:
            check_cancellation()
            progress_sink.update(
                processed_samples=processed_end_sample,
                total_samples=None,
            )
            check_cancellation()

        def flush() -> None:
            nonlocal pending
            nonlocal pending_pcm_samples
            nonlocal next_record_index
            if not pending:
                return
            check_cancellation()
            segments = tuple(pending)
            batch_result = self._adapter.transcribe_batch(
                tuple(segment.pcm for segment in segments),
                language=language,
            )
            durable_fence()
            if (
                type(batch_result) is not tuple
                or len(batch_result) != len(segments)
                or any(not isinstance(result, AsrResult) for result in batch_result)
            ):
                raise PipelineError(
                    "invalid_model_output",
                    "ASR model returned an invalid batch result",
                )
            for segment, result in zip(
                segments,
                batch_result,
                strict=True,
            ):
                if result.text == "":
                    continue
                segment_sink.append(
                    SegmentRecord(
                        index=next_record_index,
                        start_sample=segment.span.start_sample,
                        end_sample=segment.span.end_sample,
                        text=result.text,
                        language=result.language,
                        annotations=result.annotations,
                    )
                )
                next_record_index += 1
            pending = []
            pending_pcm_samples = 0

        def enqueue(segment: BufferedSpeechSegment) -> None:
            nonlocal pending_pcm_samples
            if not isinstance(segment, BufferedSpeechSegment):
                raise PipelineError(
                    "invalid_model_output",
                    "Speech segmenter returned an invalid segment",
                )
            exceeds_batch = bool(pending) and (
                len(pending) + 1 > ASR_BATCH_MAX_SEGMENTS
                or pending_pcm_samples + len(segment.pcm) > ASR_BATCH_MAX_PCM_SAMPLES
                or segment.span.end_sample - pending[0].span.start_sample
                > ASR_BATCH_MAX_WALL_SAMPLES
            )
            if exceeds_batch:
                flush()
            pending.append(segment)
            pending_pcm_samples += len(segment.pcm)

        iterator = iter(blocks)
        try:
            current = next(iterator)
        except StopIteration:
            progress_sink.update(
                processed_samples=0,
                total_samples=None,
            )
            return 0

        while True:
            try:
                following = next(iterator)
            except StopIteration:
                following = None
            is_final = following is None

            if current.start_sample != processed_end_sample:
                raise PipelineError(
                    "invalid_audio",
                    "Decoded audio blocks are not contiguous",
                )
            if not is_final and len(current.pcm) < BLOCK_SAMPLES:
                raise PipelineError(
                    "invalid_audio",
                    "Decoded audio has a non-final short block",
                )
            next_processed_end_sample = processed_end_sample + len(current.pcm)
            if next_processed_end_sample > effective_max_audio_samples:
                raise PipelineError(
                    "audio_too_long",
                    "Decoded audio exceeds the configured duration limit",
                )
            check_cancellation()
            emitted = segmenter.process(current, is_final=is_final)
            durable_fence()
            if type(emitted) is not tuple:
                raise PipelineError(
                    "invalid_model_output",
                    "Speech segmenter returned an invalid result",
                )
            for segment in emitted:
                enqueue(segment)

            processed_end_sample = next_processed_end_sample
            progress_sink.update(
                processed_samples=processed_end_sample,
                total_samples=None,
            )
            if is_final:
                break
            current = following

        flush()
        return processed_end_sample

    def _process_direct(
        self,
        blocks: DecodedBlocks,
        cancellation: Cancellation,
        progress_sink: ProgressSink,
        segment_sink: SegmentSink,
        *,
        language: str,
        effective_direct_max_audio_samples: int,
    ) -> int:
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
            if next_count > effective_direct_max_audio_samples:
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
            return 0
        int16_pcm = np.concatenate(chunks)
        if (
            int16_pcm.dtype != np.int16
            or int16_pcm.ndim != 1
            or not int16_pcm.flags.c_contiguous
            or len(int16_pcm) > effective_direct_max_audio_samples
        ):
            raise PipelineError("invalid_audio", "Decoded audio is invalid")
        batch_result = self._adapter.transcribe_batch(
            (int16_pcm,),
            language=language,
        )
        if cancellation.cancelled:
            raise PipelineError("cancelled", "Audio processing was cancelled")
        progress_sink.update(
            processed_samples=sample_count,
            total_samples=None,
        )
        if cancellation.cancelled:
            raise PipelineError("cancelled", "Audio processing was cancelled")
        if (
            type(batch_result) is not tuple
            or len(batch_result) != 1
            or not isinstance(batch_result[0], AsrResult)
        ):
            raise PipelineError(
                "invalid_model_output",
                "ASR model returned an invalid result",
            )
        result = batch_result[0]
        if result.text == "":
            return sample_count
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
        return sample_count


def _validate_effective_sample_caps(
    effective_max_audio_samples: int,
    effective_direct_max_audio_samples: int,
) -> None:
    if type(effective_max_audio_samples) is not int:
        raise TypeError("effective maximum audio samples must be an integer")
    if type(effective_direct_max_audio_samples) is not int:
        raise TypeError("effective direct maximum audio samples must be an integer")
    if not 1 <= effective_max_audio_samples <= MAX_AUDIO_SAMPLES:
        raise ValueError("effective maximum audio samples is outside the allowed range")
    if not (
        1
        <= effective_direct_max_audio_samples
        <= min(effective_max_audio_samples, DIRECT_MAX_SAMPLES)
    ):
        raise ValueError(
            "effective direct maximum audio samples is outside the allowed range"
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
