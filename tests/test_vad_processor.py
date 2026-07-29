from __future__ import annotations

from collections.abc import Callable
from inspect import signature
from pathlib import Path

import numpy as np
import pytest

from botified_asr import pipeline, speakers
from botified_asr.audio import BLOCK_SAMPLES, Cancellation, DecodedBlock, MediaProbe
from botified_asr.contracts import (
    DIARIZATION_MAX_AUDIO_SAMPLES,
    DIRECT_MAX_SAMPLES,
    MAX_AUDIO_SAMPLES,
    CanonicalOptions,
)
from botified_asr.speaker_profiles import SpeakerEmbedding
from botified_asr.speaker_matching import (
    KnownSpeakerMatch,
    KnownSpeakerMatchPolicy,
    SpeakerLabelMapping,
    SpeakerLabelResolution,
    SpeakerMatchInputError,
)
from botified_asr.speaker_snapshot import SelectedSpeaker, SelectedSpeakerSnapshot


EMPTY_SELECTED_SNAPSHOT = SelectedSpeakerSnapshot(())


class ProgressFenceStop(RuntimeError):
    pass


class FakeDecoder:
    def __init__(
        self,
        blocks: tuple[DecodedBlock, ...],
        *,
        events: list[str] | None = None,
    ) -> None:
        self.blocks = blocks
        self.closed = 0
        self.events = events

    def __iter__(self):
        yield from self.blocks

    def close(self) -> None:
        self.closed += 1
        if self.events is not None:
            self.events.append("decoder_close")


class SharedZeroBoundaryDecoder(FakeDecoder):
    def __init__(
        self,
        *,
        full_blocks: int,
        tail_samples: int = 0,
        guard_after_tail: bool = False,
    ) -> None:
        super().__init__(())
        self.full_blocks = full_blocks
        self.tail_samples = tail_samples
        self.guard_after_tail = guard_after_tail
        self.yielded = 0
        self.guard_requests = 0

    def __iter__(self):
        shared = np.zeros(BLOCK_SAMPLES, dtype=np.int16)
        for index in range(self.full_blocks):
            self.yielded += 1
            yield DecodedBlock(index * BLOCK_SAMPLES, shared)
        if self.tail_samples:
            self.yielded += 1
            yield DecodedBlock(
                self.full_blocks * BLOCK_SAMPLES,
                shared[: self.tail_samples],
            )
        if self.guard_after_tail:
            self.guard_requests += 1
            raise AssertionError("decoder iterated beyond the overflow block")


class FakeFrontend:
    def __init__(self, decoder: FakeDecoder) -> None:
        self.decoder = decoder
        self.probe_calls = 0
        self.decode_calls = 0

    def probe(self, _input_path: Path, _cancellation: Cancellation) -> MediaProbe:
        self.probe_calls += 1
        return MediaProbe(1.0, "wav")

    def decode(
        self,
        _input_path: Path,
        _probe: MediaProbe,
        _cancellation: Cancellation,
    ) -> FakeDecoder:
        self.decode_calls += 1
        return self.decoder


class FakeAsrAdapter:
    def __init__(
        self,
        result_factory: Callable[[tuple[np.ndarray, ...]], object] | None = None,
        *,
        events: list[str] | None = None,
    ) -> None:
        self._result_factory = result_factory or (
            lambda pcms: tuple(
                _result(f"text-{index}") for index, _pcm in enumerate(pcms)
            )
        )
        self.events = events
        self.calls: list[tuple[tuple[np.ndarray, ...], str]] = []

    def transcribe(self, _pcm: np.ndarray) -> pipeline.AsrResult:
        raise AssertionError("Processor must not fall back to scalar ASR")

    def transcribe_batch(
        self,
        pcms: tuple[np.ndarray, ...],
        *,
        language: str,
    ) -> object:
        self.calls.append((pcms, language))
        if self.events is not None:
            self.events.append("asr")
        return self._result_factory(pcms)


class RecordingProgress:
    def __init__(
        self,
        *,
        fail_on_call: int | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.updates: list[tuple[int, int | None]] = []
        self.fail_on_call = fail_on_call
        self.events = events

    def update(self, *, processed_samples: int, total_samples: int | None) -> None:
        self.updates.append((processed_samples, total_samples))
        if self.events is not None:
            self.events.append(f"progress:{processed_samples}:{total_samples}")
        if len(self.updates) == self.fail_on_call:
            raise ProgressFenceStop("stopped at durable progress fence")


class RecordingSink:
    def __init__(self, events: list[str] | None = None) -> None:
        self.records: list[pipeline.SegmentRecord] = []
        self.finalized = 0
        self.aborted = 0
        self.ref = object()
        self.events = events

    def append(self, record: pipeline.SegmentRecord) -> None:
        if self.events is not None:
            self.events.append("append")
        self.records.append(record)

    def finalize(self) -> object:
        if self.events is not None:
            self.events.append("sink_finalize")
        self.finalized += 1
        return self.ref

    def abort(self) -> None:
        self.aborted += 1


class ScriptedSegmenter:
    def __init__(
        self,
        adapter: object,
        outputs: tuple[object, ...],
        *,
        completed_outputs: tuple[tuple[pipeline.SpeechSpan, ...], ...] | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.adapter = adapter
        self.outputs = outputs
        self.completed_outputs = completed_outputs
        self.events = events
        self.calls: list[tuple[DecodedBlock, bool]] = []

    def process(
        self,
        block: DecodedBlock,
        *,
        is_final: bool,
    ) -> object:
        output = self.outputs[len(self.calls)]
        self.calls.append((block, is_final))
        if self.events is not None:
            self.events.append("fsmn")
        completed = (
            ()
            if self.completed_outputs is None
            else self.completed_outputs[len(self.calls) - 1]
        )
        return output, completed


class FinalSampleSegmenter:
    def __init__(
        self,
        adapter: object,
        *,
        emit_final_speech: bool,
        emit_first_speech: bool,
    ) -> None:
        self.adapter = adapter
        self.emit_final_speech = emit_final_speech
        self.emit_first_speech = emit_first_speech
        self.calls: list[tuple[DecodedBlock, bool]] = []

    def process(
        self,
        block: DecodedBlock,
        *,
        is_final: bool,
    ) -> tuple[
        tuple[pipeline.BufferedSpeechSegment, ...],
        tuple[pipeline.SpeechSpan, ...],
    ]:
        self.calls.append((block, is_final))
        if self.emit_first_speech and len(self.calls) == 1:
            start = block.start_sample
        elif is_final and self.emit_final_speech:
            start = block.start_sample + len(block.pcm) - 1
        else:
            return (), ()
        span = pipeline.SpeechSpan(start, start + 1)
        return (_speech_segment(start, np.ones(1, dtype=np.int16)),), (span,)


def _install_segmenter(
    monkeypatch: pytest.MonkeyPatch,
    outputs: tuple[object, ...],
    *,
    completed_outputs: tuple[tuple[pipeline.SpeechSpan, ...], ...] | None = None,
    events: list[str] | None = None,
) -> list[ScriptedSegmenter]:
    instances: list[ScriptedSegmenter] = []

    def factory(adapter: object) -> ScriptedSegmenter:
        instance = ScriptedSegmenter(
            adapter,
            outputs,
            completed_outputs=completed_outputs,
            events=events,
        )
        instances.append(instance)
        return instance

    monkeypatch.setattr(pipeline, "StreamingSpeechSegmenter", factory)
    return instances


def _install_final_sample_segmenter(
    monkeypatch: pytest.MonkeyPatch,
    *,
    emit_final_speech: bool,
    emit_first_speech: bool = False,
) -> list[FinalSampleSegmenter]:
    instances: list[FinalSampleSegmenter] = []

    def factory(adapter: object) -> FinalSampleSegmenter:
        instance = FinalSampleSegmenter(
            adapter,
            emit_final_speech=emit_final_speech,
            emit_first_speech=emit_first_speech,
        )
        instances.append(instance)
        return instance

    monkeypatch.setattr(pipeline, "StreamingSpeechSegmenter", factory)
    return instances


def _result(text: str) -> pipeline.AsrResult:
    return pipeline.AsrResult(text, "en", pipeline.RichAnnotations())


def _segment(
    start_sample: int,
    end_sample: int,
    *,
    pcm_start_sample: int | None = None,
) -> pipeline.BufferedSpeechSegment:
    pcm_start = start_sample if pcm_start_sample is None else pcm_start_sample
    return pipeline.BufferedSpeechSegment(
        span=pipeline.SpeechSpan(start_sample, end_sample),
        pcm_start_sample=pcm_start,
        pcm=np.arange(end_sample - pcm_start, dtype=np.int64).astype(np.int16),
    )


def _blocks(count: int) -> tuple[DecodedBlock, ...]:
    return tuple(
        DecodedBlock(
            index * BLOCK_SAMPLES,
            np.full(BLOCK_SAMPLES, index, dtype=np.int16),
        )
        for index in range(count)
    )


def _unit(*components: float) -> np.ndarray:
    embedding = np.zeros(speakers.SPEAKER_EMBEDDING_DIMENSION, dtype=np.float32)
    embedding[: len(components)] = components
    embedding /= np.linalg.norm(embedding)
    return embedding


class RecordingExactSpeakerAdapter:
    def __init__(
        self,
        *,
        invalid_output: bool = False,
        events: list[str] | None = None,
    ) -> None:
        self.calls: list[tuple[np.ndarray, ...]] = []
        self.invalid_output = invalid_output
        self.events = events

    def embed_exact_windows(
        self,
        pcms: tuple[np.ndarray, ...],
    ) -> tuple[np.ndarray, ...]:
        self.calls.append(tuple(pcm.copy() for pcm in pcms))
        if self.events is not None:
            self.events.append("cam")
        if self.invalid_output:
            return ()
        return tuple(_unit(1.0, float(index + 1)) for index, _pcm in enumerate(pcms))

    def embed_windows(
        self,
        _pcm: np.ndarray,
    ) -> tuple[speakers.SpeakerEmbeddingWindow, ...]:
        raise AssertionError("global-grid assembly must use exact windows")


def _speech_segment(start: int, values: np.ndarray) -> pipeline.BufferedSpeechSegment:
    return pipeline.BufferedSpeechSegment(
        span=pipeline.SpeechSpan(start, start + len(values)),
        pcm_start_sample=start,
        pcm=np.ascontiguousarray(values, dtype=np.int16),
    )


def _clustering_policy() -> speakers.AnonymousSpeakerClusteringPolicy:
    return speakers.AnonymousSpeakerClusteringPolicy(
        pruning_p=0.5,
        low_frequency_beta=1.0,
        normalized_gap_gamma=0.1,
    )


def _selected_snapshot(
    profile_ids: tuple[str, ...] = ("00000001",),
) -> SelectedSpeakerSnapshot:
    return SelectedSpeakerSnapshot(
        tuple(
            SelectedSpeaker(
                profile_id,
                f"Speaker {profile_id}",
                SpeakerEmbedding.from_numpy(
                    _unit(1.0, 0.0),
                    dimension=speakers.SPEAKER_EMBEDDING_DIMENSION,
                ),
            )
            for profile_id in profile_ids
        )
    )


def _options(
    *,
    model: str = "sensevoice",
    language: str = "auto",
    chunking_strategy: str | None = "auto",
    known_speaker_ids: tuple[str, ...] = (),
) -> CanonicalOptions:
    return CanonicalOptions(
        model=model,
        language=language,
        response_format=("diarized_json" if model == "sensevoice-diarize" else "json"),
        chunking_strategy=chunking_strategy,
        include=(),
        known_speaker_ids=known_speaker_ids,
    )


def _process(
    processor: pipeline.Processor,
    sink: RecordingSink,
    *,
    selected_speaker_snapshot: SelectedSpeakerSnapshot,
    language: str = "auto",
    progress: RecordingProgress | None = None,
    canonical_options: CanonicalOptions | None = None,
    effective_max_audio_samples: int = MAX_AUDIO_SAMPLES,
    effective_direct_max_audio_samples: int = DIRECT_MAX_SAMPLES,
) -> object:
    return processor.process(
        Path("/internal/input.ready"),
        canonical_options or _options(language=language),
        Cancellation(),
        progress or RecordingProgress(),
        sink,
        effective_max_audio_samples=effective_max_audio_samples,
        effective_direct_max_audio_samples=effective_direct_max_audio_samples,
        selected_speaker_snapshot=selected_speaker_snapshot,
    )


def _assert_empty_processor_result(result: object, ref: object) -> None:
    assert type(result) is pipeline.ProcessorResult
    assert result.artifact_ref is ref
    assert result.speaker_mapping == SpeakerLabelMapping(())


def test_global_speaker_windows_follow_absolute_grid_and_preserve_silence() -> None:
    adapter = RecordingExactSpeakerAdapter()
    segments = (
        _speech_segment(1_000, np.full(1_000, 1, dtype=np.int16)),
        _speech_segment(70_000, np.full(1_000, 2, dtype=np.int16)),
        _speech_segment(71_500, np.full(501, 3, dtype=np.int16)),
    )

    windows = pipeline._embed_global_speaker_windows(
        tuple(segment.span for segment in segments),
        segments,
        total_samples=72_001,
        speaker_adapter=adapter,
        post_batch_fence=lambda: None,
    )

    assert [(window.start_sample, window.end_sample) for window in windows] == [
        (0, 24_000),
        (48_000, 72_000),
        (60_000, 84_000),
        (72_000, 96_000),
    ]
    assert len(adapter.calls) == 1
    pcms = adapter.calls[0]
    assert [len(pcm) for pcm in pcms] == [24_000, 24_000, 12_001, 1]
    assert np.all(pcms[0][1_000:2_000] == 1)
    assert np.count_nonzero(pcms[0][:1_000]) == 0
    assert np.count_nonzero(pcms[0][2_000:]) == 0
    assert np.all(pcms[1][22_000:23_000] == 2)
    assert np.count_nonzero(pcms[1][23_000:23_500]) == 0
    assert np.all(pcms[1][23_500:] == 3)
    assert np.all(pcms[2][10_000:11_000] == 2)
    assert np.count_nonzero(pcms[2][11_000:11_500]) == 0
    assert np.all(pcms[2][11_500:] == 3)
    assert pcms[3].tolist() == [3]
    assert 48_001 not in tuple(window.start_sample for window in windows)


def test_global_speaker_windows_batch_at_existing_exact_window_bound() -> None:
    adapter = RecordingExactSpeakerAdapter()
    total_samples = 39 * speakers.SPEAKER_WINDOW_SHIFT_SAMPLES + 1
    fences: list[None] = []

    windows = pipeline._embed_global_speaker_windows(
        (pipeline.SpeechSpan(0, total_samples),),
        (
            _speech_segment(
                0,
                np.ones(total_samples, dtype=np.int16),
            ),
        ),
        total_samples=total_samples,
        speaker_adapter=adapter,
        post_batch_fence=lambda: fences.append(None),
    )

    assert len(windows) == 40
    assert [len(call) for call in adapter.calls] == [39, 1]
    assert fences == [None, None]
    assert windows[-1].start_sample == 468_000
    assert windows[-1].end_sample == 492_000


def test_global_speaker_windows_empty_and_invalid_adapter_output() -> None:
    empty_adapter = RecordingExactSpeakerAdapter()
    assert (
        pipeline._embed_global_speaker_windows(
            (),
            (),
            total_samples=72_001,
            speaker_adapter=empty_adapter,
            post_batch_fence=lambda: None,
        )
        == ()
    )
    assert empty_adapter.calls == []

    with pytest.raises(pipeline.PipelineError) as extra_segment:
        pipeline._embed_global_speaker_windows(
            (),
            (_speech_segment(0, np.ones(1, dtype=np.int16)),),
            total_samples=1,
            speaker_adapter=empty_adapter,
            post_batch_fence=lambda: None,
        )
    assert extra_segment.value.code == "invalid_model_output"

    with pytest.raises(pipeline.PipelineError) as source_gap:
        pipeline._embed_global_speaker_windows(
            (pipeline.SpeechSpan(0, 3),),
            (
                _speech_segment(0, np.ones(1, dtype=np.int16)),
                _speech_segment(2, np.ones(1, dtype=np.int16)),
            ),
            total_samples=3,
            speaker_adapter=empty_adapter,
            post_batch_fence=lambda: None,
        )
    assert source_gap.value.code == "invalid_model_output"

    invalid_adapter = RecordingExactSpeakerAdapter(invalid_output=True)
    with pytest.raises(pipeline.PipelineError) as caught:
        pipeline._embed_global_speaker_windows(
            (pipeline.SpeechSpan(0, 1),),
            (_speech_segment(0, np.ones(1, dtype=np.int16)),),
            total_samples=1,
            speaker_adapter=invalid_adapter,
            post_batch_fence=lambda: None,
        )

    assert caught.value.code == "invalid_model_output"


def test_global_speaker_windows_mask_chunk_spill_outside_true_island() -> None:
    adapter = RecordingExactSpeakerAdapter()

    windows = pipeline._embed_global_speaker_windows(
        (pipeline.SpeechSpan(456_000, 468_000),),
        (
            _speech_segment(
                456_000,
                np.full(24_000, 7, dtype=np.int16),
            ),
        ),
        total_samples=480_000,
        speaker_adapter=adapter,
        post_batch_fence=lambda: None,
    )

    assert [(window.start_sample, window.end_sample) for window in windows] == [
        (444_000, 468_000),
        (456_000, 480_000),
    ]
    assert len(adapter.calls) == 1
    first, second = adapter.calls[0]
    assert np.count_nonzero(first[:12_000]) == 0
    assert np.all(first[12_000:] == 7)
    assert np.all(second[:12_000] == 7)
    assert np.count_nonzero(second[12_000:]) == 0


def test_speaker_region_projection_handles_empty_and_single_window() -> None:
    assert (
        pipeline._project_speaker_regions(
            (),
            (),
            speakers.AnonymousSpeakerClusteringResult((), ()),
            total_samples=0,
        )
        == ()
    )

    island = pipeline.SpeechSpan(100, 9_000)
    assert pipeline._project_speaker_regions(
        (island,),
        (
            speakers.SpeakerEmbeddingWindow(
                0,
                speakers.SPEAKER_WINDOW_MAX_SAMPLES,
                _unit(1.0),
            ),
        ),
        speakers.AnonymousSpeakerClusteringResult(
            (0,),
            (speakers.AnonymousSpeakerCluster("A", (1.0,)),),
        ),
        total_samples=10_000,
    ) == ((island, 0),)


def test_speaker_region_projection_uses_global_midpoints_and_merges_labels() -> None:
    windows = tuple(
        speakers.SpeakerEmbeddingWindow(
            start,
            start + speakers.SPEAKER_WINDOW_MAX_SAMPLES,
            _unit(1.0),
        )
        for start in (0, 12_000, 24_000)
    )

    assert pipeline._project_speaker_regions(
        (pipeline.SpeechSpan(10_000, 40_000),),
        windows,
        speakers.AnonymousSpeakerClusteringResult(
            (0, 0, 1),
            (
                speakers.AnonymousSpeakerCluster("A", (1.0,)),
                speakers.AnonymousSpeakerCluster("B", (1.0,)),
            ),
        ),
        total_samples=40_000,
    ) == (
        (pipeline.SpeechSpan(10_000, 30_000), 0),
        (pipeline.SpeechSpan(30_000, 40_000), 1),
    )


def test_speaker_region_projection_does_not_merge_touching_islands() -> None:
    islands = (
        pipeline.SpeechSpan(0, 18_000),
        pipeline.SpeechSpan(18_000, 30_000),
    )

    assert pipeline._project_speaker_regions(
        islands,
        (
            speakers.SpeakerEmbeddingWindow(
                0,
                speakers.SPEAKER_WINDOW_MAX_SAMPLES,
                _unit(1.0),
            ),
        ),
        speakers.AnonymousSpeakerClusteringResult(
            (0,),
            (speakers.AnonymousSpeakerCluster("A", (1.0,)),),
        ),
        total_samples=30_000,
    ) == (
        (islands[0], 0),
        (islands[1], 0),
    )


def test_speaker_region_projection_splits_only_over_direct_limit() -> None:
    window = speakers.SpeakerEmbeddingWindow(
        0,
        speakers.SPEAKER_WINDOW_MAX_SAMPLES,
        _unit(1.0),
    )
    exact = pipeline.SpeechSpan(7, 7 + DIRECT_MAX_SAMPLES)
    over = pipeline.SpeechSpan(7, 8 + DIRECT_MAX_SAMPLES)

    assert pipeline._project_speaker_regions(
        (exact,),
        (window,),
        speakers.AnonymousSpeakerClusteringResult(
            (0,),
            (speakers.AnonymousSpeakerCluster("A", (1.0,)),),
        ),
        total_samples=exact.end_sample,
    ) == ((exact, 0),)
    assert pipeline._project_speaker_regions(
        (over,),
        (window,),
        speakers.AnonymousSpeakerClusteringResult(
            (0,),
            (speakers.AnonymousSpeakerCluster("A", (1.0,)),),
        ),
        total_samples=over.end_sample,
    ) == (
        (pipeline.SpeechSpan(7, 7 + DIRECT_MAX_SAMPLES), 0),
        (
            pipeline.SpeechSpan(
                7 + DIRECT_MAX_SAMPLES,
                8 + DIRECT_MAX_SAMPLES,
            ),
            0,
        ),
    )


@pytest.mark.parametrize(
    ("islands", "windows", "clustering_result", "total_samples"),
    (
        (
            (pipeline.SpeechSpan(0, 100),),
            (
                speakers.SpeakerEmbeddingWindow(
                    0,
                    speakers.SPEAKER_WINDOW_MAX_SAMPLES,
                    _unit(1.0),
                ),
            ),
            speakers.AnonymousSpeakerClusteringResult(
                (),
                (speakers.AnonymousSpeakerCluster("A", (1.0,)),),
            ),
            100,
        ),
        (
            (pipeline.SpeechSpan(0, 100),),
            (
                speakers.SpeakerEmbeddingWindow(
                    12_000,
                    36_000,
                    _unit(1.0),
                ),
            ),
            speakers.AnonymousSpeakerClusteringResult(
                (0,),
                (speakers.AnonymousSpeakerCluster("A", (1.0,)),),
            ),
            36_000,
        ),
        (
            (pipeline.SpeechSpan(0, 100),),
            (
                speakers.SpeakerEmbeddingWindow(
                    0,
                    speakers.SPEAKER_WINDOW_MAX_SAMPLES,
                    _unit(1.0),
                ),
            ),
            speakers.AnonymousSpeakerClusteringResult(
                (1,),
                (speakers.AnonymousSpeakerCluster("A", (1.0,)),),
            ),
            100,
        ),
        (
            (
                pipeline.SpeechSpan(0, 100),
                pipeline.SpeechSpan(24_000, 24_100),
            ),
            (
                speakers.SpeakerEmbeddingWindow(
                    0,
                    speakers.SPEAKER_WINDOW_MAX_SAMPLES,
                    _unit(1.0),
                ),
            ),
            speakers.AnonymousSpeakerClusteringResult(
                (0,),
                (speakers.AnonymousSpeakerCluster("A", (1.0,)),),
            ),
            24_100,
        ),
    ),
    ids=(
        "cardinality",
        "window-outside-speech",
        "ordinal-out-of-range",
        "island-without-window",
    ),
)
def test_speaker_region_projection_rejects_inconsistent_inputs(
    islands: tuple[pipeline.SpeechSpan, ...],
    windows: tuple[speakers.SpeakerEmbeddingWindow, ...],
    clustering_result: speakers.AnonymousSpeakerClusteringResult,
    total_samples: int,
) -> None:
    with pytest.raises(pipeline.PipelineError) as caught:
        pipeline._project_speaker_regions(
            islands,
            windows,
            clustering_result,
            total_samples=total_samples,
        )

    assert caught.value.code == "invalid_model_output"


def test_auto_without_vad_fails_before_probe() -> None:
    decoder = FakeDecoder(())
    frontend = FakeFrontend(decoder)
    sink = RecordingSink()

    with pytest.raises(pipeline.PipelineNotReady):
        _process(
            pipeline.Processor(
                frontend,
                FakeAsrAdapter(),
            ),
            sink,
            selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
        )

    assert frontend.probe_calls == 0
    assert frontend.decode_calls == 0
    assert decoder.closed == 0
    assert sink.finalized == 0
    assert sink.aborted == 1


def test_empty_auto_succeeds_without_vad_or_asr_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances = _install_segmenter(monkeypatch, ())
    events: list[str] = []
    decoder = FakeDecoder((), events=events)
    asr = FakeAsrAdapter()
    progress = RecordingProgress(events=events)
    sink = RecordingSink()

    result = _process(
        pipeline.Processor(
            FakeFrontend(decoder),
            asr,
            vad_adapter=object(),
        ),
        sink,
        selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
        progress=progress,
    )

    _assert_empty_processor_result(result, sink.ref)
    assert sum(len(instance.calls) for instance in instances) == 0
    assert asr.calls == []
    assert progress.updates == [(0, None), (0, 0)]
    assert events[-2:] == ["decoder_close", "progress:0:0"]
    assert decoder.closed == 1
    assert sink.finalized == 1
    assert sink.aborted == 0


def test_auto_uses_lookahead_language_and_canonical_span_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment = _segment(5_920, 24_000, pcm_start_sample=2_720)
    events: list[str] = []
    instances = _install_segmenter(
        monkeypatch,
        ((), (), (segment,)),
        events=events,
    )
    blocks = _blocks(3)
    decoder = FakeDecoder(blocks, events=events)
    asr = FakeAsrAdapter(lambda _pcms: (_result("mapped"),))
    progress = RecordingProgress(events=events)
    sink = RecordingSink()

    _process(
        pipeline.Processor(
            FakeFrontend(decoder),
            asr,
            vad_adapter=object(),
        ),
        sink,
        selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
        language="ja",
        progress=progress,
    )

    assert len(instances) == 1
    assert [is_final for _block, is_final in instances[0].calls] == [
        False,
        False,
        True,
    ]
    assert [block for block, _is_final in instances[0].calls] == list(blocks)
    assert len(asr.calls) == 1
    assert asr.calls[0][1] == "ja"
    np.testing.assert_array_equal(asr.calls[0][0][0], segment.pcm)
    assert sink.records == [
        pipeline.SegmentRecord(
            index=0,
            start_sample=5_920,
            end_sample=24_000,
            text="mapped",
            language="en",
            annotations=pipeline.RichAnnotations(),
        )
    ]
    assert progress.updates == [
        (0, None),
        (BLOCK_SAMPLES, None),
        (BLOCK_SAMPLES, None),
        (2 * BLOCK_SAMPLES, None),
        (2 * BLOCK_SAMPLES, None),
        (3 * BLOCK_SAMPLES, None),
        (3 * BLOCK_SAMPLES, None),
        (3 * BLOCK_SAMPLES, 3 * BLOCK_SAMPLES),
    ]
    fsmn_positions = [
        index for index, event in enumerate(events) if event == "fsmn"
    ]
    assert [events[index + 1] for index in fsmn_positions] == [
        "progress:0:None",
        f"progress:{BLOCK_SAMPLES}:None",
        f"progress:{2 * BLOCK_SAMPLES}:None",
    ]
    assert events[-2:] == [
        "decoder_close",
        f"progress:{3 * BLOCK_SAMPLES}:{3 * BLOCK_SAMPLES}",
    ]


def test_auto_post_fsmn_progress_fence_precedes_result_interpretation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances = _install_segmenter(monkeypatch, (object(),))
    asr = FakeAsrAdapter()
    progress = RecordingProgress(fail_on_call=1)
    sink = RecordingSink()

    with pytest.raises(
        ProgressFenceStop,
        match="stopped at durable progress fence",
    ):
        _process(
            pipeline.Processor(
                FakeFrontend(FakeDecoder(_blocks(1))),
                asr,
                vad_adapter=object(),
            ),
            sink,
            selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
            progress=progress,
        )

    assert len(instances) == 1
    assert len(instances[0].calls) == 1
    assert progress.updates == [(0, None)]
    assert asr.calls == []
    assert sink.records == []
    assert sink.finalized == 0
    assert sink.aborted == 1


@pytest.mark.parametrize(
    (
        "overall_cap",
        "blocks",
        "expected_processed",
        "expected_fsmn_calls",
    ),
    (
        (
            BLOCK_SAMPLES - 1,
            _blocks(1),
            (),
            0,
        ),
        (
            BLOCK_SAMPLES,
            (
                *_blocks(1),
                DecodedBlock(BLOCK_SAMPLES, np.ones(1, dtype=np.int16)),
            ),
            (),
            0,
        ),
    ),
    ids=("first_block_exceeds", "first_sample_after_cap"),
)
def test_auto_stops_before_processing_audio_beyond_the_effective_overall_cap(
    monkeypatch: pytest.MonkeyPatch,
    overall_cap: int,
    blocks: tuple[DecodedBlock, ...],
    expected_processed: tuple[tuple[int, int | None], ...],
    expected_fsmn_calls: int,
) -> None:
    instances = _install_segmenter(monkeypatch, ((), ()))
    decoder = FakeDecoder(blocks)
    asr = FakeAsrAdapter()
    progress = RecordingProgress()
    sink = RecordingSink()

    with pytest.raises(pipeline.PipelineError) as caught:
        _process(
            pipeline.Processor(
                FakeFrontend(decoder),
                asr,
                vad_adapter=object(),
            ),
            sink,
            selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
            progress=progress,
            effective_max_audio_samples=overall_cap,
            effective_direct_max_audio_samples=min(
                overall_cap,
                DIRECT_MAX_SAMPLES,
            ),
        )

    assert caught.value.code == "audio_too_long"
    assert len(instances) == 1
    assert len(instances[0].calls) == expected_fsmn_calls
    assert progress.updates == list(expected_processed)
    assert asr.calls == []
    assert decoder.closed == 1
    assert sink.finalized == 0
    assert sink.aborted == 1


@pytest.mark.parametrize(
    ("segments", "expected_batch_sizes"),
    (
        (
            tuple(_segment(index, index + 1) for index in range(33)),
            [32, 1],
        ),
        (
            (
                _segment(0, 480_000),
                _segment(480_000, 960_000),
                _segment(960_000, 960_001),
            ),
            [2, 1],
        ),
        (
            (
                _segment(0, 1),
                _segment(4_799_999, 4_800_000),
                _segment(4_800_000, 4_800_001),
            ),
            [2, 1],
        ),
    ),
    ids=("32_segments", "960000_pcm", "4800000_wall"),
)
def test_auto_preflushes_before_a_candidate_exceeds_each_inclusive_batch_cap(
    monkeypatch: pytest.MonkeyPatch,
    segments: tuple[pipeline.BufferedSpeechSegment, ...],
    expected_batch_sizes: list[int],
) -> None:
    _install_segmenter(monkeypatch, (segments,))
    events: list[str] = []
    asr = FakeAsrAdapter(events=events)
    progress = RecordingProgress(events=events)
    sink = RecordingSink(events)

    _process(
        pipeline.Processor(
            FakeFrontend(FakeDecoder(_blocks(1))),
            asr,
            vad_adapter=object(),
        ),
        sink,
        selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
        progress=progress,
    )

    assert [len(pcms) for pcms, _language in asr.calls] == expected_batch_sizes
    assert all(language == "auto" for _pcms, language in asr.calls)
    asr_positions = [
        index for index, event in enumerate(events) if event == "asr"
    ]
    assert len(asr_positions) == len(expected_batch_sizes)
    assert all(
        events[index + 1].startswith("progress:")
        for index in asr_positions
    )
    assert len(sink.records) == len(segments)
    assert sink.finalized == 1
    assert sink.aborted == 0


@pytest.mark.parametrize(
    "invalid_result",
    (
        [_result("first"), _result("second")],
        (_result("wrong cardinality"),),
        (_result("first"), object()),
    ),
    ids=("list", "cardinality", "element_type"),
)
def test_auto_rejects_invalid_batch_atomically_and_aborts(
    monkeypatch: pytest.MonkeyPatch,
    invalid_result: object,
) -> None:
    segments = (_segment(100, 200), _segment(300, 400))
    _install_segmenter(monkeypatch, (segments,))
    decoder = FakeDecoder(_blocks(1))
    asr = FakeAsrAdapter(lambda _pcms: invalid_result)
    sink = RecordingSink()

    with pytest.raises(pipeline.PipelineError) as caught:
        _process(
            pipeline.Processor(
                FakeFrontend(decoder),
                asr,
                vad_adapter=object(),
            ),
            sink,
            selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
        )

    assert caught.value.code == "invalid_model_output"
    assert sink.records == []
    assert sink.finalized == 0
    assert sink.aborted == 1
    assert decoder.closed == 1


def test_auto_post_inference_progress_fence_precedes_result_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment = _segment(100, 200, pcm_start_sample=90)
    _install_segmenter(monkeypatch, ((segment,),))
    progress = RecordingProgress(fail_on_call=3)
    sink = RecordingSink()
    asr = FakeAsrAdapter(lambda _pcms: object())

    with pytest.raises(
        ProgressFenceStop,
        match="stopped at durable progress fence",
    ):
        _process(
            pipeline.Processor(
                FakeFrontend(FakeDecoder(_blocks(1))),
                asr,
                vad_adapter=object(),
            ),
            sink,
            selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
            progress=progress,
        )

    assert progress.updates == [
        (0, None),
        (BLOCK_SAMPLES, None),
        (BLOCK_SAMPLES, None),
    ]
    assert len(asr.calls) == 1
    assert sink.records == []
    assert sink.finalized == 0
    assert sink.aborted == 1


def test_auto_skips_empty_results_positionally_with_dense_record_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segments = (
        _segment(100, 200),
        _segment(300, 400),
        _segment(500, 600),
    )
    _install_segmenter(monkeypatch, (segments,))
    results = (
        pipeline.AsrResult(
            "first",
            "zh",
            pipeline.RichAnnotations("happy", None),
        ),
        _result(""),
        pipeline.AsrResult(
            "third",
            "ko",
            pipeline.RichAnnotations(None, "speech"),
        ),
    )
    decoder = FakeDecoder(_blocks(1))
    asr = FakeAsrAdapter(lambda _pcms: results)
    sink = RecordingSink()

    result = _process(
        pipeline.Processor(
            FakeFrontend(decoder),
            asr,
            vad_adapter=object(),
        ),
        sink,
        selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
        language="zh",
    )

    _assert_empty_processor_result(result, sink.ref)
    assert asr.calls[0][1] == "zh"
    assert [
        (record.index, record.start_sample, record.end_sample, record.text)
        for record in sink.records
    ] == [
        (0, 100, 200, "first"),
        (1, 500, 600, "third"),
    ]
    assert decoder.closed == 1
    assert sink.finalized == 1
    assert sink.aborted == 0


def test_processor_constructor_rejects_removed_online_speaker_dependencies() -> None:
    parameters = signature(pipeline.Processor).parameters
    assert "speaker_adapter" in parameters
    assert "speaker_clustering_policy" in parameters
    assert "known_speaker_match_policy" in parameters
    assert "speaker_policy" not in parameters
    assert "known_speaker_policy" not in parameters


def test_speaker_embedding_adapter_is_a_structural_window_protocol() -> None:
    adapter_protocol = speakers.SpeakerEmbeddingAdapter

    assert getattr(adapter_protocol, "_is_protocol", False)
    assert callable(adapter_protocol.embed_windows)
    assert callable(adapter_protocol.embed_exact_windows)


def test_diarize_readiness_fails_before_probe() -> None:
    decoder = FakeDecoder(_blocks(1))
    frontend = FakeFrontend(decoder)
    sink = RecordingSink()

    with pytest.raises(pipeline.PipelineNotReady):
        _process(
            pipeline.Processor(
                frontend,
                FakeAsrAdapter(),
                vad_adapter=object(),
            ),
            sink,
            selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
            canonical_options=_options(model="sensevoice-diarize"),
        )

    assert frontend.probe_calls == 0
    assert frontend.decode_calls == 0
    assert decoder.closed == 0
    assert sink.records == []
    assert sink.finalized == 0
    assert sink.aborted == 1


def test_known_diarize_without_match_policy_is_not_ready_before_probe() -> None:
    decoder = FakeDecoder(_blocks(1))
    frontend = FakeFrontend(decoder)
    sink = RecordingSink()

    with pytest.raises(pipeline.PipelineNotReady):
        _process(
            pipeline.Processor(
                frontend,
                FakeAsrAdapter(),
                vad_adapter=object(),
                speaker_adapter=RecordingExactSpeakerAdapter(),
                speaker_clustering_policy=_clustering_policy(),
            ),
            sink,
            selected_speaker_snapshot=_selected_snapshot(),
            canonical_options=_options(
                model="sensevoice-diarize",
                known_speaker_ids=("00000001",),
            ),
        )

    assert frontend.probe_calls == 0
    assert frontend.decode_calls == 0
    assert decoder.closed == 0
    assert sink.finalized == 0
    assert sink.aborted == 1


def test_known_diarize_returns_mapping_without_changing_anonymous_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    island = pipeline.SpeechSpan(0, BLOCK_SAMPLES)
    source = _speech_segment(0, np.ones(BLOCK_SAMPLES, dtype=np.int16))
    _install_segmenter(
        monkeypatch,
        ((source,),),
        completed_outputs=((island,),),
    )
    clusters = (
        speakers.AnonymousSpeakerCluster(
            "A",
            tuple(float(value) for value in _unit(1.0, 0.0)),
        ),
        speakers.AnonymousSpeakerCluster(
            "B",
            tuple(float(value) for value in _unit(0.0, 1.0)),
        ),
    )
    clustering_result = speakers.AnonymousSpeakerClusteringResult(
        (0, 1),
        clusters,
    )
    monkeypatch.setattr(
        pipeline,
        "cluster_anonymous_speakers",
        lambda *_args, **_kwargs: clustering_result,
    )
    monkeypatch.setattr(
        pipeline,
        "_project_speaker_regions",
        lambda *_args, **_kwargs: (
            (pipeline.SpeechSpan(0, BLOCK_SAMPLES // 2), 0),
            (pipeline.SpeechSpan(BLOCK_SAMPLES // 2, BLOCK_SAMPLES), 1),
        ),
    )
    sink = RecordingSink()
    snapshot = _selected_snapshot()

    result = _process(
        pipeline.Processor(
            FakeFrontend(FakeDecoder(_blocks(1))),
            FakeAsrAdapter(),
            vad_adapter=object(),
            speaker_adapter=RecordingExactSpeakerAdapter(),
            speaker_clustering_policy=_clustering_policy(),
            known_speaker_match_policy=KnownSpeakerMatchPolicy(0.31),
        ),
        sink,
        selected_speaker_snapshot=snapshot,
        canonical_options=_options(
            model="sensevoice-diarize",
            known_speaker_ids=("00000001",),
        ),
    )

    assert [record.anonymous_speaker for record in sink.records] == ["A", "B"]
    assert result.speaker_mapping == SpeakerLabelMapping(
        (
            SpeakerLabelResolution(
                "A",
                KnownSpeakerMatch(
                    "00000001",
                    "Speaker 00000001",
                    1.0,
                ),
            ),
            SpeakerLabelResolution("B", None),
        )
    )
    assert sink.finalized == 1
    assert sink.aborted == 0


def test_known_match_input_error_aborts_as_invalid_model_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    island = pipeline.SpeechSpan(0, BLOCK_SAMPLES)
    source = _speech_segment(0, np.ones(BLOCK_SAMPLES, dtype=np.int16))
    _install_segmenter(
        monkeypatch,
        ((source,),),
        completed_outputs=((island,),),
    )
    clustering_result = speakers.AnonymousSpeakerClusteringResult(
        (0, 0),
        (
            speakers.AnonymousSpeakerCluster(
                "A",
                tuple(float(value) for value in _unit(1.0, 0.0)),
            ),
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "cluster_anonymous_speakers",
        lambda *_args, **_kwargs: clustering_result,
    )
    monkeypatch.setattr(
        pipeline,
        "_project_speaker_regions",
        lambda *_args, **_kwargs: ((island, 0),),
    )
    monkeypatch.setattr(
        pipeline,
        "match_selected_speakers",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            SpeakerMatchInputError()
        ),
        raising=False,
    )
    sink = RecordingSink()

    with pytest.raises(pipeline.PipelineError) as caught:
        _process(
            pipeline.Processor(
                FakeFrontend(FakeDecoder(_blocks(1))),
                FakeAsrAdapter(),
                vad_adapter=object(),
                speaker_adapter=RecordingExactSpeakerAdapter(),
                speaker_clustering_policy=_clustering_policy(),
                known_speaker_match_policy=KnownSpeakerMatchPolicy(0.31),
            ),
            sink,
            selected_speaker_snapshot=_selected_snapshot(),
            canonical_options=_options(
                model="sensevoice-diarize",
                known_speaker_ids=("00000001",),
            ),
        )

    assert caught.value.code == "invalid_model_output"
    assert sink.finalized == 0
    assert sink.aborted == 1


def test_diarize_runs_offline_after_decoder_close_with_bounded_cam_and_asr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    total_samples = 480_001
    split = 480_000
    island_start = 3_200
    source_segments = (
        pipeline.BufferedSpeechSegment(
            span=pipeline.SpeechSpan(island_start, split),
            pcm_start_sample=0,
            pcm=np.concatenate(
                [
                    np.zeros(island_start, dtype=np.int16),
                    np.ones(split - island_start, dtype=np.int16),
                ]
            ),
        ),
        _speech_segment(
            split,
            np.full(total_samples - split, 2, dtype=np.int16),
        ),
    )
    island = pipeline.SpeechSpan(island_start, total_samples)
    events: list[str] = []
    decoded_blocks = tuple(
        DecodedBlock(
            start,
            np.zeros(
                min(BLOCK_SAMPLES, total_samples - start),
                dtype=np.int16,
            ),
        )
        for start in range(0, total_samples, BLOCK_SAMPLES)
    )
    _install_segmenter(
        monkeypatch,
        ((),) * (len(decoded_blocks) - 2)
        + ((source_segments[0],), (source_segments[1],)),
        completed_outputs=((),) * (len(decoded_blocks) - 1) + ((island,),),
        events=events,
    )
    decoder = FakeDecoder(decoded_blocks, events=events)
    speaker_adapter = RecordingExactSpeakerAdapter(events=events)
    clustered_shapes: list[tuple[int, ...]] = []

    def cluster(
        embeddings: np.ndarray,
        *,
        policy: speakers.AnonymousSpeakerClusteringPolicy,
    ) -> speakers.AnonymousSpeakerClusteringResult:
        assert policy == _clustering_policy()
        assert embeddings.dtype == np.float32
        assert embeddings.flags.c_contiguous
        clustered_shapes.append(embeddings.shape)
        events.append("cluster")
        return speakers.AnonymousSpeakerClusteringResult(
            (0,) * len(embeddings),
            (speakers.AnonymousSpeakerCluster("A", (1.0,)),),
        )

    monkeypatch.setattr(
        pipeline,
        "cluster_anonymous_speakers",
        cluster,
        raising=False,
    )
    project_speaker_regions = pipeline._project_speaker_regions

    def project(
        speech_islands: tuple[pipeline.SpeechSpan, ...],
        windows: tuple[speakers.SpeakerEmbeddingWindow, ...],
        clustering_result: speakers.AnonymousSpeakerClusteringResult,
        *,
        total_samples: int,
    ) -> tuple[tuple[pipeline.SpeechSpan, int], ...]:
        events.append("project")
        return project_speaker_regions(
            speech_islands,
            windows,
            clustering_result,
            total_samples=total_samples,
        )

    monkeypatch.setattr(pipeline, "_project_speaker_regions", project)
    asr = FakeAsrAdapter(events=events)
    progress = RecordingProgress(events=events)
    sink = RecordingSink(events)

    result = _process(
        pipeline.Processor(
            FakeFrontend(decoder),
            asr,
            vad_adapter=object(),
            speaker_adapter=speaker_adapter,
            speaker_clustering_policy=_clustering_policy(),
        ),
        sink,
        selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
        canonical_options=_options(model="sensevoice-diarize"),
        progress=progress,
    )

    _assert_empty_processor_result(result, sink.ref)
    assert decoder.closed == 1
    assert [len(call) for call in speaker_adapter.calls] == [39, 2]
    assert clustered_shapes == [
        (41, speakers.SPEAKER_EMBEDDING_DIMENSION),
    ]
    assert len(asr.calls) == 1
    np.testing.assert_array_equal(
        asr.calls[0][0][0],
        np.concatenate(
            [
                np.ones(split - island_start, dtype=np.int16),
                np.full(total_samples - split, 2, dtype=np.int16),
            ]
        ),
    )
    assert sink.records == [
        pipeline.SegmentRecord(
            index=0,
            start_sample=island_start,
            end_sample=total_samples,
            text="text-0",
            language="en",
            annotations=pipeline.RichAnnotations(),
            anonymous_speaker="A",
        )
    ]
    assert events.index("decoder_close") < events.index("cam")
    cam_positions = [index for index, event in enumerate(events) if event == "cam"]
    assert len(cam_positions) == 2
    assert all(
        events[index + 1] == f"progress:{total_samples}:None"
        for index in cam_positions
    )
    assert max(
        index for index, event in enumerate(events) if event == "fsmn"
    ) < events.index("decoder_close")
    assert (
        cam_positions[-1]
        < events.index("cluster")
        < events.index("project")
        < events.index("asr")
    )
    asr_position = events.index("asr")
    assert events[asr_position + 1] == f"progress:{total_samples}:None"
    assert events[asr_position + 2] == "append"


def test_empty_diarize_skips_offline_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_segmenter(monkeypatch, ((),))
    decoder = FakeDecoder(_blocks(1))
    speaker_adapter = RecordingExactSpeakerAdapter()
    asr = FakeAsrAdapter()
    monkeypatch.setattr(
        pipeline,
        "cluster_anonymous_speakers",
        lambda *_args, **_kwargs: pytest.fail("empty speech must not cluster"),
        raising=False,
    )
    sink = RecordingSink()

    result = _process(
        pipeline.Processor(
            FakeFrontend(decoder),
            asr,
            vad_adapter=object(),
            speaker_adapter=speaker_adapter,
            speaker_clustering_policy=_clustering_policy(),
        ),
        sink,
        selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
        canonical_options=_options(model="sensevoice-diarize"),
    )

    _assert_empty_processor_result(result, sink.ref)
    assert decoder.closed == 1
    assert speaker_adapter.calls == []
    assert asr.calls == []
    assert sink.records == []
    assert sink.finalized == 1
    assert sink.aborted == 0


def test_diarize_actual_cap_is_inclusive_and_enters_offline_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_blocks = DIARIZATION_MAX_AUDIO_SAMPLES // BLOCK_SAMPLES
    decoder = SharedZeroBoundaryDecoder(full_blocks=full_blocks)
    instances = _install_final_sample_segmenter(
        monkeypatch,
        emit_final_speech=True,
    )
    speaker_adapter = RecordingExactSpeakerAdapter()
    asr = FakeAsrAdapter()
    sink = RecordingSink()

    result = _process(
        pipeline.Processor(
            FakeFrontend(decoder),
            asr,
            vad_adapter=object(),
            speaker_adapter=speaker_adapter,
            speaker_clustering_policy=_clustering_policy(),
        ),
        sink,
        selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
        canonical_options=_options(model="sensevoice-diarize"),
    )

    _assert_empty_processor_result(result, sink.ref)
    assert len(instances) == 1
    assert len(instances[0].calls) == full_blocks
    assert instances[0].calls[-1][1] is True
    assert decoder.yielded == full_blocks
    assert decoder.closed == 1
    assert speaker_adapter.calls
    assert asr.calls
    assert sink.records[0].end_sample == DIARIZATION_MAX_AUDIO_SAMPLES


def test_diarize_stops_at_first_sample_past_actual_cap_without_lookahead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_blocks = DIARIZATION_MAX_AUDIO_SAMPLES // BLOCK_SAMPLES
    decoder = SharedZeroBoundaryDecoder(
        full_blocks=full_blocks,
        tail_samples=1,
        guard_after_tail=True,
    )
    instances = _install_final_sample_segmenter(
        monkeypatch,
        emit_final_speech=True,
        emit_first_speech=True,
    )
    speaker_adapter = RecordingExactSpeakerAdapter()
    asr = FakeAsrAdapter()
    progress = RecordingProgress()
    sink = RecordingSink()

    with pytest.raises(pipeline.PipelineError) as caught:
        _process(
            pipeline.Processor(
                FakeFrontend(decoder),
                asr,
                vad_adapter=object(),
                speaker_adapter=speaker_adapter,
                speaker_clustering_policy=_clustering_policy(),
            ),
            sink,
            selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
            canonical_options=_options(model="sensevoice-diarize"),
            progress=progress,
        )

    assert caught.value.code == "diarization_too_long"
    assert len(instances) == 1
    assert len(instances[0].calls) == full_blocks - 1
    assert (
        instances[0].calls[-1][0].start_sample
        == (full_blocks - 2) * BLOCK_SAMPLES
    )
    assert decoder.yielded == full_blocks + 1
    assert decoder.guard_requests == 0
    assert decoder.closed == 1
    assert speaker_adapter.calls == []
    assert asr.calls == []
    assert sink.records == []
    assert sink.finalized == 0
    assert sink.aborted == 1
    assert progress.updates
    assert all(total is None for _processed, total in progress.updates)


def test_ordinary_vad_does_not_apply_diarization_actual_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_blocks = DIARIZATION_MAX_AUDIO_SAMPLES // BLOCK_SAMPLES
    decoder = SharedZeroBoundaryDecoder(
        full_blocks=full_blocks,
        tail_samples=1,
    )
    instances = _install_final_sample_segmenter(
        monkeypatch,
        emit_final_speech=False,
    )
    sink = RecordingSink()

    result = _process(
        pipeline.Processor(
            FakeFrontend(decoder),
            FakeAsrAdapter(),
            vad_adapter=object(),
        ),
        sink,
        selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
    )

    _assert_empty_processor_result(result, sink.ref)
    assert len(instances[0].calls) == full_blocks + 1
    assert instances[0].calls[-1][1] is True
    assert decoder.closed == 1
    assert sink.finalized == 1
    assert sink.aborted == 0


@pytest.mark.parametrize(
    "snapshot",
    (
        _selected_snapshot(("00000002", "00000001")),
        _selected_snapshot(("00000001", "00000003")),
    ),
    ids=("order_mismatch", "set_mismatch"),
)
def test_known_snapshot_ids_must_exactly_match_options_before_probe(
    snapshot: SelectedSpeakerSnapshot,
) -> None:
    decoder = FakeDecoder(_blocks(1))
    frontend = FakeFrontend(decoder)
    sink = RecordingSink()

    with pytest.raises(RuntimeError):
        _process(
            pipeline.Processor(
                frontend,
                FakeAsrAdapter(),
                vad_adapter=object(),
            ),
            sink,
            selected_speaker_snapshot=snapshot,
            canonical_options=_options(
                model="sensevoice-diarize",
                known_speaker_ids=("00000001", "00000002"),
            ),
        )

    assert frontend.probe_calls == 0
    assert frontend.decode_calls == 0
    assert decoder.closed == 0
    assert sink.finalized == 0
    assert sink.aborted == 1


def test_known_options_reject_empty_snapshot_before_probe() -> None:
    decoder = FakeDecoder(_blocks(1))
    frontend = FakeFrontend(decoder)
    sink = RecordingSink()

    with pytest.raises(RuntimeError):
        _process(
            pipeline.Processor(
                frontend,
                FakeAsrAdapter(),
                vad_adapter=object(),
            ),
            sink,
            selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
            canonical_options=_options(
                model="sensevoice-diarize",
                known_speaker_ids=("00000001",),
            ),
        )

    assert frontend.probe_calls == 0
    assert frontend.decode_calls == 0
    assert decoder.closed == 0
    assert sink.finalized == 0
    assert sink.aborted == 1


def test_anonymous_options_reject_nonempty_snapshot_before_probe() -> None:
    decoder = FakeDecoder(_blocks(1))
    frontend = FakeFrontend(decoder)
    sink = RecordingSink()

    with pytest.raises(RuntimeError):
        _process(
            pipeline.Processor(
                frontend,
                FakeAsrAdapter(),
                vad_adapter=object(),
            ),
            sink,
            selected_speaker_snapshot=_selected_snapshot(),
            canonical_options=_options(model="sensevoice-diarize"),
        )

    assert frontend.probe_calls == 0
    assert frontend.decode_calls == 0
    assert decoder.closed == 0
    assert sink.finalized == 0
    assert sink.aborted == 1
