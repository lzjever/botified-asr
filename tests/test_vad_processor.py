from __future__ import annotations

from collections.abc import Callable
from inspect import signature
from pathlib import Path

import numpy as np
import pytest

from botified_asr import pipeline, speakers
from botified_asr.audio import BLOCK_SAMPLES, Cancellation, DecodedBlock, MediaProbe
from botified_asr.contracts import (
    DIRECT_MAX_SAMPLES,
    MAX_AUDIO_SAMPLES,
    CanonicalOptions,
)
from botified_asr.speaker_profiles import SpeakerEmbedding
from botified_asr.speaker_matching import SpeakerLabelMapping
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
        events: list[str] | None = None,
    ) -> None:
        self.adapter = adapter
        self.outputs = outputs
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
        return output


def _install_segmenter(
    monkeypatch: pytest.MonkeyPatch,
    outputs: tuple[object, ...],
    *,
    events: list[str] | None = None,
) -> list[ScriptedSegmenter]:
    instances: list[ScriptedSegmenter] = []

    def factory(adapter: object) -> ScriptedSegmenter:
        instance = ScriptedSegmenter(
            adapter,
            outputs,
            events=events,
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
    def __init__(self, *, invalid_output: bool = False) -> None:
        self.calls: list[tuple[np.ndarray, ...]] = []
        self.invalid_output = invalid_output

    def embed_exact_windows(
        self,
        pcms: tuple[np.ndarray, ...],
    ) -> tuple[np.ndarray, ...]:
        self.calls.append(tuple(pcm.copy() for pcm in pcms))
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
        segments,
        total_samples=72_001,
        speaker_adapter=adapter,
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

    windows = pipeline._embed_global_speaker_windows(
        (
            _speech_segment(
                0,
                np.ones(total_samples, dtype=np.int16),
            ),
        ),
        total_samples=total_samples,
        speaker_adapter=adapter,
    )

    assert len(windows) == 40
    assert [len(call) for call in adapter.calls] == [39, 1]
    assert windows[-1].start_sample == 468_000
    assert windows[-1].end_sample == 492_000


def test_global_speaker_windows_empty_and_invalid_adapter_output() -> None:
    empty_adapter = RecordingExactSpeakerAdapter()
    assert (
        pipeline._embed_global_speaker_windows(
            (),
            total_samples=72_001,
            speaker_adapter=empty_adapter,
        )
        == ()
    )
    assert empty_adapter.calls == []

    invalid_adapter = RecordingExactSpeakerAdapter(invalid_output=True)
    with pytest.raises(pipeline.PipelineError) as caught:
        pipeline._embed_global_speaker_windows(
            (_speech_segment(0, np.ones(1, dtype=np.int16)),),
            total_samples=1,
            speaker_adapter=invalid_adapter,
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
            (
                (0, None),
                (BLOCK_SAMPLES, None),
            ),
            1,
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
    assert "speaker_adapter" not in parameters
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
