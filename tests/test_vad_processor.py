from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from botified_asr import pipeline
from botified_asr.audio import BLOCK_SAMPLES, Cancellation, DecodedBlock, MediaProbe
from botified_asr.contracts import CanonicalOptions


class FakeDecoder:
    def __init__(self, blocks: tuple[DecodedBlock, ...]) -> None:
        self.blocks = blocks
        self.closed = 0

    def __iter__(self):
        yield from self.blocks

    def close(self) -> None:
        self.closed += 1


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
    ) -> None:
        self._result_factory = result_factory or (
            lambda pcms: tuple(
                _result(f"text-{index}") for index, _pcm in enumerate(pcms)
            )
        )
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
        return self._result_factory(pcms)


class RecordingProgress:
    def __init__(self) -> None:
        self.updates: list[tuple[int, int | None]] = []

    def update(self, *, processed_samples: int, total_samples: int | None) -> None:
        self.updates.append((processed_samples, total_samples))


class RecordingSink:
    def __init__(self) -> None:
        self.records: list[pipeline.SegmentRecord] = []
        self.finalized = 0
        self.aborted = 0
        self.ref = object()

    def append(self, record: pipeline.SegmentRecord) -> None:
        self.records.append(record)

    def finalize(self) -> object:
        self.finalized += 1
        return self.ref

    def abort(self) -> None:
        self.aborted += 1


class ScriptedSegmenter:
    def __init__(
        self,
        adapter: object,
        outputs: tuple[tuple[pipeline.BufferedSpeechSegment, ...], ...],
    ) -> None:
        self.adapter = adapter
        self.outputs = outputs
        self.calls: list[tuple[DecodedBlock, bool]] = []

    def process(
        self,
        block: DecodedBlock,
        *,
        is_final: bool,
    ) -> tuple[pipeline.BufferedSpeechSegment, ...]:
        output = self.outputs[len(self.calls)]
        self.calls.append((block, is_final))
        return output


def _install_segmenter(
    monkeypatch: pytest.MonkeyPatch,
    outputs: tuple[tuple[pipeline.BufferedSpeechSegment, ...], ...],
) -> list[ScriptedSegmenter]:
    instances: list[ScriptedSegmenter] = []

    def factory(adapter: object) -> ScriptedSegmenter:
        instance = ScriptedSegmenter(adapter, outputs)
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


def _options(*, language: str = "auto") -> CanonicalOptions:
    return CanonicalOptions(
        model="sensevoice",
        language=language,
        response_format="json",
        chunking_strategy="auto",
        include=(),
        known_speaker_ids=(),
    )


def _process(
    processor: pipeline.Processor,
    sink: RecordingSink,
    *,
    language: str = "auto",
    progress: RecordingProgress | None = None,
) -> object:
    return processor.process(
        Path("/internal/input.ready"),
        _options(language=language),
        Cancellation(),
        progress or RecordingProgress(),
        sink,
    )


def test_auto_without_vad_fails_before_probe() -> None:
    decoder = FakeDecoder(())
    frontend = FakeFrontend(decoder)
    sink = RecordingSink()

    with pytest.raises(pipeline.PipelineNotReady):
        _process(pipeline.Processor(frontend, FakeAsrAdapter()), sink)

    assert frontend.probe_calls == 0
    assert frontend.decode_calls == 0
    assert decoder.closed == 0
    assert sink.finalized == 0
    assert sink.aborted == 1


def test_empty_auto_succeeds_without_vad_or_asr_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances = _install_segmenter(monkeypatch, ())
    decoder = FakeDecoder(())
    asr = FakeAsrAdapter()
    progress = RecordingProgress()
    sink = RecordingSink()

    artifact = _process(
        pipeline.Processor(
            FakeFrontend(decoder),
            asr,
            vad_adapter=object(),
        ),
        sink,
        progress=progress,
    )

    assert artifact is sink.ref
    assert sum(len(instance.calls) for instance in instances) == 0
    assert asr.calls == []
    assert progress.updates == [(0, None)]
    assert decoder.closed == 1
    assert sink.finalized == 1
    assert sink.aborted == 0


def test_auto_uses_lookahead_language_and_canonical_span_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment = _segment(5_920, 24_000, pcm_start_sample=2_720)
    instances = _install_segmenter(monkeypatch, ((), (), (segment,)))
    blocks = _blocks(3)
    decoder = FakeDecoder(blocks)
    asr = FakeAsrAdapter(lambda _pcms: (_result("mapped"),))
    progress = RecordingProgress()
    sink = RecordingSink()

    _process(
        pipeline.Processor(
            FakeFrontend(decoder),
            asr,
            vad_adapter=object(),
        ),
        sink,
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
        (BLOCK_SAMPLES, None),
        (2 * BLOCK_SAMPLES, None),
        (3 * BLOCK_SAMPLES, None),
    ]


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
    asr = FakeAsrAdapter()
    sink = RecordingSink()

    _process(
        pipeline.Processor(
            FakeFrontend(FakeDecoder(_blocks(1))),
            asr,
            vad_adapter=object(),
        ),
        sink,
    )

    assert [len(pcms) for pcms, _language in asr.calls] == expected_batch_sizes
    assert all(language == "auto" for _pcms, language in asr.calls)
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
        )

    assert caught.value.code == "invalid_model_output"
    assert sink.records == []
    assert sink.finalized == 0
    assert sink.aborted == 1
    assert decoder.closed == 1


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

    artifact = _process(
        pipeline.Processor(
            FakeFrontend(decoder),
            asr,
            vad_adapter=object(),
        ),
        sink,
        language="zh",
    )

    assert artifact is sink.ref
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
