from __future__ import annotations

import numpy as np
import pytest

from botified_asr import pipeline
from botified_asr.audio import BLOCK_SAMPLES, DecodedBlock


Marker = tuple[int | None, int | None]


class FakeStreamingVadAdapter:
    def __init__(self, outputs: tuple[tuple[Marker, ...], ...]) -> None:
        self._outputs = outputs
        self._cache: dict[str, object] | None = None
        self.calls = 0
        self.is_final_values: list[bool] = []

    def generate(
        self,
        pcm: np.ndarray,
        *,
        cache: dict[str, object],
        is_final: bool,
    ) -> tuple[pipeline.VadMarker, ...]:
        assert pcm.dtype == np.int16
        assert pcm.ndim == 1
        assert pcm.flags.c_contiguous
        if self._cache is None:
            self._cache = cache
        else:
            assert cache is self._cache
        output = self._outputs[self.calls]
        self.calls += 1
        self.is_final_values.append(is_final)
        return tuple(
            pipeline.VadMarker(start_ms=start_ms, end_ms=end_ms)
            for start_ms, end_ms in output
        )


def _blocks(source: np.ndarray) -> tuple[DecodedBlock, ...]:
    return tuple(
        DecodedBlock(
            start_sample=start,
            pcm=source[start : start + BLOCK_SAMPLES],
        )
        for start in range(0, len(source), BLOCK_SAMPLES)
    )


def _segmenter(adapter: FakeStreamingVadAdapter) -> object:
    return pipeline.StreamingSpeechSegmenter(adapter)  # type: ignore[attr-defined, no-any-return]


def _process(
    segmenter: object,
    block: DecodedBlock,
    *,
    is_final: bool,
) -> tuple[pipeline.BufferedSpeechSegment, ...]:
    return segmenter.process(block, is_final=is_final)  # type: ignore[attr-defined, no-any-return]


def test_normal_span_preserves_first_prepad_and_pcm_identity() -> None:
    source = np.arange(24_000, dtype=np.int32).astype(np.int16)
    blocks = _blocks(source)
    adapter = FakeStreamingVadAdapter(
        (
            (),
            ((370, None),),
            ((None, 1_500),),
        )
    )
    segmenter = _segmenter(adapter)

    assert _process(segmenter, blocks[0], is_final=False) == ()
    assert _process(segmenter, blocks[1], is_final=False) == ()
    emitted = _process(segmenter, blocks[2], is_final=True)

    assert len(emitted) == 1
    assert emitted[0].span == pipeline.SpeechSpan(5_920, 24_000)
    assert emitted[0].pcm_start_sample == 2_720
    np.testing.assert_array_equal(emitted[0].pcm, source[2_720:24_000])
    assert adapter.calls == 3
    assert adapter.is_final_values == [False, False, True]


def test_open_speech_forces_exactly_one_no_overlap_segment_at_480k() -> None:
    source = np.arange(480_000, dtype=np.int32).astype(np.int16)
    blocks = _blocks(source)
    adapter = FakeStreamingVadAdapter((((0, None),),) + ((),) * (len(blocks) - 1))
    segmenter = _segmenter(adapter)
    emitted: list[pipeline.BufferedSpeechSegment] = []

    for block in blocks:
        emitted.extend(_process(segmenter, block, is_final=False))

    assert len(emitted) == 1
    assert emitted[0].span == pipeline.SpeechSpan(0, 480_000)
    assert emitted[0].pcm_start_sample == 0
    np.testing.assert_array_equal(emitted[0].pcm, source)
    assert adapter.calls == len(blocks)


def test_forced_cut_clamps_late_end_and_rebegin_without_duplicate_pcm() -> None:
    source = np.arange(52 * BLOCK_SAMPLES, dtype=np.int32).astype(np.int16)
    blocks = _blocks(source)
    adapter = FakeStreamingVadAdapter(
        (((0, None),),)
        + ((),) * 49
        + (
            (
                (None, 29_800),
                (29_800, None),
            ),
            ((None, 30_600),),
        )
    )
    segmenter = _segmenter(adapter)
    emitted: list[pipeline.BufferedSpeechSegment] = []

    for index, block in enumerate(blocks):
        emitted.extend(
            _process(
                segmenter,
                block,
                is_final=index == len(blocks) - 1,
            )
        )

    assert [segment.span for segment in emitted] == [
        pipeline.SpeechSpan(0, 480_000),
        pipeline.SpeechSpan(480_000, 489_600),
    ]
    assert [segment.pcm_start_sample for segment in emitted] == [0, 480_000]
    np.testing.assert_array_equal(emitted[0].pcm, source[:480_000])
    np.testing.assert_array_equal(emitted[1].pcm, source[480_000:489_600])
    np.testing.assert_array_equal(
        np.concatenate([segment.pcm for segment in emitted]),
        source[:489_600],
    )
    assert adapter.calls == len(blocks)


def test_forced_cut_exact_end_and_rebegin_continue_without_prepad_overlap() -> None:
    source = np.arange(52 * BLOCK_SAMPLES, dtype=np.int32).astype(np.int16)
    blocks = _blocks(source)
    adapter = FakeStreamingVadAdapter(
        (((0, None),),)
        + ((),) * 49
        + (
            (
                (None, 30_000),
                (30_000, None),
            ),
            ((None, 31_000),),
        )
    )
    segmenter = _segmenter(adapter)
    emitted: list[pipeline.BufferedSpeechSegment] = []

    for index, block in enumerate(blocks):
        emitted.extend(
            _process(
                segmenter,
                block,
                is_final=index == len(blocks) - 1,
            )
        )

    assert [segment.span for segment in emitted] == [
        pipeline.SpeechSpan(0, 480_000),
        pipeline.SpeechSpan(480_000, 496_000),
    ]
    assert [segment.pcm_start_sample for segment in emitted] == [0, 480_000]
    np.testing.assert_array_equal(
        np.concatenate([segment.pcm for segment in emitted]),
        source[:496_000],
    )
    assert adapter.calls == len(blocks)


def test_forced_cut_late_end_and_rebegin_preserve_tail_without_overlap() -> None:
    source = np.arange(52 * BLOCK_SAMPLES, dtype=np.int32).astype(np.int16)
    blocks = _blocks(source)
    adapter = FakeStreamingVadAdapter(
        (((0, None),),)
        + ((),) * 49
        + (
            (
                (None, 30_010),
                (30_010, None),
            ),
            ((None, 31_000),),
        )
    )
    segmenter = _segmenter(adapter)
    emitted: list[pipeline.BufferedSpeechSegment] = []

    for index, block in enumerate(blocks):
        emitted.extend(
            _process(
                segmenter,
                block,
                is_final=index == len(blocks) - 1,
            )
        )

    assert [segment.span for segment in emitted] == [
        pipeline.SpeechSpan(0, 480_000),
        pipeline.SpeechSpan(480_000, 480_160),
        pipeline.SpeechSpan(480_160, 496_000),
    ]
    assert [segment.pcm_start_sample for segment in emitted] == [
        0,
        480_000,
        480_160,
    ]
    np.testing.assert_array_equal(
        np.concatenate([segment.pcm for segment in emitted]),
        source[:496_000],
    )
    assert adapter.calls == len(blocks)


def test_forced_cut_rejects_first_ms_grid_retreat_beyond_3200_and_is_terminal() -> None:
    source = np.arange(52 * BLOCK_SAMPLES, dtype=np.int32).astype(np.int16)
    blocks = _blocks(source)
    # VAD markers are integer milliseconds, so 3_216 samples is the first
    # representable retreat beyond the allowed 3_200-sample clamp.
    adapter = FakeStreamingVadAdapter(
        (((0, None),),)
        + ((),) * 49
        + (
            (
                (None, 29_799),
                (29_799, None),
            ),
            (),
        )
    )
    segmenter = _segmenter(adapter)

    for block in blocks[:50]:
        _process(segmenter, block, is_final=False)

    with pytest.raises(pipeline.PipelineError) as caught:
        _process(segmenter, blocks[50], is_final=False)

    assert caught.value.code == "invalid_model_output"
    calls_after_failure = adapter.calls
    with pytest.raises(pipeline.PipelineError) as terminal:
        _process(segmenter, blocks[51], is_final=False)

    assert terminal.value.code == "invalid_model_output"
    assert adapter.calls == calls_after_failure


def test_final_pending_fails_closed_without_synthesizing_an_end() -> None:
    adapter = FakeStreamingVadAdapter((((100, None),),))
    segmenter = _segmenter(adapter)
    block = DecodedBlock(0, np.arange(BLOCK_SAMPLES, dtype=np.int16))

    with pytest.raises(pipeline.PipelineError) as caught:
        _process(segmenter, block, is_final=True)

    assert caught.value.code == "invalid_model_output"
    assert adapter.calls == 1
