from __future__ import annotations

import numpy as np
import pytest

from botified_asr import pipeline
from botified_asr.audio import BLOCK_SAMPLES, DecodedBlock


def _block(source: np.ndarray, start_sample: int) -> DecodedBlock:
    return DecodedBlock(
        start_sample=start_sample,
        pcm=source[start_sample : start_sample + BLOCK_SAMPLES],
    )


def _consume(
    buffer: object,
    block: DecodedBlock,
    *,
    completed_spans: tuple[pipeline.SpeechSpan, ...] = (),
    open_start_sample: int | None = None,
) -> tuple[object, ...]:
    return buffer.consume(  # type: ignore[attr-defined, no-any-return]
        block,
        completed_spans=completed_spans,
        open_start_sample=open_start_sample,
    )


def _new_buffer() -> object:
    return pipeline.BoundedSpeechPcmBuffer()  # type: ignore[attr-defined, no-any-return]


def test_speech_buffer_release_policy_constants_include_all_latency_terms() -> None:
    actual = {
        "VAD_FRONTEND_CONTEXT_SAMPLES": getattr(
            pipeline,
            "VAD_FRONTEND_CONTEXT_SAMPLES",
            None,
        ),
        "VAD_BACKTRACK_SAMPLES": getattr(
            pipeline,
            "VAD_BACKTRACK_SAMPLES",
            None,
        ),
        "VAD_PREPADDING_SAMPLES": getattr(
            pipeline,
            "VAD_PREPADDING_SAMPLES",
            None,
        ),
        "VAD_IDLE_RING_SAMPLES": getattr(
            pipeline,
            "VAD_IDLE_RING_SAMPLES",
            None,
        ),
    }

    assert actual == {
        "VAD_FRONTEND_CONTEXT_SAMPLES": 640,
        "VAD_BACKTRACK_SAMPLES": 6_400,
        "VAD_PREPADDING_SAMPLES": 3_200,
        "VAD_IDLE_RING_SAMPLES": 19_840,
    }
    assert actual["VAD_IDLE_RING_SAMPLES"] == (
        BLOCK_SAMPLES
        + actual["VAD_FRONTEND_CONTEXT_SAMPLES"]
        + actual["VAD_BACKTRACK_SAMPLES"]
        + actual["VAD_PREPADDING_SAMPLES"]
    )


def test_long_idle_pcm_is_bounded_to_the_exact_release_ring() -> None:
    buffer = _new_buffer()
    block_count = 1_000

    for index in range(block_count):
        pcm = np.full(BLOCK_SAMPLES, index % 32_767, dtype=np.int16)
        assert (
            _consume(
                buffer,
                DecodedBlock(index * BLOCK_SAMPLES, pcm),
            )
            == ()
        )
        assert buffer.retained_sample_count <= 19_840  # type: ignore[attr-defined]

    processed_end = block_count * BLOCK_SAMPLES
    assert buffer.retained_start_sample == processed_end - 19_840  # type: ignore[attr-defined]
    assert buffer.retained_sample_count == 19_840  # type: ignore[attr-defined]


def test_cross_block_830ms_backtrack_recovers_200ms_prepad_without_pcm_loss() -> None:
    source = np.arange(24_000, dtype=np.int32).astype(np.int16)
    buffer = _new_buffer()

    assert _consume(buffer, _block(source, 0)) == ()
    assert (
        _consume(
            buffer,
            _block(source, 9_600),
            open_start_sample=5_920,
        )
        == ()
    )
    completed = pipeline.SpeechSpan(start_sample=5_920, end_sample=24_000)

    emitted = _consume(
        buffer,
        _block(source, 19_200),
        completed_spans=(completed,),
        open_start_sample=None,
    )

    assert len(emitted) == 1
    assert emitted[0].span == completed  # type: ignore[attr-defined]
    assert emitted[0].pcm_start_sample == 2_720  # type: ignore[attr-defined]
    np.testing.assert_array_equal(
        emitted[0].pcm,  # type: ignore[attr-defined]
        source[2_720:24_000],
    )


def test_marker_whose_prepad_starts_at_retained_ring_base_is_accepted() -> None:
    source = np.arange(3 * BLOCK_SAMPLES, dtype=np.int32).astype(np.int16)
    buffer = _new_buffer()

    assert _consume(buffer, _block(source, 0)) == ()
    assert _consume(buffer, _block(source, 9_600)) == ()
    assert (
        _consume(
            buffer,
            _block(source, 19_200),
            open_start_sample=12_160,
        )
        == ()
    )

    assert buffer.retained_start_sample == 8_960  # type: ignore[attr-defined]
    assert buffer.retained_sample_count == 19_840  # type: ignore[attr-defined]


def test_marker_whose_prepad_underflows_ring_by_one_vad_ms_fails_closed() -> None:
    source = np.arange(3 * BLOCK_SAMPLES, dtype=np.int32).astype(np.int16)
    buffer = _new_buffer()

    assert _consume(buffer, _block(source, 0)) == ()
    assert _consume(buffer, _block(source, 9_600)) == ()

    with pytest.raises(pipeline.PipelineError) as caught:
        _consume(
            buffer,
            _block(source, 19_200),
            open_start_sample=12_144,
        )

    assert caught.value.code == "invalid_model_output"


def test_complete_span_whose_prepad_starts_at_ring_base_is_recovered() -> None:
    source = np.arange(3 * BLOCK_SAMPLES, dtype=np.int32).astype(np.int16)
    buffer = _new_buffer()
    completed = pipeline.SpeechSpan(start_sample=12_160, end_sample=25_000)

    assert _consume(buffer, _block(source, 0)) == ()
    assert _consume(buffer, _block(source, 9_600)) == ()
    emitted = _consume(
        buffer,
        _block(source, 19_200),
        completed_spans=(completed,),
    )

    assert len(emitted) == 1
    assert emitted[0].span == completed  # type: ignore[attr-defined]
    assert emitted[0].pcm_start_sample == 8_960  # type: ignore[attr-defined]
    np.testing.assert_array_equal(
        emitted[0].pcm,  # type: ignore[attr-defined]
        source[8_960:25_000],
    )


def test_complete_span_whose_prepad_underflows_ring_by_one_ms_fails_closed() -> None:
    source = np.arange(3 * BLOCK_SAMPLES, dtype=np.int32).astype(np.int16)
    buffer = _new_buffer()

    assert _consume(buffer, _block(source, 0)) == ()
    assert _consume(buffer, _block(source, 9_600)) == ()

    with pytest.raises(pipeline.PipelineError) as caught:
        _consume(
            buffer,
            _block(source, 19_200),
            completed_spans=(
                pipeline.SpeechSpan(start_sample=12_144, end_sample=25_000),
            ),
        )

    assert caught.value.code == "invalid_model_output"


def test_open_speech_at_exact_asr_cap_is_emitted_once_without_oversize() -> None:
    source = np.arange(pipeline.DIRECT_MAX_SAMPLES, dtype=np.int32).astype(np.int16)
    buffer = _new_buffer()
    emitted: tuple[object, ...] = ()

    for start_sample in range(0, len(source), BLOCK_SAMPLES):
        current = _consume(
            buffer,
            _block(source, start_sample),
            open_start_sample=0,
        )
        if start_sample < len(source) - BLOCK_SAMPLES:
            assert current == ()
        else:
            emitted = current
        assert buffer.retained_sample_count <= 480_000  # type: ignore[attr-defined]

    assert len(emitted) == 1
    assert emitted[0].span == pipeline.SpeechSpan(  # type: ignore[attr-defined]
        start_sample=0,
        end_sample=480_000,
    )
    assert emitted[0].pcm_start_sample == 0  # type: ignore[attr-defined]
    assert len(emitted[0].pcm) == 480_000  # type: ignore[attr-defined]
    np.testing.assert_array_equal(
        emitted[0].pcm,  # type: ignore[attr-defined]
        source,
    )
    assert buffer.retained_sample_count == 0  # type: ignore[attr-defined]


def test_480001_open_speech_never_emits_oversize_and_retains_one_sample() -> None:
    source = np.arange(480_001, dtype=np.int32).astype(np.int16)
    buffer = _new_buffer()
    emitted: list[object] = []

    for start_sample in range(0, len(source), BLOCK_SAMPLES):
        emitted.extend(
            _consume(
                buffer,
                _block(source, start_sample),
                open_start_sample=0,
            )
        )
        assert buffer.retained_sample_count <= 480_000  # type: ignore[attr-defined]

    assert len(emitted) == 1
    assert emitted[0].span == pipeline.SpeechSpan(  # type: ignore[attr-defined]
        start_sample=0,
        end_sample=480_000,
    )
    assert len(emitted[0].pcm) <= 480_000  # type: ignore[attr-defined]
    assert buffer.retained_start_sample == 480_000  # type: ignore[attr-defined]
    assert buffer.retained_sample_count == 1  # type: ignore[attr-defined]


def test_non_aligned_padded_cap_preserves_the_canonical_first_start() -> None:
    decoded_end = 51 * BLOCK_SAMPLES
    source = np.arange(decoded_end, dtype=np.int32).astype(np.int16)
    buffer = _new_buffer()
    emitted: list[object] = []

    for start_sample in range(0, decoded_end, BLOCK_SAMPLES):
        emitted.extend(
            _consume(
                buffer,
                _block(source, start_sample),
                open_start_sample=5_920 if start_sample >= 9_600 else None,
            )
        )
        assert buffer.retained_sample_count <= 480_000  # type: ignore[attr-defined]

    assert len(emitted) == 1
    assert emitted[0].span == pipeline.SpeechSpan(  # type: ignore[attr-defined]
        start_sample=5_920,
        end_sample=482_720,
    )
    assert emitted[0].pcm_start_sample == 2_720  # type: ignore[attr-defined]
    assert len(emitted[0].pcm) == 480_000  # type: ignore[attr-defined]
    np.testing.assert_array_equal(
        emitted[0].pcm,  # type: ignore[attr-defined]
        source[2_720:482_720],
    )
    assert buffer.retained_start_sample == 482_720  # type: ignore[attr-defined]
    assert buffer.retained_sample_count == 6_880  # type: ignore[attr-defined]


def test_many_forced_caps_emit_lazily_without_retained_memory_growth() -> None:
    total_samples = 4 * pipeline.DIRECT_MAX_SAMPLES + 1
    buffer = _new_buffer()
    emitted_spans: list[pipeline.SpeechSpan] = []

    for start_sample in range(0, total_samples, BLOCK_SAMPLES):
        sample_count = min(BLOCK_SAMPLES, total_samples - start_sample)
        pcm = np.arange(
            start_sample,
            start_sample + sample_count,
            dtype=np.int64,
        ).astype(np.int16)
        for segment in _consume(
            buffer,
            DecodedBlock(start_sample, pcm),
            open_start_sample=0,
        ):
            assert len(segment.pcm) <= pipeline.DIRECT_MAX_SAMPLES  # type: ignore[attr-defined]
            emitted_spans.append(segment.span)  # type: ignore[attr-defined]
        assert buffer.retained_sample_count <= 480_000  # type: ignore[attr-defined]

    assert emitted_spans == [
        pipeline.SpeechSpan(index * 480_000, (index + 1) * 480_000)
        for index in range(4)
    ]
    assert buffer.retained_start_sample == 1_920_000  # type: ignore[attr-defined]
    assert buffer.retained_sample_count == 1  # type: ignore[attr-defined]
