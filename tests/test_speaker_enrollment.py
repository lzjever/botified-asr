from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from botified_asr import speaker_enrollment, speakers
from botified_asr.audio import BLOCK_SAMPLES, Cancellation, DecodedBlock, MediaProbe
from botified_asr.contracts import MAX_AUDIO_SAMPLES
from botified_asr.errors import PipelineError
from botified_asr.pipeline import BufferedSpeechSegment, SpeechSpan


MODEL_ID = "funasr/campplus"
MODEL_REVISION = "1" * 40
MIN_SPEECH_SAMPLES = 5 * 16_000
MAX_SPEECH_SAMPLES = 30 * 16_000


def _embedding_policy() -> speakers.SpeakerEmbeddingPolicy:
    return speakers.SpeakerEmbeddingPolicy(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        embedding_dimension=2,
        sample_rate=16_000,
        downmix_policy_version="ffmpeg-first-audio-stream-ac1-v1",
        window_samples=24_000,
        window_shift_samples=12_000,
        padding_policy_version="right-zero-pad-v1",
        normalization_policy_version="int16-div-32768-l2-v1",
        enrollment_aggregation_policy_version=("sample-centroid-equal-average-v1"),
    )


def _unit(x: float, y: float) -> np.ndarray:
    vector = np.array([x, y], dtype=np.float32)
    vector /= np.linalg.norm(vector)
    return vector


def _window(
    vector: np.ndarray,
    *,
    start_sample: int = 0,
) -> speakers.SpeakerEmbeddingWindow:
    return speakers.SpeakerEmbeddingWindow(
        start_sample=start_sample,
        end_sample=start_sample + 1,
        embedding=vector,
    )


def _speech_segment(
    sample_count: int,
    *,
    start_sample: int = 0,
    pcm_start_sample: int | None = None,
    fill: int = 1,
) -> BufferedSpeechSegment:
    pcm_start = start_sample if pcm_start_sample is None else pcm_start_sample
    end_sample = start_sample + sample_count
    return BufferedSpeechSegment(
        span=SpeechSpan(start_sample, end_sample),
        pcm_start_sample=pcm_start,
        pcm=np.full(end_sample - pcm_start, fill, dtype=np.int16),
    )


def _decoded_blocks(sample_count: int) -> tuple[DecodedBlock, ...]:
    return tuple(
        DecodedBlock(
            start,
            np.zeros(
                min(BLOCK_SAMPLES, sample_count - start),
                dtype=np.int16,
            ),
        )
        for start in range(0, sample_count, BLOCK_SAMPLES)
    )


class _ClosableBlocks:
    def __init__(self, blocks: tuple[DecodedBlock, ...]) -> None:
        self._blocks = blocks
        self.close_calls = 0

    def __iter__(self) -> Any:
        return iter(self._blocks)

    def close(self) -> None:
        self.close_calls += 1


class _Frontend:
    def __init__(
        self,
        decoded_sample_count: int = MAX_SPEECH_SAMPLES,
        *,
        probe_duration_seconds: float = 30.0,
    ) -> None:
        self.decoded_sample_count = decoded_sample_count
        self.probe_duration_seconds = probe_duration_seconds
        self.probes: list[tuple[Path, Cancellation]] = []
        self.decoders: list[_ClosableBlocks] = []
        self.segmenters: list[_ScriptedSegmenter] = []

    def probe(self, path: Path, cancellation: Cancellation) -> MediaProbe:
        self.probes.append((path, cancellation))
        return MediaProbe(self.probe_duration_seconds, "wav")

    def decode(
        self,
        path: Path,
        probe: MediaProbe,
        cancellation: Cancellation,
    ) -> _ClosableBlocks:
        del path, probe, cancellation
        blocks = _ClosableBlocks(_decoded_blocks(self.decoded_sample_count))
        self.decoders.append(blocks)
        return blocks


class _ScriptedSegmenter:
    def __init__(
        self,
        adapter: object,
        output: tuple[BufferedSpeechSegment, ...],
        *,
        cancellation: Cancellation | None = None,
    ) -> None:
        self.adapter = adapter
        self.output = output
        self.cancellation = cancellation
        self.final_flags: list[bool] = []

    def process(
        self,
        block: DecodedBlock,
        *,
        is_final: bool,
    ) -> tuple[BufferedSpeechSegment, ...]:
        del block
        self.final_flags.append(is_final)
        if self.cancellation is not None:
            self.cancellation.cancel()
        return self.output if is_final else ()


class _EmbeddingAdapter:
    def __init__(
        self,
        outputs: list[tuple[speakers.SpeakerEmbeddingWindow, ...] | BaseException],
    ) -> None:
        self.outputs = outputs
        self.calls: list[np.ndarray] = []

    def embed_windows(
        self,
        pcm: np.ndarray,
    ) -> tuple[speakers.SpeakerEmbeddingWindow, ...]:
        self.calls.append(pcm)
        output = self.outputs[len(self.calls) - 1]
        if isinstance(output, BaseException):
            raise output
        return output


class _VadAdapter:
    def generate(self, *_args: object, **_kwargs: object) -> tuple[object, ...]:
        raise AssertionError("scripted segmenter must own VAD behavior")


def _processor(
    monkeypatch: pytest.MonkeyPatch,
    segment_outputs: list[tuple[BufferedSpeechSegment, ...]],
    embedding_outputs: list[
        tuple[speakers.SpeakerEmbeddingWindow, ...] | BaseException
    ],
    *,
    threshold: float = -1.0,
    frontend: _Frontend | None = None,
    cancellation_on_segmenter_call: Cancellation | None = None,
) -> tuple[
    speaker_enrollment.SpeakerEnrollmentProcessor,
    _Frontend,
    _EmbeddingAdapter,
]:
    output_iterator = iter(segment_outputs)
    selected_frontend = frontend or _Frontend()

    def segmenter_factory(adapter: object) -> _ScriptedSegmenter:
        segmenter = _ScriptedSegmenter(
            adapter,
            next(output_iterator),
            cancellation=cancellation_on_segmenter_call,
        )
        selected_frontend.segmenters.append(segmenter)
        return segmenter

    monkeypatch.setattr(
        speaker_enrollment,
        "StreamingSpeechSegmenter",
        segmenter_factory,
    )
    embedding_adapter = _EmbeddingAdapter(embedding_outputs)
    processor = speaker_enrollment.SpeakerEnrollmentProcessor(
        selected_frontend,
        _VadAdapter(),
        embedding_adapter,
        _embedding_policy(),
        speaker_enrollment.SpeakerEnrollmentPolicy(threshold),
    )
    return processor, selected_frontend, embedding_adapter


@pytest.mark.parametrize("sample_count", [2, 3, 4, 5])
def test_enrollment_accepts_two_to_five_samples_and_builds_replacement(
    monkeypatch: pytest.MonkeyPatch,
    sample_count: int,
) -> None:
    segment = (_speech_segment(MIN_SPEECH_SAMPLES),)
    processor, frontend, _ = _processor(
        monkeypatch,
        [segment] * sample_count,
        [(_window(_unit(1.0, 0.0)),)] * sample_count,
    )

    replacement = processor.process(
        tuple(Path(f"sample-{index}.ready") for index in range(sample_count)),
        Cancellation(),
        effective_max_audio_samples=MAX_AUDIO_SAMPLES,
    )

    assert replacement.sample_count == sample_count
    assert replacement.embedding_model_id == MODEL_ID
    assert replacement.embedding_model_revision == MODEL_REVISION
    assert replacement.embedding_dimension == 2
    assert replacement.embedding_policy_fingerprint == _embedding_policy().fingerprint
    np.testing.assert_allclose(
        replacement.embedding.as_numpy(),
        _unit(1.0, 0.0),
        rtol=0.0,
        atol=1e-6,
    )
    assert [decoder.close_calls for decoder in frontend.decoders] == [1] * sample_count


@pytest.mark.parametrize("sample_count", [0, 1, 6])
def test_enrollment_rejects_sample_count_outside_two_to_five(
    monkeypatch: pytest.MonkeyPatch,
    sample_count: int,
) -> None:
    processor, frontend, adapter = _processor(monkeypatch, [], [])

    with pytest.raises(PipelineError) as caught:
        processor.process(
            tuple(Path(f"sample-{index}.ready") for index in range(sample_count)),
            Cancellation(),
            effective_max_audio_samples=MAX_AUDIO_SAMPLES,
        )

    assert caught.value.code == "invalid_speaker_samples"
    assert frontend.probes == []
    assert adapter.calls == []


@pytest.mark.parametrize("speech_samples", [MIN_SPEECH_SAMPLES, MAX_SPEECH_SAMPLES])
def test_enrollment_effective_speech_duration_boundaries_are_inclusive(
    monkeypatch: pytest.MonkeyPatch,
    speech_samples: int,
) -> None:
    segment = (_speech_segment(speech_samples),)
    processor, _, adapter = _processor(
        monkeypatch,
        [segment, segment],
        [
            (_window(_unit(1.0, 0.0)),),
            (_window(_unit(1.0, 0.0)),),
        ],
    )

    processor.process(
        (Path("a.ready"), Path("b.ready")),
        Cancellation(),
        effective_max_audio_samples=MAX_AUDIO_SAMPLES,
    )

    assert [len(pcm) for pcm in adapter.calls] == [speech_samples, speech_samples]


@pytest.mark.parametrize(
    "segments",
    [
        (_speech_segment(MIN_SPEECH_SAMPLES - 1),),
        (
            _speech_segment(MAX_SPEECH_SAMPLES // 2 + 1),
            _speech_segment(
                MAX_SPEECH_SAMPLES // 2,
                start_sample=MAX_SPEECH_SAMPLES // 2 + 1,
            ),
        ),
    ],
)
def test_enrollment_rejects_effective_speech_outside_duration_bounds(
    monkeypatch: pytest.MonkeyPatch,
    segments: tuple[BufferedSpeechSegment, ...],
) -> None:
    processor, frontend, adapter = _processor(
        monkeypatch,
        [segments, (_speech_segment(MIN_SPEECH_SAMPLES),)],
        [(_window(_unit(1.0, 0.0)),)],
        frontend=_Frontend(MAX_SPEECH_SAMPLES + 1),
    )

    with pytest.raises(PipelineError) as caught:
        processor.process(
            (Path("a.ready"), Path("b.ready")),
            Cancellation(),
            effective_max_audio_samples=MAX_AUDIO_SAMPLES,
        )

    assert caught.value.code == "invalid_speaker_sample_duration"
    assert adapter.calls == []
    assert frontend.decoders[0].close_calls == 1


def test_enrollment_reports_no_speech_without_calling_embedding_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor, frontend, adapter = _processor(
        monkeypatch,
        [(), (_speech_segment(MIN_SPEECH_SAMPLES),)],
        [],
    )

    with pytest.raises(PipelineError) as caught:
        processor.process(
            (Path("silent.ready"), Path("speech.ready")),
            Cancellation(),
            effective_max_audio_samples=MAX_AUDIO_SAMPLES,
        )

    assert caught.value.code == "no_speech"
    assert adapter.calls == []
    assert frontend.decoders[0].close_calls == 1


def test_enrollment_uses_canonical_speech_and_equal_sample_weighting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    padded = _speech_segment(
        MIN_SPEECH_SAMPLES,
        start_sample=16_000,
        pcm_start_sample=0,
        fill=7,
    )
    processor, _, adapter = _processor(
        monkeypatch,
        [(padded,), (_speech_segment(MIN_SPEECH_SAMPLES),)],
        [
            (
                _window(_unit(1.0, 0.0)),
                _window(_unit(1.0, 0.0), start_sample=1),
                _window(_unit(1.0, 0.0), start_sample=2),
            ),
            (_window(_unit(0.0, 1.0)),),
        ],
        threshold=0.0,
        frontend=_Frontend(MIN_SPEECH_SAMPLES + 16_000),
    )

    replacement = processor.process(
        (Path("long-window-count.ready"), Path("short-window-count.ready")),
        Cancellation(),
        effective_max_audio_samples=MAX_AUDIO_SAMPLES,
    )

    assert len(adapter.calls[0]) == MIN_SPEECH_SAMPLES
    assert np.all(adapter.calls[0] == 7)
    np.testing.assert_allclose(
        replacement.embedding.as_numpy(),
        _unit(1.0, 1.0),
        rtol=0.0,
        atol=1e-6,
    )


def test_enrollment_pair_consistency_threshold_is_inclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment = (_speech_segment(MIN_SPEECH_SAMPLES),)
    processor, _, _ = _processor(
        monkeypatch,
        [segment, segment],
        [
            (_window(_unit(1.0, 0.0)),),
            (_window(_unit(0.0, 1.0)),),
        ],
        threshold=0.0,
    )

    processor.process(
        (Path("a.ready"), Path("b.ready")),
        Cancellation(),
        effective_max_audio_samples=MAX_AUDIO_SAMPLES,
    )


def test_enrollment_rejects_pair_below_consistency_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment = (_speech_segment(MIN_SPEECH_SAMPLES),)
    processor, _, _ = _processor(
        monkeypatch,
        [segment, segment],
        [
            (_window(_unit(1.0, 0.0)),),
            (_window(_unit(0.0, 1.0)),),
        ],
        threshold=math.nextafter(0.0, math.inf),
    )

    with pytest.raises(PipelineError) as caught:
        processor.process(
            (Path("a.ready"), Path("b.ready")),
            Cancellation(),
            effective_max_audio_samples=MAX_AUDIO_SAMPLES,
        )

    assert caught.value.code == "speaker_samples_inconsistent"


@pytest.mark.parametrize(
    "windows",
    [
        (),
        (_window(np.array([math.nan, 0.0], dtype=np.float32)),),
        (_window(np.array([0.0, 0.0], dtype=np.float32)),),
        (_window(np.array([1.0], dtype=np.float32)),),
    ],
)
def test_enrollment_rejects_invalid_embedding_output(
    monkeypatch: pytest.MonkeyPatch,
    windows: tuple[speakers.SpeakerEmbeddingWindow, ...],
) -> None:
    segment = (_speech_segment(MIN_SPEECH_SAMPLES),)
    processor, frontend, _ = _processor(
        monkeypatch,
        [segment, segment],
        [windows],
    )

    with pytest.raises(PipelineError) as caught:
        processor.process(
            (Path("a.ready"), Path("b.ready")),
            Cancellation(),
            effective_max_audio_samples=MAX_AUDIO_SAMPLES,
        )

    assert caught.value.code == "invalid_speaker_embedding"
    assert frontend.decoders[0].close_calls == 1


def test_enrollment_maps_camplus_invalid_output_to_invalid_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment = (_speech_segment(MIN_SPEECH_SAMPLES),)
    processor, frontend, _ = _processor(
        monkeypatch,
        [segment, segment],
        [PipelineError("invalid_model_output", "CAM++ failed")],
    )

    with pytest.raises(PipelineError) as caught:
        processor.process(
            (Path("a.ready"), Path("b.ready")),
            Cancellation(),
            effective_max_audio_samples=MAX_AUDIO_SAMPLES,
        )

    assert caught.value.code == "invalid_speaker_embedding"
    assert frontend.decoders[0].close_calls == 1


def test_enrollment_cancellation_closes_decoder_and_stops_before_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancellation = Cancellation()
    processor, frontend, adapter = _processor(
        monkeypatch,
        [(_speech_segment(MIN_SPEECH_SAMPLES),)],
        [],
        cancellation_on_segmenter_call=cancellation,
    )

    with pytest.raises(PipelineError) as caught:
        processor.process(
            (Path("a.ready"), Path("b.ready")),
            cancellation,
            effective_max_audio_samples=MAX_AUDIO_SAMPLES,
        )

    assert caught.value.code == "cancelled"
    assert adapter.calls == []
    assert frontend.decoders[0].close_calls == 1


@pytest.mark.parametrize(
    "threshold",
    [True, math.nan, math.inf, -1.01, 1.01],
)
def test_enrollment_policy_requires_finite_bounded_real_threshold(
    threshold: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        speaker_enrollment.SpeakerEnrollmentPolicy(threshold)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "effective_max_audio_samples",
    [True, 1.0, 0, -1, MAX_AUDIO_SAMPLES + 1],
)
def test_enrollment_requires_exact_positive_bounded_audio_cap(
    monkeypatch: pytest.MonkeyPatch,
    effective_max_audio_samples: object,
) -> None:
    processor, frontend, _ = _processor(monkeypatch, [], [])

    with pytest.raises((TypeError, ValueError)):
        processor.process(
            (Path("a.ready"), Path("b.ready")),
            Cancellation(),
            effective_max_audio_samples=effective_max_audio_samples,  # type: ignore[arg-type]
        )

    assert frontend.probes == []


def test_enrollment_probe_rejects_over_cap_without_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frontend = _Frontend(
        MIN_SPEECH_SAMPLES,
        probe_duration_seconds=5.1,
    )
    processor, _, adapter = _processor(
        monkeypatch,
        [],
        [],
        frontend=frontend,
    )

    with pytest.raises(PipelineError) as caught:
        processor.process(
            (Path("a.ready"), Path("b.ready")),
            Cancellation(),
            effective_max_audio_samples=MIN_SPEECH_SAMPLES,
        )

    assert caught.value.code == "audio_too_long"
    assert frontend.decoders == []
    assert adapter.calls == []


def test_enrollment_decoded_count_is_final_and_closes_on_first_sample_over_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frontend = _Frontend(
        BLOCK_SAMPLES + 1,
        probe_duration_seconds=0.1,
    )
    processor, _, adapter = _processor(
        monkeypatch,
        [(), ()],
        [],
        frontend=frontend,
    )

    with pytest.raises(PipelineError) as caught:
        processor.process(
            (Path("a.ready"), Path("b.ready")),
            Cancellation(),
            effective_max_audio_samples=BLOCK_SAMPLES,
        )

    assert caught.value.code == "audio_too_long"
    assert len(frontend.decoders) == 1
    assert frontend.decoders[0].close_calls == 1
    assert frontend.segmenters[0].final_flags == [False]
    assert adapter.calls == []


def test_enrollment_marks_only_last_decoded_block_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment = (_speech_segment(MIN_SPEECH_SAMPLES),)
    frontend = _Frontend(
        MIN_SPEECH_SAMPLES,
        probe_duration_seconds=5.0,
    )
    processor, _, _ = _processor(
        monkeypatch,
        [segment, segment],
        [
            (_window(_unit(1.0, 0.0)),),
            (_window(_unit(1.0, 0.0)),),
        ],
        frontend=frontend,
    )

    processor.process(
        (Path("a.ready"), Path("b.ready")),
        Cancellation(),
        effective_max_audio_samples=MIN_SPEECH_SAMPLES,
    )

    expected = [False] * (len(_decoded_blocks(MIN_SPEECH_SAMPLES)) - 1) + [True]
    assert frontend.segmenters[0].final_flags == expected
