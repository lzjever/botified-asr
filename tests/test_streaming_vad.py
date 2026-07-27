from __future__ import annotations

import numpy as np
import pytest

from botified_asr import pipeline
from botified_asr.audio import DecodedBlock


class FakeAutoModel:
    def __init__(self, raw_markers: list[list[int]]) -> None:
        self.raw_markers = raw_markers
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> list[dict[str, object]]:
        self.calls.append(kwargs)
        return [{"key": "vad-fixture", "value": self.raw_markers}]


class FakeStreamingVadAdapter:
    def __init__(
        self,
        outputs: tuple[
            tuple[tuple[int | None, int | None], ...],
            ...,
        ] = (
            ((100, None),),
            (),
            ((None, 450),),
        ),
    ) -> None:
        self._outputs = outputs
        self.caches: list[dict[str, object]] = []
        self.is_final_values: list[bool] = []

    def generate(
        self,
        pcm: np.ndarray,
        *,
        cache: dict[str, object],
        is_final: bool,
    ) -> tuple[pipeline.VadMarker, ...]:
        self.caches.append(cache)
        self.is_final_values.append(is_final)
        markers = self._outputs[len(self.caches) - 1]
        return tuple(
            pipeline.VadMarker(start_ms=start_ms, end_ms=end_ms)
            for start_ms, end_ms in markers
        )


@pytest.mark.parametrize(
    ("raw_markers", "expected"),
    [
        ([], ()),
        ([[100, -1]], (pipeline.VadMarker(start_ms=100, end_ms=None),)),
        ([[-1, 450]], (pipeline.VadMarker(start_ms=None, end_ms=450),)),
        ([[100, 450]], (pipeline.VadMarker(start_ms=100, end_ms=450),)),
    ],
    ids=("empty", "begin", "end", "complete"),
)
def test_funasr_streaming_vad_adapter_maps_raw_envelope_to_typed_markers(
    raw_markers: list[list[int]],
    expected: tuple[pipeline.VadMarker, ...],
) -> None:
    from botified_asr.funasr_adapter import FunAsrStreamingVadAdapter

    pcm = np.arange(3_200, dtype=np.int16)
    original_pcm = pcm.copy()
    cache: dict[str, object] = {}
    model = FakeAutoModel(raw_markers)
    adapter = FunAsrStreamingVadAdapter(model)

    markers = adapter.generate(pcm, cache=cache, is_final=False)

    assert markers == expected
    assert len(model.calls) == 1
    call = model.calls[0]
    assert set(call) == {"input", "cache", "is_final", "chunk_size"}
    assert isinstance(call["input"], np.ndarray)
    assert call["input"].dtype == np.float32
    assert call["input"].shape == (3_200,)
    assert call["input"].flags.c_contiguous
    np.testing.assert_array_equal(
        call["input"],
        pcm.astype(np.float32) / np.float32(32_768.0),
    )
    np.testing.assert_array_equal(pcm, original_pcm)
    assert call["cache"] is cache
    assert call["is_final"] is False
    assert call["chunk_size"] == 200


@pytest.mark.parametrize(
    "raw_markers",
    (
        [[-1, -1]],
        [[450, 100]],
    ),
    ids=("neither_boundary", "inverted_complete"),
)
def test_funasr_streaming_vad_adapter_rejects_impossible_markers(
    raw_markers: list[list[int]],
) -> None:
    from botified_asr.funasr_adapter import FunAsrStreamingVadAdapter

    adapter = FunAsrStreamingVadAdapter(FakeAutoModel(raw_markers))

    with pytest.raises(pipeline.PipelineError) as caught:
        adapter.generate(
            np.zeros(3_200, dtype=np.int16),
            cache={},
            is_final=False,
        )

    assert caught.value.code == "invalid_model_output"


def test_streaming_vad_session_pairs_typed_cross_block_markers() -> None:
    blocks = [
        DecodedBlock(
            start_sample=index * 3_200,
            pcm=np.full(3_200, index, dtype=np.int16),
        )
        for index in range(3)
    ]
    adapter = FakeStreamingVadAdapter()
    session = pipeline.StreamingVadSession(adapter)

    emissions = [
        session.process(block, is_final=index == len(blocks) - 1)
        for index, block in enumerate(blocks)
    ]

    assert emissions == [
        (),
        (),
        (pipeline.SpeechSpan(start_sample=1_600, end_sample=7_200),),
    ]
    assert adapter.is_final_values == [False, False, True]
    assert len(adapter.caches) == 3
    assert adapter.caches[0] is adapter.caches[1] is adapter.caches[2]


@pytest.mark.parametrize(
    "outputs",
    (
        (
            ((100, None),),
            ((200, None),),
        ),
        (
            ((100, None),),
            ((200, 300),),
        ),
        (((None, 450),),),
        (((100, 100),),),
        (((450, 100),),),
    ),
    ids=(
        "duplicate_start",
        "pending_then_complete",
        "unmatched_end",
        "zero_length",
        "inverted",
    ),
)
def test_streaming_vad_session_rejects_invalid_marker_transitions(
    outputs: tuple[
        tuple[tuple[int | None, int | None], ...],
        ...,
    ],
) -> None:
    session = pipeline.StreamingVadSession(FakeStreamingVadAdapter(outputs))

    with pytest.raises(pipeline.PipelineError) as caught:
        for index in range(len(outputs)):
            session.process(
                DecodedBlock(
                    start_sample=index * 3_200,
                    pcm=np.zeros(3_200, dtype=np.int16),
                ),
                is_final=False,
            )

    assert caught.value.code == "invalid_model_output"


def test_streaming_vad_session_rejects_same_batch_span_rollback() -> None:
    session = pipeline.StreamingVadSession(
        FakeStreamingVadAdapter(
            (
                (
                    (100, 200),
                    (50, 150),
                ),
            )
        )
    )

    with pytest.raises(pipeline.PipelineError) as caught:
        session.process(
            DecodedBlock(0, np.zeros(3_200, dtype=np.int16)),
            is_final=False,
        )

    assert caught.value.code == "invalid_model_output"


def test_streaming_vad_session_rejects_cross_call_span_rollback() -> None:
    session = pipeline.StreamingVadSession(
        FakeStreamingVadAdapter(
            (
                ((100, 200),),
                ((50, 150),),
            )
        )
    )

    assert session.process(
        DecodedBlock(0, np.zeros(3_200, dtype=np.int16)),
        is_final=False,
    ) == (pipeline.SpeechSpan(start_sample=1_600, end_sample=3_200),)

    with pytest.raises(pipeline.PipelineError) as caught:
        session.process(
            DecodedBlock(3_200, np.zeros(3_200, dtype=np.int16)),
            is_final=False,
        )

    assert caught.value.code == "invalid_model_output"


def test_streaming_vad_session_rejects_pending_start_rollback() -> None:
    session = pipeline.StreamingVadSession(
        FakeStreamingVadAdapter(
            (
                ((100, 200),),
                ((50, None),),
            )
        )
    )

    assert session.process(
        DecodedBlock(0, np.zeros(3_200, dtype=np.int16)),
        is_final=False,
    ) == (pipeline.SpeechSpan(start_sample=1_600, end_sample=3_200),)

    with pytest.raises(pipeline.PipelineError) as caught:
        session.process(
            DecodedBlock(3_200, np.zeros(3_200, dtype=np.int16)),
            is_final=False,
        )

    assert caught.value.code == "invalid_model_output"


@pytest.mark.parametrize(
    "outputs",
    (
        (
            ((-1, None),),
            ((None, 100),),
        ),
        (((-1, 100),),),
    ),
    ids=("negative_begin", "negative_complete"),
)
def test_streaming_vad_session_rejects_negative_typed_markers(
    outputs: tuple[
        tuple[tuple[int | None, int | None], ...],
        ...,
    ],
) -> None:
    session = pipeline.StreamingVadSession(FakeStreamingVadAdapter(outputs))

    with pytest.raises(pipeline.PipelineError) as caught:
        for index in range(len(outputs)):
            session.process(
                DecodedBlock(
                    start_sample=index * 3_200,
                    pcm=np.zeros(3_200, dtype=np.int16),
                ),
                is_final=False,
            )

    assert caught.value.code == "invalid_model_output"


def test_streaming_vad_session_allows_adjacent_absolute_spans() -> None:
    session = pipeline.StreamingVadSession(
        FakeStreamingVadAdapter(
            (
                ((100, 200),),
                ((200, 300),),
            )
        )
    )

    assert session.process(
        DecodedBlock(0, np.zeros(3_200, dtype=np.int16)),
        is_final=False,
    ) == (pipeline.SpeechSpan(start_sample=1_600, end_sample=3_200),)
    assert session.process(
        DecodedBlock(3_200, np.zeros(3_200, dtype=np.int16)),
        is_final=True,
    ) == (pipeline.SpeechSpan(start_sample=3_200, end_sample=4_800),)


def test_streaming_vad_session_rejects_final_with_pending_start() -> None:
    session = pipeline.StreamingVadSession(FakeStreamingVadAdapter((((100, None),),)))

    with pytest.raises(pipeline.PipelineError) as caught:
        session.process(
            DecodedBlock(0, np.zeros(3_200, dtype=np.int16)),
            is_final=True,
        )

    assert caught.value.code == "invalid_model_output"


def test_streaming_vad_session_rejects_calls_after_successful_final() -> None:
    adapter = FakeStreamingVadAdapter(
        (
            ((100, 150),),
            (),
        )
    )
    session = pipeline.StreamingVadSession(adapter)

    assert session.process(
        DecodedBlock(0, np.zeros(3_200, dtype=np.int16)),
        is_final=True,
    ) == (pipeline.SpeechSpan(start_sample=1_600, end_sample=2_400),)

    with pytest.raises(pipeline.PipelineError) as caught:
        session.process(
            DecodedBlock(3_200, np.zeros(3_200, dtype=np.int16)),
            is_final=True,
        )

    assert caught.value.code == "invalid_model_output"
    assert len(adapter.caches) == 1
