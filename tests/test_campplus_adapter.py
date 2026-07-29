from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch

from botified_asr import funasr_adapter, speakers
from botified_asr.pipeline import PipelineError


WINDOW_SAMPLES = 24_000
WINDOW_SHIFT_SAMPLES = 12_000
EMBEDDING_DIMENSION = 192


class RecordingModel:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.result


class RecordingLane:
    def __init__(self) -> None:
        self.operations: list[Callable[[], object]] = []

    def invoke(self, operation: Callable[[], object], /) -> object:
        self.operations.append(operation)
        return operation()


def _raw_embeddings(count: int) -> torch.Tensor:
    value = torch.zeros((count, EMBEDDING_DIMENSION), dtype=torch.float32)
    value[:, 0] = 1
    return value


def _adapter(
    model: RecordingModel,
    lane: RecordingLane,
):
    return funasr_adapter.FunAsrCampPlusAdapter(
        model,
        inference_lane=lane,
    )


def test_campplus_adapter_normalizes_pcm_and_embeddings_in_one_lane_call() -> None:
    pcm = np.arange(30_000, dtype=np.int16)
    raw = _raw_embeddings(2)
    raw[0, :2] = torch.tensor([3.0, 4.0])
    raw[1, :2] = torch.tensor([-12.0, 5.0])
    model = RecordingModel([{"spk_embedding": raw}])
    lane = RecordingLane()

    windows = _adapter(model, lane).embed_windows(pcm)

    assert funasr_adapter.SpeakerEmbeddingWindow is speakers.SpeakerEmbeddingWindow
    assert all(type(window) is speakers.SpeakerEmbeddingWindow for window in windows)
    assert len(lane.operations) == 1
    assert len(model.calls) == 1
    assert set(model.calls[0]) == {"input", "batch_size"}
    assert model.calls[0]["batch_size"] == 2
    inputs = model.calls[0]["input"]
    assert isinstance(inputs, list)
    assert len(inputs) == 2
    expected_ranges = [
        (0, WINDOW_SAMPLES),
        (6_000, 30_000),
    ]
    for value, (start, end) in zip(inputs, expected_ranges, strict=True):
        assert isinstance(value, np.ndarray)
        assert value.dtype == np.float32
        assert value.flags.c_contiguous
        np.testing.assert_array_equal(
            value,
            pcm[start:end].astype(np.float32) / np.float32(32768.0),
        )

    assert [(item.start_sample, item.end_sample) for item in windows] == [
        (0, 24_000),
        (6_000, 30_000),
    ]
    assert all(item.embedding.shape == (EMBEDDING_DIMENSION,) for item in windows)
    assert all(item.embedding.dtype == np.float32 for item in windows)
    assert all(np.isfinite(item.embedding).all() for item in windows)
    assert all(np.linalg.norm(item.embedding) == pytest.approx(1.0) for item in windows)
    assert all(item.embedding.flags.owndata for item in windows)
    assert all(not item.embedding.flags.writeable for item in windows)
    assert not np.shares_memory(windows[0].embedding, windows[1].embedding)
    for item in windows:
        with pytest.raises(ValueError):
            item.embedding[0] = 0
    np.testing.assert_allclose(windows[0].embedding[:2], [0.6, 0.8])
    np.testing.assert_allclose(windows[1].embedding[:2], [-12 / 13, 5 / 13])


def test_campplus_exact_windows_use_one_padded_model_batch() -> None:
    pcms = (
        np.arange(WINDOW_SAMPLES, dtype=np.int16),
        np.array([-32_768, 16_384, 1], dtype=np.int16),
    )
    raw = _raw_embeddings(2)
    raw[1, :2] = torch.tensor([3.0, 4.0])
    model = RecordingModel([{"spk_embedding": raw}])
    lane = RecordingLane()

    embeddings = _adapter(model, lane).embed_exact_windows(pcms)

    assert len(lane.operations) == 1
    assert len(model.calls) == 1
    assert model.calls[0]["batch_size"] == 2
    inputs = model.calls[0]["input"]
    assert isinstance(inputs, list)
    assert len(inputs) == 2
    np.testing.assert_array_equal(
        inputs[0],
        pcms[0].astype(np.float32) / np.float32(32768.0),
    )
    np.testing.assert_array_equal(
        inputs[1][:3],
        pcms[1].astype(np.float32) / np.float32(32768.0),
    )
    assert inputs[1].shape == (WINDOW_SAMPLES,)
    assert np.count_nonzero(inputs[1][3:]) == 0
    assert len(embeddings) == 2
    np.testing.assert_allclose(embeddings[1][:2], [0.6, 0.8])


def test_campplus_local_windows_delegate_to_exact_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = RecordingModel([{"spk_embedding": _raw_embeddings(1)}])
    adapter = _adapter(model, RecordingLane())
    calls: list[tuple[np.ndarray, ...]] = []

    def embed_exact_windows(
        pcms: tuple[np.ndarray, ...],
    ) -> tuple[np.ndarray, ...]:
        calls.append(pcms)
        return tuple(
            np.eye(1, EMBEDDING_DIMENSION, dtype=np.float32)[0]
            for _pcm in pcms
        )

    monkeypatch.setattr(adapter, "embed_exact_windows", embed_exact_windows)

    windows = adapter.embed_windows(np.arange(30_000, dtype=np.int16))

    assert [(window.start_sample, window.end_sample) for window in windows] == [
        (0, 24_000),
        (6_000, 30_000),
    ]
    assert len(calls) == 1
    assert [len(pcm) for pcm in calls[0]] == [24_000, 24_000]
    assert model.calls == []


def test_campplus_exact_batch_bound_is_derived_from_existing_local_limit() -> None:
    assert speakers.SPEAKER_EMBEDDING_BATCH_MAX_WINDOWS == 39
    pcms = tuple(
        np.zeros(1, dtype=np.int16)
        for _ in range(speakers.SPEAKER_EMBEDDING_BATCH_MAX_WINDOWS + 1)
    )
    model = RecordingModel([{"spk_embedding": _raw_embeddings(len(pcms))}])
    lane = RecordingLane()

    with pytest.raises(PipelineError) as caught:
        _adapter(model, lane).embed_exact_windows(pcms)

    assert caught.value.code == "invalid_audio"
    assert model.calls == []
    assert lane.operations == []


@pytest.mark.parametrize(
    ("sample_count", "expected_ranges"),
    [
        (1, [(0, 1)]),
        (23_999, [(0, 23_999)]),
        (24_000, [(0, 24_000)]),
        (24_001, [(0, 24_000), (1, 24_001)]),
        (35_999, [(0, 24_000), (11_999, 35_999)]),
        (36_000, [(0, 24_000), (12_000, 36_000)]),
        (
            36_001,
            [(0, 24_000), (12_000, 36_000), (12_001, 36_001)],
        ),
        (
            480_000,
            [(start, start + WINDOW_SAMPLES) for start in range(0, 456_001, 12_000)],
        ),
    ],
)
def test_campplus_adapter_matches_sv_chunk_sample_boundaries(
    sample_count: int,
    expected_ranges: list[tuple[int, int]],
) -> None:
    pcm = np.full(sample_count, 16_384, dtype=np.int16)
    model = RecordingModel([{"spk_embedding": _raw_embeddings(len(expected_ranges))}])
    lane = RecordingLane()

    windows = _adapter(model, lane).embed_windows(pcm)

    assert len(lane.operations) == 1
    assert len(model.calls) == 1
    assert model.calls[0]["batch_size"] == len(expected_ranges)
    inputs = model.calls[0]["input"]
    assert isinstance(inputs, list)
    assert len(inputs) == len(expected_ranges)
    for value, (start, end) in zip(inputs, expected_ranges, strict=True):
        assert isinstance(value, np.ndarray)
        assert value.dtype == np.float32
        assert value.flags.c_contiguous
        assert value.shape == (WINDOW_SAMPLES,)
        np.testing.assert_array_equal(
            value[: end - start],
            np.full(end - start, 0.5),
        )
        assert np.count_nonzero(value[end - start :]) == 0
    assert [(item.start_sample, item.end_sample) for item in windows] == expected_ranges


@pytest.mark.parametrize(
    "pcm",
    [
        np.zeros(0, dtype=np.int16),
        np.zeros(480_001, dtype=np.int16),
        np.zeros(1, dtype=np.float32),
        np.zeros((1, 1), dtype=np.int16),
        np.zeros(4, dtype=np.int16)[::2],
    ],
    ids=["empty", "too-long", "wrong-dtype", "wrong-rank", "non-contiguous"],
)
def test_campplus_adapter_rejects_invalid_segments_without_model_calls(
    pcm: np.ndarray,
) -> None:
    model = RecordingModel([{"spk_embedding": _raw_embeddings(1)}])
    lane = RecordingLane()

    with pytest.raises(PipelineError) as caught:
        _adapter(model, lane).embed_windows(pcm)

    assert caught.value.code == "invalid_audio"
    assert model.calls == []
    assert lane.operations == []


_VALID_EMBEDDING = _raw_embeddings(1)
_NONFINITE_EMBEDDING = _raw_embeddings(1)
_NONFINITE_EMBEDDING[0, 1] = torch.nan


@pytest.mark.parametrize(
    "result",
    [
        {},
        [
            [],
        ],
        [{}],
        [
            {"spk_embedding": _VALID_EMBEDDING},
            {"spk_embedding": _VALID_EMBEDDING},
        ],
        [{"spk_embedding": _raw_embeddings(2)}],
        [{"spk_embedding": torch.ones((1, EMBEDDING_DIMENSION - 1))}],
        [
            {
                "spk_embedding": torch.ones(
                    (1, EMBEDDING_DIMENSION),
                    dtype=torch.int64,
                )
            }
        ],
        [
            {
                "spk_embedding": torch.ones(
                    (1, EMBEDDING_DIMENSION),
                    dtype=torch.complex64,
                )
            }
        ],
        [{"spk_embedding": _NONFINITE_EMBEDDING}],
        [{"spk_embedding": torch.zeros((1, EMBEDDING_DIMENSION))}],
    ],
    ids=[
        "not-list",
        "item-not-dict",
        "missing-embedding",
        "wrong-list-count",
        "wrong-window-count",
        "wrong-dimension",
        "integer",
        "complex",
        "non-finite",
        "zero-norm",
    ],
)
def test_campplus_adapter_rejects_invalid_model_output(result: object) -> None:
    model = RecordingModel(result)
    lane = RecordingLane()

    with pytest.raises(PipelineError) as caught:
        _adapter(model, lane).embed_windows(np.zeros(1, dtype=np.int16))

    assert caught.value.code == "invalid_model_output"
    assert len(model.calls) == 1
    assert len(lane.operations) == 1
