from __future__ import annotations

from typing import Protocol

import numpy as np

from botified_asr.pipeline import PipelineError, VadMarker


class FunAsrAutoModel(Protocol):
    def generate(self, **kwargs: object) -> object: ...


class FunAsrStreamingVadAdapter:
    def __init__(self, model: FunAsrAutoModel) -> None:
        self._model = model

    def generate(
        self,
        pcm: np.ndarray,
        *,
        cache: dict[str, object],
        is_final: bool,
    ) -> tuple[VadMarker, ...]:
        if pcm.dtype != np.int16 or pcm.ndim != 1 or not pcm.flags.c_contiguous:
            raise PipelineError("invalid_audio", "VAD input block is invalid")

        normalized = pcm.astype(np.float32)
        normalized /= np.float32(32768.0)
        result = self._model.generate(
            input=normalized,
            cache=cache,
            is_final=is_final,
            chunk_size=200,
        )
        return _decode_markers(result)


def _decode_markers(result: object) -> tuple[VadMarker, ...]:
    if (
        not isinstance(result, list)
        or len(result) != 1
        or not isinstance(result[0], dict)
    ):
        raise PipelineError(
            "invalid_model_output",
            "VAD model returned an invalid result",
        )
    raw_markers = result[0].get("value")
    if not isinstance(raw_markers, list):
        raise PipelineError(
            "invalid_model_output",
            "VAD model returned an invalid result",
        )

    markers: list[VadMarker] = []
    for raw_marker in raw_markers:
        if (
            not isinstance(raw_marker, list)
            or len(raw_marker) != 2
            or any(type(value) is not int for value in raw_marker)
        ):
            raise PipelineError(
                "invalid_model_output",
                "VAD model returned an invalid result",
            )
        raw_start, raw_end = raw_marker
        is_begin = raw_start >= 0 and raw_end == -1
        is_end = raw_start == -1 and raw_end >= 0
        is_complete = raw_start >= 0 and raw_end > raw_start
        if not (is_begin or is_end or is_complete):
            raise PipelineError(
                "invalid_model_output",
                "VAD model returned an invalid result",
            )
        markers.append(
            VadMarker(
                start_ms=None if raw_start == -1 else raw_start,
                end_ms=None if raw_end == -1 else raw_end,
            )
        )
    return tuple(markers)
