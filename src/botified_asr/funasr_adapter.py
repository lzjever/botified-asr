from __future__ import annotations

import re
from typing import Protocol

import numpy as np
import torch

from botified_asr.errors import PipelineError
from botified_asr.inference import InferenceLane
from botified_asr.pipeline import (
    DIRECT_MAX_SAMPLES,
    AsrResult,
    RichAnnotations,
    VadMarker,
)
from botified_asr.speakers import (
    SPEAKER_EMBEDDING_BATCH_MAX_WINDOWS as CAMPLUS_BATCH_MAX_WINDOWS,
    SPEAKER_EMBEDDING_DIMENSION as CAMPLUS_EMBEDDING_DIMENSION,
    SPEAKER_WINDOW_MAX_SAMPLES as CAMPLUS_WINDOW_SAMPLES,
    SPEAKER_WINDOW_SHIFT_SAMPLES as CAMPLUS_WINDOW_SHIFT_SAMPLES,
    SpeakerEmbeddingWindow,
)

_LANGUAGES = frozenset({"auto", "zh", "en", "yue", "ja", "ko"})
_OUTPUT_LANGUAGES = _LANGUAGES - {"auto"}
_EMOTIONS = {
    "HAPPY": "happy",
    "SAD": "sad",
    "ANGRY": "angry",
    "NEUTRAL": "neutral",
    "FEARFUL": "fearful",
    "DISGUSTED": "disgusted",
    "SURPRISED": "surprised",
}
_AUDIO_EVENTS = {
    "Speech": "speech",
    "BGM": "bgm",
    "Applause": "applause",
    "Laughter": "laughter",
    "Cry": "cry",
    "Sneeze": "sneeze",
    "Breath": "breath",
    "Cough": "cough",
}
_EMOTION_TAGS = frozenset((*_EMOTIONS, "EMO_UNKNOWN"))
_AUDIO_EVENT_TAGS = frozenset((*_AUDIO_EVENTS, "Event_UNK"))
_CONTROL_TAGS = frozenset((*_LANGUAGES, "nospeech", "withitn", "woitn"))
_CONTROL_PREFIX = re.compile(
    r"\A<\|([^|<>]{1,64})\|>"
    r"<\|([^|<>]{1,64})\|>"
    r"<\|([^|<>]{1,64})\|>"
    r"<\|([^|<>]{1,64})\|>"
)
_UNKNOWN_TAG = re.compile(r"\A[A-Za-z][A-Za-z0-9_]{0,63}\Z")


class FunAsrAutoModel(Protocol):
    def generate(self, **kwargs: object) -> object: ...


class FunAsrCampPlusAdapter:
    def __init__(
        self,
        model: FunAsrAutoModel,
        *,
        inference_lane: InferenceLane,
    ) -> None:
        self._model = model
        self._inference_lane = inference_lane

    def embed_windows(
        self,
        pcm: np.ndarray,
    ) -> tuple[SpeakerEmbeddingWindow, ...]:
        if not _valid_campplus_pcm(pcm):
            raise PipelineError(
                "invalid_audio",
                "CAM++ input segment is invalid",
            )

        sample_ranges = _campplus_sample_ranges(len(pcm))
        embeddings = self.embed_exact_windows(
            tuple(
                pcm[start_sample:end_sample]
                for start_sample, end_sample in sample_ranges
            )
        )
        return tuple(
            SpeakerEmbeddingWindow(
                start_sample=start_sample,
                end_sample=end_sample,
                embedding=embedding,
            )
            for (start_sample, end_sample), embedding in zip(
                sample_ranges,
                embeddings,
                strict=True,
            )
        )

    def embed_exact_windows(
        self,
        pcms: tuple[np.ndarray, ...],
    ) -> tuple[np.ndarray, ...]:
        if (
            type(pcms) is not tuple
            or not 1 <= len(pcms) <= CAMPLUS_BATCH_MAX_WINDOWS
            or any(not _valid_exact_campplus_pcm(pcm) for pcm in pcms)
        ):
            raise PipelineError(
                "invalid_audio",
                "CAM++ input windows are invalid",
            )

        normalized = [
            _normalize_campplus_window(pcm, 0, len(pcm))
            for pcm in pcms
        ]
        raw_result = self._inference_lane.invoke(
            lambda: self._model.generate(
                input=normalized,
                batch_size=len(normalized),
            )
        )
        embeddings = _decode_campplus_embeddings(
            raw_result,
            expected_count=len(pcms),
        )
        return embeddings


class FunAsrSenseVoiceBatchAdapter:
    def __init__(
        self,
        model: FunAsrAutoModel,
        *,
        inference_lane: InferenceLane,
    ) -> None:
        self._model = model
        self._inference_lane = inference_lane

    def transcribe(self, pcm: np.ndarray) -> AsrResult:
        return self.transcribe_batch((pcm,), language="auto")[0]

    def transcribe_batch(
        self,
        pcms: tuple[np.ndarray, ...],
        *,
        language: str,
    ) -> tuple[AsrResult, ...]:
        if not pcms:
            return ()
        if language not in _LANGUAGES:
            raise PipelineError("invalid_audio", "ASR language is invalid")
        if any(not _valid_asr_pcm(pcm) for pcm in pcms):
            raise PipelineError("invalid_audio", "ASR input segment is invalid")

        normalized: list[np.ndarray] = []
        for pcm in pcms:
            item = pcm.astype(np.float32)
            item /= np.float32(32768.0)
            normalized.append(item)

        raw_result = self._inference_lane.invoke(
            lambda: self._model.generate(
                input=normalized,
                language=language,
                use_itn=True,
                batch_size=len(normalized),
                ban_emo_unk=False,
            )
        )
        return _decode_sensevoice_batch(
            raw_result,
            expected_count=len(normalized),
            language=language,
        )


class FunAsrStreamingVadAdapter:
    def __init__(
        self,
        model: FunAsrAutoModel,
        *,
        inference_lane: InferenceLane,
    ) -> None:
        self._model = model
        self._inference_lane = inference_lane

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
        result = self._inference_lane.invoke(
            lambda: self._model.generate(
                input=normalized,
                cache=cache,
                is_final=is_final,
                chunk_size=200,
            )
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


def _valid_asr_pcm(pcm: object) -> bool:
    return (
        isinstance(pcm, np.ndarray)
        and pcm.dtype == np.int16
        and pcm.ndim == 1
        and pcm.flags.c_contiguous
        and 1 <= len(pcm) <= DIRECT_MAX_SAMPLES
    )


def _valid_campplus_pcm(pcm: object) -> bool:
    return (
        isinstance(pcm, np.ndarray)
        and pcm.dtype == np.int16
        and pcm.ndim == 1
        and pcm.flags.c_contiguous
        and 1 <= len(pcm) <= DIRECT_MAX_SAMPLES
    )


def _valid_exact_campplus_pcm(pcm: object) -> bool:
    return (
        isinstance(pcm, np.ndarray)
        and pcm.dtype == np.int16
        and pcm.ndim == 1
        and pcm.flags.c_contiguous
        and 1 <= len(pcm) <= CAMPLUS_WINDOW_SAMPLES
    )


def _campplus_sample_ranges(sample_count: int) -> tuple[tuple[int, int], ...]:
    if sample_count <= CAMPLUS_WINDOW_SAMPLES:
        return ((0, sample_count),)

    final_start = sample_count - CAMPLUS_WINDOW_SAMPLES
    starts = list(
        range(
            0,
            final_start + 1,
            CAMPLUS_WINDOW_SHIFT_SAMPLES,
        )
    )
    if starts[-1] != final_start:
        starts.append(final_start)
    return tuple(
        (start_sample, start_sample + CAMPLUS_WINDOW_SAMPLES) for start_sample in starts
    )


def _normalize_campplus_window(
    pcm: np.ndarray,
    start_sample: int,
    end_sample: int,
) -> np.ndarray:
    window = np.zeros(CAMPLUS_WINDOW_SAMPLES, dtype=np.float32)
    real_sample_count = end_sample - start_sample
    window[:real_sample_count] = pcm[start_sample:end_sample]
    window /= np.float32(32768.0)
    return window


def _decode_campplus_embeddings(
    raw_result: object,
    *,
    expected_count: int,
) -> tuple[np.ndarray, ...]:
    if (
        not isinstance(raw_result, list)
        or len(raw_result) != 1
        or not isinstance(raw_result[0], dict)
    ):
        _raise_invalid_campplus_output()
    raw_embeddings = raw_result[0].get("spk_embedding")
    if not isinstance(raw_embeddings, torch.Tensor) or not torch.is_floating_point(
        raw_embeddings
    ):
        _raise_invalid_campplus_output()
    try:
        embeddings = (
            raw_embeddings.detach().to(device="cpu", dtype=torch.float32).numpy()
        )
    except (RuntimeError, TypeError):
        _raise_invalid_campplus_output()
    if (
        not isinstance(embeddings, np.ndarray)
        or embeddings.shape != (expected_count, CAMPLUS_EMBEDDING_DIMENSION)
        or not np.issubdtype(embeddings.dtype, np.number)
    ):
        _raise_invalid_campplus_output()

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    if not np.isfinite(embeddings).all():
        _raise_invalid_campplus_output()
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        _raise_invalid_campplus_output()
    normalized = embeddings / norms[:, np.newaxis]
    if not np.isfinite(normalized).all():
        _raise_invalid_campplus_output()
    decoded: list[np.ndarray] = []
    for embedding in normalized:
        owned_embedding = np.array(
            embedding,
            dtype=np.float32,
            order="C",
            copy=True,
        )
        owned_embedding.setflags(write=False)
        decoded.append(owned_embedding)
    return tuple(decoded)


def _raise_invalid_campplus_output() -> None:
    raise PipelineError(
        "invalid_model_output",
        "CAM++ model returned an invalid result",
    )


def _decode_sensevoice_batch(
    raw_result: object,
    *,
    expected_count: int,
    language: str,
) -> tuple[AsrResult, ...]:
    if (
        type(expected_count) is not int
        or expected_count < 0
        or language not in _LANGUAGES
        or not isinstance(raw_result, list)
        or len(raw_result) != expected_count
    ):
        _raise_invalid_sensevoice_output()

    decoded: list[AsrResult] = []
    for raw_item in raw_result:
        if not isinstance(raw_item, dict):
            _raise_invalid_sensevoice_output()
        raw_text = raw_item.get("text")
        if not isinstance(raw_text, str):
            _raise_invalid_sensevoice_output()
        decoded.append(_decode_sensevoice_text(raw_text, requested_language=language))
    return tuple(decoded)


def _decode_sensevoice_text(
    raw_text: str,
    *,
    requested_language: str,
) -> AsrResult:
    prefix = _CONTROL_PREFIX.match(raw_text)
    if prefix is None:
        _raise_invalid_sensevoice_output()
    raw_language, raw_emotion, raw_event, raw_itn = prefix.groups()
    body = raw_text[prefix.end() :]

    if "<|" in body:
        _raise_invalid_sensevoice_output()
    if raw_language == "nospeech":
        return _decode_nospeech(
            raw_emotion=raw_emotion,
            raw_event=raw_event,
            raw_itn=raw_itn,
            body=body,
        )
    if (
        raw_language not in _OUTPUT_LANGUAGES
        or requested_language != "auto"
        and raw_language != requested_language
        or raw_itn != "withitn"
    ):
        _raise_invalid_sensevoice_output()

    return AsrResult(
        text=body,
        language=raw_language,
        annotations=RichAnnotations(
            emotion=_decode_emotion(raw_emotion),
            audio_event=_decode_audio_event(raw_event),
        ),
    )


def _decode_nospeech(
    *,
    raw_emotion: str,
    raw_event: str,
    raw_itn: str,
    body: str,
) -> AsrResult:
    if (
        raw_emotion != "EMO_UNKNOWN"
        or raw_event != "Event_UNK"
        or raw_itn != "withitn"
        or body.strip() not in {"", "."}
    ):
        _raise_invalid_sensevoice_output()
    return AsrResult(
        text="",
        language=None,
        annotations=RichAnnotations(
            emotion=_unknown_label("emotion", raw_emotion),
            audio_event=_unknown_label("audio_event", raw_event),
        ),
    )


def _decode_emotion(raw_emotion: str) -> str:
    known = _EMOTIONS.get(raw_emotion)
    if known is not None:
        return known
    if raw_emotion in _AUDIO_EVENT_TAGS or raw_emotion in _CONTROL_TAGS:
        _raise_invalid_sensevoice_output()
    return _unknown_label("emotion", raw_emotion)


def _decode_audio_event(raw_event: str) -> str:
    known = _AUDIO_EVENTS.get(raw_event)
    if known is not None:
        return known
    if raw_event in _EMOTION_TAGS or raw_event in _CONTROL_TAGS:
        _raise_invalid_sensevoice_output()
    return _unknown_label("audio_event", raw_event)


def _unknown_label(slot: str, raw_tag: str) -> str:
    if _UNKNOWN_TAG.fullmatch(raw_tag) is None:
        _raise_invalid_sensevoice_output()
    return f"unknown:sensevoice:{slot}:{raw_tag}"


def _raise_invalid_sensevoice_output() -> None:
    raise PipelineError(
        "invalid_model_output",
        "SenseVoice model returned an invalid result",
    )
