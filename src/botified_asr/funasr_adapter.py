from __future__ import annotations

import re
from typing import Protocol

import numpy as np

from botified_asr.pipeline import (
    DIRECT_MAX_SAMPLES,
    AsrResult,
    PipelineError,
    RichAnnotations,
    VadMarker,
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


class FunAsrSenseVoiceBatchAdapter:
    def __init__(self, model: FunAsrAutoModel) -> None:
        self._model = model

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

        raw_result = self._model.generate(
            input=normalized,
            language=language,
            use_itn=True,
            batch_size=len(normalized),
            ban_emo_unk=False,
        )
        return _decode_sensevoice_batch(
            raw_result,
            expected_count=len(normalized),
            language=language,
        )


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


def _valid_asr_pcm(pcm: object) -> bool:
    return (
        isinstance(pcm, np.ndarray)
        and pcm.dtype == np.int16
        and pcm.ndim == 1
        and pcm.flags.c_contiguous
        and 1 <= len(pcm) <= DIRECT_MAX_SAMPLES
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
