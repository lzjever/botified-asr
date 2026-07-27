from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from botified_asr.funasr_adapter import (
    FunAsrAutoModel,
    FunAsrSenseVoiceBatchAdapter,
    FunAsrStreamingVadAdapter,
)
from botified_asr.model_artifacts import (
    FSMN_VAD_SPEC,
    SENSEVOICE_SPEC,
    ModelArtifactSpec,
    ResolvedModelSnapshot,
)

_LOAD_ERROR_MESSAGE = "FunASR model bundle could not be loaded"
_WARMUP_SAMPLES = 16_000


class FunAsrModelLoadError(RuntimeError):
    pass


class _ModelArtifactResolver(Protocol):
    def resolve(self, spec: ModelArtifactSpec) -> ResolvedModelSnapshot: ...


AutoModelFactory = Callable[..., FunAsrAutoModel]


@dataclass(frozen=True)
class FunAsrModelBundle:
    asr: FunAsrSenseVoiceBatchAdapter
    vad: FunAsrStreamingVadAdapter


def load_funasr_model_bundle(
    resolver: _ModelArtifactResolver,
    *,
    device: str,
    auto_model_factory: AutoModelFactory | None = None,
) -> FunAsrModelBundle:
    sensevoice_snapshot = resolver.resolve(SENSEVOICE_SPEC)
    vad_snapshot = resolver.resolve(FSMN_VAD_SPEC)

    try:
        if auto_model_factory is None:
            from funasr import AutoModel

            auto_model_factory = AutoModel

        sensevoice_root = str(sensevoice_snapshot.root.absolute())
        vad_root = str(vad_snapshot.root.absolute())
        asr_model = auto_model_factory(
            model=sensevoice_root,
            model_path=sensevoice_root,
            hub="hf",
            device=device,
            disable_update=True,
            trust_remote_code=False,
            disable_pbar=True,
            disable_log=True,
        )
        vad_model = auto_model_factory(
            model=vad_root,
            model_path=vad_root,
            hub="hf",
            device=device,
            disable_update=True,
            trust_remote_code=False,
            disable_pbar=True,
            disable_log=True,
            max_single_segment_time=29_790,
        )

        asr = FunAsrSenseVoiceBatchAdapter(asr_model)
        vad = FunAsrStreamingVadAdapter(vad_model)
        silence = np.zeros(_WARMUP_SAMPLES, dtype=np.int16)
        asr.transcribe(silence)
        markers = vad.generate(silence, cache={}, is_final=True)
    except Exception as error:
        raise FunAsrModelLoadError(_LOAD_ERROR_MESSAGE) from error

    if markers:
        raise FunAsrModelLoadError(_LOAD_ERROR_MESSAGE)
    return FunAsrModelBundle(asr=asr, vad=vad)
