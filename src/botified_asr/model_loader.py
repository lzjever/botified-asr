from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from botified_asr.funasr_adapter import (
    FunAsrAutoModel,
    FunAsrCampPlusAdapter,
    FunAsrSenseVoiceBatchAdapter,
    FunAsrStreamingVadAdapter,
)
from botified_asr.inference import MAX_INFERENCE_LANES, SerialInferenceLane
from botified_asr.model_artifacts import (
    CAMPLUS_SPEC,
    FSMN_VAD_SPEC,
    SENSEVOICE_SPEC,
    ModelArtifactSpec,
    ResolvedModelSnapshot,
    model_snapshot_manifest_digest,
)
from botified_asr.result_artifact import RESULT_ENVELOPE_VERSION
from botified_asr.speaker_snapshot import SPEAKER_SNAPSHOT_WIRE_VERSION
from botified_asr.speakers import (
    SPEAKER_DOWNMIX_POLICY_VERSION,
    SPEAKER_EMBEDDING_DIMENSION,
    SPEAKER_ENROLLMENT_AGGREGATION_POLICY_VERSION,
    SPEAKER_NORMALIZATION_POLICY_VERSION,
    SPEAKER_PADDING_POLICY_VERSION,
    SPEAKER_SAMPLE_RATE,
    SPEAKER_WINDOW_MAX_SAMPLES,
    SPEAKER_WINDOW_SHIFT_SAMPLES,
    SpeakerEmbeddingPolicy,
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
    speaker: FunAsrCampPlusAdapter
    speaker_embedding_policy: SpeakerEmbeddingPolicy
    processor_fingerprint: str


@dataclass(frozen=True)
class FunAsrModelPool:
    bundles: tuple[FunAsrModelBundle, ...]

    def __post_init__(self) -> None:
        if not self.bundles:
            raise ValueError("FunASR model pool must contain at least one bundle")

    @property
    def speaker_embedding_policy(self) -> SpeakerEmbeddingPolicy:
        return self.bundles[0].speaker_embedding_policy

    @property
    def processor_fingerprint(self) -> str:
        return self.bundles[0].processor_fingerprint


def _build_processor_fingerprint(
    sensevoice_snapshot: ResolvedModelSnapshot,
    vad_snapshot: ResolvedModelSnapshot,
    campplus_snapshot: ResolvedModelSnapshot,
) -> str:
    manifest = {
        "model_snapshot_manifest_digests": {
            "campplus": model_snapshot_manifest_digest(campplus_snapshot),
            "fsmn_vad": model_snapshot_manifest_digest(vad_snapshot),
            "sensevoice": model_snapshot_manifest_digest(sensevoice_snapshot),
        },
        "processor_compatibility_version": 1,
        "result_envelope_version": RESULT_ENVELOPE_VERSION,
        "speaker_snapshot_wire_version": SPEAKER_SNAPSHOT_WIRE_VERSION,
        "version": 1,
    }
    canonical = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_funasr_model_bundle(
    resolver: _ModelArtifactResolver,
    *,
    device: str,
    auto_model_factory: AutoModelFactory | None = None,
) -> FunAsrModelBundle:
    return load_funasr_model_pool(
        resolver,
        device=device,
        inference_lanes=1,
        auto_model_factory=auto_model_factory,
    ).bundles[0]


def load_funasr_model_pool(
    resolver: _ModelArtifactResolver,
    *,
    device: str,
    inference_lanes: int,
    auto_model_factory: AutoModelFactory | None = None,
) -> FunAsrModelPool:
    if type(inference_lanes) is not int:
        raise TypeError("inference lanes must be an integer")
    if not 1 <= inference_lanes <= MAX_INFERENCE_LANES:
        raise ValueError(
            f"inference lanes must be from 1 to {MAX_INFERENCE_LANES}"
        )
    sensevoice_snapshot = resolver.resolve(SENSEVOICE_SPEC)
    vad_snapshot = resolver.resolve(FSMN_VAD_SPEC)
    speaker_snapshot = resolver.resolve(CAMPLUS_SPEC)

    speaker_embedding_policy = SpeakerEmbeddingPolicy(
        model_id=speaker_snapshot.spec.model_id,
        model_revision=speaker_snapshot.spec.revision,
        embedding_dimension=SPEAKER_EMBEDDING_DIMENSION,
        sample_rate=SPEAKER_SAMPLE_RATE,
        downmix_policy_version=SPEAKER_DOWNMIX_POLICY_VERSION,
        window_samples=SPEAKER_WINDOW_MAX_SAMPLES,
        window_shift_samples=SPEAKER_WINDOW_SHIFT_SAMPLES,
        padding_policy_version=SPEAKER_PADDING_POLICY_VERSION,
        normalization_policy_version=SPEAKER_NORMALIZATION_POLICY_VERSION,
        enrollment_aggregation_policy_version=(
            SPEAKER_ENROLLMENT_AGGREGATION_POLICY_VERSION
        ),
    )
    processor_fingerprint = _build_processor_fingerprint(
        sensevoice_snapshot,
        vad_snapshot,
        speaker_snapshot,
    )
    bundles: list[FunAsrModelBundle] = []
    try:
        if auto_model_factory is None:
            from funasr import AutoModel

            auto_model_factory = AutoModel

        sensevoice_root = str(sensevoice_snapshot.root.absolute())
        vad_root = str(vad_snapshot.root.absolute())
        speaker_root = str(speaker_snapshot.root.absolute())
        for _ in range(inference_lanes):
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
            speaker_model = auto_model_factory(
                model=speaker_root,
                model_path=speaker_root,
                hub="hf",
                device=device,
                disable_update=True,
                trust_remote_code=False,
                disable_pbar=True,
                disable_log=True,
            )

            inference_lane = SerialInferenceLane()
            asr = FunAsrSenseVoiceBatchAdapter(
                asr_model,
                inference_lane=inference_lane,
            )
            vad = FunAsrStreamingVadAdapter(
                vad_model,
                inference_lane=inference_lane,
            )
            speaker = FunAsrCampPlusAdapter(
                speaker_model,
                inference_lane=inference_lane,
            )
            silence = np.zeros(_WARMUP_SAMPLES, dtype=np.int16)
            asr.transcribe(silence)
            markers = vad.generate(silence, cache={}, is_final=True)
            if markers:
                raise FunAsrModelLoadError(_LOAD_ERROR_MESSAGE)
            speaker.embed_windows(np.zeros(24_000, dtype=np.int16))
            bundles.append(
                FunAsrModelBundle(
                    asr=asr,
                    vad=vad,
                    speaker=speaker,
                    speaker_embedding_policy=speaker_embedding_policy,
                    processor_fingerprint=processor_fingerprint,
                )
            )
    except FunAsrModelLoadError:
        raise
    except Exception as error:
        raise FunAsrModelLoadError(_LOAD_ERROR_MESSAGE) from error

    return FunAsrModelPool(
        bundles=tuple(bundles),
    )
