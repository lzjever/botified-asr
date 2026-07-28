from __future__ import annotations

import builtins
import importlib
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import numpy as np
import pytest
import torch

from botified_asr import funasr_adapter, model_artifacts, speakers
from botified_asr.funasr_adapter import (
    FunAsrSenseVoiceBatchAdapter,
    FunAsrStreamingVadAdapter,
)
from botified_asr.inference import MAX_INFERENCE_LANES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cpu"
ASR_NOSPEECH = "<|nospeech|><|EMO_UNKNOWN|><|Event_UNK|><|withitn|>"
LOAD_ERROR_MESSAGE = "FunASR model bundle could not be loaded"
PROCESSOR_FINGERPRINT = (
    "15912cff0389a1a0e38828063e4de2018e4787c53ee60a25ca74f7bedfbdd633"
)
CAMPLUS_SPEC = getattr(model_artifacts, "CAMPLUS_SPEC", object())


def _model_loader():
    return importlib.import_module("botified_asr.model_loader")


class RecordingResolver:
    def __init__(
        self,
        snapshots: dict[
            model_artifacts.ModelArtifactSpec,
            model_artifacts.ResolvedModelSnapshot,
        ],
        events: list[object],
        *,
        failure_index: int | None = None,
        failure: Exception | None = None,
    ) -> None:
        self._snapshots = snapshots
        self._events = events
        self._failure_index = failure_index
        self._failure = failure
        self.calls: list[model_artifacts.ModelArtifactSpec] = []

    def resolve(
        self,
        spec: model_artifacts.ModelArtifactSpec,
    ) -> model_artifacts.ResolvedModelSnapshot:
        self.calls.append(spec)
        self._events.append(("resolve", spec))
        if len(self.calls) - 1 == self._failure_index:
            assert self._failure is not None
            raise self._failure
        return self._snapshots[spec]


class RecordingAutoModel:
    def __init__(
        self,
        role: str,
        events: list[object],
        *,
        failure: Exception | None = None,
        vad_markers: list[list[int]] | None = None,
    ) -> None:
        self.role = role
        self._events = events
        self._failure = failure
        self._vad_markers = vad_markers
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        self._events.append(("warmup", self.role))
        if self._failure is not None:
            raise self._failure
        if self.role == "asr":
            return [{"text": ASR_NOSPEECH}]
        if self.role == "vad":
            return [{"value": self._vad_markers or []}]
        return [{"spk_embedding": torch.ones((1, 192), dtype=torch.float32)}]


class RecordingAutoModelFactory:
    def __init__(
        self,
        events: list[object],
        *,
        construct_failure_role: str | None = None,
        warmup_failure_role: str | None = None,
        warmup_failure_index: int | None = None,
        failure: Exception | None = None,
        vad_markers: list[list[int]] | None = None,
    ) -> None:
        self._events = events
        self._construct_failure_role = construct_failure_role
        self._warmup_failure_role = warmup_failure_role
        self._warmup_failure_index = warmup_failure_index
        self._failure = failure
        self._vad_markers = vad_markers
        self.calls: list[dict[str, object]] = []
        self.models: list[RecordingAutoModel] = []

    def __call__(self, **kwargs: object) -> RecordingAutoModel:
        index = len(self.calls)
        role = ("asr", "vad", "speaker")[index % 3]
        self.calls.append(kwargs)
        self._events.append(("construct", role))
        if role == self._construct_failure_role:
            assert self._failure is not None
            raise self._failure
        model = RecordingAutoModel(
            role,
            self._events,
            failure=(
                self._failure
                if (
                    role == self._warmup_failure_role
                    or index == self._warmup_failure_index
                )
                else None
            ),
            vad_markers=self._vad_markers if role == "vad" else None,
        )
        self.models.append(model)
        return model


class RecordingInferenceLane:
    def __init__(self) -> None:
        self.operations: list[object] = []

    def invoke(self, operation, /):
        self.operations.append(operation)
        return operation()


def _snapshots(
    tmp_path: Path,
) -> dict[
    model_artifacts.ModelArtifactSpec,
    model_artifacts.ResolvedModelSnapshot,
]:
    sensevoice_root = tmp_path / "verified SenseVoice root"
    vad_root = tmp_path / "verified FSMN root"
    speaker_root = tmp_path / "verified CAM++ root"
    sensevoice_root.mkdir()
    vad_root.mkdir()
    speaker_root.mkdir()
    assert sensevoice_root.is_absolute()
    assert vad_root.is_absolute()
    assert speaker_root.is_absolute()
    return {
        model_artifacts.SENSEVOICE_SPEC: model_artifacts.ResolvedModelSnapshot(
            spec=model_artifacts.SENSEVOICE_SPEC,
            root=sensevoice_root,
        ),
        model_artifacts.FSMN_VAD_SPEC: model_artifacts.ResolvedModelSnapshot(
            spec=model_artifacts.FSMN_VAD_SPEC,
            root=vad_root,
        ),
        CAMPLUS_SPEC: model_artifacts.ResolvedModelSnapshot(
            spec=CAMPLUS_SPEC,
            root=speaker_root,
        ),
    }


def _fingerprint_snapshots(root: Path) -> tuple:
    return tuple(
        model_artifacts.ResolvedModelSnapshot(spec, root / name)
        for spec, name in (
            (model_artifacts.SENSEVOICE_SPEC, "sensevoice"),
            (model_artifacts.FSMN_VAD_SPEC, "vad"),
            (CAMPLUS_SPEC, "speaker"),
        )
    )


def test_processor_fingerprint_is_exact_and_root_independent(
    tmp_path: Path,
) -> None:
    loader = _model_loader()
    snapshots = _fingerprint_snapshots(tmp_path / "missing-a")
    other_roots = _fingerprint_snapshots(tmp_path / "missing-b")

    assert loader._build_processor_fingerprint(*snapshots) == (
        PROCESSOR_FINGERPRINT
    )
    assert loader._build_processor_fingerprint(*other_roots) == (
        PROCESSOR_FINGERPRINT
    )


def _expected_kwargs(
    root: Path,
    *,
    vad: bool = False,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "model": str(root),
        "model_path": str(root),
        "hub": "hf",
        "device": DEVICE,
        "disable_update": True,
        "trust_remote_code": False,
        "disable_pbar": True,
        "disable_log": True,
    }
    if vad:
        kwargs["max_single_segment_time"] = 29_790
    return kwargs


def test_default_factory_import_is_lazy_and_resolver_failure_does_not_import_funasr() -> (
    None
):
    script = """
import builtins

from botified_asr import model_artifacts

real_import = builtins.__import__
funasr_imports = []

def guarded_import(name, *args, **kwargs):
    if name == "funasr" or name.startswith("funasr."):
        funasr_imports.append(name)
        raise AssertionError("funasr was imported before verified resolution")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from botified_asr import model_loader

failure = model_artifacts.ModelArtifactUnavailable("missing")

class FailingResolver:
    def resolve(self, _spec):
        raise failure

try:
    model_loader.load_funasr_model_bundle(FailingResolver(), device="cpu")
except model_artifacts.ModelArtifactUnavailable as error:
    assert error is failure
else:
    raise AssertionError("artifact failure was not propagated")

assert funasr_imports == []
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"{result.stdout}{result.stderr}"


def test_loader_resolves_three_pinned_specs_before_exact_distinct_construction_and_warmup(
    tmp_path: Path,
) -> None:
    loader = _model_loader()
    events: list[object] = []
    snapshots = _snapshots(tmp_path)
    resolver = RecordingResolver(snapshots, events)
    factory = RecordingAutoModelFactory(events)

    bundle = loader.load_funasr_model_bundle(
        resolver,
        device=DEVICE,
        auto_model_factory=factory,
    )
    events.append(("observed", "return"))

    assert resolver.calls == [
        model_artifacts.SENSEVOICE_SPEC,
        model_artifacts.FSMN_VAD_SPEC,
        CAMPLUS_SPEC,
    ]
    assert factory.calls == [
        _expected_kwargs(snapshots[model_artifacts.SENSEVOICE_SPEC].root),
        _expected_kwargs(
            snapshots[model_artifacts.FSMN_VAD_SPEC].root,
            vad=True,
        ),
        _expected_kwargs(snapshots[CAMPLUS_SPEC].root),
    ]
    assert events == [
        ("resolve", model_artifacts.SENSEVOICE_SPEC),
        ("resolve", model_artifacts.FSMN_VAD_SPEC),
        ("resolve", CAMPLUS_SPEC),
        ("construct", "asr"),
        ("construct", "vad"),
        ("construct", "speaker"),
        ("warmup", "asr"),
        ("warmup", "vad"),
        ("warmup", "speaker"),
        ("observed", "return"),
    ]
    assert len(factory.models) == 3
    assert len({id(model) for model in factory.models}) == 3
    assert isinstance(bundle, loader.FunAsrModelBundle)
    assert isinstance(bundle.asr, FunAsrSenseVoiceBatchAdapter)
    assert isinstance(bundle.vad, FunAsrStreamingVadAdapter)
    assert isinstance(bundle.speaker, funasr_adapter.FunAsrCampPlusAdapter)
    assert len({id(bundle.asr), id(bundle.vad), id(bundle.speaker)}) == 3
    expected_speaker_policy = speakers.SpeakerEmbeddingPolicy(
        model_id=snapshots[CAMPLUS_SPEC].spec.model_id,
        model_revision=snapshots[CAMPLUS_SPEC].spec.revision,
        embedding_dimension=speakers.SPEAKER_EMBEDDING_DIMENSION,
        sample_rate=16_000,
        downmix_policy_version=speakers.SPEAKER_DOWNMIX_POLICY_VERSION,
        window_samples=speakers.SPEAKER_WINDOW_MAX_SAMPLES,
        window_shift_samples=speakers.SPEAKER_WINDOW_SHIFT_SAMPLES,
        padding_policy_version=speakers.SPEAKER_PADDING_POLICY_VERSION,
        normalization_policy_version=(speakers.SPEAKER_NORMALIZATION_POLICY_VERSION),
        enrollment_aggregation_policy_version=(
            speakers.SPEAKER_ENROLLMENT_AGGREGATION_POLICY_VERSION
        ),
    )
    assert bundle.speaker_embedding_policy == expected_speaker_policy
    assert bundle.speaker_embedding_policy is not expected_speaker_policy
    assert bundle.processor_fingerprint == PROCESSOR_FINGERPRINT
    assert not hasattr(bundle.speaker_embedding_policy, "threshold")
    assert not hasattr(bundle.speaker_embedding_policy, "top_two_margin")
    with pytest.raises(FrozenInstanceError):
        bundle.asr = bundle.asr

    asr_call = factory.models[0].calls
    vad_call = factory.models[1].calls
    speaker_call = factory.models[2].calls
    assert len(asr_call) == 1
    assert len(vad_call) == 1
    assert len(speaker_call) == 1
    assert set(asr_call[0]) == {
        "input",
        "language",
        "use_itn",
        "batch_size",
        "ban_emo_unk",
    }
    assert asr_call[0]["language"] == "auto"
    assert asr_call[0]["use_itn"] is True
    assert asr_call[0]["batch_size"] == 1
    assert asr_call[0]["ban_emo_unk"] is False
    asr_inputs = asr_call[0]["input"]
    assert isinstance(asr_inputs, list)
    assert len(asr_inputs) == 1
    _assert_normalized_one_second_silence(asr_inputs[0])

    assert set(vad_call[0]) == {
        "input",
        "cache",
        "is_final",
        "chunk_size",
    }
    _assert_normalized_one_second_silence(vad_call[0]["input"])
    assert vad_call[0]["cache"] == {}
    assert vad_call[0]["is_final"] is True
    assert vad_call[0]["chunk_size"] == 200
    assert set(speaker_call[0]) == {"input", "batch_size"}
    assert speaker_call[0]["batch_size"] == 1
    speaker_inputs = speaker_call[0]["input"]
    assert isinstance(speaker_inputs, list)
    assert len(speaker_inputs) == 1
    _assert_normalized_speaker_silence(speaker_inputs[0])


def test_loader_shares_one_lane_across_all_three_adapters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _model_loader()
    events: list[object] = []
    lanes: list[RecordingInferenceLane] = []

    def lane_factory() -> RecordingInferenceLane:
        lane = RecordingInferenceLane()
        lanes.append(lane)
        return lane

    monkeypatch.setattr(loader, "SerialInferenceLane", lane_factory)
    factory = RecordingAutoModelFactory(events)
    bundle = loader.load_funasr_model_bundle(
        RecordingResolver(_snapshots(tmp_path), events),
        device=DEVICE,
        auto_model_factory=factory,
    )

    assert len(lanes) == 1
    assert len(lanes[0].operations) == 3

    bundle.asr.transcribe(np.zeros(16_000, dtype=np.int16))
    bundle.vad.generate(
        np.zeros(3_200, dtype=np.int16),
        cache={},
        is_final=True,
    )
    bundle.speaker.embed_windows(np.zeros(24_000, dtype=np.int16))

    assert len(lanes[0].operations) == 6
    assert len(factory.models[0].calls) == 2
    assert len(factory.models[1].calls) == 2
    assert len(factory.models[2].calls) == 2


def test_pool_resolves_once_and_builds_two_fully_independent_warmed_lanes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _model_loader()
    events: list[object] = []
    snapshots = _snapshots(tmp_path)
    resolver = RecordingResolver(snapshots, events)
    factory = RecordingAutoModelFactory(events)
    lanes: list[RecordingInferenceLane] = []

    def lane_factory() -> RecordingInferenceLane:
        lane = RecordingInferenceLane()
        lanes.append(lane)
        return lane

    monkeypatch.setattr(loader, "SerialInferenceLane", lane_factory)

    pool = loader.load_funasr_model_pool(
        resolver,
        device=DEVICE,
        inference_lanes=MAX_INFERENCE_LANES,
        auto_model_factory=factory,
    )

    assert isinstance(pool, loader.FunAsrModelPool)
    assert resolver.calls == [
        model_artifacts.SENSEVOICE_SPEC,
        model_artifacts.FSMN_VAD_SPEC,
        CAMPLUS_SPEC,
    ]
    assert len(pool.bundles) == 2
    assert len(factory.models) == 6
    assert len({id(model) for model in factory.models}) == 6
    assert len(lanes) == 2
    assert lanes[0] is not lanes[1]
    for index, bundle in enumerate(pool.bundles):
        lane = lanes[index]
        assert bundle.asr._inference_lane is lane
        assert bundle.vad._inference_lane is lane
        assert bundle.speaker._inference_lane is lane
        assert len(lane.operations) == 3
        assert bundle.speaker_embedding_policy is pool.speaker_embedding_policy
        assert bundle.processor_fingerprint == pool.processor_fingerprint
    assert pool.processor_fingerprint == loader._build_processor_fingerprint(
        snapshots[model_artifacts.SENSEVOICE_SPEC],
        snapshots[model_artifacts.FSMN_VAD_SPEC],
        snapshots[CAMPLUS_SPEC],
    )
    assert events == [
        ("resolve", model_artifacts.SENSEVOICE_SPEC),
        ("resolve", model_artifacts.FSMN_VAD_SPEC),
        ("resolve", CAMPLUS_SPEC),
        ("construct", "asr"),
        ("construct", "vad"),
        ("construct", "speaker"),
        ("warmup", "asr"),
        ("warmup", "vad"),
        ("warmup", "speaker"),
        ("construct", "asr"),
        ("construct", "vad"),
        ("construct", "speaker"),
        ("warmup", "asr"),
        ("warmup", "vad"),
        ("warmup", "speaker"),
    ]


def test_pool_stores_only_non_empty_bundles_and_derives_metadata(
    tmp_path: Path,
) -> None:
    loader = _model_loader()
    events: list[object] = []
    pool = loader.load_funasr_model_pool(
        RecordingResolver(_snapshots(tmp_path), events),
        device=DEVICE,
        inference_lanes=MAX_INFERENCE_LANES,
        auto_model_factory=RecordingAutoModelFactory(events),
    )

    assert tuple(field.name for field in fields(pool)) == ("bundles",)
    assert vars(pool) == {"bundles": pool.bundles}
    assert (
        pool.speaker_embedding_policy
        is pool.bundles[0].speaker_embedding_policy
    )
    assert pool.processor_fingerprint == pool.bundles[0].processor_fingerprint

    with pytest.raises(ValueError, match="at least one bundle"):
        loader.FunAsrModelPool(bundles=())


def test_pool_later_lane_warmup_failure_is_fail_closed(
    tmp_path: Path,
) -> None:
    loader = _model_loader()
    events: list[object] = []
    failure = RuntimeError("second lane ASR warmup failed")
    factory = RecordingAutoModelFactory(
        events,
        warmup_failure_index=3,
        failure=failure,
    )
    not_returned = object()
    result: object = not_returned

    with pytest.raises(loader.FunAsrModelLoadError) as caught:
        result = loader.load_funasr_model_pool(
            RecordingResolver(_snapshots(tmp_path), events),
            device=DEVICE,
            inference_lanes=MAX_INFERENCE_LANES,
            auto_model_factory=factory,
        )

    assert result is not_returned
    assert caught.value.__cause__ is failure
    assert len(factory.models) == 6
    assert events == [
        ("resolve", model_artifacts.SENSEVOICE_SPEC),
        ("resolve", model_artifacts.FSMN_VAD_SPEC),
        ("resolve", CAMPLUS_SPEC),
        ("construct", "asr"),
        ("construct", "vad"),
        ("construct", "speaker"),
        ("warmup", "asr"),
        ("warmup", "vad"),
        ("warmup", "speaker"),
        ("construct", "asr"),
        ("construct", "vad"),
        ("construct", "speaker"),
        ("warmup", "asr"),
    ]


def test_relative_snapshot_roots_are_passed_as_equal_absolute_model_paths() -> None:
    loader = _model_loader()
    events: list[object] = []
    sensevoice_root = Path("relative verified SenseVoice root")
    vad_root = Path("relative verified FSMN root")
    speaker_root = Path("relative verified CAM++ root")
    assert not sensevoice_root.is_absolute()
    assert not vad_root.is_absolute()
    assert not speaker_root.is_absolute()
    snapshots = {
        model_artifacts.SENSEVOICE_SPEC: model_artifacts.ResolvedModelSnapshot(
            spec=model_artifacts.SENSEVOICE_SPEC,
            root=sensevoice_root,
        ),
        model_artifacts.FSMN_VAD_SPEC: model_artifacts.ResolvedModelSnapshot(
            spec=model_artifacts.FSMN_VAD_SPEC,
            root=vad_root,
        ),
        CAMPLUS_SPEC: model_artifacts.ResolvedModelSnapshot(
            spec=CAMPLUS_SPEC,
            root=speaker_root,
        ),
    }
    factory = RecordingAutoModelFactory(events)

    loader.load_funasr_model_bundle(
        RecordingResolver(snapshots, events),
        device=DEVICE,
        auto_model_factory=factory,
    )

    assert factory.calls == [
        _expected_kwargs(sensevoice_root.absolute()),
        _expected_kwargs(vad_root.absolute(), vad=True),
        _expected_kwargs(speaker_root.absolute()),
    ]
    for kwargs in factory.calls:
        assert kwargs["model"] == kwargs["model_path"]
        assert Path(kwargs["model"]).is_absolute()


def _assert_normalized_one_second_silence(value: object) -> None:
    assert isinstance(value, np.ndarray)
    assert value.dtype == np.float32
    assert value.shape == (16_000,)
    assert value.flags.c_contiguous
    assert np.count_nonzero(value) == 0


def _assert_normalized_speaker_silence(value: object) -> None:
    assert isinstance(value, np.ndarray)
    assert value.dtype == np.float32
    assert value.shape == (24_000,)
    assert value.flags.c_contiguous
    assert np.count_nonzero(value) == 0


@pytest.mark.parametrize(
    "error_type",
    (
        model_artifacts.ModelArtifactUnavailable,
        model_artifacts.ModelArtifactIntegrityError,
    ),
)
def test_second_artifact_resolution_error_is_unchanged_and_factory_is_not_called(
    tmp_path: Path,
    error_type: type[model_artifacts.ModelArtifactError],
) -> None:
    loader = _model_loader()
    events: list[object] = []
    failure = error_type("artifact failure")
    resolver = RecordingResolver(
        _snapshots(tmp_path),
        events,
        failure_index=1,
        failure=failure,
    )
    factory = RecordingAutoModelFactory(events)

    with pytest.raises(error_type) as caught:
        loader.load_funasr_model_bundle(
            resolver,
            device=DEVICE,
            auto_model_factory=factory,
        )

    assert caught.value is failure
    assert factory.calls == []
    assert resolver.calls == [
        model_artifacts.SENSEVOICE_SPEC,
        model_artifacts.FSMN_VAD_SPEC,
    ]


def test_third_artifact_resolution_error_is_unchanged_and_factory_is_not_called(
    tmp_path: Path,
) -> None:
    loader = _model_loader()
    events: list[object] = []
    failure = model_artifacts.ModelArtifactUnavailable("CAM++ artifact failure")
    resolver = RecordingResolver(
        _snapshots(tmp_path),
        events,
        failure_index=2,
        failure=failure,
    )
    factory = RecordingAutoModelFactory(events)

    with pytest.raises(model_artifacts.ModelArtifactUnavailable) as caught:
        loader.load_funasr_model_bundle(
            resolver,
            device=DEVICE,
            auto_model_factory=factory,
        )

    assert caught.value is failure
    assert factory.calls == []
    assert resolver.calls == [
        model_artifacts.SENSEVOICE_SPEC,
        model_artifacts.FSMN_VAD_SPEC,
        CAMPLUS_SPEC,
    ]


def test_default_factory_import_failure_is_stably_wrapped_after_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _model_loader()
    events: list[object] = []
    resolver = RecordingResolver(_snapshots(tmp_path), events)
    failure = ImportError("funasr import failed")
    real_import = builtins.__import__

    def fail_funasr_import(name, *args, **kwargs):
        if name == "funasr" or name.startswith("funasr."):
            raise failure
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_funasr_import)

    with pytest.raises(loader.FunAsrModelLoadError) as caught:
        loader.load_funasr_model_bundle(resolver, device=DEVICE)

    assert type(caught.value) is loader.FunAsrModelLoadError
    assert str(caught.value) == LOAD_ERROR_MESSAGE
    assert caught.value.__cause__ is failure
    assert resolver.calls == [
        model_artifacts.SENSEVOICE_SPEC,
        model_artifacts.FSMN_VAD_SPEC,
        CAMPLUS_SPEC,
    ]
    assert events == [
        ("resolve", model_artifacts.SENSEVOICE_SPEC),
        ("resolve", model_artifacts.FSMN_VAD_SPEC),
        ("resolve", CAMPLUS_SPEC),
    ]


def test_non_empty_vad_warmup_markers_fail_with_the_stable_load_error(
    tmp_path: Path,
) -> None:
    loader = _model_loader()
    events: list[object] = []
    factory = RecordingAutoModelFactory(
        events,
        vad_markers=[[0, 1]],
    )
    not_returned = object()
    result: object = not_returned

    with pytest.raises(loader.FunAsrModelLoadError) as caught:
        result = loader.load_funasr_model_bundle(
            RecordingResolver(_snapshots(tmp_path), events),
            device=DEVICE,
            auto_model_factory=factory,
        )

    assert result is not_returned
    assert type(caught.value) is loader.FunAsrModelLoadError
    assert str(caught.value) == LOAD_ERROR_MESSAGE
    assert caught.value.__cause__ is None
    assert events == [
        ("resolve", model_artifacts.SENSEVOICE_SPEC),
        ("resolve", model_artifacts.FSMN_VAD_SPEC),
        ("resolve", CAMPLUS_SPEC),
        ("construct", "asr"),
        ("construct", "vad"),
        ("construct", "speaker"),
        ("warmup", "asr"),
        ("warmup", "vad"),
    ]


@pytest.mark.parametrize(
    ("failure_stage", "expected_events"),
    (
        (
            "construct_asr",
            (
                ("resolve", model_artifacts.SENSEVOICE_SPEC),
                ("resolve", model_artifacts.FSMN_VAD_SPEC),
                ("resolve", CAMPLUS_SPEC),
                ("construct", "asr"),
            ),
        ),
        (
            "construct_vad",
            (
                ("resolve", model_artifacts.SENSEVOICE_SPEC),
                ("resolve", model_artifacts.FSMN_VAD_SPEC),
                ("resolve", CAMPLUS_SPEC),
                ("construct", "asr"),
                ("construct", "vad"),
            ),
        ),
        (
            "construct_speaker",
            (
                ("resolve", model_artifacts.SENSEVOICE_SPEC),
                ("resolve", model_artifacts.FSMN_VAD_SPEC),
                ("resolve", CAMPLUS_SPEC),
                ("construct", "asr"),
                ("construct", "vad"),
                ("construct", "speaker"),
            ),
        ),
        (
            "warmup_asr",
            (
                ("resolve", model_artifacts.SENSEVOICE_SPEC),
                ("resolve", model_artifacts.FSMN_VAD_SPEC),
                ("resolve", CAMPLUS_SPEC),
                ("construct", "asr"),
                ("construct", "vad"),
                ("construct", "speaker"),
                ("warmup", "asr"),
            ),
        ),
        (
            "warmup_vad",
            (
                ("resolve", model_artifacts.SENSEVOICE_SPEC),
                ("resolve", model_artifacts.FSMN_VAD_SPEC),
                ("resolve", CAMPLUS_SPEC),
                ("construct", "asr"),
                ("construct", "vad"),
                ("construct", "speaker"),
                ("warmup", "asr"),
                ("warmup", "vad"),
            ),
        ),
        (
            "warmup_speaker",
            (
                ("resolve", model_artifacts.SENSEVOICE_SPEC),
                ("resolve", model_artifacts.FSMN_VAD_SPEC),
                ("resolve", CAMPLUS_SPEC),
                ("construct", "asr"),
                ("construct", "vad"),
                ("construct", "speaker"),
                ("warmup", "asr"),
                ("warmup", "vad"),
                ("warmup", "speaker"),
            ),
        ),
    ),
)
def test_construct_and_warmup_failures_are_stably_wrapped_without_later_stages(
    tmp_path: Path,
    failure_stage: str,
    expected_events: tuple[object, ...],
) -> None:
    loader = _model_loader()
    events: list[object] = []
    failure = RuntimeError(failure_stage)
    factory = RecordingAutoModelFactory(
        events,
        construct_failure_role=(
            failure_stage.removeprefix("construct_")
            if failure_stage.startswith("construct_")
            else None
        ),
        warmup_failure_role=(
            failure_stage.removeprefix("warmup_")
            if failure_stage.startswith("warmup_")
            else None
        ),
        failure=failure,
    )

    with pytest.raises(loader.FunAsrModelLoadError) as caught:
        loader.load_funasr_model_bundle(
            RecordingResolver(_snapshots(tmp_path), events),
            device=DEVICE,
            auto_model_factory=factory,
        )

    assert type(caught.value) is loader.FunAsrModelLoadError
    assert str(caught.value) == LOAD_ERROR_MESSAGE
    assert caught.value.__cause__ is failure
    assert events == list(expected_events)
