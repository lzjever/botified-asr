from __future__ import annotations

import importlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


main_module = importlib.import_module("botified_asr.main")

CONFIG_PATH = Path("/config/botified-asr.yaml")
DATA_DIR = Path("/data/botified-asr")
MODEL_CACHE_DIR = Path("/cache/botified-asr/models")
API_KEY = "startup-test-token"
PROCESSOR_FINGERPRINT = "3" * 64
PROCESS_GENERATION = "process-generation"
PROCESS_NOW = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)


def install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    listen: str = "127.0.0.1:19001",
    failure_site: str | None = None,
    track_executor: bool = False,
) -> SimpleNamespace:
    events: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
    failure = RuntimeError("startup failed")
    limits = object()
    fetcher = object()
    resolver = object()
    frontend = SimpleNamespace(probe=object())
    asrs = (object(), object())
    vads = (object(), object())
    speakers = (object(), object())
    speaker_embedding_policy = object()
    bundles = tuple(
        SimpleNamespace(
            asr=asr,
            vad=vad,
            speaker=speaker,
        )
        for asr, vad, speaker in zip(asrs, vads, speakers, strict=True)
    )
    model_pool = SimpleNamespace(
        bundles=bundles,
        speaker_embedding_policy=speaker_embedding_policy,
        processor_fingerprint=PROCESSOR_FINGERPRINT,
    )
    processors = (object(), object())
    sync_processor = object()
    async_processor = object()
    processor_pool = SimpleNamespace(
        sync_processor=sync_processor,
        async_processor=async_processor,
    )
    executor = SimpleNamespace(ready=False)
    executor_clocks: list[object] = []
    readiness = SimpleNamespace(
        database=True,
        models=True,
        executor=False,
        ready=False,
    )
    app = object()
    config = SimpleNamespace(
        server=SimpleNamespace(listen=listen),
        runtime=SimpleNamespace(
            device="cpu",
            model_cache_dir=MODEL_CACHE_DIR,
            inference_lanes=2,
        ),
        storage=SimpleNamespace(data_dir=DATA_DIR),
        limits=limits,
    )

    class FakeStorage:
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            events.append(("storage.close", (), {}))

    storage = FakeStorage()

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            assert tz is timezone.utc
            return PROCESS_NOW

    def fake(
        name: str,
        result: object,
    ):
        def call(*args: Any, **kwargs: Any) -> object:
            events.append((name, args, kwargs))
            if failure_site == name:
                raise failure
            return result

        return call

    monkeypatch.setattr(sys, "argv", ["botified-asr", "--config", str(CONFIG_PATH)])
    monkeypatch.setattr(
        main_module,
        "load_config",
        fake("load_config", config),
    )
    monkeypatch.setattr(
        main_module,
        "load_api_key",
        fake("load_api_key", API_KEY),
    )
    monkeypatch.setattr(
        main_module,
        "Storage",
        fake("Storage", storage),
    )
    monkeypatch.setattr(
        main_module,
        "HuggingFaceSnapshotFetcher",
        fake("HuggingFaceSnapshotFetcher", fetcher),
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "ModelArtifactResolver",
        fake("ModelArtifactResolver", resolver),
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "load_funasr_model_pool",
        fake("load_funasr_model_pool", model_pool),
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "FfmpegAudioFrontend",
        fake("FfmpegAudioFrontend", frontend),
    )
    next_processor = 0

    def build_processor(*args: Any, **kwargs: Any) -> object:
        nonlocal next_processor
        events.append(("Processor", args, kwargs))
        result = processors[next_processor]
        next_processor += 1
        return result

    monkeypatch.setattr(main_module, "Processor", build_processor)
    monkeypatch.setattr(
        main_module,
        "TranscriptionProcessorPool",
        fake("TranscriptionProcessorPool", processor_pool),
        raising=False,
    )

    def generate_process_generation(*args: Any, **kwargs: Any) -> str:
        if track_executor:
            events.append(("secrets.token_urlsafe", args, kwargs))
        return PROCESS_GENERATION

    def build_job_executor(
        actual_storage: object,
        actual_processor: object,
        actual_policy: object,
        generation: str,
        now: object,
    ) -> object:
        if track_executor:
            events.append(
                (
                    "JobExecutor",
                    (
                        actual_storage,
                        actual_processor,
                        actual_policy,
                        generation,
                    ),
                    {},
                )
            )
        executor_clocks.append(now)
        return executor

    monkeypatch.setattr(
        main_module,
        "secrets",
        SimpleNamespace(token_urlsafe=generate_process_generation),
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "datetime",
        FixedDateTime,
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "timezone",
        timezone,
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "JobExecutor",
        build_job_executor,
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "Readiness",
        fake("Readiness", readiness),
    )
    monkeypatch.setattr(
        main_module,
        "create_app",
        fake("create_app", app),
    )
    monkeypatch.setattr(
        main_module.uvicorn,
        "run",
        fake("uvicorn.run", None),
    )
    return SimpleNamespace(
        events=events,
        failure=failure,
        config=config,
        limits=limits,
        storage=storage,
        fetcher=fetcher,
        resolver=resolver,
        frontend=frontend,
        asrs=asrs,
        vads=vads,
        speakers=speakers,
        speaker_embedding_policy=speaker_embedding_policy,
        bundles=bundles,
        model_pool=model_pool,
        processors=processors,
        processor_pool=processor_pool,
        sync_processor=sync_processor,
        async_processor=async_processor,
        executor=executor,
        executor_clocks=executor_clocks,
        readiness=readiness,
        app=app,
    )


def expected_success_events(
    scenario: SimpleNamespace,
) -> list[tuple[str, tuple[Any, ...], dict[str, Any]]]:
    return [
        ("load_config", (CONFIG_PATH,), {}),
        ("load_api_key", (), {}),
        ("HuggingFaceSnapshotFetcher", (), {}),
        (
            "ModelArtifactResolver",
            (MODEL_CACHE_DIR, scenario.fetcher),
            {},
        ),
        (
            "load_funasr_model_pool",
            (scenario.resolver,),
            {"device": "cpu", "inference_lanes": 2},
        ),
        (
            "Storage",
            (DATA_DIR, scenario.limits),
            {"current_processor_fingerprint": PROCESSOR_FINGERPRINT},
        ),
        ("FfmpegAudioFrontend", (), {}),
        (
            "Processor",
            (scenario.frontend, scenario.asrs[0]),
            {
                "vad_adapter": scenario.vads[0],
                "known_speaker_policy": None,
            },
        ),
        (
            "Processor",
            (scenario.frontend, scenario.asrs[1]),
            {
                "vad_adapter": scenario.vads[1],
                "known_speaker_policy": None,
            },
        ),
        (
            "TranscriptionProcessorPool",
            (scenario.processors,),
            {},
        ),
        ("secrets.token_urlsafe", (), {}),
        (
            "JobExecutor",
            (
                scenario.storage,
                scenario.async_processor,
                scenario.speaker_embedding_policy,
                PROCESS_GENERATION,
            ),
            {},
        ),
        (
            "Readiness",
            (),
            {"database": True, "models": True, "executor": False},
        ),
        (
            "create_app",
            (),
            {
                "api_key": API_KEY,
                "readiness": scenario.readiness,
                "storage": scenario.storage,
                "processor": scenario.sync_processor,
                "audio_prober": scenario.frontend.probe,
                "processor_fingerprint": PROCESSOR_FINGERPRINT,
                "speaker_embedding_policy": (scenario.speaker_embedding_policy),
                "job_executor": scenario.executor,
                "close_storage_on_shutdown": False,
            },
        ),
        (
            "uvicorn.run",
            (scenario.app,),
            {"host": "127.0.0.1", "port": 19001, "workers": 1},
        ),
        ("storage.close", (), {}),
    ]


def test_main_composes_loaded_models_before_serving_and_closes_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = install_fakes(monkeypatch, track_executor=True)

    main_module.main()

    assert scenario.events == expected_success_events(scenario)
    assert len(scenario.executor_clocks) == 1
    assert callable(scenario.executor_clocks[0])
    assert scenario.executor_clocks[0]() == PROCESS_NOW
    assert scenario.readiness.ready is False
    assert scenario.storage.close_calls == 1


@pytest.mark.parametrize(
    ("failure_site", "expected_names"),
    [
        (
            "Storage",
            [
                "load_config",
                "load_api_key",
                "HuggingFaceSnapshotFetcher",
                "ModelArtifactResolver",
                "load_funasr_model_pool",
                "Storage",
            ],
        ),
        (
            "load_funasr_model_pool",
            [
                "load_config",
                "load_api_key",
                "HuggingFaceSnapshotFetcher",
                "ModelArtifactResolver",
                "load_funasr_model_pool",
            ],
        ),
        (
            "create_app",
            [
                "load_config",
                "load_api_key",
                "HuggingFaceSnapshotFetcher",
                "ModelArtifactResolver",
                "load_funasr_model_pool",
                "Storage",
                "FfmpegAudioFrontend",
                "Processor",
                "Processor",
                "TranscriptionProcessorPool",
                "Readiness",
                "create_app",
                "storage.close",
            ],
        ),
        (
            "uvicorn.run",
            [
                "load_config",
                "load_api_key",
                "HuggingFaceSnapshotFetcher",
                "ModelArtifactResolver",
                "load_funasr_model_pool",
                "Storage",
                "FfmpegAudioFrontend",
                "Processor",
                "Processor",
                "TranscriptionProcessorPool",
                "Readiness",
                "create_app",
                "uvicorn.run",
                "storage.close",
            ],
        ),
    ],
)
def test_main_propagates_startup_failures_and_closes_storage_once(
    monkeypatch: pytest.MonkeyPatch,
    failure_site: str,
    expected_names: list[str],
) -> None:
    scenario = install_fakes(monkeypatch, failure_site=failure_site)

    with pytest.raises(RuntimeError) as caught:
        main_module.main()

    assert caught.value is scenario.failure
    assert [name for name, _, _ in scenario.events] == expected_names
    assert scenario.storage.close_calls == (
        0 if failure_site in {"load_funasr_model_pool", "Storage"} else 1
    )


def test_main_parses_port_before_api_key_or_resource_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = install_fakes(
        monkeypatch,
        listen="127.0.0.1:not-a-port",
    )

    with pytest.raises(ValueError):
        main_module.main()

    assert scenario.events == [("load_config", (CONFIG_PATH,), {})]
    assert scenario.storage.close_calls == 0
