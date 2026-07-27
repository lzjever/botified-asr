from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


main_module = importlib.import_module("botified_asr.main")

CONFIG_PATH = Path("/config/botified-asr.yaml")
DATA_DIR = Path("/data/botified-asr")
MODEL_CACHE_DIR = Path("/cache/botified-asr/models")
API_KEY = "startup-test-token"


def install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    listen: str = "127.0.0.1:19001",
    failure_site: str | None = None,
) -> SimpleNamespace:
    events: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
    failure = RuntimeError("startup failed")
    limits = object()
    fetcher = object()
    resolver = object()
    frontend = SimpleNamespace(probe=object())
    asr = object()
    vad = object()
    speaker = object()
    speaker_embedding_policy = object()
    bundle = SimpleNamespace(
        asr=asr,
        vad=vad,
        speaker=speaker,
        speaker_embedding_policy=speaker_embedding_policy,
        processor_fingerprint="3" * 64,
    )
    processor = object()
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
        "load_funasr_model_bundle",
        fake("load_funasr_model_bundle", bundle),
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "FfmpegAudioFrontend",
        fake("FfmpegAudioFrontend", frontend),
    )
    monkeypatch.setattr(
        main_module,
        "Processor",
        fake("Processor", processor),
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
        asr=asr,
        vad=vad,
        speaker=speaker,
        speaker_embedding_policy=speaker_embedding_policy,
        processor=processor,
        readiness=readiness,
        app=app,
    )


def expected_success_events(
    scenario: SimpleNamespace,
) -> list[tuple[str, tuple[Any, ...], dict[str, Any]]]:
    return [
        ("load_config", (CONFIG_PATH,), {}),
        ("load_api_key", (), {}),
        ("Storage", (DATA_DIR, scenario.limits), {}),
        ("HuggingFaceSnapshotFetcher", (), {}),
        (
            "ModelArtifactResolver",
            (MODEL_CACHE_DIR, scenario.fetcher),
            {},
        ),
        (
            "load_funasr_model_bundle",
            (scenario.resolver,),
            {"device": "cpu"},
        ),
        ("FfmpegAudioFrontend", (), {}),
        (
            "Processor",
            (scenario.frontend, scenario.asr),
            {
                "vad_adapter": scenario.vad,
                "known_speaker_policy": None,
            },
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
                "processor": scenario.processor,
                "audio_prober": scenario.frontend.probe,
                "processor_fingerprint": "3" * 64,
                "speaker_embedding_policy": (scenario.speaker_embedding_policy),
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
    scenario = install_fakes(monkeypatch)

    main_module.main()

    assert scenario.events == expected_success_events(scenario)
    assert scenario.readiness.ready is False
    assert scenario.storage.close_calls == 1


@pytest.mark.parametrize(
    ("failure_site", "expected_names"),
    [
        (
            "load_funasr_model_bundle",
            [
                "load_config",
                "load_api_key",
                "Storage",
                "HuggingFaceSnapshotFetcher",
                "ModelArtifactResolver",
                "load_funasr_model_bundle",
                "storage.close",
            ],
        ),
        (
            "create_app",
            [
                "load_config",
                "load_api_key",
                "Storage",
                "HuggingFaceSnapshotFetcher",
                "ModelArtifactResolver",
                "load_funasr_model_bundle",
                "FfmpegAudioFrontend",
                "Processor",
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
                "Storage",
                "HuggingFaceSnapshotFetcher",
                "ModelArtifactResolver",
                "load_funasr_model_bundle",
                "FfmpegAudioFrontend",
                "Processor",
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
    assert scenario.storage.close_calls == 1


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
