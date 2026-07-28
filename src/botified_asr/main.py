from __future__ import annotations

import argparse
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

import uvicorn

from botified_asr.api import Readiness, create_app
from botified_asr.audio import FfmpegAudioFrontend
from botified_asr.config import load_api_key, load_config
from botified_asr.huggingface_fetcher import HuggingFaceSnapshotFetcher
from botified_asr.model_artifacts import ModelArtifactResolver
from botified_asr.model_loader import load_funasr_model_bundle
from botified_asr.pipeline import Processor
from botified_asr.runtime import JobExecutor
from botified_asr.storage import Storage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=_default_config_path())
    args = parser.parse_args()
    config = load_config(args.config)
    host, port_text = config.server.listen.rsplit(":", 1)
    port = int(port_text)
    api_key = load_api_key()
    storage: Storage | None = None
    try:
        fetcher = HuggingFaceSnapshotFetcher()
        resolver = ModelArtifactResolver(
            config.runtime.model_cache_dir,
            fetcher,
        )
        bundle = load_funasr_model_bundle(
            resolver,
            device=config.runtime.device,
        )
        storage = Storage(
            config.storage.data_dir,
            config.limits,
            current_processor_fingerprint=bundle.processor_fingerprint,
        )
        frontend = FfmpegAudioFrontend()
        processor = Processor(
            frontend,
            bundle.asr,
            vad_adapter=bundle.vad,
            known_speaker_policy=None,
        )
        generation = secrets.token_urlsafe()
        job_executor = JobExecutor(
            storage,
            processor,
            bundle.speaker_embedding_policy,
            generation,
            lambda: datetime.now(timezone.utc),
        )
        readiness = Readiness(
            database=True,
            models=True,
            executor=False,
        )
        app = create_app(
            api_key=api_key,
            readiness=readiness,
            storage=storage,
            processor=processor,
            audio_prober=frontend.probe,
            processor_fingerprint=bundle.processor_fingerprint,
            speaker_embedding_policy=bundle.speaker_embedding_policy,
            job_executor=job_executor,
            close_storage_on_shutdown=False,
        )
        uvicorn.run(app, host=host, port=port, workers=1)
    finally:
        if storage is not None:
            storage.close()


def _default_config_path() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME")
    root = Path(config_home) if config_home else Path("~/.config").expanduser()
    return root / "botified-asr" / "config.yaml"
