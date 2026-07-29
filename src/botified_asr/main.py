from __future__ import annotations

import argparse
import os
import secrets
from datetime import datetime, timezone
from importlib.metadata import version as distribution_version
from pathlib import Path
from types import FrameType

import uvicorn

from botified_asr.api import Readiness, create_app
from botified_asr.audio import FfmpegAudioFrontend
from botified_asr.composition import TranscriptionProcessorPool
from botified_asr.config import load_api_key, load_config
from botified_asr.huggingface_fetcher import HuggingFaceSnapshotFetcher
from botified_asr.model_artifacts import ModelArtifactResolver
from botified_asr.model_loader import load_funasr_model_pool
from botified_asr.pipeline import Processor
from botified_asr.runtime import JobExecutor
from botified_asr.storage import Storage


class _ShutdownAwareServer(uvicorn.Server):
    def __init__(
        self,
        config: uvicorn.Config,
        job_executor: JobExecutor,
    ) -> None:
        super().__init__(config)
        self._job_executor = job_executor

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        super().handle_exit(sig, frame)
        try:
            self._job_executor.begin_shutdown()
        except BaseException:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {distribution_version('botified-asr')}",
    )
    parser.add_argument("--config", type=Path, default=_default_config_path())
    args = parser.parse_args()
    config = load_config(args.config)
    host, port_text = config.server.listen.rsplit(":", 1)
    port = int(port_text)
    api_key = load_api_key()
    storage: Storage | None = None
    job_executor: JobExecutor | None = None
    try:
        fetcher = HuggingFaceSnapshotFetcher()
        resolver = ModelArtifactResolver(
            config.runtime.model_cache_dir,
            fetcher,
        )
        model_pool = load_funasr_model_pool(
            resolver,
            device=config.runtime.device,
            inference_lanes=config.runtime.inference_lanes,
        )
        storage = Storage(
            config.storage.data_dir,
            config.limits,
            current_processor_fingerprint=model_pool.processor_fingerprint,
        )
        frontend = FfmpegAudioFrontend()
        processor_pool = TranscriptionProcessorPool(
            tuple(
                Processor(
                    frontend,
                    bundle.asr,
                    vad_adapter=bundle.vad,
                )
                for bundle in model_pool.bundles
            )
        )
        generation = secrets.token_urlsafe()
        job_executor = JobExecutor(
            storage,
            processor_pool.async_processor,
            model_pool.speaker_embedding_policy,
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
            processor=processor_pool.sync_processor,
            audio_prober=frontend.probe,
            processor_fingerprint=model_pool.processor_fingerprint,
            speaker_embedding_policy=model_pool.speaker_embedding_policy,
            job_executor=job_executor,
            close_storage_on_shutdown=False,
        )
        uvicorn_config = uvicorn.Config(
            app,
            host=host,
            port=port,
            workers=1,
            timeout_graceful_shutdown=30,
        )
        _ShutdownAwareServer(uvicorn_config, job_executor).run()
    finally:
        try:
            if job_executor is not None:
                job_executor.stop()
        finally:
            if storage is not None:
                storage.close()


def _default_config_path() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME")
    root = Path(config_home) if config_home else Path("~/.config").expanduser()
    return root / "botified-asr" / "config.yaml"
