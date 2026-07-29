from __future__ import annotations

import argparse
import json
import logging
import math
import os
import secrets
import sys
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
from botified_asr.speaker_enrollment import (
    SpeakerEnrollmentPolicy,
    SpeakerEnrollmentProcessor,
)
from botified_asr.speaker_matching import KnownSpeakerMatchPolicy
from botified_asr.speakers import AnonymousSpeakerClusteringPolicy
from botified_asr.storage import Storage


class _JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, timezone.utc)
        payload: dict[str, object] = {
            "ts": timestamp.isoformat(timespec="milliseconds").replace(
                "+00:00",
                "Z",
            ),
            "level": record.levelname,
            "event": "log_message",
        }
        event = getattr(record, "event", None)
        if record.name == "botified_asr.service" and event in {
            "service_started",
            "service_stopped",
        }:
            payload["event"] = event
        elif record.name == "botified_asr.http" and event == (
            "http_request_completed"
        ):
            request_id = getattr(record, "request_id", None)
            method = getattr(record, "method", None)
            route = getattr(record, "route", None)
            status = getattr(record, "status", None)
            elapsed_ms = getattr(record, "elapsed_ms", None)
            error_code = getattr(record, "error_code", None)
            if (
                type(request_id) is str
                and type(method) is str
                and type(route) is str
                and type(status) is int
                and type(elapsed_ms) is float
                and math.isfinite(elapsed_ms)
                and elapsed_ms >= 0
                and (error_code is None or type(error_code) is str)
            ):
                payload.update(
                    {
                        "event": event,
                        "request_id": request_id,
                        "method": method,
                        "route": route,
                        "status": status,
                        "elapsed_ms": elapsed_ms,
                        "error_code": error_code,
                    }
                )
        elif record.name == "botified_asr.job":
            job_id = getattr(record, "job_id", None)
            if event == "job_started":
                attempt = getattr(record, "attempt", None)
                model = getattr(record, "model", None)
                queue_wait_ms = getattr(record, "queue_wait_ms", None)
                if (
                    type(job_id) is str
                    and type(attempt) is int
                    and attempt >= 1
                    and type(model) is str
                    and type(queue_wait_ms) is float
                    and math.isfinite(queue_wait_ms)
                    and queue_wait_ms >= 0
                ):
                    payload.update(
                        {
                            "event": event,
                            "job_id": job_id,
                            "attempt": attempt,
                            "model": model,
                            "queue_wait_ms": queue_wait_ms,
                        }
                    )
            elif event == "job_finished":
                attempt = getattr(record, "attempt", None)
                model = getattr(record, "model", None)
                status = getattr(record, "status", None)
                error_code = getattr(record, "error_code", None)
                elapsed_ms = getattr(record, "elapsed_ms", None)
                audio_duration_seconds = getattr(
                    record,
                    "audio_duration_seconds",
                    None,
                )
                if (
                    type(job_id) is str
                    and type(attempt) is int
                    and attempt >= 1
                    and type(model) is str
                    and type(status) is str
                    and status in {"succeeded", "failed", "cancelled"}
                    and (error_code is None or type(error_code) is str)
                    and type(elapsed_ms) is float
                    and math.isfinite(elapsed_ms)
                    and elapsed_ms >= 0
                    and (
                        audio_duration_seconds is None
                        or (
                            type(audio_duration_seconds) is float
                            and math.isfinite(audio_duration_seconds)
                            and audio_duration_seconds >= 0
                        )
                    )
                ):
                    payload.update(
                        {
                            "event": event,
                            "job_id": job_id,
                            "attempt": attempt,
                            "model": model,
                            "status": status,
                            "error_code": error_code,
                            "elapsed_ms": elapsed_ms,
                            "audio_duration_seconds": (
                                audio_duration_seconds
                            ),
                        }
                    )
            elif event == "job_executor_failed":
                exception_type = getattr(record, "exception_type", None)
                if (
                    type(job_id) is str
                    and type(exception_type) is str
                ):
                    payload.update(
                        {
                            "event": event,
                            "job_id": job_id,
                            "exception_type": exception_type,
                        }
                    )
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )


def _configure_logging() -> logging.Logger:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonLogFormatter())
    application_logger = logging.getLogger("botified_asr")
    application_logger.handlers.clear()
    application_logger.addHandler(handler)
    application_logger.setLevel(logging.INFO)
    application_logger.propagate = False
    return logging.getLogger("botified_asr.service")


def _log_service_event(logger: logging.Logger, event: str) -> None:
    try:
        logger.info(event, extra={"event": event})
    except Exception:
        pass


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
    service_logger = _configure_logging()
    config = load_config(args.config)
    host, port_text = config.server.listen.rsplit(":", 1)
    port = int(port_text)
    api_key = load_api_key()
    storage: Storage | None = None
    job_executor: JobExecutor | None = None
    service_started = False
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
        speaker_clustering_policy = AnonymousSpeakerClusteringPolicy(
            pruning_p=1.0,
            low_frequency_beta=2.0,
            normalized_gap_gamma=0.5,
        )
        speaker_similarity_threshold = 0.31
        speaker_enrollment_policy = SpeakerEnrollmentPolicy(
            consistency_threshold=speaker_similarity_threshold,
        )
        known_speaker_match_policy = KnownSpeakerMatchPolicy(
            match_threshold=speaker_similarity_threshold,
        )
        processor_pool = TranscriptionProcessorPool(
            tuple(
                Processor(
                    frontend,
                    bundle.asr,
                    vad_adapter=bundle.vad,
                    speaker_adapter=bundle.speaker,
                    speaker_clustering_policy=speaker_clustering_policy,
                    known_speaker_match_policy=known_speaker_match_policy,
                )
                for bundle in model_pool.bundles
            ),
            tuple(
                SpeakerEnrollmentProcessor(
                    frontend,
                    bundle.vad,
                    bundle.speaker,
                    model_pool.speaker_embedding_policy,
                    speaker_enrollment_policy,
                )
                for bundle in model_pool.bundles
            ),
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
            speaker_enrollment_processor=(
                processor_pool.speaker_enrollment_processor
            ),
            job_executor=job_executor,
            close_storage_on_shutdown=False,
        )
        uvicorn_config = uvicorn.Config(
            app,
            host=host,
            port=port,
            workers=1,
            timeout_graceful_shutdown=30,
            log_config=None,
            access_log=False,
        )
        _log_service_event(service_logger, "service_started")
        service_started = True
        _ShutdownAwareServer(uvicorn_config, job_executor).run()
    finally:
        try:
            if job_executor is not None:
                job_executor.stop()
        finally:
            try:
                if storage is not None:
                    storage.close()
            finally:
                if service_started:
                    _log_service_event(service_logger, "service_stopped")


def _default_config_path() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME")
    root = Path(config_home) if config_home else Path("~/.config").expanduser()
    return root / "botified-asr" / "config.yaml"
