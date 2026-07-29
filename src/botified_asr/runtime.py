from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import datetime

from botified_asr.audio import SAMPLE_RATE, AudioError, Cancellation
from botified_asr.canonical_options import parse_canonical_options_json
from botified_asr.composition import (
    TranscriptionProcessor,
    execute_claimed_job_attempt,
)
from botified_asr.errors import PipelineError
from botified_asr.jobs import DurableJob, JobStatus
from botified_asr.speakers import SpeakerEmbeddingPolicy
from botified_asr.storage import Storage

_IDLE_WAIT_SECONDS = 1.0
_JOB_LOGGER = logging.getLogger("botified_asr.job")
_TERMINAL_JOB_STATUSES = {
    JobStatus.SUCCEEDED,
    JobStatus.FAILED,
    JobStatus.CANCELLED,
}


def _log_job_event(event: str, fields: dict[str, object]) -> None:
    try:
        _JOB_LOGGER.info(event, extra={"event": event, **fields})
    except Exception:
        pass


def _log_started_job(job: DurableJob) -> None:
    try:
        if job.started_at is None or job.canonical_options_json is None:
            return
        model = parse_canonical_options_json(job.canonical_options_json).model
        queue_wait_ms = (
            job.started_at - job.created_at
        ).total_seconds() * 1_000
        _log_job_event(
            "job_started",
            {
                "job_id": job.id,
                "attempt": job.attempt_no,
                "model": model,
                "queue_wait_ms": queue_wait_ms,
            },
        )
    except Exception:
        pass


def _log_finished_job(storage: Storage, running_job: DurableJob) -> None:
    try:
        job = storage.get_visible_job(running_job.id)
        if (
            type(job) is not DurableJob
            or job.status not in _TERMINAL_JOB_STATUSES
            or job.started_at is None
            or job.finished_at is None
            or job.canonical_options_json is None
        ):
            return
        model = parse_canonical_options_json(job.canonical_options_json).model
        elapsed_ms = (
            job.finished_at - job.started_at
        ).total_seconds() * 1_000
        audio_duration_seconds = (
            None
            if job.total_samples is None
            else job.total_samples / SAMPLE_RATE
        )
        _log_job_event(
            "job_finished",
            {
                "job_id": job.id,
                "attempt": job.attempt_no,
                "model": model,
                "status": job.status.value,
                "error_code": job.error_code,
                "elapsed_ms": elapsed_ms,
                "audio_duration_seconds": audio_duration_seconds,
            },
        )
    except Exception:
        pass


def _log_job_executor_failure(job_id: str, error: BaseException) -> None:
    _log_job_event(
        "job_executor_failed",
        {
            "job_id": job_id,
            "exception_type": type(error).__name__,
        },
    )


class JobExecutor:
    def __init__(
        self,
        storage: Storage,
        processor: TranscriptionProcessor,
        speaker_embedding_policy: SpeakerEmbeddingPolicy,
        generation: str,
        now: Callable[[], datetime],
    ) -> None:
        self._storage = storage
        self._processor = processor
        self._speaker_embedding_policy = speaker_embedding_policy
        self._generation = generation
        self._now = now
        self._lock = threading.Lock()
        self._wake_event = threading.Event()
        self._marker_done = threading.Event()
        self._stop_complete = threading.Event()
        self._thread: threading.Thread | None = None
        self._marker_thread: threading.Thread | None = None
        self._started = False
        self._stopping = False
        self._stop_started = False
        self._ready = False
        self._failure: BaseException | None = None
        self._stop_error: BaseException | None = None
        self._marker_written = False
        self._active_job_id: str | None = None
        self._active_cancellation: Cancellation | None = None

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._ready

    @property
    def failure(self) -> BaseException | None:
        with self._lock:
            return self._failure

    def start(self) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError("job executor was already started")
            if self._stopping:
                raise RuntimeError("job executor is stopped")
            thread = threading.Thread(
                target=self._run,
                name="botified-asr-job-executor",
            )
            self._thread = thread
            self._started = True
            self._ready = True
            try:
                thread.start()
            except BaseException:
                self._ready = False
                self._started = False
                self._thread = None
                raise

    def wake(self) -> None:
        self._wake_event.set()

    def notify_cancellation(self, job_id: str) -> None:
        with self._lock:
            cancellation = (
                self._active_cancellation
                if self._active_job_id == job_id
                else None
            )
        if cancellation is not None:
            cancellation.cancel()

    def begin_shutdown(self) -> None:
        active_cancellation: Cancellation | None = None
        marker_required = False
        try:
            with self._lock:
                if self._stopping:
                    return
                self._stopping = True
                self._ready = False
                active_cancellation = self._active_cancellation
                marker_required = self._started and self._failure is None

            if active_cancellation is not None:
                active_cancellation.cancel()
            self._wake_event.set()

            if not marker_required:
                self._marker_done.set()
                return
            try:
                marker_thread = threading.Thread(
                    target=self._write_shutdown_marker,
                    name="botified-asr-shutdown-marker",
                )
                with self._lock:
                    self._marker_thread = marker_thread
                marker_thread.start()
            except BaseException as error:
                with self._lock:
                    self._marker_thread = None
                self._record_stop_error(error)
                self._marker_done.set()
        except BaseException as error:
            self._record_stop_error(error)
            try:
                self._wake_event.set()
            except BaseException:
                pass
            self._marker_done.set()

    def stop(self) -> None:
        self.begin_shutdown()
        with self._lock:
            if self._stop_started:
                owns_stop = False
            else:
                owns_stop = True
                self._stop_started = True

        if not owns_stop:
            self._stop_complete.wait()
            self._raise_stop_error()
            return

        current_thread = threading.current_thread()
        try:
            self._marker_done.wait()
            with self._lock:
                marker_thread = self._marker_thread
                thread = self._thread
            if marker_thread is not None and marker_thread is not current_thread:
                try:
                    marker_thread.join()
                except BaseException as error:
                    self._record_stop_error(error)
            if thread is not None and thread is not current_thread:
                try:
                    thread.join()
                except BaseException as error:
                    self._record_stop_error(error)
            elif thread is current_thread:
                self._record_stop_error(
                    RuntimeError("job executor cannot join its worker thread")
                )
        finally:
            self._stop_complete.set()
        self._raise_stop_error()

    def _write_shutdown_marker(self) -> None:
        try:
            self._storage.write_shutdown_marker(
                self._generation,
                self._now(),
            )
        except BaseException as error:
            self._record_stop_error(error)
        else:
            with self._lock:
                self._marker_written = True
        finally:
            self._marker_done.set()

    def _record_stop_error(self, error: BaseException) -> None:
        try:
            with self._lock:
                if self._stop_error is None:
                    self._stop_error = error
        except BaseException:
            pass

    def _run(self) -> None:
        while True:
            self._wake_event.clear()
            with self._lock:
                if self._stopping or self._failure is not None:
                    return

            try:
                running_job = self._storage.claim_next_job(
                    self._generation,
                    self._now(),
                )
            except BaseException as error:
                with self._lock:
                    if self._stopping:
                        if self._stop_error is None:
                            self._stop_error = error
                    else:
                        self._failure = error
                        self._ready = False
                return

            if running_job is None:
                with self._lock:
                    if self._stopping or self._failure is not None:
                        return
                self._wake_event.wait(_IDLE_WAIT_SECONDS)
                continue

            cancellation = Cancellation()
            with self._lock:
                stopping = self._stopping
                if stopping:
                    cancellation.cancel()
                self._active_job_id = running_job.id
                self._active_cancellation = cancellation

            runner_error: BaseException | None = None
            if not stopping:
                _log_started_job(running_job)
                try:
                    execute_claimed_job_attempt(
                        self._storage,
                        self._processor,
                        running_job,
                        cancellation,
                        speaker_embedding_policy=(
                            self._speaker_embedding_policy
                        ),
                        now=self._now,
                    )
                except BaseException as error:
                    runner_error = error

            with self._lock:
                stopping = self._stopping
                if not stopping and runner_error is not None:
                    self._failure = runner_error
                    self._ready = False
                elif (
                    stopping
                    and runner_error is not None
                    and not (
                        cancellation.cancelled
                        and isinstance(
                            runner_error,
                            (PipelineError, AudioError),
                        )
                        and runner_error.code == "cancelled"
                    )
                    and self._stop_error is None
                ):
                    self._stop_error = runner_error
                if not stopping:
                    self._active_job_id = None
                    self._active_cancellation = None

            if not stopping:
                if runner_error is not None:
                    _log_job_executor_failure(running_job.id, runner_error)
                    return
                _log_finished_job(self._storage, running_job)
                continue

            self._marker_done.wait()
            with self._lock:
                marker_written = self._marker_written
            if marker_written:
                try:
                    requeued = self._storage.requeue_job_at_shutdown(
                        running_job.id,
                        running_job.attempt_token,
                        self._generation,
                    )
                    if not requeued:
                        if running_job.started_at is None:
                            raise RuntimeError(
                                "claimed job has no start timestamp"
                            )
                        self._storage.commit_job_cancellation(
                            running_job.id,
                            running_job.attempt_token,
                            max(self._now(), running_job.started_at),
                        )
                except BaseException as error:
                    with self._lock:
                        if self._stop_error is None:
                            self._stop_error = error
            with self._lock:
                self._active_job_id = None
                self._active_cancellation = None
            return

    def _raise_stop_error(self) -> None:
        with self._lock:
            error = self._stop_error
        if error is not None:
            raise error
