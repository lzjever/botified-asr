from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import datetime

from botified_asr.audio import AudioError, Cancellation
from botified_asr.composition import (
    TranscriptionProcessor,
    execute_claimed_job_attempt,
)
from botified_asr.errors import PipelineError
from botified_asr.speakers import SpeakerEmbeddingPolicy
from botified_asr.storage import Storage

_IDLE_WAIT_SECONDS = 1.0


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
        self._started = False
        self._stopping = False
        self._ready = False
        self._failure: BaseException | None = None
        self._stop_error: BaseException | None = None
        self._marker_written = False
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

    def stop(self) -> None:
        with self._lock:
            if self._stopping:
                owns_stop = False
            else:
                owns_stop = True
                self._stopping = True
                self._ready = False
                started = self._started
                fatal = self._failure is not None

        if not owns_stop:
            self._stop_complete.wait()
            self._raise_stop_error()
            return

        if not started:
            self._marker_done.set()
            self._stop_complete.set()
            return

        if not fatal:
            try:
                self._storage.write_shutdown_marker(
                    self._generation,
                    self._now(),
                )
            except BaseException as error:
                with self._lock:
                    if self._stop_error is None:
                        self._stop_error = error
            else:
                with self._lock:
                    self._marker_written = True

        self._marker_done.set()
        with self._lock:
            active_cancellation = self._active_cancellation
            thread = self._thread
        if active_cancellation is not None:
            active_cancellation.cancel()
        self._wake_event.set()
        assert thread is not None
        thread.join()
        self._stop_complete.set()
        self._raise_stop_error()

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
                    self._failure = error
                    self._ready = False
                    return
                if running_job is not None:
                    cancellation = Cancellation()
                    self._active_cancellation = cancellation

            if running_job is None:
                self._wake_event.wait(_IDLE_WAIT_SECONDS)
                continue

            runner_error: BaseException | None = None
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
                    self._active_cancellation = None

            if not stopping:
                if runner_error is not None:
                    return
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
                self._active_cancellation = None
            return

    def _raise_stop_error(self) -> None:
        with self._lock:
            error = self._stop_error
        if error is not None:
            raise error
