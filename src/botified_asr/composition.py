from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol

from botified_asr.audio import SAMPLE_RATE, AudioError, Cancellation, MediaProbe
from botified_asr.canonical_options import parse_canonical_options_json
from botified_asr.contracts import (
    DIRECT_MAX_SAMPLES,
    MAX_AUDIO_SAMPLES,
    CanonicalOptions,
)
from botified_asr.errors import PipelineError
from botified_asr.inference import inference_session
from botified_asr.jobs import (
    DurableJob,
    JobCancellationRequestedError,
    JobProgressOutcome,
    JobSuccessOutcome,
    JobTerminalOutcome,
    StaleJobAttemptError,
)
from botified_asr.pipeline import (
    CanonicalJsonlSegmentSink,
    ProcessorResult,
    ProgressSink,
    SegmentSink,
)
from botified_asr.result_artifact import (
    CanonicalJsonlReader,
    Projection,
    RESULT_ENVELOPE_VERSION,
    ResultEnvelopeManifest,
    ResultProjector,
    finalize_result_envelope,
)
from botified_asr.speaker_matching import SpeakerLabelMapping
from botified_asr.speaker_profiles import SpeakerEmbeddingReplacement
from botified_asr.speaker_snapshot import (
    SelectedSpeakerSnapshot,
    parse_selected_speaker_snapshot,
    resolve_selected_speaker_snapshot,
)
from botified_asr.speakers import SpeakerEmbeddingPolicy
from botified_asr.storage import (
    RUNTIME_JOB_FAILURE_CODES,
    ArtifactRef,
    ReservedByteWriter,
    Storage,
    StorageAdmissionError,
    StorageSchemaError,
)


class TranscriptionProcessor(Protocol):
    def process(
        self,
        input_path: Path,
        canonical_options: CanonicalOptions,
        cancellation: Cancellation,
        progress_sink: ProgressSink,
        segment_sink: SegmentSink,
        *,
        selected_speaker_snapshot: SelectedSpeakerSnapshot,
        effective_max_audio_samples: int,
        effective_direct_max_audio_samples: int,
        media_probe: MediaProbe | None = None,
    ) -> ProcessorResult: ...


class SpeakerEnrollmentProcessor(Protocol):
    def process(
        self,
        sample_paths: tuple[Path, ...],
        cancellation: Cancellation,
        *,
        effective_max_audio_samples: int,
    ) -> SpeakerEmbeddingReplacement: ...


class _LeastActiveProcessorSelector:
    def __init__(self, lane_count: int) -> None:
        if type(lane_count) is not int or lane_count <= 0:
            raise ValueError("processor lane count must be positive")
        self._lock = threading.Lock()
        self._active_sessions = [0] * lane_count
        self._cursor = 0

    def acquire(self) -> int:
        with self._lock:
            minimum = min(self._active_sessions)
            for offset in range(len(self._active_sessions)):
                index = (self._cursor + offset) % len(self._active_sessions)
                if self._active_sessions[index] == minimum:
                    break
            self._active_sessions[index] += 1
            self._cursor = (index + 1) % len(self._active_sessions)
            return index

    def release(self, index: int) -> None:
        with self._lock:
            if self._active_sessions[index] <= 0:
                raise RuntimeError("processor selector release is unbalanced")
            self._active_sessions[index] -= 1


class _SessionTranscriptionProcessor:
    def __init__(
        self,
        selector: _LeastActiveProcessorSelector,
        processors: tuple[TranscriptionProcessor, ...],
        category: Literal["sync", "async"],
    ) -> None:
        self._selector = selector
        self._processors = processors
        self._category = category

    def process(
        self,
        input_path: Path,
        canonical_options: CanonicalOptions,
        cancellation: Cancellation,
        progress_sink: ProgressSink,
        segment_sink: SegmentSink,
        *,
        selected_speaker_snapshot: SelectedSpeakerSnapshot,
        effective_max_audio_samples: int,
        effective_direct_max_audio_samples: int,
        media_probe: MediaProbe | None = None,
    ) -> ProcessorResult:
        index = self._selector.acquire()
        processor = self._processors[index]
        try:
            with inference_session(self._category, cancellation):
                return processor.process(
                    input_path,
                    canonical_options,
                    cancellation,
                    progress_sink,
                    segment_sink,
                    selected_speaker_snapshot=selected_speaker_snapshot,
                    effective_max_audio_samples=effective_max_audio_samples,
                    effective_direct_max_audio_samples=(
                        effective_direct_max_audio_samples
                    ),
                    media_probe=media_probe,
                )
        finally:
            self._selector.release(index)


class _SessionSpeakerEnrollmentProcessor:
    def __init__(
        self,
        selector: _LeastActiveProcessorSelector,
        processors: tuple[SpeakerEnrollmentProcessor, ...],
    ) -> None:
        self._selector = selector
        self._processors = processors

    def process(
        self,
        sample_paths: tuple[Path, ...],
        cancellation: Cancellation,
        *,
        effective_max_audio_samples: int,
    ) -> SpeakerEmbeddingReplacement:
        index = self._selector.acquire()
        processor = self._processors[index]
        try:
            with inference_session("sync", cancellation):
                return processor.process(
                    sample_paths,
                    cancellation,
                    effective_max_audio_samples=effective_max_audio_samples,
                )
        finally:
            self._selector.release(index)


class TranscriptionProcessorPool:
    def __init__(
        self,
        lane_processors: tuple[TranscriptionProcessor, ...],
        speaker_enrollment_processors: (
            tuple[SpeakerEnrollmentProcessor, ...] | None
        ) = None,
    ) -> None:
        if type(lane_processors) is not tuple:
            raise TypeError("lane processors must be a tuple")
        if not lane_processors:
            raise ValueError("lane processors must not be empty")
        if any(
            not callable(getattr(processor, "process", None))
            for processor in lane_processors
        ):
            raise TypeError("each lane processor must have a callable process")
        if speaker_enrollment_processors is not None:
            if type(speaker_enrollment_processors) is not tuple:
                raise TypeError("speaker enrollment processors must be a tuple")
            if len(speaker_enrollment_processors) != len(lane_processors):
                raise ValueError(
                    "speaker enrollment processors must match processor lanes"
                )
            if any(
                not callable(getattr(processor, "process", None))
                for processor in speaker_enrollment_processors
            ):
                raise TypeError(
                    "each speaker enrollment processor must have a callable process"
                )
        selector = _LeastActiveProcessorSelector(len(lane_processors))
        self.sync_processor: TranscriptionProcessor = _SessionTranscriptionProcessor(
            selector, lane_processors, "sync"
        )
        self.async_processor: TranscriptionProcessor = _SessionTranscriptionProcessor(
            selector, lane_processors, "async"
        )
        self.speaker_enrollment_processor: SpeakerEnrollmentProcessor | None = (
            None
            if speaker_enrollment_processors is None
            else _SessionSpeakerEnrollmentProcessor(
                selector,
                speaker_enrollment_processors,
            )
        )


class ProjectionBuilder(Protocol):
    def prepare(
        self,
        reader: CanonicalJsonlReader,
        options: CanonicalOptions,
        total_samples: int,
        *,
        speaker_mapping: SpeakerLabelMapping,
    ) -> Projection: ...


class StorageArtifactByteWriter:
    def __init__(
        self,
        storage: Storage,
        writer: ReservedByteWriter,
    ) -> None:
        self._storage = storage
        self._writer = writer
        self._state = "OPEN"
        self._sealed_ref: ArtifactRef | None = None

    @property
    def state(self) -> str:
        return self._state

    @property
    def sealed_ref(self) -> ArtifactRef | None:
        return self._sealed_ref

    def write(self, payload: bytes) -> None:
        self._require_open("write")
        self._storage.append_artifact(self._writer, payload)

    def seal(self) -> ArtifactRef:
        self._require_open("seal")
        ref = self._storage.seal_artifact(self._writer)
        self._sealed_ref = ref
        self._state = "SEALED"
        return ref

    def abort(self) -> None:
        if self._state == "ABORTED":
            return
        self._require_open("abort")
        self._storage.abort_artifact(self._writer)
        self._state = "ABORTED"

    def _discard(self) -> None:
        if self._state in {"ABORTED", "RELEASED"}:
            return
        if self._state == "OPEN":
            self._storage.abort_artifact(self._writer)
            self._state = "ABORTED"
            return
        if self._sealed_ref is None:
            raise RuntimeError("sealed artifact writer has no reference")
        self._storage.release_artifact(self._sealed_ref)
        self._state = "RELEASED"

    def _require_open(self, operation: str) -> None:
        if self._state != "OPEN":
            raise RuntimeError(
                f"cannot {operation} artifact writer in {self._state} state"
            )


class ProgressAccumulator:
    def __init__(self) -> None:
        self._last_processed: int | None = None
        self._exact_total: int | None = None

    def update(
        self,
        *,
        processed_samples: int,
        total_samples: int | None,
    ) -> None:
        if self._exact_total is not None:
            raise RuntimeError("progress was updated after EOF")
        _validate_sample_count(
            processed_samples,
            name="processed sample count",
        )
        if total_samples is not None:
            _validate_sample_count(
                total_samples,
                name="total sample count",
            )
            if total_samples != processed_samples:
                raise ValueError(
                    "EOF progress must have equal processed and total sample counts"
                )
        if (
            self._last_processed is not None
            and processed_samples < self._last_processed
        ):
            raise ValueError("processed sample count must be monotonic")
        self._last_processed = processed_samples
        if total_samples is not None:
            self._exact_total = total_samples

    def finish(self) -> int:
        if self._last_processed is None:
            raise RuntimeError("progress did not report processed samples")
        if self._exact_total is None:
            raise RuntimeError("progress did not report EOF")
        return self._exact_total


class _DurableJobProgress:
    def __init__(
        self,
        storage: Storage,
        job_id: str,
        attempt_token: str,
        cancellation: Cancellation,
    ) -> None:
        self._storage = storage
        self._job_id = job_id
        self._attempt_token = attempt_token
        self._cancellation = cancellation
        self._local = ProgressAccumulator()

    def update(
        self,
        *,
        processed_samples: int,
        total_samples: int | None,
    ) -> None:
        self._local.update(
            processed_samples=processed_samples,
            total_samples=total_samples,
        )
        outcome = self._storage.update_job_progress(
            self._job_id,
            self._attempt_token,
            processed_samples,
            total_samples=total_samples,
        )
        if outcome is JobProgressOutcome.UPDATED:
            return
        self._cancellation.cancel()
        if outcome is JobProgressOutcome.CANCEL_REQUESTED:
            raise JobCancellationRequestedError(
                "job cancellation was requested"
            )
        raise StaleJobAttemptError(
            "job attempt is no longer running"
        )

    def finish(self) -> int:
        return self._local.finish()


def _validate_processor_result(
    result: object,
    writer: StorageArtifactByteWriter,
) -> SpeakerLabelMapping:
    if type(result) is not ProcessorResult:
        raise RuntimeError("processor returned an invalid result")
    speaker_mapping = result.speaker_mapping
    if (
        type(speaker_mapping) is not SpeakerLabelMapping
        or type(speaker_mapping.resolutions) is not tuple
    ):
        raise RuntimeError("processor returned an invalid speaker mapping")
    if writer.sealed_ref is None or result.artifact_ref is not writer.sealed_ref:
        raise RuntimeError("processor returned an unexpected artifact reference")
    return speaker_mapping


def _validate_sample_count(value: int, *, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value <= MAX_AUDIO_SAMPLES:
        raise ValueError(f"{name} is outside the allowed range")


class PreparedSyncResponse:
    def __init__(
        self,
        projection: Projection,
        artifact_writer: StorageArtifactByteWriter,
    ) -> None:
        self.content_type = projection.content_type
        self._body_factory = projection.body_factory
        self._artifact_writer = artifact_writer
        self._body_claimed = False
        self._closed = False
        self._close_lock = threading.Lock()

    def iter_body(self) -> Iterator[bytes]:
        with self._close_lock:
            if self._closed:
                raise RuntimeError("prepared response is closed")
            if self._body_claimed:
                raise RuntimeError("prepared response body was already requested")
            self._body_claimed = True
        try:
            iterator = iter(self._body_factory())
        except BaseException:
            self.close()
            raise

        def body() -> Iterator[bytes]:
            try:
                yield from iterator
            finally:
                try:
                    close = getattr(iterator, "close", None)
                    if close is not None:
                        close()
                finally:
                    self.close()

        return body()

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._artifact_writer._discard()
            self._closed = True


def prepare_sync_transcription(
    storage: Storage,
    processor: TranscriptionProcessor,
    input_path: Path,
    owner_id: str,
    options: CanonicalOptions,
    cancellation: Cancellation,
    projector: ProjectionBuilder | None = None,
    *,
    speaker_embedding_policy: SpeakerEmbeddingPolicy,
    media_probe: MediaProbe,
) -> PreparedSyncResponse:
    effective_max_audio_samples = (
        storage.limits.max_audio_duration_secs * SAMPLE_RATE
    )
    effective_direct_max_audio_samples = min(
        storage.limits.direct_max_audio_duration_secs * SAMPLE_RATE,
        effective_max_audio_samples,
        DIRECT_MAX_SAMPLES,
    )
    selected_speaker_snapshot = resolve_selected_speaker_snapshot(
        storage,
        options.known_speaker_ids,
        speaker_embedding_policy,
    )
    reserved_writer = storage.begin_artifact(
        "segment_jsonl",
        owner_kind="sync",
        owner_id=owner_id,
    )
    writer = StorageArtifactByteWriter(storage, reserved_writer)
    transferred = False
    try:
        sink = CanonicalJsonlSegmentSink(writer)
        progress = ProgressAccumulator()
        result = processor.process(
            input_path,
            options,
            cancellation,
            progress,
            sink,
            selected_speaker_snapshot=selected_speaker_snapshot,
            effective_max_audio_samples=effective_max_audio_samples,
            effective_direct_max_audio_samples=effective_direct_max_audio_samples,
            media_probe=media_probe,
        )
        speaker_mapping = _validate_processor_result(result, writer)
        total_samples = progress.finish()
        artifact_path = storage.resolve_artifact(writer.sealed_ref)
        reader = CanonicalJsonlReader(artifact_path)
        projection_builder = ResultProjector() if projector is None else projector
        projection = projection_builder.prepare(
            reader,
            options,
            total_samples,
            speaker_mapping=speaker_mapping,
        )
        response = PreparedSyncResponse(projection, writer)
        transferred = True
        return response
    finally:
        if not transferred:
            writer._discard()


def execute_claimed_job_attempt(
    storage: Storage,
    processor: TranscriptionProcessor,
    running_job: DurableJob,
    cancellation: Cancellation,
    *,
    speaker_embedding_policy: SpeakerEmbeddingPolicy,
    now: Callable[[], datetime],
) -> None:
    if type(running_job) is not DurableJob:
        raise TypeError(
            "execute_claimed_job_attempt requires a DurableJob"
        )
    attempt_token = running_job.attempt_token
    if attempt_token is None:
        raise ValueError("claimed job has no attempt token")

    def finished_at(job: DurableJob) -> datetime:
        if job.started_at is None:
            raise StorageSchemaError(
                "running job has no attempt start time"
            )
        return max(now(), job.started_at)

    def commit_cancellation(
        job: DurableJob,
        when: datetime,
    ) -> JobTerminalOutcome:
        return storage.commit_job_cancellation(
            job.id,
            attempt_token,
            when,
        )

    def commit_failure(
        job: DurableJob,
        error_code: str,
        when: datetime,
    ) -> None:
        outcome = storage.commit_job_failure(
            job.id,
            attempt_token,
            error_code,
            when,
        )
        if outcome is JobTerminalOutcome.CANCEL_REQUESTED:
            commit_cancellation(job, when)

    try:
        current_job, input_path = storage.resolve_job_attempt_input(
            running_job.id,
            attempt_token,
        )
    except StaleJobAttemptError:
        cancellation.cancel()
        return
    except JobCancellationRequestedError:
        cancellation.cancel()
        commit_cancellation(running_job, finished_at(running_job))
        return

    if (
        current_job.canonical_options_json is None
        or current_job.selected_speaker_snapshot is None
        or current_job.effective_max_audio_samples is None
        or current_job.effective_direct_max_audio_samples is None
    ):
        raise StorageSchemaError(
            "running job execution metadata is incomplete"
        )
    try:
        options = parse_canonical_options_json(
            current_job.canonical_options_json
        )
        selected_speaker_snapshot = parse_selected_speaker_snapshot(
            current_job.selected_speaker_snapshot,
            speaker_embedding_policy,
            expected_ids=options.known_speaker_ids,
        )
    except ValueError as error:
        raise StorageSchemaError(
            "running job execution metadata is corrupt"
        ) from error

    try:
        reserved_writer = storage.begin_job_attempt_artifact(
            current_job.id,
            attempt_token,
        )
    except StaleJobAttemptError:
        cancellation.cancel()
        return
    except JobCancellationRequestedError:
        cancellation.cancel()
        commit_cancellation(current_job, finished_at(current_job))
        return

    writer = StorageArtifactByteWriter(storage, reserved_writer)
    segment_release_attempted = False
    try:
        sink = CanonicalJsonlSegmentSink(writer)
        progress = _DurableJobProgress(
            storage,
            current_job.id,
            attempt_token,
            cancellation,
        )
        try:
            result = processor.process(
                input_path,
                options,
                cancellation,
                progress,
                sink,
                selected_speaker_snapshot=selected_speaker_snapshot,
                effective_max_audio_samples=(
                    current_job.effective_max_audio_samples
                ),
                effective_direct_max_audio_samples=(
                    current_job.effective_direct_max_audio_samples
                ),
            )
            speaker_mapping = _validate_processor_result(
                result,
                writer,
            )
            total_samples = progress.finish()
        except StaleJobAttemptError:
            cancellation.cancel()
            return
        except JobCancellationRequestedError:
            cancellation.cancel()
            commit_cancellation(
                current_job,
                finished_at(current_job),
            )
            return
        except (PipelineError, AudioError) as error:
            if error.code == "cancelled":
                outcome = commit_cancellation(
                    current_job,
                    finished_at(current_job),
                )
                if outcome is JobTerminalOutcome.COMMITTED:
                    return
                if outcome is not JobTerminalOutcome.STALE:
                    raise RuntimeError(
                        "job cancellation commit returned an invalid outcome"
                    )
                raise
            error_code = (
                error.code
                if error.code in RUNTIME_JOB_FAILURE_CODES
                else "internal_error"
            )
            commit_failure(
                current_job,
                error_code,
                finished_at(current_job),
            )
            return
        except StorageSchemaError:
            raise
        except Exception:
            commit_failure(
                current_job,
                "internal_error",
                finished_at(current_job),
            )
            return

        segment_ref = writer.sealed_ref
        if segment_ref is None:
            raise RuntimeError(
                "processor did not seal its segment artifact"
            )
        reader = CanonicalJsonlReader(
            storage.resolve_artifact(segment_ref)
        )
        when = finished_at(current_job)
        if (
            current_job.request_fingerprint is None
            or current_job.processor_fingerprint is None
        ):
            raise StorageSchemaError(
                "running job fingerprints are incomplete"
            )
        manifest = ResultEnvelopeManifest(
            version=RESULT_ENVELOPE_VERSION,
            job_id=current_job.id,
            attempt_no=current_job.attempt_no,
            request_fingerprint=current_job.request_fingerprint,
            processor_fingerprint=current_job.processor_fingerprint,
            finished_at=when,
        )
        try:
            result_reserved_writer = (
                storage.begin_job_result_artifact(
                    current_job.id,
                    attempt_token,
                )
            )
            result_writer = StorageArtifactByteWriter(
                storage,
                result_reserved_writer,
            )
            result_ref = finalize_result_envelope(
                reader,
                options,
                total_samples,
                writer=result_writer,
                manifest=manifest,
                speaker_mapping=speaker_mapping,
            )
        except StaleJobAttemptError:
            cancellation.cancel()
            return
        except JobCancellationRequestedError:
            cancellation.cancel()
            commit_cancellation(current_job, when)
            return
        except StorageAdmissionError:
            commit_failure(
                current_job,
                "internal_error",
                when,
            )
            return

        segment_release_attempted = True
        writer._discard()
        if type(result_ref) is not ArtifactRef:
            raise RuntimeError(
                "result envelope returned an invalid artifact"
            )
        outcome = storage.commit_job_success(
            current_job.id,
            attempt_token,
            result_ref,
        )
        if outcome is JobSuccessOutcome.CANCEL_REQUESTED:
            cancellation.cancel()
            commit_cancellation(current_job, when)
        elif outcome not in {
            JobSuccessOutcome.COMMITTED,
            JobSuccessOutcome.STALE,
        }:
            raise RuntimeError("job success commit returned an invalid outcome")
    finally:
        if not segment_release_attempted:
            writer._discard()
