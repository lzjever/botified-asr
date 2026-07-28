from __future__ import annotations

import hashlib
import hmac
import re
from asyncio import Event, Task, create_task, wait_for
from collections.abc import Callable, Collection, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import anyio
from python_multipart import MultipartParser
from python_multipart.multipart import parse_options_header
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from botified_asr.audio import SAMPLE_RATE, AudioError, Cancellation, MediaProbe
from botified_asr.canonical_options import (
    CanonicalOptionsValidationError,
    MODEL_VALUES,
    canonicalize_option_values,
    serialize_canonical_options,
)
from botified_asr.composition import (
    PreparedSyncResponse,
    SpeakerEnrollmentProcessor,
    TranscriptionProcessor,
    prepare_sync_transcription,
)
from botified_asr.contracts import DIRECT_MAX_SAMPLES, CanonicalOptions
from botified_asr.errors import InferenceSaturated
from botified_asr.jobs import (
    JobDeletionOutcome,
    JobStatus,
    QueuedJobSpec,
    generate_public_id,
    validate_job_id,
)
from botified_asr.pipeline import PipelineError, PipelineNotReady
from botified_asr.result_artifact import CanonicalArtifactError
from botified_asr.runtime import JobExecutor
from botified_asr.speaker_profiles import (
    KEEP_EXISTING,
    MAX_SPEAKER_SAMPLES,
    MIN_SPEAKER_SAMPLES,
    ReservedSpeakerProfileNameError,
    SPEAKER_PROFILE_DESCRIPTION_MAX_CHARS,
    SPEAKER_PROFILE_NAME_MAX_CHARS,
    SpeakerEmbeddingReplacement,
    SpeakerProfile,
    SpeakerProfileUpdate,
    canonicalize_speaker_profile_name,
)
from botified_asr.speaker_snapshot import (
    SelectedSpeakerIncompatibleError,
    SelectedSpeakerNotFoundError,
)
from botified_asr.speakers import SpeakerEmbeddingPolicy
from botified_asr.storage import (
    InputRef,
    SpeakerProfileIdCollisionError,
    SpeakerProfileLimitReachedError,
    SpeakerProfileNameConflictError,
    Storage,
    StorageAdmissionError,
    StoredJobResult,
    UploadLease,
    JobInputRef,
    JobUploadLease,
)

MODEL_CREATED = 1785024000
MAX_PARTS = 64
MAX_PART_HEADER_BYTES = 32 * 1024
MAX_MULTIPART_OVERHEAD_BYTES = 1024 * 1024
PARSER_FEED_BYTES = 64 * 1024
SCALAR_FIELDS = {
    "model",
    "language",
    "response_format",
    "chunking_strategy",
}
ARRAY_FIELDS = {"include[]", "known_speaker_ids[]"}
ALL_FIELDS = SCALAR_FIELDS | ARRAY_FIELDS | {"file"}
SPEAKER_METADATA_FIELDS = {"name", "description"}
SPEAKER_FIELDS = SPEAKER_METADATA_FIELDS | {"samples[]"}
SPEAKER_EMBEDDING_MODEL_ALIAS = "cam++"
MAX_SPEAKER_SAMPLE_BYTES = 20 * 1024 * 1024
SPEAKER_ID_GENERATION_ATTEMPTS = 16
_LOWERCASE_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_RETENTION_SWEEP_BATCH_SIZE = 32
_RETENTION_SWEEP_INTERVAL_SECONDS = 60.0


class ApiError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        param: str | None = None,
        error_type: str = "invalid_request_error",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.param = param
        self.error_type = error_type


@dataclass
class Readiness:
    database: bool
    models: bool
    executor: bool

    @property
    def ready(self) -> bool:
        return self.database and self.models and self.executor


def canonicalize_options(
    fields: Mapping[str, Sequence[str]],
) -> CanonicalOptions:
    unknown = sorted(set(fields) - SCALAR_FIELDS - ARRAY_FIELDS)
    if unknown:
        raise _invalid_multipart(f"Unknown multipart field: {unknown[0]}")
    scalar = {name: _one(fields, name) for name in SCALAR_FIELDS if name in fields}
    option_values: dict[str, object] = {
        "model": scalar.get("model"),
        "include": tuple(fields.get("include[]", ())),
        "known_speaker_ids": tuple(fields.get("known_speaker_ids[]", ())),
    }
    option_values.update(
        (name, scalar[name])
        for name in (
            "language",
            "response_format",
            "chunking_strategy",
        )
        if name in scalar
    )
    try:
        return canonicalize_option_values(**option_values)
    except CanonicalOptionsValidationError as error:
        raise ApiError(
            400,
            error.code,
            error.message,
            param=error.param,
        ) from error


def create_app(
    *,
    api_key: str,
    readiness: Readiness,
    storage: Storage,
    processor: TranscriptionProcessor,
    speaker_embedding_policy: SpeakerEmbeddingPolicy,
    audio_prober: Callable[[Path, Cancellation], MediaProbe],
    processor_fingerprint: str,
    job_executor: JobExecutor | None = None,
    close_storage_on_shutdown: bool = True,
    speaker_enrollment_processor: SpeakerEnrollmentProcessor | None = None,
) -> Starlette:
    if not api_key:
        raise ValueError("api_key must not be empty")
    if type(speaker_embedding_policy) is not SpeakerEmbeddingPolicy:
        raise TypeError("speaker embedding policy is invalid")
    if not callable(audio_prober):
        raise TypeError("audio prober must be callable")
    if type(processor_fingerprint) is not str:
        raise TypeError("processor fingerprint must be a string")
    if _LOWERCASE_SHA256.fullmatch(processor_fingerprint) is None:
        raise ValueError("processor fingerprint is invalid")
    if speaker_enrollment_processor is not None and (
        not hasattr(speaker_enrollment_processor, "process")
        or not callable(speaker_enrollment_processor.process)
    ):
        raise TypeError("speaker enrollment processor is invalid")
    expected_api_key_digest = hashlib.sha256(api_key.encode("utf-8")).digest()

    def authenticate(request: Request) -> None:
        values = [
            value
            for name, value in request.scope.get("headers", ())
            if name.lower() == b"authorization"
        ]
        authorization = values[0] if len(values) == 1 else b""
        prefix = b"Bearer "
        candidate = (
            authorization[len(prefix) :]
            if authorization.startswith(prefix)
            else b""
        )
        candidate_digest = hashlib.sha256(candidate).digest()
        valid_ascii = bool(candidate) and candidate.isascii()
        if not (
            len(values) == 1
            and valid_ascii
            and hmac.compare_digest(candidate_digest, expected_api_key_digest)
        ):
            raise ApiError(
                401,
                "invalid_api_key",
                "Invalid authentication credentials",
                error_type="authentication_error",
            )

    def reject_not_ready() -> None:
        raise ApiError(
            503,
            "service_not_ready",
            "Service is not ready",
            error_type="server_error",
        )

    def require_ready() -> None:
        if job_executor is not None and not job_executor.ready:
            readiness.executor = False
        if not readiness.ready:
            reject_not_ready()

    async def live(_: Request) -> Response:
        return JSONResponse({"status": "ok"})

    async def ready(request: Request) -> Response:
        authenticate(request)
        require_ready()
        if not await run_in_threadpool(storage.probe_readiness):
            reject_not_ready()
        require_ready()
        return JSONResponse({"status": "ready"})

    async def list_models(request: Request) -> Response:
        authenticate(request)
        return JSONResponse({"object": "list", "data": _model_objects()})

    async def get_model(request: Request) -> Response:
        authenticate(request)
        model_id = request.path_params["model_id"]
        for item in _model_objects():
            if item["id"] == model_id:
                return JSONResponse(item)
        raise ApiError(
            404,
            "model_not_found",
            "Model not found",
            param="model_id",
        )

    async def list_speakers(request: Request) -> Response:
        authenticate(request)
        require_ready()
        profiles = storage.list_speaker_profiles()
        return JSONResponse(
            {
                "object": "list",
                "data": [_speaker_profile_object(profile) for profile in profiles],
            }
        )

    async def get_speaker(request: Request) -> Response:
        authenticate(request)
        require_ready()
        profile = storage.get_speaker_profile(request.path_params["speaker_id"])
        if profile is None:
            raise _speaker_not_found()
        return JSONResponse(_speaker_profile_object(profile))

    def require_speaker_enrollment() -> SpeakerEnrollmentProcessor:
        if speaker_enrollment_processor is None:
            raise _speaker_enrollment_unavailable()
        return speaker_enrollment_processor

    async def create_speaker(request: Request) -> Response:
        authenticate(request)
        require_ready()
        enrollment_processor = require_speaker_enrollment()
        prepared = await _prepare_speaker_change(
            request,
            storage,
            enrollment_processor,
            speaker_embedding_policy,
            require_samples=True,
            description_default=None,
        )
        now = datetime.now(timezone.utc)
        for _ in range(SPEAKER_ID_GENERATION_ATTEMPTS):
            replacement = prepared.embedding
            assert isinstance(replacement, SpeakerEmbeddingReplacement)
            profile = SpeakerProfile(
                id=generate_public_id(),
                name=prepared.name,
                description=prepared.description,
                embedding=replacement.embedding,
                embedding_model_id=replacement.embedding_model_id,
                embedding_model_revision=replacement.embedding_model_revision,
                embedding_dimension=replacement.embedding_dimension,
                embedding_policy_fingerprint=(replacement.embedding_policy_fingerprint),
                sample_count=replacement.sample_count,
                created_at=now,
                updated_at=now,
            )
            try:
                created = storage.create_speaker_profile(profile)
            except SpeakerProfileIdCollisionError:
                continue
            except SpeakerProfileNameConflictError as error:
                raise _speaker_name_conflict() from error
            except SpeakerProfileLimitReachedError as error:
                raise ApiError(
                    409,
                    "speaker_profile_limit_reached",
                    "Speaker profile limit reached",
                ) from error
            return JSONResponse(
                _speaker_profile_object(created),
                status_code=201,
            )
        raise ApiError(
            500,
            "internal_error",
            "Internal server error",
            error_type="server_error",
        )

    async def update_speaker(request: Request) -> Response:
        authenticate(request)
        require_ready()
        prepared = await _prepare_speaker_change(
            request,
            storage,
            speaker_enrollment_processor,
            speaker_embedding_policy,
            require_samples=False,
            description_default=KEEP_EXISTING,
        )

        try:
            profile = storage.update_speaker_profile(
                request.path_params["speaker_id"],
                SpeakerProfileUpdate(
                    name=prepared.name,
                    description=prepared.description,
                    embedding=prepared.embedding,
                    updated_at=datetime.now(timezone.utc),
                ),
            )
        except SpeakerProfileNameConflictError as error:
            raise _speaker_name_conflict() from error
        if profile is None:
            raise _speaker_not_found()
        return JSONResponse(_speaker_profile_object(profile))

    async def delete_speaker(request: Request) -> Response:
        authenticate(request)
        require_ready()
        deleted = storage.delete_speaker_profile(request.path_params["speaker_id"])
        if not deleted:
            raise _speaker_not_found()
        return Response(status_code=204)

    async def transcriptions(request: Request) -> Response:
        authenticate(request)
        require_ready()
        prefer_async = _prefers_async(request.headers)
        if prefer_async:
            try:
                return await _submit_async_transcription(
                    request,
                    storage,
                    audio_prober,
                    processor_fingerprint,
                    speaker_embedding_policy,
                    job_executor,
                )
            except StorageAdmissionError as exc:
                raise _storage_admission_error(exc) from exc
        prepared: PreparedSyncResponse | None = None
        response_owns_artifact = False
        try:
            lease = storage.begin_upload("transcription")
        except StorageAdmissionError as exc:
            raise _storage_admission_error(exc) from exc

        input_ref = None
        input_cleanup_complete = False
        try:
            fields = await _ingest_multipart(
                request,
                storage,
                lambda payload: storage.append(lease, payload),
                prefer_async=prefer_async,
            )
            options = canonicalize_options(fields)
            input_ref = storage.seal_upload(lease)
            input_path = storage.resolve_input(input_ref)
            cancellation = Cancellation()
            try:
                prepared = await _prepare_while_watching_disconnect(
                    request,
                    storage,
                    processor,
                    input_path,
                    input_ref.id,
                    options,
                    cancellation,
                    speaker_embedding_policy,
                    audio_prober,
                )
            except StorageAdmissionError:
                raise
            except SelectedSpeakerNotFoundError as exc:
                raise ApiError(
                    404,
                    "speaker_not_found",
                    "One or more known speakers were not found",
                    param="known_speaker_ids[]",
                ) from exc
            except SelectedSpeakerIncompatibleError as exc:
                raise ApiError(
                    409,
                    "speaker_profile_incompatible",
                    "One or more known speakers are incompatible",
                    param="known_speaker_ids[]",
                ) from exc
            except (
                AudioError,
                PipelineError,
                CanonicalArtifactError,
            ) as exc:
                raise _processing_api_error(exc) from exc
            except ApiError:
                raise
            except Exception as exc:
                raise ApiError(
                    500,
                    "internal_error",
                    "Internal server error",
                    error_type="server_error",
                ) from exc
            storage.release_input(input_ref)
            input_cleanup_complete = True
            response = _PreparedStreamingResponse(prepared)
            response_owns_artifact = True
            return response
        except StorageAdmissionError as exc:
            raise _storage_admission_error(exc) from exc
        finally:
            try:
                if not input_cleanup_complete:
                    if input_ref is None:
                        storage.abort_upload(lease)
                    else:
                        storage.release_input(input_ref)
            finally:
                if prepared is not None and not response_owns_artifact:
                    _close_prepared(prepared)

    async def get_transcription_job(request: Request) -> Response:
        authenticate(request)
        require_ready()
        job_id = request.path_params["job_id"]
        try:
            job = storage.get_visible_job(job_id)
        except (TypeError, ValueError):
            raise _job_not_found() from None
        if job is None:
            raise _job_not_found()
        if job.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
            return JSONResponse(
                {
                    "id": job.id,
                    "status": job.status.value,
                    "progress": {
                        "processed_audio_secs": (
                            job.processed_samples / SAMPLE_RATE
                        ),
                        "total_audio_secs": (
                            None
                            if job.total_samples is None
                            else job.total_samples / SAMPLE_RATE
                        ),
                    },
                },
                status_code=202,
            )
        if job.status is JobStatus.SUCCEEDED:
            try:
                stored_result = storage.open_succeeded_job_result(
                    job.id
                )
            except CanonicalArtifactError as error:
                raise _processing_api_error(error) from error
            if stored_result is None:
                raise _job_not_found()
            return _StoredJobStreamingResponse(job.id, stored_result)
        if job.status is JobStatus.FAILED:
            assert job.finished_at is not None
            known_error = job.error_code == "worker_crashed"
            return JSONResponse(
                {
                    "id": job.id,
                    "status": "failed",
                    "error": {
                        "message": (
                            "The transcription worker crashed"
                            if known_error
                            else "Internal server error"
                        ),
                        "type": "server_error",
                        "param": None,
                        "code": (
                            "worker_crashed"
                            if known_error
                            else "internal_error"
                        ),
                    },
                    "finished_at": _public_utc_timestamp(
                        job.finished_at
                    ),
                }
            )
        if job.status is JobStatus.CANCELLED:
            assert job.finished_at is not None
            return JSONResponse(
                {
                    "id": job.id,
                    "status": "cancelled",
                    "finished_at": _public_utc_timestamp(
                        job.finished_at
                    ),
                }
            )
        raise RuntimeError("visible job has an unsupported status")

    async def delete_transcription_job(request: Request) -> Response:
        authenticate(request)
        require_ready()
        try:
            job_id = validate_job_id(request.path_params["job_id"])
        except (TypeError, ValueError):
            raise _job_not_found() from None
        outcome = await run_in_threadpool(
            storage.delete_or_cancel_job,
            job_id,
            datetime.now(timezone.utc),
        )
        if outcome is JobDeletionOutcome.QUEUED_CANCELLED:
            return JSONResponse(
                {"id": job_id, "status": "cancelled"},
                status_code=202,
                background=BackgroundTask(
                    storage.cleanup_cancelled_job_input,
                    job_id,
                ),
            )
        if outcome is JobDeletionOutcome.RUNNING_CANCEL_REQUESTED:
            if job_executor is not None:
                job_executor.notify_cancellation(job_id)
            return JSONResponse(
                {"id": job_id, "status": "running"},
                status_code=202,
            )
        if outcome is JobDeletionOutcome.TERMINAL_DELETED:
            return Response(status_code=204)
        if outcome is JobDeletionOutcome.NOT_FOUND:
            raise _job_not_found()
        raise RuntimeError("storage returned an unsupported deletion outcome")

    async def sweep_expired_jobs() -> None:
        sweep_at = datetime.now(timezone.utc)
        for _ in range(_RETENTION_SWEEP_BATCH_SIZE):
            deleted = await run_in_threadpool(
                storage.delete_next_expired_terminal_job,
                sweep_at,
            )
            if not deleted:
                return

    @asynccontextmanager
    async def lifespan(_: Starlette):
        executor_started = False
        maintenance_stop = Event()
        maintenance_task: Task[None] | None = None

        async def maintain_retention() -> None:
            last_error: Exception | None = None
            while True:
                try:
                    await wait_for(
                        maintenance_stop.wait(),
                        _RETENTION_SWEEP_INTERVAL_SECONDS,
                    )
                except TimeoutError:
                    try:
                        await sweep_expired_jobs()
                    except Exception as error:
                        readiness.database = False
                        last_error = error
                    else:
                        readiness.database = True
                        last_error = None
                else:
                    if last_error is not None:
                        raise last_error
                    return

        try:
            if job_executor is not None:
                try:
                    await sweep_expired_jobs()
                except Exception:
                    readiness.database = False
                    raise
                job_executor.start()
                executor_started = True
                readiness.executor = job_executor.ready
                maintenance_task = create_task(maintain_retention())
            yield
        finally:
            readiness.executor = False
            maintenance_stop.set()
            try:
                if maintenance_task is not None:
                    await maintenance_task
            finally:
                try:
                    if executor_started:
                        await run_in_threadpool(job_executor.stop)
                finally:
                    if close_storage_on_shutdown:
                        storage.close()

    app = Starlette(
        debug=False,
        routes=[
            Route("/health/live", live, methods=["GET"]),
            Route("/health/ready", ready, methods=["GET"]),
            Route("/v1/models", list_models, methods=["GET"]),
            Route("/v1/models/{model_id}", get_model, methods=["GET"]),
            Route("/v1/speakers", list_speakers, methods=["GET"]),
            Route("/v1/speakers", create_speaker, methods=["POST"]),
            Route(
                "/v1/speakers/{speaker_id}",
                get_speaker,
                methods=["GET"],
            ),
            Route(
                "/v1/speakers/{speaker_id}",
                update_speaker,
                methods=["PUT"],
            ),
            Route(
                "/v1/speakers/{speaker_id}",
                delete_speaker,
                methods=["DELETE"],
            ),
            Route(
                "/v1/audio/transcriptions",
                transcriptions,
                methods=["POST"],
            ),
            Route(
                "/v1/audio/transcriptions/{job_id}",
                get_transcription_job,
                methods=["GET"],
            ),
            Route(
                "/v1/audio/transcriptions/{job_id}",
                delete_transcription_job,
                methods=["DELETE"],
            ),
        ],
        exception_handlers={
            ApiError: _api_error_response,
            HTTPException: _http_error_response,
            Exception: _internal_error_response,
        },
        lifespan=lifespan,
    )
    return app


def _speaker_profile_object(profile: SpeakerProfile) -> dict[str, object]:
    return {
        "id": profile.id,
        "object": "speaker",
        "name": profile.name,
        "description": profile.description,
        "sample_count": profile.sample_count,
        "embedding_model": {
            "id": SPEAKER_EMBEDDING_MODEL_ALIAS,
            "revision": profile.embedding_model_revision,
            "dimension": profile.embedding_dimension,
            "policy_fingerprint": profile.embedding_policy_fingerprint,
        },
        "created_at": _public_utc_timestamp(profile.created_at),
        "updated_at": _public_utc_timestamp(profile.updated_at),
    }


def _public_utc_timestamp(value: datetime) -> str:
    if value.tzinfo is not timezone.utc:
        raise ValueError("speaker profile timestamp is not canonical UTC")
    fractional = f".{value.microsecond:06d}" if value.microsecond else ""
    return (
        f"{value.year:04d}-{value.month:02d}-{value.day:02d}"
        f"T{value.hour:02d}:{value.minute:02d}:{value.second:02d}"
        f"{fractional}Z"
    )


def _speaker_not_found() -> ApiError:
    return ApiError(
        404,
        "speaker_not_found",
        "Speaker not found",
        param="speaker_id",
    )


def _speaker_name_conflict() -> ApiError:
    return ApiError(
        409,
        "speaker_name_conflict",
        "Speaker name already exists",
        param="name",
    )


@dataclass(frozen=True, slots=True)
class _PreparedSpeakerChange:
    name: str
    description: object
    embedding: object


class _SpeakerSampleUploads:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage
        self.current: UploadLease | None = None
        self.refs: list[InputRef] = []

    def begin(self) -> None:
        self.current = self.storage.begin_upload(
            "speaker_sample",
            owner_kind="sync",
        )

    def append(self, payload: bytes) -> None:
        assert self.current is not None
        self.storage.append(self.current, payload)

    def finish(self, size: int) -> None:
        if size == 0:
            raise _invalid_speaker_samples()
        assert self.current is not None
        self.refs.append(self.storage.seal_upload(self.current))
        self.current = None

    def paths(self) -> tuple[Path, ...]:
        return tuple(self.storage.resolve_input(ref) for ref in self.refs)

    def cleanup(self) -> None:
        cleanup_error: BaseException | None = None
        try:
            if self.current is not None:
                try:
                    self.storage.abort_upload(self.current)
                except BaseException as error:
                    cleanup_error = error
            for ref in self.refs:
                try:
                    self.storage.release_input(ref)
                except BaseException as error:
                    if cleanup_error is None:
                        cleanup_error = error
        finally:
            self.current = None
            self.refs.clear()
        if cleanup_error is not None:
            raise cleanup_error


async def _prepare_speaker_change(
    request: Request,
    storage: Storage,
    enrollment_processor: SpeakerEnrollmentProcessor | None,
    speaker_embedding_policy: SpeakerEmbeddingPolicy,
    *,
    require_samples: bool,
    description_default: object,
) -> _PreparedSpeakerChange:
    uploads = _SpeakerSampleUploads(storage)

    def begin_sample_upload() -> None:
        if enrollment_processor is None:
            raise _speaker_enrollment_unavailable()
        uploads.begin()

    handler = _MultipartFileHandler(
        field_name="samples[]",
        minimum=MIN_SPEAKER_SAMPLES,
        maximum=MAX_SPEAKER_SAMPLES,
        byte_limit=MAX_SPEAKER_SAMPLE_BYTES,
        required=require_samples,
        begin=begin_sample_upload,
        append=uploads.append,
        finish=uploads.finish,
        cardinality_error=_invalid_speaker_samples,
        too_large_error=lambda: ApiError(
            413,
            "upload_too_large",
            "Speaker sample exceeds 20 MiB",
            param="samples[]",
        ),
    )
    try:
        try:
            fields = await _ingest_multipart(
                request,
                storage,
                None,
                prefer_async=False,
                allowed_fields=SPEAKER_FIELDS,
                require_file=False,
                file_handler=handler,
            )
        except StorageAdmissionError as error:
            raise _storage_admission_error(error) from error
        name, description = _speaker_metadata(
            fields,
            description_default=description_default,
        )
        embedding: object = KEEP_EXISTING
        if uploads.refs:
            if enrollment_processor is None:
                raise _speaker_enrollment_unavailable()
            cancellation = Cancellation()
            try:
                sample_paths = uploads.paths()
                embedding = await _run_blocking_while_watching_disconnect(
                    request,
                    cancellation,
                    lambda: enrollment_processor.process(
                        sample_paths,
                        cancellation,
                        effective_max_audio_samples=(
                            storage.limits.sync_max_audio_duration_secs * SAMPLE_RATE
                        ),
                    ),
                )
            except (AudioError, PipelineError) as error:
                raise _speaker_processing_api_error(error) from error
            if not (
                isinstance(embedding, SpeakerEmbeddingReplacement)
                and _speaker_replacement_matches_policy(
                    embedding,
                    speaker_embedding_policy,
                )
            ):
                raise ApiError(
                    500,
                    "internal_error",
                    "Internal server error",
                    error_type="server_error",
                )
        return _PreparedSpeakerChange(name, description, embedding)
    finally:
        uploads.cleanup()


def _speaker_replacement_matches_policy(
    replacement: SpeakerEmbeddingReplacement,
    policy: SpeakerEmbeddingPolicy,
) -> bool:
    return (
        replacement.embedding_model_id == policy.model_id
        and replacement.embedding_model_revision == policy.model_revision
        and replacement.embedding_dimension == policy.embedding_dimension
        and replacement.embedding_policy_fingerprint == policy.fingerprint
    )


def _speaker_metadata(
    fields: Mapping[str, Sequence[str]],
    *,
    description_default: object,
) -> tuple[str, object]:
    if "name" not in fields:
        raise ApiError(
            400,
            "invalid_speaker_name",
            "Speaker name is required",
            param="name",
        )
    try:
        name = canonicalize_speaker_profile_name(_one(fields, "name"))
    except ReservedSpeakerProfileNameError as error:
        raise ApiError(
            400,
            "reserved_speaker_name",
            "Speaker name is reserved",
            param="name",
        ) from error
    except (TypeError, ValueError) as error:
        raise ApiError(
            400,
            "invalid_speaker_name",
            "Speaker name must contain 1 to "
            f"{SPEAKER_PROFILE_NAME_MAX_CHARS} characters",
            param="name",
        ) from error

    if "description" not in fields:
        description = description_default
    else:
        raw_description = _one(fields, "description")
        if len(raw_description) > SPEAKER_PROFILE_DESCRIPTION_MAX_CHARS:
            raise ApiError(
                400,
                "invalid_speaker_description",
                "Speaker description must not exceed "
                f"{SPEAKER_PROFILE_DESCRIPTION_MAX_CHARS} characters",
                param="description",
            )
        description = None if raw_description == "" else raw_description
    return name, description


def _invalid_speaker_samples() -> ApiError:
    return ApiError(
        400,
        "invalid_speaker_samples",
        "Speaker enrollment requires 2 to 5 non-empty samples",
        param="samples[]",
    )


def _speaker_enrollment_unavailable() -> ApiError:
    return ApiError(
        503,
        "pipeline_not_ready",
        "The requested audio pipeline is not ready",
        error_type="server_error",
    )


def _speaker_processing_api_error(
    error: AudioError | PipelineError,
) -> ApiError:
    if isinstance(error, InferenceSaturated):
        return ApiError(
            429,
            "inference_saturated",
            "Inference capacity is temporarily unavailable",
            error_type="rate_limit_error",
        )
    if isinstance(error, PipelineNotReady):
        return ApiError(
            503,
            "pipeline_not_ready",
            "The requested audio pipeline is not ready",
            error_type="server_error",
        )
    invalid_messages = {
        "invalid_audio": "Speaker sample is not valid audio",
        "invalid_speaker_samples": "Speaker enrollment samples are invalid",
        "invalid_speaker_sample_duration": "Speaker sample duration is invalid",
        "no_speech": "Speaker sample contains no speech",
        "invalid_speaker_embedding": (
            "Speaker enrollment produced an invalid embedding"
        ),
        "speaker_samples_inconsistent": "Speaker samples are inconsistent",
    }
    if error.code in invalid_messages:
        return ApiError(
            400,
            error.code,
            invalid_messages[error.code],
            param="samples[]",
        )
    if error.code == "audio_too_long":
        return ApiError(
            413,
            error.code,
            "Speaker sample exceeds the configured audio duration limit",
            param="samples[]",
        )
    if isinstance(error, AudioError) and error.code in {
        "audio_tool_unavailable",
        "audio_probe_timeout",
        "audio_decode_timeout",
    }:
        return ApiError(
            503,
            error.code,
            "Audio processing is unavailable",
            error_type="server_error",
        )
    return ApiError(
        500,
        "internal_error",
        "Internal server error",
        error_type="server_error",
    )


def _job_not_found() -> ApiError:
    return ApiError(
        404,
        "job_not_found",
        "Transcription job not found",
        param="job_id",
    )


class _PreparedStreamingResponse(StreamingResponse):
    def __init__(self, prepared: PreparedSyncResponse) -> None:
        self._prepared = prepared
        super().__init__(
            prepared.iter_body(),
            headers={"Content-Type": prepared.content_type},
            background=BackgroundTask(prepared.close),
        )

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            _close_prepared(self._prepared)


class _StoredJobStreamingResponse(StreamingResponse):
    def __init__(
        self,
        job_id: str,
        stored_result: StoredJobResult,
    ) -> None:
        self._stored_result = stored_result

        def body():
            yield (
                b'{"id":"'
                + job_id.encode("ascii")
                + b'","status":"succeeded","result":'
            )
            yield from stored_result.iter_body()
            yield b"}"

        self._stream = body()
        super().__init__(
            self._stream,
            headers={"Content-Type": "application/json"},
        )

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                self._stream.close()
            finally:
                self._stored_result.close()


def _close_prepared(prepared: PreparedSyncResponse) -> None:
    for attempt in range(2):
        try:
            prepared.close()
            return
        except Exception:
            if attempt == 1:
                raise


async def _submit_async_transcription(
    request: Request,
    storage: Storage,
    audio_prober: Callable[[Path, Cancellation], MediaProbe],
    processor_fingerprint: str,
    speaker_embedding_policy: SpeakerEmbeddingPolicy,
    job_executor: JobExecutor | None,
) -> Response:
    lease = storage.begin_job_upload(datetime.now(timezone.utc))
    cleanup_handle: JobUploadLease | JobInputRef = lease
    ownership_transferred = False
    try:
        fields = await _ingest_multipart(
            request,
            storage,
            lambda payload: storage.append_job_upload(lease, payload),
            prefer_async=True,
        )
        options = canonicalize_options(fields)
        input_ref = storage.seal_job_upload(lease)
        cleanup_handle = input_ref
        cancellation = Cancellation()
        try:
            probe = await run_in_threadpool(
                audio_prober,
                input_ref.path,
                cancellation,
            )
        except AudioError as exc:
            raise _processing_api_error(exc) from exc
        _validate_transcription_preflight(
            storage,
            options,
            probe,
            prefer_async=True,
        )
        spec = QueuedJobSpec(
            canonical_options_json=serialize_canonical_options(options),
            effective_max_audio_samples=(
                storage.limits.max_audio_duration_secs * SAMPLE_RATE
            ),
            effective_direct_max_audio_samples=min(
                storage.limits.direct_max_audio_duration_secs * SAMPLE_RATE,
                storage.limits.max_audio_duration_secs * SAMPLE_RATE,
                DIRECT_MAX_SAMPLES,
            ),
            processor_fingerprint=processor_fingerprint,
        )
        try:
            published = storage.publish_job(
                input_ref,
                spec,
                speaker_embedding_policy=speaker_embedding_policy,
            )
        except SelectedSpeakerNotFoundError as exc:
            raise ApiError(
                404,
                "speaker_not_found",
                "One or more known speakers were not found",
                param="known_speaker_ids[]",
            ) from exc
        except SelectedSpeakerIncompatibleError as exc:
            raise ApiError(
                409,
                "speaker_profile_incompatible",
                "One or more known speakers are incompatible",
                param="known_speaker_ids[]",
            ) from exc
        ownership_transferred = True
        if job_executor is not None:
            job_executor.wake()
        return JSONResponse(
            {
                "id": published.id,
                "status": published.status.value,
                "created_at": _public_utc_timestamp(published.created_at),
            },
            status_code=202,
            headers={
                "Preference-Applied": "respond-async",
                "Location": f"/v1/audio/transcriptions/{published.id}",
            },
        )
    finally:
        if not ownership_transferred:
            storage.abort_job_upload(cleanup_handle)


def _validate_transcription_preflight(
    storage: Storage,
    options: CanonicalOptions,
    probe: MediaProbe,
    *,
    prefer_async: bool,
) -> None:
    if probe.duration_seconds > storage.limits.max_audio_duration_secs:
        raise ApiError(
            413,
            "audio_too_long",
            "Audio exceeds max_audio_duration_secs",
            param="file",
        )
    if options.chunking_strategy is None:
        if (
            probe.duration_seconds
            > storage.limits.direct_max_audio_duration_secs
        ):
            raise ApiError(
                422,
                "long_audio_requires_vad",
                "chunking_strategy=auto is required for long audio",
                param="chunking_strategy",
            )
    if (
        not prefer_async
        and probe.duration_seconds
        > storage.limits.sync_max_audio_duration_secs
    ):
        raise ApiError(
            422,
            "async_required",
            "Prefer: respond-async is required for this audio duration",
            param="file",
        )


async def _prepare_while_watching_disconnect(
    request: Request,
    storage: Storage,
    processor: TranscriptionProcessor,
    input_path: Path,
    owner_id: str,
    options: CanonicalOptions,
    cancellation: Cancellation,
    speaker_embedding_policy: SpeakerEmbeddingPolicy,
    audio_prober: Callable[[Path, Cancellation], MediaProbe],
) -> PreparedSyncResponse:
    def prepare_in_worker() -> PreparedSyncResponse:
        media_probe = audio_prober(input_path, cancellation)
        _validate_transcription_preflight(
            storage,
            options,
            media_probe,
            prefer_async=False,
        )
        result = prepare_sync_transcription(
            storage,
            processor,
            input_path,
            owner_id,
            options,
            cancellation,
            speaker_embedding_policy=speaker_embedding_policy,
            media_probe=media_probe,
        )
        return result

    def discard_prepared(value: object) -> None:
        if isinstance(value, PreparedSyncResponse):
            _close_prepared(value)

    result = await _run_blocking_while_watching_disconnect(
        request,
        cancellation,
        prepare_in_worker,
        discard=discard_prepared,
    )
    if not isinstance(result, PreparedSyncResponse):
        raise RuntimeError("transcription preparation returned an invalid response")
    return result


async def _run_blocking_while_watching_disconnect(
    request: Request,
    cancellation: Cancellation,
    work: Callable[[], object],
    *,
    discard: Callable[[object], None] | None = None,
) -> object:
    result: list[object] = []
    worker_errors: list[BaseException] = []
    worker_finished = anyio.Event()
    propagated_cancellation: BaseException | None = None
    disconnected = False
    work_finished = False
    ownership_transferred = False

    async def run_worker() -> None:
        try:
            result.append(await run_in_threadpool(work))
        except BaseException as error:
            worker_errors.append(error)
        finally:
            worker_finished.set()

    async def watch_disconnect() -> None:
        nonlocal disconnected
        try:
            while True:
                message = await request.receive()
                if message["type"] == "http.disconnect":
                    disconnected = True
                    cancellation.cancel()
                    return
        finally:
            if not work_finished:
                cancellation.cancel()

    try:
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(watch_disconnect)
            task_group.start_soon(run_worker)
            try:
                await worker_finished.wait()
            except anyio.get_cancelled_exc_class() as error:
                propagated_cancellation = error
                cancellation.cancel()
                with anyio.CancelScope(shield=True):
                    await worker_finished.wait()
            finally:
                work_finished = True
                task_group.cancel_scope.cancel()
                if disconnected and propagated_cancellation is None:
                    try:
                        await anyio.lowlevel.checkpoint()
                    except anyio.get_cancelled_exc_class() as error:
                        propagated_cancellation = error

        if propagated_cancellation is not None:
            raise propagated_cancellation
        if disconnected:
            raise RuntimeError("disconnect cancellation was not propagated")
        if worker_errors:
            raise worker_errors[0]
        if not result:
            raise RuntimeError("blocking worker returned no result")
        try:
            await anyio.lowlevel.checkpoint_if_cancelled()
        except anyio.get_cancelled_exc_class():
            cancellation.cancel()
            raise
        ownership_transferred = True
        return result[0]
    finally:
        if not worker_finished.is_set():
            cancellation.cancel()
            with anyio.CancelScope(shield=True):
                await worker_finished.wait()
        if result and not ownership_transferred and discard is not None:
            discard(result[0])


def _processing_api_error(
    exc: AudioError | PipelineError | CanonicalArtifactError,
) -> ApiError:
    if isinstance(exc, InferenceSaturated):
        return ApiError(
            429,
            "inference_saturated",
            "Inference capacity is temporarily unavailable",
            error_type="rate_limit_error",
        )
    if isinstance(exc, PipelineNotReady):
        return ApiError(
            503,
            "pipeline_not_ready",
            "The requested audio pipeline is not ready",
            error_type="server_error",
        )
    if isinstance(exc, CanonicalArtifactError):
        return ApiError(
            500,
            "invalid_result_artifact",
            "The transcription result artifact is invalid",
            error_type="server_error",
        )
    if isinstance(exc, PipelineError):
        if exc.code == "long_audio_requires_vad":
            return ApiError(
                422,
                exc.code,
                "chunking_strategy=auto is required for long audio",
                param="chunking_strategy",
            )
        if exc.code == "invalid_audio":
            return ApiError(
                400,
                exc.code,
                "The uploaded file is not valid audio",
                param="file",
            )
        if exc.code == "audio_too_long":
            return ApiError(
                413,
                exc.code,
                "Audio exceeds max_audio_duration_secs",
                param="file",
            )
        return ApiError(
            500,
            "internal_error",
            "Internal server error",
            error_type="server_error",
        )
    if exc.code == "invalid_audio":
        return ApiError(
            400,
            exc.code,
            "The uploaded file is not valid audio",
            param="file",
        )
    unavailable_messages = {
        "audio_tool_unavailable": "Audio processing is unavailable",
        "audio_probe_timeout": "Audio probing timed out",
        "audio_decode_timeout": "Audio decoding timed out",
    }
    if exc.code in unavailable_messages:
        return ApiError(
            503,
            exc.code,
            unavailable_messages[exc.code],
            error_type="server_error",
        )
    return ApiError(
        500,
        "internal_error",
        "Internal server error",
        error_type="server_error",
    )


async def _ingest_multipart(
    request: Request,
    storage: Storage,
    append_file: Callable[[bytes], None] | None,
    *,
    prefer_async: bool,
    allowed_fields: Collection[str] = ALL_FIELDS,
    require_file: bool = True,
    file_handler: _MultipartFileHandler | None = None,
) -> dict[str, list[str]]:
    content_type = request.headers.get("content-type", "")
    media_type, options = parse_options_header(content_type)
    boundary = options.get(b"boundary")
    if media_type != b"multipart/form-data" or not boundary:
        raise _invalid_multipart("multipart/form-data with a boundary is required")

    if file_handler is not None and append_file is not None:
        raise TypeError("multipart accepts only one file handler")
    if file_handler is None and append_file is not None:
        limits = storage.limits
        if prefer_async or (limits.sync_max_upload_bytes == limits.max_upload_bytes):
            byte_limit = limits.max_upload_bytes
            too_large_status = 413
            too_large_code = "upload_too_large"
            too_large_message = "Upload exceeds max_upload_bytes"
        else:
            byte_limit = limits.sync_max_upload_bytes
            too_large_status = 422
            too_large_code = "async_required"
            too_large_message = "Prefer: respond-async is required for this upload size"

        def too_large_error() -> ApiError:
            return ApiError(
                too_large_status,
                too_large_code,
                too_large_message,
                param="file",
            )

        file_handler = _MultipartFileHandler(
            field_name="file",
            minimum=1,
            maximum=1,
            byte_limit=byte_limit,
            required=require_file,
            begin=lambda: None,
            append=append_file,
            finish=lambda _size: None,
            cardinality_error=lambda: _invalid_multipart(
                "Exactly one file part is required"
            ),
            too_large_error=too_large_error,
        )

    state = _MultipartState(
        file_handler,
        allowed_fields=allowed_fields,
    )
    parser = MultipartParser(boundary, state.callbacks)
    raw_body_limit = (
        file_handler.maximum * file_handler.byte_limit
        if file_handler is not None
        else 0
    ) + MAX_MULTIPART_OVERHEAD_BYTES
    try:
        async for chunk in request.stream():
            offset = 0
            while offset < len(chunk):
                remaining = raw_body_limit - state.body_bytes
                if remaining <= 0:
                    raise _invalid_multipart(
                        "Multipart body exceeds file plus overhead limits"
                    )
                feed_size = min(
                    PARSER_FEED_BYTES,
                    len(chunk) - offset,
                    remaining,
                )
                feed = chunk[offset : offset + feed_size]
                state.body_bytes += len(feed)
                parser.write(feed)
                offset += feed_size
                if offset < len(chunk) and state.body_bytes == raw_body_limit:
                    raise _invalid_multipart(
                        "Multipart body exceeds file plus overhead limits"
                    )
        parser.finalize()
    except ClientDisconnect as exc:
        raise ApiError(
            400, "client_disconnected", "Client disconnected during upload"
        ) from exc
    except StorageAdmissionError:
        raise
    except ApiError:
        raise
    except Exception as exc:
        raise _invalid_multipart("Malformed multipart body") from exc

    if not state.ended:
        raise _invalid_multipart("Multipart body did not terminate")
    state.validate_file_count()
    if state.body_bytes - state.total_file_bytes > MAX_MULTIPART_OVERHEAD_BYTES:
        raise _invalid_multipart("Multipart overhead exceeds 1 MiB")
    return state.fields


@dataclass(frozen=True, slots=True)
class _MultipartFileHandler:
    field_name: str
    minimum: int
    maximum: int
    byte_limit: int
    required: bool
    begin: Callable[[], None]
    append: Callable[[bytes], None]
    finish: Callable[[int], None]
    cardinality_error: Callable[[], ApiError]
    too_large_error: Callable[[], ApiError]


class _MultipartState:
    def __init__(
        self,
        file_handler: _MultipartFileHandler | None,
        *,
        allowed_fields: Collection[str],
    ) -> None:
        self.file_handler = file_handler
        self.allowed_fields = frozenset(allowed_fields)
        self.body_bytes = 0
        self.total_file_bytes = 0
        self.file_count = 0
        self.scalar_bytes = 0
        self.part_count = 0
        self.ended = False
        self.fields: dict[str, list[str]] = {}
        self._header_field = bytearray()
        self._header_value = bytearray()
        self._headers: dict[bytes, bytes] = {}
        self._header_bytes = 0
        self._name: str | None = None
        self._is_file = False
        self._file_bytes = 0
        self._value = bytearray()
        self.callbacks = {
            "on_part_begin": self.on_part_begin,
            "on_header_field": self.on_header_field,
            "on_header_value": self.on_header_value,
            "on_header_end": self.on_header_end,
            "on_headers_finished": self.on_headers_finished,
            "on_part_data": self.on_part_data,
            "on_part_end": self.on_part_end,
            "on_end": self.on_end,
        }

    def on_part_begin(self) -> None:
        self.part_count += 1
        if self.part_count > MAX_PARTS:
            raise _invalid_multipart("Multipart contains more than 64 parts")
        self._header_field.clear()
        self._header_value.clear()
        self._headers.clear()
        self._header_bytes = 2
        self._name = None
        self._is_file = False
        self._file_bytes = 0
        self._value.clear()

    def on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._header_field.extend(data[start:end])
        self._check_header_size()

    def on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._header_value.extend(data[start:end])
        self._check_header_size()

    def on_header_end(self) -> None:
        field = bytes(self._header_field).lower()
        value = bytes(self._header_value)
        self._header_bytes += len(field) + len(value) + 4
        self._check_header_size()
        if field in self._headers:
            raise _invalid_multipart("Duplicate part header")
        self._headers[field] = value
        self._header_field.clear()
        self._header_value.clear()

    def on_headers_finished(self) -> None:
        disposition = self._headers.get(b"content-disposition")
        if disposition is None:
            raise _invalid_multipart("Missing Content-Disposition")
        kind, options = parse_options_header(disposition)
        name_bytes = options.get(b"name")
        if kind != b"form-data" or not name_bytes:
            raise _invalid_multipart("Invalid Content-Disposition")
        try:
            name = name_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _invalid_multipart("Part name must be UTF-8") from exc
        if name not in self.allowed_fields:
            raise _invalid_multipart("Unknown multipart part")
        filename = options.get(b"filename")
        if self.file_handler is not None and name == self.file_handler.field_name:
            if filename is None:
                raise _invalid_multipart("File part requires a filename")
            if self.file_count >= self.file_handler.maximum:
                raise self.file_handler.cardinality_error()
            self.file_count += 1
            self._is_file = True
            self.file_handler.begin()
        elif filename is not None:
            raise _invalid_multipart("Unexpected file part")
        elif name not in ARRAY_FIELDS and name in self.fields:
            raise _invalid_multipart("Scalar fields must not be repeated")
        self._name = name

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        payload = data[start:end]
        if self._is_file:
            self._append_file(payload)
            return
        if len(self._value) + len(payload) > MAX_MULTIPART_OVERHEAD_BYTES:
            raise _invalid_multipart("Multipart overhead exceeds 1 MiB")
        self.scalar_bytes += len(payload)
        if self.scalar_bytes > MAX_MULTIPART_OVERHEAD_BYTES:
            raise _invalid_multipart("Multipart overhead exceeds 1 MiB")
        self._value.extend(payload)

    def on_part_end(self) -> None:
        if self._name is None:
            raise _invalid_multipart("Part headers were not completed")
        if self._is_file:
            assert self.file_handler is not None
            self.file_handler.finish(self._file_bytes)
            return
        try:
            value = self._value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _invalid_multipart("Multipart field must be UTF-8") from exc
        self.fields.setdefault(self._name, []).append(value)
        self._value.clear()

    def on_end(self) -> None:
        self.ended = True

    def _append_file(self, payload: bytes) -> None:
        assert self.file_handler is not None
        remaining = self.file_handler.byte_limit - self._file_bytes
        if len(payload) > remaining:
            if remaining > 0:
                prefix = payload[:remaining]
                self.file_handler.append(prefix)
                self._file_bytes += len(prefix)
                self.total_file_bytes += len(prefix)
            raise self.file_handler.too_large_error()
        self.file_handler.append(payload)
        self._file_bytes += len(payload)
        self.total_file_bytes += len(payload)

    def validate_file_count(self) -> None:
        if self.file_handler is None:
            return
        if self.file_count == 0 and not self.file_handler.required:
            return
        if self.file_count < self.file_handler.minimum:
            raise self.file_handler.cardinality_error()

    def _check_header_size(self) -> None:
        current = (
            self._header_bytes
            + len(self._header_field)
            + len(self._header_value)
        )
        if current > MAX_PART_HEADER_BYTES:
            raise _invalid_multipart("Part header exceeds 32 KiB")



def _one(fields: Mapping[str, Sequence[str]], name: str) -> str:
    values = fields[name]
    if len(values) != 1:
        raise _invalid_multipart(f"{name} must not be repeated")
    return values[0]


def _model_objects() -> list[dict[str, object]]:
    return [
        {
            "id": model_id,
            "object": "model",
            "created": MODEL_CREATED,
            "owned_by": "botified-asr",
        }
        for model_id in MODEL_VALUES
    ]


def _prefers_async(headers: Headers) -> bool:
    return any(
        item.strip().lower() == "respond-async"
        for item in headers.get("prefer", "").split(",")
    )


def _invalid_multipart(message: str) -> ApiError:
    return ApiError(
        400, "invalid_multipart", message, param=None
    )


def _storage_admission_error(exc: StorageAdmissionError) -> ApiError:
    return ApiError(
        429,
        exc.code,
        "Storage admission is saturated",
        error_type="rate_limit_error",
    )


def _api_error_response(_: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "message": exc.message,
                "type": exc.error_type,
                "param": exc.param,
                "code": exc.code,
            }
        },
        status_code=exc.status_code,
    )


def _http_error_response(_: Request, exc: HTTPException) -> JSONResponse:
    code = "not_found" if exc.status_code == 404 else "http_error"
    return _api_error_response(
        _,
        ApiError(
            exc.status_code,
            code,
            "Not found" if exc.status_code == 404 else "HTTP error",
        ),
    )


def _internal_error_response(_: Request, __: Exception) -> JSONResponse:
    return _api_error_response(
        _,
        ApiError(
            500,
            "internal_error",
            "Internal server error",
            error_type="server_error",
        ),
    )
