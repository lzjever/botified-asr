from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Callable, Mapping, Sequence
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
    canonicalize_option_values,
    serialize_canonical_options,
)
from botified_asr.composition import (
    PreparedSyncResponse,
    TranscriptionProcessor,
    prepare_sync_transcription,
)
from botified_asr.contracts import DIRECT_MAX_SAMPLES, CanonicalOptions
from botified_asr.errors import InferenceSaturated
from botified_asr.jobs import JobStatus, QueuedJobSpec
from botified_asr.pipeline import PipelineError, PipelineNotReady
from botified_asr.result_artifact import CanonicalArtifactError
from botified_asr.runtime import JobExecutor
from botified_asr.speaker_profiles import SpeakerProfile
from botified_asr.speaker_snapshot import (
    SelectedSpeakerIncompatibleError,
    SelectedSpeakerNotFoundError,
)
from botified_asr.speakers import SpeakerEmbeddingPolicy
from botified_asr.storage import (
    Storage,
    StorageAdmissionError,
    StoredJobResult,
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
SPEAKER_EMBEDDING_MODEL_ALIAS = "cam++"
_LOWERCASE_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")


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

    def require_ready() -> None:
        if job_executor is not None and not job_executor.ready:
            readiness.executor = False
        if not readiness.ready:
            raise ApiError(
                503,
                "service_not_ready",
                "Service is not ready",
                error_type="server_error",
            )

    async def live(_: Request) -> Response:
        return JSONResponse({"status": "ok"})

    async def ready(request: Request) -> Response:
        authenticate(request)
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
                }
            )
        if job.status is JobStatus.SUCCEEDED:
            try:
                stored_result = storage.open_succeeded_job_result(
                    job.id
                )
            except CanonicalArtifactError as error:
                raise _processing_api_error(error) from error
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

    @asynccontextmanager
    async def lifespan(_: Starlette):
        executor_started = False
        try:
            if job_executor is not None:
                job_executor.start()
                executor_started = True
                readiness.executor = job_executor.ready
            yield
        finally:
            readiness.executor = False
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
            Route(
                "/v1/speakers/{speaker_id}",
                get_speaker,
                methods=["GET"],
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
        _validate_async_preflight(storage, options, probe)
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


def _validate_async_preflight(
    storage: Storage,
    options: CanonicalOptions,
    probe: MediaProbe,
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


async def _prepare_while_watching_disconnect(
    request: Request,
    storage: Storage,
    processor: TranscriptionProcessor,
    input_path: Path,
    owner_id: str,
    options: CanonicalOptions,
    cancellation: Cancellation,
    speaker_embedding_policy: SpeakerEmbeddingPolicy,
) -> PreparedSyncResponse:
    prepared: PreparedSyncResponse | None = None
    worker_errors: list[BaseException] = []
    propagated_cancellation: BaseException | None = None
    disconnected = False
    preparation_finished = False
    ownership_transferred = False
    worker_result: list[PreparedSyncResponse] = []
    worker_finished = anyio.Event()

    def prepare_in_worker() -> PreparedSyncResponse:
        result = prepare_sync_transcription(
            storage,
            processor,
            input_path,
            owner_id,
            options,
            cancellation,
            speaker_embedding_policy=speaker_embedding_policy,
        )
        worker_result.append(result)
        return result

    async def run_worker() -> None:
        try:
            await run_in_threadpool(prepare_in_worker)
        except BaseException as exc:
            worker_errors.append(exc)
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
            if not preparation_finished:
                cancellation.cancel()

    try:
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(watch_disconnect)
            task_group.start_soon(run_worker)
            try:
                await worker_finished.wait()
            except anyio.get_cancelled_exc_class() as exc:
                propagated_cancellation = exc
                cancellation.cancel()
                with anyio.CancelScope(shield=True):
                    await worker_finished.wait()
            finally:
                if worker_result:
                    prepared = worker_result[0]
                preparation_finished = True
                task_group.cancel_scope.cancel()
                if disconnected and propagated_cancellation is None:
                    try:
                        await anyio.lowlevel.checkpoint()
                    except anyio.get_cancelled_exc_class() as exc:
                        propagated_cancellation = exc

        if propagated_cancellation is not None:
            raise propagated_cancellation
        if disconnected:
            raise RuntimeError(
                "disconnect cancellation was not propagated"
            )
        if worker_errors:
            raise worker_errors[0]
        if prepared is None:
            raise RuntimeError(
                "transcription preparation returned no response"
            )
        try:
            await anyio.lowlevel.checkpoint_if_cancelled()
        except anyio.get_cancelled_exc_class():
            cancellation.cancel()
            raise
        ownership_transferred = True
        return prepared
    finally:
        if not worker_finished.is_set():
            cancellation.cancel()
            with anyio.CancelScope(shield=True):
                await worker_finished.wait()
        if prepared is None and worker_result:
            prepared = worker_result[0]
        if prepared is not None and not ownership_transferred:
            _close_prepared(prepared)


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
    append_file: Callable[[bytes], None],
    *,
    prefer_async: bool,
) -> dict[str, list[str]]:
    content_type = request.headers.get("content-type", "")
    media_type, options = parse_options_header(content_type)
    boundary = options.get(b"boundary")
    if media_type != b"multipart/form-data" or not boundary:
        raise _invalid_multipart("multipart/form-data with a boundary is required")

    state = _MultipartState(
        storage,
        append_file,
        prefer_async=prefer_async,
    )
    parser = MultipartParser(boundary, state.callbacks)
    raw_body_limit = (
        state.file_byte_limit + MAX_MULTIPART_OVERHEAD_BYTES
    )
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
    except ApiError:
        raise
    except Exception as exc:
        raise _invalid_multipart("Malformed multipart body") from exc

    if not state.ended or not state.file_seen:
        raise _invalid_multipart("Exactly one file part is required")
    if state.body_bytes - state.file_bytes > MAX_MULTIPART_OVERHEAD_BYTES:
        raise _invalid_multipart("Multipart overhead exceeds 1 MiB")
    return state.fields


class _MultipartState:
    def __init__(
        self,
        storage: Storage,
        append_file: Callable[[bytes], None],
        *,
        prefer_async: bool,
    ) -> None:
        self.storage = storage
        self.append_file = append_file
        self.prefer_async = prefer_async
        self.body_bytes = 0
        self.file_bytes = 0
        self.file_seen = False
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
        if name not in ALL_FIELDS:
            raise _invalid_multipart("Unknown multipart part")
        filename = options.get(b"filename")
        if name == "file":
            if filename is None or self.file_seen:
                raise _invalid_multipart("Exactly one file part is allowed")
            self.file_seen = True
            self._is_file = True
        elif filename is not None:
            raise _invalid_multipart("Only file may be a file part")
        elif name in SCALAR_FIELDS and name in self.fields:
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
        limits = self.storage.limits
        if self.prefer_async or (
            limits.sync_max_upload_bytes == limits.max_upload_bytes
        ):
            byte_limit = limits.max_upload_bytes
            status = 413
            code = "upload_too_large"
            message = "Upload exceeds max_upload_bytes"
        else:
            byte_limit = limits.sync_max_upload_bytes
            status = 422
            code = "async_required"
            message = "Prefer: respond-async is required for this upload size"

        remaining = byte_limit - self.file_bytes
        if len(payload) > remaining:
            if remaining > 0:
                prefix = payload[:remaining]
                self.append_file(prefix)
                self.file_bytes += len(prefix)
            raise ApiError(status, code, message, param="file")
        self.append_file(payload)
        self.file_bytes += len(payload)

    @property
    def file_byte_limit(self) -> int:
        limits = self.storage.limits
        if self.prefer_async or (
            limits.sync_max_upload_bytes == limits.max_upload_bytes
        ):
            return limits.max_upload_bytes
        return limits.sync_max_upload_bytes

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
        for model_id in ("sensevoice", "sensevoice-diarize")
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
