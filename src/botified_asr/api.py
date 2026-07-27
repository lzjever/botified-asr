from __future__ import annotations

import hashlib
import hmac
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from python_multipart import MultipartParser
from python_multipart.multipart import parse_options_header
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

from botified_asr.contracts import CanonicalOptions
from botified_asr.storage import (
    Storage,
    StorageAdmissionError,
    UploadLease,
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
INCLUDE_ORDER = ("funasr.emotion", "funasr.audio_events")
SPEAKER_ID_PATTERN = re.compile(r"^[0-9A-HJKMNP-TV-Z]{8}$")


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


Transcriber = Callable[[Path, CanonicalOptions], Mapping[str, object]]


def canonicalize_options(
    fields: Mapping[str, Sequence[str]],
) -> CanonicalOptions:
    unknown = sorted(set(fields) - SCALAR_FIELDS - ARRAY_FIELDS)
    if unknown:
        raise _invalid_multipart(f"Unknown multipart field: {unknown[0]}")
    scalar = {
        name: _one(fields, name)
        for name in SCALAR_FIELDS
        if name in fields
    }
    model = scalar.get("model")
    if model not in {"sensevoice", "sensevoice-diarize"}:
        raise ApiError(
            400,
            "invalid_model",
            "model must be sensevoice or sensevoice-diarize",
            param="model",
        )

    language = scalar.get("language", "auto")
    if language not in {"auto", "zh", "en", "yue", "ja", "ko"}:
        raise ApiError(
            400,
            "invalid_language",
            "unsupported language",
            param="language",
        )

    response_format = scalar.get("response_format")
    chunking_strategy = scalar.get("chunking_strategy")
    if chunking_strategy not in {None, "auto"}:
        raise ApiError(
            400,
            "invalid_chunking_strategy",
            "chunking_strategy must be auto when provided",
            param="chunking_strategy",
        )

    if model == "sensevoice-diarize":
        if chunking_strategy != "auto":
            raise ApiError(
                400,
                "diarization_requires_vad",
                "sensevoice-diarize requires explicit chunking_strategy=auto",
                param="chunking_strategy",
            )
        if response_format != "diarized_json":
            raise ApiError(
                400,
                "diarization_requires_format",
                "sensevoice-diarize requires explicit response_format=diarized_json",
                param="response_format",
            )
    else:
        response_format = response_format or "json"
        if response_format == "diarized_json":
            raise ApiError(
                400,
                "diarized_format_requires_model",
                "diarized_json requires model=sensevoice-diarize",
                param="response_format",
            )

    if response_format not in {"json", "text", "verbose_json", "diarized_json"}:
        raise ApiError(
            400,
            "invalid_response_format",
            "unsupported response_format",
            param="response_format",
        )

    include_values = list(fields.get("include[]", ()))
    invalid_include = next(
        (value for value in include_values if value not in INCLUDE_ORDER), None
    )
    if invalid_include is not None:
        raise ApiError(
            400,
            "invalid_include",
            "unsupported include value",
            param="include[]",
        )
    includes = tuple(value for value in INCLUDE_ORDER if value in include_values)
    if response_format == "text" and includes:
        raise ApiError(
            400,
            "incompatible_response_format",
            "response_format=text cannot be combined with include[]",
            param="response_format",
        )

    known_ids = list(fields.get("known_speaker_ids[]", ()))
    if len(known_ids) != len(set(known_ids)):
        raise ApiError(
            400,
            "invalid_known_speaker_ids",
            "known_speaker_ids[] must not contain duplicates",
            param="known_speaker_ids[]",
        )
    if len(known_ids) > 32 or any(
        SPEAKER_ID_PATTERN.fullmatch(value) is None for value in known_ids
    ):
        raise ApiError(
            400,
            "invalid_known_speaker_ids",
            "known_speaker_ids[] contains an invalid speaker ID",
            param="known_speaker_ids[]",
        )
    if known_ids and model != "sensevoice-diarize":
        raise ApiError(
            400,
            "known_speakers_require_diarization",
            "known_speaker_ids[] requires model=sensevoice-diarize",
            param="known_speaker_ids[]",
        )

    return CanonicalOptions(
        model=model,
        language=language,
        response_format=response_format,
        chunking_strategy=chunking_strategy,
        include=includes,
        known_speaker_ids=tuple(sorted(known_ids)),
    )


def create_app(
    *,
    api_key: str,
    readiness: Readiness,
    storage: Storage,
    transcriber: Transcriber,
    close_storage_on_shutdown: bool = True,
) -> Starlette:
    if not api_key:
        raise ValueError("api_key must not be empty")
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

    async def transcriptions(request: Request) -> Response:
        authenticate(request)
        require_ready()
        prefer_async = _prefers_async(request.headers)
        try:
            lease = storage.begin_upload("transcription")
        except StorageAdmissionError as exc:
            raise ApiError(
                429,
                exc.code,
                "Upload admission is saturated",
                error_type="rate_limit_error",
            ) from exc

        try:
            fields = await _ingest_multipart(
                request, storage, lease, prefer_async=prefer_async
            )
            options = canonicalize_options(fields)
            input_path = storage.complete_upload(lease)
            if prefer_async:
                raise ApiError(
                    503,
                    "service_not_ready",
                    "Async job executor is not ready",
                    error_type="server_error",
                )
            result = dict(
                await run_in_threadpool(transcriber, input_path, options)
            )
            if options.response_format == "text":
                return PlainTextResponse(str(result.get("text", "")))
            return JSONResponse(result)
        finally:
            storage.abort_upload(lease)

    @asynccontextmanager
    async def lifespan(_: Starlette):
        try:
            yield
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
            Route(
                "/v1/audio/transcriptions",
                transcriptions,
                methods=["POST"],
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


async def _ingest_multipart(
    request: Request,
    storage: Storage,
    lease: UploadLease,
    *,
    prefer_async: bool,
) -> dict[str, list[str]]:
    content_type = request.headers.get("content-type", "")
    media_type, options = parse_options_header(content_type)
    boundary = options.get(b"boundary")
    if media_type != b"multipart/form-data" or not boundary:
        raise _invalid_multipart("multipart/form-data with a boundary is required")

    state = _MultipartState(storage, lease, prefer_async=prefer_async)
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
        lease: UploadLease,
        *,
        prefer_async: bool,
    ) -> None:
        self.storage = storage
        self.lease = lease
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
                self.storage.append(self.lease, prefix)
                self.file_bytes += len(prefix)
            raise ApiError(status, code, message, param="file")
        self.storage.append(self.lease, payload)
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
