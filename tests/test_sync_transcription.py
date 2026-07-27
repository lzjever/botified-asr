from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import anyio
import pytest
from starlette.testclient import TestClient

from botified_asr.api import Readiness, create_app
from botified_asr.audio import AudioError
from botified_asr.config import LimitsConfig, RESERVATION_QUANTUM
from botified_asr.pipeline import (
    PipelineError,
    PipelineNotReady,
    RichAnnotations,
    SegmentRecord,
)
from botified_asr.storage import ArtifactRef, Storage


AUTHORIZATION = (b"authorization", b"Bearer test-secret")


class SpyProcessor:
    def __init__(self, behavior: str = "success") -> None:
        self.behavior = behavior
        self.calls = 0
        self.input_bytes: bytes | None = None
        self.ran_off_event_loop = False
        self.cancelled = False
        self.cancellation = None
        self.started = threading.Event()
        self.cancel_observed = threading.Event()
        self.allow_finish = threading.Event()
        self.finalized_ref: ArtifactRef | None = None
        self.finished = threading.Event()

    def process(
        self,
        input_path,
        _options,
        cancellation,
        progress,
        sink,
    ):
        self.calls += 1
        self.started.set()
        self.cancellation = cancellation
        self.input_bytes = input_path.read_bytes()
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self.ran_off_event_loop = True
        try:
            if self.behavior == "wait_for_cancel":
                assert cancellation._event.wait(timeout=2)
                self.cancelled = cancellation.cancelled
                raise PipelineError(
                    "cancelled",
                    "cancelled /private/input.wav",
                )
            if self.behavior in {
                "success_after_cancel",
                "success_after_cancel_gate",
            }:
                cancellation._event.wait(timeout=0.25)
                self.cancelled = cancellation.cancelled
                self.cancel_observed.set()
                if self.behavior == "success_after_cancel_gate":
                    assert self.allow_finish.wait(timeout=2)
            if self.behavior == "long":
                progress.update(
                    processed_samples=480_001,
                    total_samples=None,
                )
                raise PipelineError(
                    "long_audio_requires_vad",
                    "private long-audio detail",
                )
            if self.behavior == "model":
                raise RuntimeError("model failed at /private/model.bin")
            if self.behavior == "not_ready":
                raise PipelineNotReady()
            if self.behavior == "invalid_audio":
                raise AudioError(
                    "invalid_audio",
                    "bad input /private/input.wav",
                )
            if self.behavior.startswith("audio_"):
                raise AudioError(
                    self.behavior,
                    "tool detail /usr/bin/ffmpeg",
                )
            if self.behavior == "invalid_artifact":
                progress.update(processed_samples=1, total_samples=None)
                sink._writer.write(b"not canonical jsonl\n")
                return sink._writer.seal()

            sink.append(
                SegmentRecord(
                    0,
                    0,
                    4,
                    "hello",
                    "en",
                    RichAnnotations("happy", "speech"),
                )
            )
            progress.update(processed_samples=4, total_samples=None)
            ref = sink.finalize()
            assert isinstance(ref, ArtifactRef)
            self.finalized_ref = ref
            return ref
        finally:
            self.finished.set()


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    value = Storage(
        tmp_path,
        LimitsConfig(
            max_upload_bytes=1024 * 1024,
            sync_max_upload_bytes=1024 * 1024,
            max_job_storage_bytes=2 * RESERVATION_QUANTUM,
            min_filesystem_free_bytes=1,
        ),
        free_bytes=lambda _: 1 << 40,
    )
    yield value
    value.close()


def _app(storage: Storage, processor: Any):
    return create_app(
        api_key="test-secret",
        readiness=Readiness(True, True, True),
        storage=storage,
        processor=processor,
        close_storage_on_shutdown=False,
    )


def _files(response_format: str = "json") -> dict[str, tuple]:
    return {
        "file": ("audio.wav", b"stored-input", "audio/wav"),
        "model": (None, "sensevoice"),
        "response_format": (None, response_format),
    }


def _assert_no_resources(storage: Storage) -> None:
    assert storage.total_reserved_bytes() == 0
    assert not list(storage.staging_dir.iterdir())
    assert not list(storage.artifact_dir.iterdir())
    assert (
        storage._connection.execute("SELECT COUNT(*) FROM storage_leases").fetchone()[0]
        == 0
    )


@pytest.mark.parametrize(
    ("response_format", "content_type"),
    [
        ("json", "application/json"),
        ("text", "text/plain; charset=utf-8"),
        ("verbose_json", "application/json"),
    ],
)
def test_sync_projection_is_exact_and_cleans_after_normal_consumption(
    storage: Storage,
    response_format: str,
    content_type: str,
) -> None:
    processor = SpyProcessor()
    with TestClient(_app(storage, processor)) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer test-secret"},
            files=_files(response_format),
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == content_type
    if response_format == "text":
        assert response.content == b"hello"
    else:
        payload = response.json()
        assert payload["text"] == "hello"
        if response_format == "verbose_json":
            assert payload["duration"] == 4 / 16_000
            assert payload["segments"] == [
                {
                    "id": "0",
                    "start": 0.0,
                    "end": 4 / 16_000,
                    "text": "hello",
                }
            ]
    assert processor.calls == 1
    assert processor.input_bytes == b"stored-input"
    assert processor.ran_off_event_loop
    assert not processor.cancellation.cancelled
    _assert_no_resources(storage)


@pytest.mark.parametrize(
    ("behavior", "status", "code", "param"),
    [
        ("long", 422, "long_audio_requires_vad", "chunking_strategy"),
        ("model", 500, "internal_error", None),
        ("not_ready", 503, "pipeline_not_ready", None),
        ("invalid_audio", 400, "invalid_audio", "file"),
        ("audio_tool_unavailable", 503, "audio_tool_unavailable", None),
        ("audio_probe_timeout", 503, "audio_probe_timeout", None),
        ("audio_decode_timeout", 503, "audio_decode_timeout", None),
        ("invalid_artifact", 500, "invalid_result_artifact", None),
    ],
)
def test_processing_and_prepare_errors_are_stable_redacted_and_clean(
    storage: Storage,
    behavior: str,
    status: int,
    code: str,
    param: str | None,
) -> None:
    with TestClient(_app(storage, SpyProcessor(behavior))) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer test-secret"},
            files=_files(),
        )

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert response.json()["error"]["param"] == param
    assert "/private" not in response.text
    assert "/usr/bin" not in response.text
    _assert_no_resources(storage)


@pytest.mark.parametrize("fault", ["write", "seal"])
def test_artifact_storage_faults_are_internal_and_clean(
    storage: Storage,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    method = "append_artifact" if fault == "write" else "seal_artifact"

    def fail(*_args: object) -> None:
        raise OSError(f"{fault} failed /private/storage")

    monkeypatch.setattr(storage, method, fail)
    with TestClient(_app(storage, SpyProcessor())) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer test-secret"},
            files=_files(),
        )

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "/private" not in response.text
    _assert_no_resources(storage)


def test_artifact_admission_after_input_seal_is_429_and_cleans(
    tmp_path: Path,
) -> None:
    constrained = Storage(
        tmp_path,
        LimitsConfig(
            max_upload_bytes=1024,
            sync_max_upload_bytes=1024,
            max_job_storage_bytes=RESERVATION_QUANTUM,
            min_filesystem_free_bytes=1,
        ),
        free_bytes=lambda _: 1 << 40,
    )
    processor = SpyProcessor()
    try:
        with TestClient(_app(constrained, processor)) as client:
            response = client.post(
                "/v1/audio/transcriptions",
                headers={"Authorization": "Bearer test-secret"},
                files=_files(),
            )

        assert response.status_code == 429
        assert response.json()["error"]["code"] == "storage_capacity_exceeded"
        assert processor.calls == 0
        _assert_no_resources(constrained)
    finally:
        constrained.close()


def _multipart_body() -> tuple[bytes, bytes]:
    boundary = b"sync-lifecycle-boundary"
    body = (
        b"--"
        + boundary
        + b'\r\nContent-Disposition: form-data; name="file"; filename="a.wav"'
        + b"\r\nContent-Type: audio/wav\r\n\r\nstored-input"
        + b"\r\n--"
        + boundary
        + b'\r\nContent-Disposition: form-data; name="model"\r\n\r\nsensevoice'
        + b"\r\n--"
        + boundary
        + b"--\r\n"
    )
    return boundary, body


def _scope(boundary: bytes) -> dict[str, object]:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/audio/transcriptions",
        "raw_path": b"/v1/audio/transcriptions",
        "query_string": b"",
        "headers": [
            AUTHORIZATION,
            (
                b"content-type",
                b"multipart/form-data; boundary=" + boundary,
            ),
        ],
        "client": ("test", 1),
        "server": ("testserver", 80),
    }


def test_input_is_released_while_artifact_is_owned_before_first_body(
    storage: Storage,
) -> None:
    async def run() -> list[dict[str, object]]:
        boundary, body = _multipart_body()
        app = _app(storage, SpyProcessor())
        sent: list[dict[str, object]] = []
        request_sent = False

        async def receive() -> dict[str, object]:
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {
                    "type": "http.request",
                    "body": body,
                    "more_body": False,
                }
            await anyio.sleep_forever()

        async def send(message: dict[str, object]) -> None:
            if message["type"] == "http.response.start":
                assert not list(storage.staging_dir.iterdir())
                assert len(list(storage.artifact_dir.iterdir())) == 1
                rows = storage._connection.execute(
                    "SELECT lease_type FROM storage_leases"
                ).fetchall()
                assert [row["lease_type"] for row in rows] == ["artifact"]
            sent.append(message)

        await app(_scope(boundary), receive, send)
        return sent

    messages = asyncio.run(run())
    assert sum(message["type"] == "http.response.start" for message in messages) == 1
    _assert_no_resources(storage)


@pytest.mark.parametrize(
    "failure",
    ["never_first_next", "send_body", "body"],
)
def test_stream_failures_release_without_second_response_start(
    storage: Storage,
    failure: str,
) -> None:
    async def run() -> tuple[list[dict[str, object]], BaseException | None]:
        boundary, body = _multipart_body()
        app = _app(storage, SpyProcessor())
        sent: list[dict[str, object]] = []
        request_sent = False

        async def receive() -> dict[str, object]:
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {
                    "type": "http.request",
                    "body": body,
                    "more_body": False,
                }
            await anyio.sleep_forever()

        async def send(message: dict[str, object]) -> None:
            sent.append(message)
            if message["type"] == "http.response.start":
                if failure == "body":
                    artifact = next(storage.artifact_dir.iterdir())
                    artifact.write_bytes(b"corrupt after prepare\n")
                if failure == "never_first_next":
                    raise OSError("send start failed")
            elif (
                message["type"] == "http.response.body"
                and message.get("body")
                and failure == "send_body"
            ):
                raise OSError("send body failed")

        try:
            await app(_scope(boundary), receive, send)
        except BaseException as exc:
            return sent, exc
        return sent, None

    sent, caught = asyncio.run(run())
    assert caught is not None
    assert sum(message["type"] == "http.response.start" for message in sent) == 1
    wire_body = b"".join(
        message.get("body", b"")
        for message in sent
        if message["type"] == "http.response.body"
    )
    assert b"corrupt" not in wire_body
    assert b"/private" not in wire_body
    _assert_no_resources(storage)


def test_response_disconnect_after_start_releases_artifact(
    storage: Storage,
) -> None:
    async def run() -> list[dict[str, object]]:
        boundary, body = _multipart_body()
        app = _app(storage, SpyProcessor())
        request_sent = False
        response_started = anyio.Event()
        sent: list[dict[str, object]] = []

        async def receive() -> dict[str, object]:
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {
                    "type": "http.request",
                    "body": body,
                    "more_body": False,
                }
            await response_started.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, object]) -> None:
            sent.append(message)
            if message["type"] == "http.response.start":
                response_started.set()

        await app(_scope(boundary), receive, send)
        return sent

    messages = asyncio.run(run())
    assert sum(message["type"] == "http.response.start" for message in messages) == 1
    _assert_no_resources(storage)


def test_processing_disconnect_cancels_and_waits_for_cleanup(
    storage: Storage,
) -> None:
    async def run() -> None:
        boundary, body = _multipart_body()
        processor = SpyProcessor("wait_for_cancel")
        app = _app(storage, processor)
        events = [
            {
                "type": "http.request",
                "body": body,
                "more_body": False,
            },
            {"type": "http.disconnect"},
        ]
        sent: list[dict[str, object]] = []

        async def receive() -> dict[str, object]:
            return events.pop(0)

        async def send(message: dict[str, object]) -> None:
            sent.append(message)

        try:
            await app(_scope(boundary), receive, send)
        except anyio.get_cancelled_exc_class():
            pass
        else:
            pytest.fail("processing disconnect did not propagate cancellation")
        assert processor.calls == 1
        assert processor.cancelled
        assert processor.finished.is_set()
        assert sent == []

    asyncio.run(run())
    _assert_no_resources(storage)


def test_processing_task_cancellation_triggers_token_and_waits(
    storage: Storage,
) -> None:
    async def run() -> None:
        boundary, body = _multipart_body()
        processor = SpyProcessor("wait_for_cancel")
        app = _app(storage, processor)
        request_sent = False
        processor_scope: anyio.CancelScope | None = None
        sent: list[dict[str, object]] = []

        async def receive() -> dict[str, object]:
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {
                    "type": "http.request",
                    "body": body,
                    "more_body": False,
                }
            await anyio.sleep_forever()

        async def send(message: dict[str, object]) -> None:
            sent.append(message)

        async def call_app() -> None:
            nonlocal processor_scope
            with anyio.CancelScope() as scope:
                processor_scope = scope
                await app(_scope(boundary), receive, send)

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(call_app)
            await anyio.to_thread.run_sync(processor.started.wait)
            assert processor_scope is not None
            processor_scope.cancel()

        assert processor.cancelled
        assert processor.finished.is_set()
        assert sent == []

    asyncio.run(run())
    _assert_no_resources(storage)


def test_processing_task_cancellation_cleans_successful_local_prepared(
    storage: Storage,
) -> None:
    async def run() -> None:
        boundary, body = _multipart_body()
        processor = SpyProcessor("success_after_cancel")
        app = _app(storage, processor)
        request_sent = False
        processor_scope: anyio.CancelScope | None = None
        sent: list[dict[str, object]] = []

        async def receive() -> dict[str, object]:
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {
                    "type": "http.request",
                    "body": body,
                    "more_body": False,
                }
            await anyio.sleep_forever()

        async def send(message: dict[str, object]) -> None:
            sent.append(message)

        async def call_app() -> None:
            nonlocal processor_scope
            with anyio.CancelScope() as scope:
                processor_scope = scope
                await app(_scope(boundary), receive, send)

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(call_app)
            await anyio.to_thread.run_sync(processor.started.wait)
            assert processor_scope is not None
            processor_scope.cancel()

        assert processor.cancelled
        assert processor.finalized_ref is not None
        assert processor.finished.is_set()
        assert sent == []

    asyncio.run(run())
    _assert_no_resources(storage)


def test_native_asyncio_task_cancel_waits_for_successful_worker_cleanup(
    storage: Storage,
) -> None:
    async def run() -> None:
        boundary, body = _multipart_body()
        processor = SpyProcessor("success_after_cancel")
        app = _app(storage, processor)
        request_sent = False
        sent: list[dict[str, object]] = []

        async def receive() -> dict[str, object]:
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {
                    "type": "http.request",
                    "body": body,
                    "more_body": False,
                }
            await anyio.sleep_forever()

        async def send(message: dict[str, object]) -> None:
            sent.append(message)

        task = asyncio.create_task(app(_scope(boundary), receive, send))
        await anyio.to_thread.run_sync(processor.started.wait)
        task.cancel()
        try:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)
            assert processor.cancelled
            assert processor.finalized_ref is not None
            assert processor.finished.is_set()
            assert sent == []
        finally:
            assert await anyio.to_thread.run_sync(
                processor.finished.wait,
                2,
            )

    asyncio.run(run())
    _assert_no_resources(storage)


def test_repeated_native_task_cancel_still_recovers_local_prepared(
    storage: Storage,
) -> None:
    async def run() -> None:
        boundary, body = _multipart_body()
        processor = SpyProcessor("success_after_cancel_gate")
        app = _app(storage, processor)
        request_sent = False
        sent: list[dict[str, object]] = []

        async def receive() -> dict[str, object]:
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {
                    "type": "http.request",
                    "body": body,
                    "more_body": False,
                }
            await anyio.sleep_forever()

        async def send(message: dict[str, object]) -> None:
            sent.append(message)

        task = asyncio.create_task(app(_scope(boundary), receive, send))
        await anyio.to_thread.run_sync(processor.started.wait)
        task.cancel()
        await anyio.to_thread.run_sync(processor.cancel_observed.wait)
        task.cancel()
        processor.allow_finish.set()
        try:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)
            assert processor.cancelled
            assert processor.finalized_ref is not None
            assert processor.finished.is_set()
            assert sent == []
        finally:
            processor.allow_finish.set()
            assert await anyio.to_thread.run_sync(
                processor.finished.wait,
                2,
            )

    asyncio.run(run())
    _assert_no_resources(storage)


def test_third_native_cancel_cannot_interrupt_bounded_owner_close(
    storage: Storage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_release = storage.release_artifact
    release_started = threading.Event()
    allow_release = threading.Event()
    release_finished = threading.Event()
    release_calls = 0
    captured_ref: ArtifactRef | None = None

    def block_then_fail_once(ref: ArtifactRef) -> None:
        nonlocal captured_ref, release_calls
        captured_ref = ref
        release_calls += 1
        release_started.set()
        try:
            assert allow_release.wait(timeout=2)
            if release_calls == 1:
                raise OSError("first release failed")
            original_release(ref)
        finally:
            release_finished.set()

    monkeypatch.setattr(
        storage,
        "release_artifact",
        block_then_fail_once,
    )

    async def run() -> None:
        boundary, body = _multipart_body()
        processor = SpyProcessor("success_after_cancel_gate")
        app = _app(storage, processor)
        request_sent = False
        sent: list[dict[str, object]] = []

        async def receive() -> dict[str, object]:
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {
                    "type": "http.request",
                    "body": body,
                    "more_body": False,
                }
            await anyio.sleep_forever()

        async def send(message: dict[str, object]) -> None:
            sent.append(message)

        task = asyncio.create_task(app(_scope(boundary), receive, send))
        loop = asyncio.get_running_loop()

        def request_third_cancel() -> None:
            assert release_started.wait(timeout=2)
            loop.call_soon_threadsafe(task.cancel)
            allow_release.set()

        cancel_thread = threading.Thread(
            target=request_third_cancel,
        )
        cancel_thread.start()
        await anyio.to_thread.run_sync(processor.started.wait)
        task.cancel()
        await anyio.to_thread.run_sync(processor.cancel_observed.wait)
        task.cancel()
        processor.allow_finish.set()
        try:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)
            assert processor.cancelled
            assert processor.finalized_ref is not None
            assert processor.finished.is_set()
            assert release_calls == 2
            assert sent == []
        finally:
            processor.allow_finish.set()
            allow_release.set()
            await anyio.to_thread.run_sync(release_finished.wait)
            await anyio.to_thread.run_sync(cancel_thread.join)

    try:
        asyncio.run(run())
        _assert_no_resources(storage)
    finally:
        if storage.total_reserved_bytes() and captured_ref is not None:
            original_release(captured_ref)


def test_response_task_cancellation_after_start_still_releases(
    storage: Storage,
) -> None:
    async def run() -> list[dict[str, object]]:
        boundary, body = _multipart_body()
        app = _app(storage, SpyProcessor())
        request_sent = False
        response_scope: anyio.CancelScope | None = None
        sent: list[dict[str, object]] = []

        async def receive() -> dict[str, object]:
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {
                    "type": "http.request",
                    "body": body,
                    "more_body": False,
                }
            await anyio.sleep_forever()

        async def send(message: dict[str, object]) -> None:
            sent.append(message)
            if message["type"] == "http.response.start":
                assert response_scope is not None
                response_scope.cancel()

        with anyio.CancelScope() as scope:
            response_scope = scope
            await app(_scope(boundary), receive, send)
        return sent

    messages = asyncio.run(run())
    assert sum(message["type"] == "http.response.start" for message in messages) == 1
    _assert_no_resources(storage)


@pytest.mark.parametrize(
    ("release_failures", "expected_error"),
    [
        (1, "send start failed"),
        (2, "release failed 2"),
    ],
)
def test_send_start_failure_has_bounded_release_retry_and_priority(
    storage: Storage,
    monkeypatch: pytest.MonkeyPatch,
    release_failures: int,
    expected_error: str,
) -> None:
    original_release = storage.release_artifact
    release_calls = 0
    captured_ref = None

    def fail_release(ref) -> None:
        nonlocal captured_ref, release_calls
        captured_ref = ref
        release_calls += 1
        if release_calls <= release_failures:
            raise OSError(f"release failed {release_calls}")
        original_release(ref)

    monkeypatch.setattr(storage, "release_artifact", fail_release)

    async def run() -> tuple[list[dict[str, object]], BaseException | None]:
        boundary, body = _multipart_body()
        app = _app(storage, SpyProcessor())
        request_sent = False
        sent: list[dict[str, object]] = []

        async def receive() -> dict[str, object]:
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {
                    "type": "http.request",
                    "body": body,
                    "more_body": False,
                }
            await anyio.sleep_forever()

        async def send(message: dict[str, object]) -> None:
            sent.append(message)
            if message["type"] == "http.response.start":
                raise OSError("send start failed")

        try:
            await app(_scope(boundary), receive, send)
        except BaseException as exc:
            return sent, exc
        return sent, None

    sent, caught = asyncio.run(run())
    try:
        assert caught is not None
        assert str(caught) == expected_error
        assert release_calls == 2
        assert sum(message["type"] == "http.response.start" for message in sent) == 1
        if release_failures == 1:
            _assert_no_resources(storage)
        else:
            assert storage.total_reserved_bytes() > 0
    finally:
        if release_failures == 2 and captured_ref is not None:
            original_release(captured_ref)


def test_production_has_no_legacy_transcriber_or_mapping_result_path() -> None:
    api_source = Path("src/botified_asr/api.py").read_text(encoding="utf-8")
    main_source = Path("src/botified_asr/main.py").read_text(encoding="utf-8")

    assert "Transcriber" not in api_source
    assert "transcriber=" not in api_source
    assert "transcriber=" not in main_source
    assert "Mapping[str, object]" not in api_source
    assert "PlainTextResponse" not in api_source
