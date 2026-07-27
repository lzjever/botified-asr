from __future__ import annotations

import io
import wave
from pathlib import Path

import httpx
from openai import OpenAI
from starlette.testclient import TestClient

from botified_asr import pipeline as pipeline_module
from botified_asr.api import Readiness, create_app
from botified_asr.config import LimitsConfig, RESERVATION_QUANTUM
from botified_asr.pipeline import RichAnnotations, SegmentRecord
from botified_asr.speaker_matching import SpeakerLabelMapping
from botified_asr.storage import Storage


class SdkProcessor:
    def process(
        self,
        _input_path,
        _options,
        _cancellation,
        progress,
        sink,
    ):
        sink.append(
            SegmentRecord(
                0,
                0,
                1,
                "SDK works",
                "en",
                RichAnnotations(),
            )
        )
        progress.update(processed_samples=1, total_samples=None)
        ref = sink.finalize()
        return pipeline_module.ProcessorResult(
            ref,
            SpeakerLabelMapping(()),
        )


def test_uv_lock_uses_only_official_pypi() -> None:
    lock = Path("uv.lock").read_text(encoding="utf-8")
    assert "pypi.tuna.tsinghua.edu.cn" not in lock
    assert 'registry = "https://pypi.org/simple"' in lock


def test_openai_sdk_basic_sync_text_smoke(tmp_path) -> None:
    storage = Storage(
        tmp_path,
        LimitsConfig(
            max_upload_bytes=1024,
            sync_max_upload_bytes=1024,
            max_job_storage_bytes=2 * RESERVATION_QUANTUM,
            min_filesystem_free_bytes=1,
        ),
        free_bytes=lambda _: 1 << 40,
    )
    app = create_app(
        api_key="sdk-secret",
        readiness=Readiness(True, True, True),
        storage=storage,
        processor=SdkProcessor(),
        close_storage_on_shutdown=False,
    )
    service = TestClient(app)

    def forward(request: httpx.Request) -> httpx.Response:
        response = service.request(
            request.method,
            request.url.raw_path.decode("ascii"),
            headers=dict(request.headers),
            content=request.read(),
        )
        return httpx.Response(
            response.status_code,
            headers=response.headers,
            content=response.content,
        )

    sdk = OpenAI(
        api_key="sdk-secret",
        base_url="http://testserver/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(forward)),
    )
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\x00\x00" * 16)
    try:
        result = sdk.audio.transcriptions.create(
            model="sensevoice",
            file=("audio.wav", wav_buffer.getvalue(), "audio/wav"),
        )
        assert result.text == "SDK works"
    finally:
        sdk.close()
        service.close()
        storage.close()
