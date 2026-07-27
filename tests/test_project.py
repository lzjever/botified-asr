from __future__ import annotations

import httpx
from openai import OpenAI
from pathlib import Path
from starlette.testclient import TestClient

from botified_asr.api import Readiness, create_app
from botified_asr.config import LimitsConfig
from botified_asr.storage import Storage


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
            max_job_storage_bytes=8 * 1024 * 1024,
            min_filesystem_free_bytes=1,
        ),
        free_bytes=lambda _: 1 << 40,
    )
    app = create_app(
        api_key="sdk-secret",
        readiness=Readiness(True, True, True),
        storage=storage,
        transcriber=lambda _path, _options: {"text": "SDK works"},
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
    try:
        result = sdk.audio.transcriptions.create(
            model="sensevoice",
            file=("audio.wav", b"1234", "audio/wav"),
        )
        assert result.text == "SDK works"
    finally:
        sdk.close()
        service.close()
        storage.close()
