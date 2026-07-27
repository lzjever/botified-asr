from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request

import pytest

from botified_asr import huggingface_fetcher, model_artifacts


REVISION = "0123456789abcdef0123456789abcdef01234567"


def _file(
    relative_path: str,
    payload: bytes,
) -> model_artifacts.ModelArtifactFile:
    return model_artifacts.ModelArtifactFile(
        relative_path=relative_path,
        sha256=hashlib.sha256(payload).hexdigest(),
        expected_bytes=len(payload),
    )


def _spec(
    *files: model_artifacts.ModelArtifactFile,
) -> model_artifacts.ModelArtifactSpec:
    return model_artifacts.ModelArtifactSpec(
        provider="huggingface",
        model_id="example/model",
        revision=REVISION,
        files=tuple(files),
    )


class FakeResponse:
    def __init__(
        self,
        payload: bytes,
        *,
        headers: dict[str, str] | None = None,
        status: int = 200,
        final_url: str | None = None,
        max_return_bytes: int | None = None,
        fail_on_read: int | None = None,
    ) -> None:
        self._payload = payload
        self._offset = 0
        self._max_return_bytes = max_return_bytes
        self._fail_on_read = fail_on_read
        self.headers = headers or {}
        self.status = status
        self.final_url = final_url
        self.read_sizes: list[int] = []
        self.returned_sizes: list[int] = []
        self.closed = False

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self.closed = True

    def geturl(self) -> str:
        assert self.final_url is not None
        return self.final_url

    def read(self, size: int) -> bytes:
        assert type(size) is int
        assert size > 0
        self.read_sizes.append(size)
        if self._fail_on_read == len(self.read_sizes):
            raise OSError("response read failed")
        returned = size
        if self._max_return_bytes is not None:
            returned = min(returned, self._max_return_bytes)
        chunk = self._payload[self._offset : self._offset + returned]
        self._offset += len(chunk)
        self.returned_sizes.append(len(chunk))
        return chunk


class FakeOpener:
    def __init__(self, *outcomes: FakeResponse | Exception) -> None:
        self._outcomes = outcomes
        self.calls: list[tuple[Request, float]] = []

    def __call__(self, request: Request, *, timeout: float) -> FakeResponse:
        self.calls.append((request, timeout))
        outcome = self._outcomes[len(self.calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        if outcome.final_url is None:
            outcome.final_url = request.full_url
        return outcome


def test_fetcher_requests_only_declared_files_in_manifest_order(
    tmp_path: Path,
) -> None:
    payloads = (b"model", b"sidecar")
    files = (
        _file("model.bin", payloads[0]),
        _file("dir/a b#?.bin", payloads[1]),
    )
    responses = tuple(
        FakeResponse(
            payload,
            headers={"Content-Length": str(len(payload))},
        )
        for payload in payloads
    )
    opener = FakeOpener(*responses)
    destination = tmp_path / "staging"
    destination.mkdir()

    huggingface_fetcher.HuggingFaceSnapshotFetcher(
        opener=opener,
        timeout_seconds=30,
    ).fetch(_spec(*files), destination)

    assert [request.full_url for request, _timeout in opener.calls] == [
        f"https://huggingface.co/example/model/resolve/{REVISION}/model.bin",
        (
            "https://huggingface.co/example/model/resolve/"
            f"{REVISION}/dir/a%20b%23%3F.bin"
        ),
    ]
    assert [timeout for _request, timeout in opener.calls] == [30, 30]
    assert all(
        request.get_header("Accept-encoding") == "identity"
        for request, _timeout in opener.calls
    )
    assert (destination / "model.bin").read_bytes() == payloads[0]
    assert (destination / "dir/a b#?.bin").read_bytes() == payloads[1]
    assert sorted(
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    ) == ["dir/a b#?.bin", "model.bin"]
    assert all(response.closed for response in responses)


def test_fetcher_streams_with_public_bounded_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(huggingface_fetcher, "DOWNLOAD_CHUNK_BYTES", 4)
    payload = b"abcdefghijk"
    response = FakeResponse(payload)
    opener = FakeOpener(response)
    destination = tmp_path / "staging"
    destination.mkdir()

    huggingface_fetcher.HuggingFaceSnapshotFetcher(
        opener=opener,
        timeout_seconds=30,
    ).fetch(_spec(_file("model.bin", payload)), destination)

    assert len(response.read_sizes) >= 3
    remaining = len(payload)
    for requested, returned in zip(
        response.read_sizes,
        response.returned_sizes,
        strict=True,
    ):
        assert 0 < requested <= huggingface_fetcher.DOWNLOAD_CHUNK_BYTES
        assert requested <= remaining + 1
        remaining -= returned
    assert remaining == 0
    assert (destination / "model.bin").read_bytes() == payload
    assert response.closed


@pytest.mark.parametrize(
    "mode",
    (
        "short",
        "oversize",
        "content_length",
        "open_error",
        "read_error",
        "bad_status",
        "insecure_final_url",
    ),
)
def test_fetcher_stops_after_first_transport_or_size_failure(
    tmp_path: Path,
    mode: str,
) -> None:
    expected = b"abcd"
    response: FakeResponse | None
    if mode == "short":
        response = FakeResponse(b"abc")
        first: FakeResponse | Exception = response
    elif mode == "oversize":
        response = FakeResponse(b"abcde")
        first = response
    elif mode == "content_length":
        response = FakeResponse(expected, headers={"Content-Length": "5"})
        first = response
    elif mode == "open_error":
        response = None
        first = OSError("open failed")
    elif mode == "read_error":
        response = FakeResponse(
            expected,
            max_return_bytes=2,
            fail_on_read=2,
        )
        first = response
    elif mode == "bad_status":
        response = FakeResponse(expected, status=503)
        first = response
    else:
        response = FakeResponse(
            expected,
            final_url="http://cdn.example.invalid/model.bin",
        )
        first = response
    opener = FakeOpener(first, FakeResponse(b"x"))
    destination = tmp_path / "staging"
    destination.mkdir()
    spec = _spec(
        _file("model.bin", expected),
        _file("never-requested.bin", b"x"),
    )

    with pytest.raises(OSError):
        huggingface_fetcher.HuggingFaceSnapshotFetcher(
            opener=opener,
            timeout_seconds=30,
        ).fetch(spec, destination)

    assert len(opener.calls) == 1
    assert not (destination / "never-requested.bin").exists()
    downloaded = destination / "model.bin"
    if downloaded.exists():
        assert downloaded.stat().st_size <= len(expected)
    if response is not None:
        assert response.closed


def test_default_fetcher_limits_https_redirects_and_rejects_http() -> None:
    fetcher = huggingface_fetcher.HuggingFaceSnapshotFetcher()
    opener = fetcher._opener.__self__
    redirect_handlers = tuple(
        handler
        for handler in opener.handlers
        if isinstance(handler, HTTPRedirectHandler)
    )
    assert len(redirect_handlers) == 1
    redirect_handler = redirect_handlers[0]

    class EchoParent:
        def open(self, request: Request, *, timeout: float) -> Request:
            assert timeout == 30
            request.timeout = timeout
            return request

    redirect_handler.add_parent(EchoParent())
    request = Request("https://origin.example/model.bin")
    request.timeout = 30
    for hop in range(1, 6):
        redirected = redirect_handler.http_error_302(
            request,
            BytesIO(),
            302,
            "Found",
            {"location": f"https://cdn{hop}.example/model.bin"},
        )
        assert isinstance(redirected, Request)
        request = redirected

    assert len(request.redirect_dict) == 5
    with pytest.raises(HTTPError) as too_many:
        redirect_handler.http_error_302(
            request,
            BytesIO(),
            302,
            "Found",
            {"location": "https://cdn6.example/model.bin"},
        )
    assert type(too_many.value) is HTTPError

    insecure_request = Request("https://origin.example/model.bin")
    insecure_request.timeout = 30
    with pytest.raises(HTTPError) as insecure:
        redirect_handler.http_error_302(
            insecure_request,
            BytesIO(),
            302,
            "Found",
            {"location": "http://cdn.example/model.bin"},
        )
    assert type(insecure.value) is HTTPError


def test_resolver_maps_fetcher_oversize_to_unavailable_and_cleans_staging(
    tmp_path: Path,
) -> None:
    expected = b"abcd"
    response = FakeResponse(b"abcde")
    fetcher = huggingface_fetcher.HuggingFaceSnapshotFetcher(
        opener=FakeOpener(response),
        timeout_seconds=30,
    )
    cache_root = tmp_path / "cache"
    spec = _spec(_file("model.bin", expected))

    with pytest.raises(model_artifacts.ModelArtifactUnavailable) as caught:
        model_artifacts.ModelArtifactResolver(cache_root, fetcher).resolve(spec)

    assert type(caught.value) is model_artifacts.ModelArtifactUnavailable
    assert caught.value.__cause__ is not None
    target = cache_root / "example" / "model" / REVISION
    assert not target.exists()
    assert not any(path.name.endswith(".partial") for path in cache_root.rglob("*"))
    assert response.closed
