from __future__ import annotations

import math
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from botified_asr.model_artifacts import ModelArtifactFile, ModelArtifactSpec

DOWNLOAD_CHUNK_BYTES = 1024 * 1024
_MAX_REDIRECTS = 5


class _HttpsOnlyRedirectHandler(HTTPRedirectHandler):
    max_redirections = _MAX_REDIRECTS
    max_repeats = _MAX_REDIRECTS

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        if urlsplit(newurl).scheme != "https":
            raise HTTPError(
                newurl,
                code,
                "Redirect to a non-HTTPS URL is not allowed",
                headers,
                fp,
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class HuggingFaceSnapshotFetcher:
    def __init__(
        self,
        *,
        opener: Callable[..., Any] | None = None,
        timeout_seconds: float = 30,
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("HTTP timeout must be a finite positive number")
        if opener is not None and not callable(opener):
            raise TypeError("HTTP opener must be callable")

        self._timeout_seconds = timeout_seconds
        self._opener = (
            opener
            if opener is not None
            else build_opener(_HttpsOnlyRedirectHandler()).open
        )

    def fetch(self, spec: ModelArtifactSpec, destination: Path) -> None:
        destination = Path(destination)
        _require_safe_directory(destination)
        for artifact in spec.files:
            self._fetch_artifact(spec, artifact, destination)

    def _fetch_artifact(
        self,
        spec: ModelArtifactSpec,
        artifact: ModelArtifactFile,
        destination: Path,
    ) -> None:
        target = _prepare_target(destination, artifact.relative_path)
        request = Request(
            _artifact_url(spec, artifact),
            headers={"Accept-Encoding": "identity"},
        )

        with self._opener(
            request,
            timeout=self._timeout_seconds,
        ) as response:
            if response.status != 200:
                raise OSError(f"Unexpected HTTP status: {response.status}")
            if urlsplit(response.geturl()).scheme != "https":
                raise OSError("Artifact response URL is not HTTPS")
            _validate_content_length(response.headers, artifact.expected_bytes)

            remaining = artifact.expected_bytes
            with target.open("xb") as output:
                while True:
                    chunk = response.read(min(DOWNLOAD_CHUNK_BYTES, remaining + 1))
                    if not chunk:
                        break
                    if len(chunk) > remaining:
                        raise OSError("Artifact response exceeds its expected size")
                    output.write(chunk)
                    remaining -= len(chunk)

                if remaining != 0:
                    raise OSError("Artifact response is shorter than expected")


def _artifact_url(
    spec: ModelArtifactSpec,
    artifact: ModelArtifactFile,
) -> str:
    model_path = "/".join(
        quote(component, safe="") for component in spec.model_id.split("/")
    )
    artifact_path = "/".join(
        quote(component, safe="") for component in artifact.relative_path.split("/")
    )
    revision = quote(spec.revision, safe="")
    return f"https://huggingface.co/{model_path}/resolve/{revision}/{artifact_path}"


def _prepare_target(destination: Path, relative_path: str) -> Path:
    parts = relative_path.split("/")
    current = destination
    for part in parts[:-1]:
        current = current / part
        try:
            current.mkdir()
        except FileExistsError:
            pass
        _require_safe_directory(current)
    return current / parts[-1]


def _require_safe_directory(path: Path) -> None:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise OSError("Download destination is not a safe directory")


def _validate_content_length(headers: Any, expected_bytes: int) -> None:
    content_length = headers.get("Content-Length")
    if content_length is None:
        return
    if (
        not isinstance(content_length, str)
        or not content_length
        or not content_length.isascii()
        or not content_length.isdigit()
        or int(content_length) != expected_bytes
    ):
        raise OSError("Artifact Content-Length does not match its manifest")
