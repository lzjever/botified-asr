from __future__ import annotations

import math
import stat
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from botified_asr.model_artifacts import ModelArtifactFile, ModelArtifactSpec

DOWNLOAD_CHUNK_BYTES = 1024 * 1024
_MAX_REDIRECTS = 5
_MAX_ARTIFACT_DOWNLOAD_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = (1, 2)


class _TransientArtifactDownloadError(Exception):
    pass


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

        for attempt in range(_MAX_ARTIFACT_DOWNLOAD_ATTEMPTS):
            owns_partial = False
            request = Request(
                _artifact_url(spec, artifact),
                headers={"Accept-Encoding": "identity"},
            )
            try:
                try:
                    response = self._opener(
                        request,
                        timeout=self._timeout_seconds,
                    )
                except HTTPError:
                    raise
                except OSError as error:
                    if not _is_transient_network_error(error):
                        raise
                    raise _TransientArtifactDownloadError(
                        "Artifact request failed"
                    ) from error

                with response:
                    if response.status != 200:
                        raise OSError(f"Unexpected HTTP status: {response.status}")
                    if urlsplit(response.geturl()).scheme != "https":
                        raise OSError("Artifact response URL is not HTTPS")
                    _validate_content_length(
                        response.headers,
                        artifact.expected_bytes,
                    )

                    remaining = artifact.expected_bytes
                    with target.open("xb") as output:
                        owns_partial = True
                        while True:
                            try:
                                chunk = response.read(
                                    min(DOWNLOAD_CHUNK_BYTES, remaining + 1)
                                )
                            except HTTPError:
                                raise
                            except OSError as error:
                                if not _is_transient_network_error(error):
                                    raise
                                raise _TransientArtifactDownloadError(
                                    "Artifact response read failed"
                                ) from error
                            if not chunk:
                                break
                            if len(chunk) > remaining:
                                raise OSError(
                                    "Artifact response exceeds its expected size"
                                )
                            output.write(chunk)
                            remaining -= len(chunk)

                        if remaining != 0:
                            raise _TransientArtifactDownloadError(
                                "Artifact response is shorter than expected"
                            )
            except _TransientArtifactDownloadError as error:
                if owns_partial:
                    _remove_partial(target)
                if attempt == _MAX_ARTIFACT_DOWNLOAD_ATTEMPTS - 1:
                    cause = error.__cause__
                    if isinstance(cause, OSError):
                        raise cause
                    raise OSError(str(error)) from error
                time.sleep(_RETRY_BACKOFF_SECONDS[attempt])
            else:
                return


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


def _is_transient_network_error(error: OSError) -> bool:
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    return isinstance(error, URLError) and isinstance(
        error.reason,
        (ConnectionError, TimeoutError),
    )


def _remove_partial(target: Path) -> None:
    try:
        target.unlink(missing_ok=True)
    except OSError as error:
        raise OSError("Artifact partial could not be removed") from error


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
