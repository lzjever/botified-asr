from __future__ import annotations

import errno
import hashlib
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

_MODEL_ID_PART = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_REVISION = re.compile(r"\A[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_HASH_CHUNK_BYTES = 1024 * 1024


class ModelArtifactError(Exception):
    pass


class ModelManifestError(ModelArtifactError, ValueError):
    pass


class ModelArtifactUnavailable(ModelArtifactError):
    pass


class ModelArtifactIntegrityError(ModelArtifactError):
    pass


@dataclass(frozen=True)
class ModelArtifactFile:
    relative_path: str
    sha256: str
    expected_bytes: int

    def __post_init__(self) -> None:
        if not _valid_relative_path(self.relative_path):
            raise ModelManifestError("Model artifact path is unsafe")
        if not isinstance(self.sha256, str) or _SHA256.fullmatch(self.sha256) is None:
            raise ModelManifestError("Model artifact SHA-256 is invalid")
        if type(self.expected_bytes) is not int or self.expected_bytes <= 0:
            raise ModelManifestError("Model artifact byte length is invalid")


@dataclass(frozen=True)
class ModelArtifactSpec:
    provider: str
    model_id: str
    revision: str
    files: tuple[ModelArtifactFile, ...]

    def __post_init__(self) -> None:
        if self.provider != "huggingface":
            raise ModelManifestError("Model artifact provider is unsupported")
        if not _valid_model_id(self.model_id):
            raise ModelManifestError("Model ID is invalid")
        if (
            not isinstance(self.revision, str)
            or _REVISION.fullmatch(self.revision) is None
        ):
            raise ModelManifestError("Model revision must be a full lowercase commit")
        if (
            type(self.files) is not tuple
            or not self.files
            or any(not isinstance(item, ModelArtifactFile) for item in self.files)
        ):
            raise ModelManifestError("Model artifact file manifest is invalid")
        paths = tuple(item.relative_path for item in self.files)
        if len(set(paths)) != len(paths):
            raise ModelManifestError("Model artifact paths must be unique")
        path_set = set(paths)
        if any(
            "/".join(parts[:index]) in path_set
            for path in paths
            for parts in (path.split("/"),)
            for index in range(1, len(parts))
        ):
            raise ModelManifestError("Model artifact paths must not overlap")


@dataclass(frozen=True)
class ResolvedModelSnapshot:
    spec: ModelArtifactSpec
    root: Path


class SnapshotFetcher(Protocol):
    def fetch(self, spec: ModelArtifactSpec, destination: Path) -> None: ...


class ModelArtifactResolver:
    def __init__(
        self,
        cache_root: Path,
        fetcher: SnapshotFetcher,
    ) -> None:
        self._cache_root = Path(cache_root)
        self._fetcher = fetcher

    def resolve(self, spec: ModelArtifactSpec) -> ResolvedModelSnapshot:
        if not isinstance(spec, ModelArtifactSpec):
            raise ModelManifestError("Model artifact spec is invalid")

        target = self._target(spec)
        self._ensure_snapshot_parent(spec)
        if _path_entry_exists(target):
            self._validate_snapshot(
                target,
                spec,
                missing_is_unavailable=False,
            )
            return ResolvedModelSnapshot(spec=spec, root=target)

        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{spec.revision}.",
                suffix=".partial",
                dir=target.parent,
            )
        )
        try:
            try:
                self._fetcher.fetch(spec, staging)
            except Exception as error:
                raise ModelArtifactUnavailable(
                    "Model snapshot could not be fetched"
                ) from error

            self._validate_snapshot(
                staging,
                spec,
                missing_is_unavailable=True,
            )

            if _path_entry_exists(target):
                self._validate_snapshot(
                    target,
                    spec,
                    missing_is_unavailable=False,
                )
                return ResolvedModelSnapshot(spec=spec, root=target)

            try:
                staging.rename(target)
            except OSError as error:
                if error.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise ModelArtifactUnavailable(
                        "Model snapshot could not be published"
                    ) from error
                self._validate_snapshot(
                    target,
                    spec,
                    missing_is_unavailable=False,
                )
                return ResolvedModelSnapshot(spec=spec, root=target)

            staging = None
            return ResolvedModelSnapshot(spec=spec, root=target)
        finally:
            if staging is not None:
                _remove_owned_staging(staging)

    def _target(self, spec: ModelArtifactSpec) -> Path:
        organization, repository = spec.model_id.split("/")
        return self._cache_root / organization / repository / spec.revision

    def _ensure_snapshot_parent(self, spec: ModelArtifactSpec) -> None:
        try:
            self._cache_root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ModelArtifactUnavailable(
                "Model cache directory is unavailable"
            ) from error
        _require_directory(self._cache_root)

        organization, repository = spec.model_id.split("/")
        current = self._cache_root
        for part in (organization, repository):
            current = current / part
            try:
                current.mkdir()
            except FileExistsError:
                pass
            except OSError as error:
                raise ModelArtifactUnavailable(
                    "Model cache directory is unavailable"
                ) from error
            _require_directory(current)

    def _validate_snapshot(
        self,
        root: Path,
        spec: ModelArtifactSpec,
        *,
        missing_is_unavailable: bool,
    ) -> None:
        if not _path_entry_exists(root):
            _raise_missing(missing_is_unavailable)
        _require_directory(root)
        for artifact in spec.files:
            path = root
            parts = artifact.relative_path.split("/")
            for part in parts[:-1]:
                path = path / part
                if not _path_entry_exists(path):
                    _raise_missing(missing_is_unavailable)
                _require_directory(path)

            path = path / parts[-1]
            if not _path_entry_exists(path):
                _raise_missing(missing_is_unavailable)
            try:
                metadata = path.lstat()
            except OSError as error:
                raise ModelArtifactIntegrityError(
                    "Model artifact could not be inspected"
                ) from error
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise ModelArtifactIntegrityError(
                    "Model artifact is not a regular file"
                )
            if metadata.st_size != artifact.expected_bytes:
                _raise_size_mismatch(missing_is_unavailable)
            actual_bytes, actual_sha256 = _file_sha256(path)
            if actual_bytes != artifact.expected_bytes:
                _raise_size_mismatch(missing_is_unavailable)
            if actual_sha256 != artifact.sha256:
                raise ModelArtifactIntegrityError(
                    "Model artifact SHA-256 does not match its manifest"
                )


def _valid_model_id(model_id: object) -> bool:
    if not isinstance(model_id, str):
        return False
    parts = model_id.split("/")
    return (
        len(parts) == 2
        and all(_MODEL_ID_PART.fullmatch(part) is not None for part in parts)
        and all(part not in {".", ".."} for part in parts)
    )


def _valid_relative_path(relative_path: object) -> bool:
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or relative_path.startswith("/")
        or relative_path.endswith("/")
        or "\\" in relative_path
        or "\x00" in relative_path
    ):
        return False
    return all(part not in {"", ".", ".."} for part in relative_path.split("/"))


def _path_entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise ModelArtifactIntegrityError(
            "Model cache path could not be inspected"
        ) from error
    return True


def _require_directory(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ModelArtifactIntegrityError(
            "Model cache directory could not be inspected"
        ) from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ModelArtifactIntegrityError("Model cache path is not a safe directory")


def _raise_missing(missing_is_unavailable: bool) -> None:
    if missing_is_unavailable:
        raise ModelArtifactUnavailable("Fetched model snapshot is incomplete")
    raise ModelArtifactIntegrityError("Existing model snapshot is incomplete")


def _raise_size_mismatch(missing_is_unavailable: bool) -> None:
    if missing_is_unavailable:
        raise ModelArtifactUnavailable("Fetched model artifact has an invalid size")
    raise ModelArtifactIntegrityError("Existing model artifact has an invalid size")


def _file_sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    actual_bytes = 0
    try:
        with path.open("rb") as source:
            while chunk := source.read(_HASH_CHUNK_BYTES):
                actual_bytes += len(chunk)
                digest.update(chunk)
    except OSError as error:
        raise ModelArtifactIntegrityError("Model artifact could not be read") from error
    return actual_bytes, digest.hexdigest()


def _remove_owned_staging(staging: Path) -> None:
    try:
        metadata = staging.lstat()
    except FileNotFoundError:
        return
    except OSError:
        return
    try:
        if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
            shutil.rmtree(staging)
        else:
            staging.unlink()
    except OSError:
        pass


SENSEVOICE_SPEC = ModelArtifactSpec(
    provider="huggingface",
    model_id="FunAudioLLM/SenseVoiceSmall",
    revision="3847d57b6bdf2dd8875cb1508d2af43d80a16bf7",
    files=(
        ModelArtifactFile(
            relative_path="model.pt",
            sha256="833ca2dcfdf8ec91bd4f31cfac36d6124e0c459074d5e909aec9cabe6204a3ea",
            expected_bytes=936_291_369,
        ),
        ModelArtifactFile(
            relative_path="configuration.json",
            sha256="02810a7f8e9e8aee10370a265f7e799728ce25b4c00cdbf4602b303ee395a38e",
            expected_bytes=396,
        ),
        ModelArtifactFile(
            relative_path="config.yaml",
            sha256="f71e239ba36705564b5bf2d2ffd07eece07b8e3f2bbf6d2c99d8df856339ac19",
            expected_bytes=1_855,
        ),
        ModelArtifactFile(
            relative_path="am.mvn",
            sha256="29b3c740a2c0cfc6b308126d31d7f265fa2be74f3bb095cd2f143ea970896ae5",
            expected_bytes=11_203,
        ),
        ModelArtifactFile(
            relative_path="chn_jpn_yue_eng_ko_spectok.bpe.model",
            sha256="aa87f86064c3730d799ddf7af3c04659151102cba548bce325cf06ba4da4e6a8",
            expected_bytes=377_341,
        ),
    ),
)

FSMN_VAD_SPEC = ModelArtifactSpec(
    provider="huggingface",
    model_id="funasr/fsmn-vad",
    revision="df20e6b30c653645fa4ff125cacfcabd1020a669",
    files=(
        ModelArtifactFile(
            relative_path="model.pt",
            sha256="b3be75be477f0780277f3bae0fe489f48718f585f3a6e45d7dd1fbb1a4255fc5",
            expected_bytes=1_721_366,
        ),
        ModelArtifactFile(
            relative_path="configuration.json",
            sha256="7bce8867e37d55c3dd8f672695ced18077a2be199ea529a5d432d5350fc0acba",
            expected_bytes=365,
        ),
        ModelArtifactFile(
            relative_path="config.yaml",
            sha256="486861ca26ddb79081663b6179cb204c6bfae71c52f04aafc48a9e9d8dde1e93",
            expected_bytes=1_215,
        ),
        ModelArtifactFile(
            relative_path="am.mvn",
            sha256="df189fd5f4352df84a0fd464eeab4e450a5e645665d6b38f13c832492261a739",
            expected_bytes=8_033,
        ),
    ),
)

CAMPLUS_SPEC = ModelArtifactSpec(
    provider="huggingface",
    model_id="funasr/campplus",
    revision="e4b6ede7ce16997aff4ae69fbca1f0175e2afede",
    files=(
        ModelArtifactFile(
            relative_path="campplus_cn_common.bin",
            sha256="3388cf5fd3493c9ac9c69851d8e7a8badcfb4f3dc631020c4961371646d5ada8",
            expected_bytes=28_036_335,
        ),
        ModelArtifactFile(
            relative_path="configuration.json",
            sha256="6f7acaf1e81ca121f4a3c71b6ddb66beec24350a3ef330e2c846f17829176a8f",
            expected_bytes=581,
        ),
        ModelArtifactFile(
            relative_path="config.yaml",
            sha256="17342041bd5b22f6fd7e32f6e7a267b0bf65f018c0a721bada6547e3d28fbfc9",
            expected_bytes=537,
        ),
    ),
)
