from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import yaml


MIB = 1024 * 1024
RESERVATION_QUANTUM = 8 * MIB
MAX_UPLOAD_BYTES = 1024 * MIB
MAX_AUDIO_DURATION_SECS = 12 * 60 * 60
B64TOKEN_PATTERN = re.compile(r"[A-Za-z0-9\-._~+/]+={0,}")


class ConfigError(ValueError):
    """A stable startup configuration error."""


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _canonical_directory(value: object, field_name: str) -> Path:
    try:
        path = Path(value).expanduser()
    except (TypeError, ValueError, RuntimeError) as error:
        raise ConfigError(f"{field_name} must be a valid path") from error
    if not path.is_absolute():
        raise ConfigError(f"{field_name} must be an absolute path")
    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise ConfigError(f"{field_name} could not be resolved") from error


@dataclass(frozen=True)
class ServerConfig:
    listen: str = "127.0.0.1:8090"
    public_base_url: str = "http://127.0.0.1:8090"


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "auto"
    model_cache_dir: Path = Path("~/.cache/botified-asr/models")
    max_speakers: int = 32

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or self.device not in {"auto", "cpu"}:
            raise ConfigError("runtime.device must be auto or cpu")
        object.__setattr__(self, "device", "cpu")
        object.__setattr__(
            self,
            "model_cache_dir",
            _canonical_directory(
                self.model_cache_dir,
                "runtime.model_cache_dir",
            ),
        )
        if not _is_int(self.max_speakers) or not 1 <= self.max_speakers <= 32:
            raise ConfigError("runtime.max_speakers must be an integer from 1 to 32")


@dataclass(frozen=True)
class StorageConfig:
    data_dir: Path = Path("~/.local/share/botified-asr")

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "data_dir",
            _canonical_directory(self.data_dir, "storage.data_dir"),
        )


@dataclass(frozen=True)
class LimitsConfig:
    max_upload_bytes: int = MAX_UPLOAD_BYTES
    max_audio_duration_secs: int = MAX_AUDIO_DURATION_SECS
    direct_max_audio_duration_secs: int = 30
    sync_max_upload_bytes: int = 64 * MIB
    sync_max_audio_duration_secs: int = 3600
    max_active_uploads: int = 4
    max_queued_jobs: int = 16
    max_job_storage_bytes: int = 20 * 1024 * MIB
    min_filesystem_free_bytes: int = 2 * 1024 * MIB
    result_retention_hours: int = 24

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not _is_int(value) or value <= 0:
                raise ConfigError(f"limits.{field.name} must be a positive integer")

        if not (
            self.direct_max_audio_duration_secs <= 30
            and self.sync_max_audio_duration_secs <= 3600
            and self.direct_max_audio_duration_secs
            <= self.sync_max_audio_duration_secs
            <= self.max_audio_duration_secs
            <= MAX_AUDIO_DURATION_SECS
        ):
            raise ConfigError(
                "duration limits must satisfy direct <= sync <= max <= 43200"
            )

        if not (
            self.sync_max_upload_bytes
            <= self.max_upload_bytes
            <= MAX_UPLOAD_BYTES
        ):
            raise ConfigError(
                "upload byte limits must satisfy sync <= max <= 1073741824"
            )

        staging_reservation = (
            math.ceil(self.max_upload_bytes / RESERVATION_QUANTUM)
            * RESERVATION_QUANTUM
        )
        if self.max_job_storage_bytes < staging_reservation:
            raise ConfigError(
                "limits.max_job_storage_bytes must cover the max upload rounded "
                "up to the 8 MiB reservation quantum"
            )


@dataclass(frozen=True)
class Config:
    server: ServerConfig = ServerConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    storage: StorageConfig = StorageConfig()
    limits: LimitsConfig = LimitsConfig()

    def __post_init__(self) -> None:
        model_cache_dir = self.runtime.model_cache_dir
        data_dir = self.storage.data_dir
        if (
            model_cache_dir == data_dir
            or model_cache_dir.is_relative_to(data_dir)
            or data_dir.is_relative_to(model_cache_dir)
        ):
            raise ConfigError(
                "runtime.model_cache_dir and storage.data_dir must not overlap"
            )


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"unable to load config: {config_path}") from exc

    if raw is None:
        raw = {}
    root = _mapping(raw, "config")
    _reject_unknown(root, {"server", "runtime", "storage", "limits"}, "")

    server = _load_dataclass(ServerConfig, root.get("server"), "server")
    runtime = _load_dataclass(RuntimeConfig, root.get("runtime"), "runtime")
    storage = _load_dataclass(StorageConfig, root.get("storage"), "storage")
    limits = _load_dataclass(LimitsConfig, root.get("limits"), "limits")
    return Config(server=server, runtime=runtime, storage=storage, limits=limits)


def load_api_key(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = source.get("BOTIFIED_ASR_API_KEY", "")
    if B64TOKEN_PATTERN.fullmatch(value) is None:
        raise ConfigError(
            "BOTIFIED_ASR_API_KEY must be a non-empty RFC 6750 b64token"
        )
    return value


def _load_dataclass(cls, value: Any, path: str):
    mapping = {} if value is None else _mapping(value, path)
    allowed = {field.name for field in fields(cls)}
    _reject_unknown(mapping, allowed, f"{path}.")
    try:
        return cls(**mapping)
    except TypeError as exc:
        raise ConfigError(f"invalid {path} configuration") from exc


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        raise ConfigError(f"{path} must be a YAML mapping")
    return value


def _reject_unknown(
    mapping: Mapping[str, Any], allowed: set[str], prefix: str
) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigError(f"unknown configuration field: {prefix}{unknown[0]}")
