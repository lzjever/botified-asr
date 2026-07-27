from __future__ import annotations

from pathlib import Path

import pytest

from botified_asr.config import ConfigError, load_api_key, load_config


MIB = 1024 * 1024


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_unknown_yaml_key_fails_fast(tmp_path: Path) -> None:
    path = write_config(tmp_path, "server:\n  listen: '127.0.0.1:8090'\n  typo: true\n")

    with pytest.raises(ConfigError, match=r"server\.typo"):
        load_config(path)


def test_yaml_api_key_is_rejected(tmp_path: Path) -> None:
    path = write_config(tmp_path, "api_key: plaintext-is-forbidden\n")

    with pytest.raises(ConfigError, match="api_key"):
        load_config(path)


def test_data_dir_tilde_is_expanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    path = write_config(tmp_path, "storage:\n  data_dir: '~/asr-data'\n")

    config = load_config(path)

    assert config.storage.data_dir == tmp_path / "asr-data"


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ("max_active_uploads: 0", "positive integer"),
        ("result_retention_hours: true", "positive integer"),
        (
            "direct_max_audio_duration_secs: 31\n  sync_max_audio_duration_secs: 30",
            "duration limits",
        ),
        (
            "sync_max_upload_bytes: 9\n  max_upload_bytes: 8",
            "upload byte limits",
        ),
        ("max_audio_duration_secs: 43201", "duration limits"),
        ("max_upload_bytes: 1073741825", "upload byte limits"),
    ],
)
def test_limit_constraints_are_validated(
    tmp_path: Path, override: str, match: str
) -> None:
    path = write_config(tmp_path, f"limits:\n  {override}\n")

    with pytest.raises(ConfigError, match=match):
        load_config(path)


def test_storage_limit_rounds_max_upload_to_reservation_quantum(
    tmp_path: Path,
) -> None:
    too_small = write_config(
        tmp_path,
        "limits:\n"
        f"  max_upload_bytes: {8 * MIB + 1}\n"
        f"  sync_max_upload_bytes: {8 * MIB}\n"
        f"  max_job_storage_bytes: {8 * MIB + 1}\n",
    )

    with pytest.raises(ConfigError, match="reservation quantum"):
        load_config(too_small)

    enough = write_config(
        tmp_path,
        "limits:\n"
        f"  max_upload_bytes: {8 * MIB + 1}\n"
        f"  sync_max_upload_bytes: {8 * MIB}\n"
        f"  max_job_storage_bytes: {16 * MIB}\n",
    )
    assert load_config(enough).limits.max_job_storage_bytes == 16 * MIB


@pytest.mark.parametrize(
    "value",
    [
        "abcXYZ019-._~+/",
        "token=",
        "token===",
    ],
)
def test_api_key_accepts_rfc6750_b64token(value: str) -> None:
    assert load_api_key({"BOTIFIED_ASR_API_KEY": value}) == value


@pytest.mark.parametrize(
    "value",
    [
        "",
        " ",
        " token",
        "token ",
        "to ken",
        "tökén",
        "token\n",
        "to=ken",
        "=token",
    ],
)
def test_api_key_rejects_non_b64token_without_echoing_secret(value: str) -> None:
    with pytest.raises(ConfigError, match="BOTIFIED_ASR_API_KEY") as caught:
        load_api_key({"BOTIFIED_ASR_API_KEY": value})
    if value and not value.isspace():
        assert value not in str(caught.value)
