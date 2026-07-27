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


def write_model_paths_config(
    tmp_path: Path, *, data_dir: Path, model_cache_dir: Path
) -> Path:
    return write_config(
        tmp_path,
        "runtime:\n"
        f"  model_cache_dir: '{model_cache_dir}'\n"
        "storage:\n"
        f"  data_dir: '{data_dir}'\n",
    )


def test_default_model_cache_dir_is_absolute_and_canonical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    config = load_config(write_config(tmp_path, "{}\n"))

    expected = (tmp_path / ".cache/botified-asr/models").resolve()
    assert config.runtime.model_cache_dir == expected
    assert config.runtime.model_cache_dir.is_absolute()


def test_explicit_model_cache_dir_expands_tilde_and_is_canonical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    path = write_config(
        tmp_path,
        "runtime:\n  model_cache_dir: '~/cache-parent/../asr-models'\n",
    )

    config = load_config(path)

    assert config.runtime.model_cache_dir == (tmp_path / "asr-models").resolve()


@pytest.mark.parametrize(
    ("yaml_text", "match"),
    [
        (
            "storage:\n  data_dir: 'relative-data'\n",
            r"storage\.data_dir.*absolute",
        ),
        (
            "runtime:\n  model_cache_dir: 'relative-cache'\n",
            r"runtime\.model_cache_dir.*absolute",
        ),
    ],
)
def test_runtime_paths_reject_relative_values(
    tmp_path: Path, yaml_text: str, match: str
) -> None:
    with pytest.raises(ConfigError, match=match):
        load_config(write_config(tmp_path, yaml_text))


@pytest.mark.parametrize(
    ("yaml_text", "match"),
    [
        (
            'runtime:\n  model_cache_dir: "/tmp/\\0model-cache"\n',
            r"runtime\.model_cache_dir",
        ),
        (
            'storage:\n  data_dir: "/tmp/\\0runtime-data"\n',
            r"storage\.data_dir",
        ),
    ],
)
def test_runtime_paths_wrap_malformed_absolute_values(
    tmp_path: Path, yaml_text: str, match: str
) -> None:
    with pytest.raises(ConfigError, match=match):
        load_config(write_config(tmp_path, yaml_text))


@pytest.mark.parametrize(
    "relationship",
    ["equal", "cache_under_data", "data_under_cache"],
)
def test_data_and_model_cache_dirs_must_not_overlap(
    tmp_path: Path, relationship: str
) -> None:
    root = (tmp_path / "runtime-data").resolve()
    if relationship == "equal":
        data_dir, model_cache_dir = root, root
    elif relationship == "cache_under_data":
        data_dir, model_cache_dir = root, root / "models"
    else:
        data_dir, model_cache_dir = root / "jobs", root

    path = write_model_paths_config(
        tmp_path,
        data_dir=data_dir,
        model_cache_dir=model_cache_dir,
    )

    with pytest.raises(ConfigError, match="must not overlap"):
        load_config(path)


def test_data_and_model_cache_dirs_reject_symlink_alias(tmp_path: Path) -> None:
    model_cache_dir = tmp_path / "model-cache"
    model_cache_dir.mkdir()
    data_dir = tmp_path / "runtime-data-link"
    data_dir.symlink_to(model_cache_dir, target_is_directory=True)
    path = write_model_paths_config(
        tmp_path,
        data_dir=data_dir,
        model_cache_dir=model_cache_dir,
    )

    with pytest.raises(ConfigError, match="must not overlap"):
        load_config(path)


def test_disjoint_sibling_data_and_model_cache_dirs_succeed(
    tmp_path: Path,
) -> None:
    data_dir = (tmp_path / "runtime-data").resolve()
    model_cache_dir = (tmp_path / "model-cache").resolve()
    path = write_model_paths_config(
        tmp_path,
        data_dir=data_dir,
        model_cache_dir=model_cache_dir,
    )

    config = load_config(path)

    assert config.storage.data_dir == data_dir
    assert config.runtime.model_cache_dir == model_cache_dir


@pytest.mark.parametrize("device", ["auto", "cpu"])
def test_cpu_runtime_normalizes_supported_device(
    tmp_path: Path, device: str
) -> None:
    path = write_config(tmp_path, f"runtime:\n  device: {device}\n")

    assert load_config(path).runtime.device == "cpu"


@pytest.mark.parametrize(
    "yaml_value",
    ["cuda", "cuda:0", "true"],
    ids=["cuda", "cuda-index", "boolean"],
)
def test_cpu_runtime_rejects_unsupported_device(
    tmp_path: Path, yaml_value: str
) -> None:
    path = write_config(tmp_path, f"runtime:\n  device: {yaml_value}\n")

    with pytest.raises(ConfigError, match=r"runtime\.device"):
        load_config(path)


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
            "direct_max_audio_duration_secs: 31\n"
            "  sync_max_audio_duration_secs: 31\n"
            "  max_audio_duration_secs: 31",
            "duration limits",
        ),
        (
            "sync_max_audio_duration_secs: 3601\n"
            "  max_audio_duration_secs: 3601",
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


def test_duration_release_boundaries_are_accepted(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        "limits:\n"
        "  direct_max_audio_duration_secs: 30\n"
        "  sync_max_audio_duration_secs: 3600\n"
        "  max_audio_duration_secs: 43200\n",
    )

    limits = load_config(path).limits

    assert limits.direct_max_audio_duration_secs == 30
    assert limits.sync_max_audio_duration_secs == 3600
    assert limits.max_audio_duration_secs == 43_200


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
