from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = PROJECT_ROOT / "skills" / "botified-asr"
HELPER = SKILL_ROOT / "scripts" / "botified-asr"
TOKEN = "test-token+/=="


def _install_fake_curl(
    tmp_path: Path,
    *,
    body: bytes = b'{"status":"ready"}',
    stderr: bytes = b"",
    exit_code: int = 0,
) -> tuple[dict[str, str], Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    curl = fake_bin / "curl"
    curl.write_text(
        """#!/bin/sh
: >"$FAKE_CURL_ARGS"
for argument do
    printf '%s\\0' "$argument" >>"$FAKE_CURL_ARGS"
done
/bin/cat >"$FAKE_CURL_STDIN"
/usr/bin/env >"$FAKE_CURL_ENV"
/bin/cat "$FAKE_CURL_BODY"
/bin/cat "$FAKE_CURL_STDERR" >&2
exit "$FAKE_CURL_EXIT"
""",
        encoding="utf-8",
    )
    curl.chmod(0o755)
    body_path = tmp_path / "curl-body"
    body_path.write_bytes(body)
    stderr_path = tmp_path / "curl-stderr"
    stderr_path.write_bytes(stderr)
    args_path = tmp_path / "curl-args"
    stdin_path = tmp_path / "curl-stdin"
    config_home = tmp_path / "config home"
    home = tmp_path / "home"
    home.mkdir()
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "XDG_CONFIG_HOME": str(config_home),
            "HOME": str(home),
            "FAKE_CURL_ARGS": str(args_path),
            "FAKE_CURL_STDIN": str(stdin_path),
            "FAKE_CURL_BODY": str(body_path),
            "FAKE_CURL_STDERR": str(stderr_path),
            "FAKE_CURL_EXIT": str(exit_code),
            "FAKE_CURL_ENV": str(tmp_path / "curl-env"),
        }
    )
    environment.pop("BOTIFIED_ASR_BASE_URL", None)
    environment.pop("BOTIFIED_ASR_API_KEY", None)
    return environment, args_path, stdin_path


def _write_client_config(
    environment: dict[str, str],
    content: bytes,
    *,
    mode: int = 0o600,
) -> Path:
    path = (
        Path(environment["XDG_CONFIG_HOME"])
        / "botified-asr"
        / "client.env"
    )
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    path.chmod(mode)
    return path


def _valid_config(
    *,
    base_url: str = "https://asr.example:17770/",
    token: str = TOKEN,
) -> bytes:
    return (
        f"BOTIFIED_ASR_BASE_URL={base_url}\n"
        f"BOTIFIED_ASR_API_KEY={token}\n"
    ).encode()


def _run(
    environment: dict[str, str],
    *arguments: str,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [HELPER, *arguments],
        env=environment,
        capture_output=True,
        check=False,
    )


def _error_code(result: subprocess.CompletedProcess[bytes]) -> str:
    payload = json.loads(result.stdout)
    return payload["error"]["code"]


def test_skill_has_only_the_release_shape_and_minimal_metadata() -> None:
    files = {
        path.relative_to(SKILL_ROOT).as_posix()
        for path in SKILL_ROOT.rglob("*")
        if path.is_file()
    }
    assert files == {
        "SKILL.md",
        "agents/openai.yaml",
        "references/api.md",
        "scripts/botified-asr",
    }
    assert HELPER.stat().st_mode & 0o111

    skill_text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    _, frontmatter, body = skill_text.split("---", 2)
    assert yaml.safe_load(frontmatter).keys() == {"name", "description"}
    assert "TODO" not in skill_text
    assert "relative to this `SKILL.md` (the skill root)" in body
    assert "scripts/botified-asr health" in body
    assert "references/api.md" in body

    assert yaml.safe_load(
        (SKILL_ROOT / "agents" / "openai.yaml").read_text(encoding="utf-8")
    ) == {
        "interface": {
            "display_name": "Botified ASR",
            "short_description": (
                "Check configured Botified ASR service readiness"
            ),
            "default_prompt": (
                "Use $botified-asr to check whether my configured "
                "Botified ASR service is ready."
            ),
        }
    }


@pytest.mark.parametrize("arguments", [(), ("unknown",), ("health", "extra")])
def test_invalid_command_precedes_the_curl_dependency_check(
    tmp_path: Path,
    arguments: tuple[str, ...],
) -> None:
    environment = os.environ.copy()
    environment["PATH"] = str(tmp_path)

    result = _run(environment, *arguments)

    assert result.returncode == 64
    assert _error_code(result) == "invalid_command"
    assert result.stderr == b""


def test_missing_curl_is_stable(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["PATH"] = str(tmp_path)

    result = _run(environment, "health")

    assert result.returncode == 69
    assert _error_code(result) == "curl_not_found"
    assert result.stderr == b""


def test_missing_config_returns_a_json_safe_canonical_path(tmp_path: Path) -> None:
    environment, args_path, _ = _install_fake_curl(tmp_path)
    config_home = tmp_path / 'config "quoted" \\ root'
    environment["XDG_CONFIG_HOME"] = str(config_home)

    result = _run(environment, "health")

    payload = json.loads(result.stdout)
    assert result.returncode == 78
    assert payload["error"]["code"] == "client_not_configured"
    assert payload["client_config_path"] == str(
        config_home / "botified-asr" / "client.env"
    )
    assert result.stderr == b""
    assert not args_path.exists()


@pytest.mark.parametrize(
    ("xdg_config_home", "home"),
    [
        ("relative/config", "/valid/home"),
        ("/bad\tconfig", "/valid/home"),
        (None, None),
        (None, ""),
        (None, "relative/home"),
    ],
)
def test_invalid_client_config_roots_fail_before_file_or_curl(
    tmp_path: Path,
    xdg_config_home: str | None,
    home: str | None,
) -> None:
    environment, args_path, _ = _install_fake_curl(tmp_path)
    if xdg_config_home is None:
        environment.pop("XDG_CONFIG_HOME", None)
    else:
        environment["XDG_CONFIG_HOME"] = xdg_config_home
    if home is None:
        environment.pop("HOME", None)
    else:
        environment["HOME"] = home

    result = _run(environment, "health")

    assert result.returncode == 78
    assert _error_code(result) == "invalid_client_config"
    assert result.stderr == b""
    assert not args_path.exists()


def test_empty_xdg_config_home_falls_back_to_home(tmp_path: Path) -> None:
    environment, _, _ = _install_fake_curl(tmp_path)
    environment["XDG_CONFIG_HOME"] = ""
    home = tmp_path / "fallback-home"
    environment["HOME"] = str(home)
    path = home / ".config" / "botified-asr" / "client.env"
    path.parent.mkdir(parents=True)
    path.write_bytes(_valid_config())
    path.chmod(0o600)

    result = _run(environment, "health")

    assert result.returncode == 0
    assert result.stdout == b'{"status":"ready"}'
    assert result.stderr == b""


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"\nBOTIFIED_ASR_BASE_URL=https://asr.example\n"
        b"BOTIFIED_ASR_API_KEY=token\n",
        b"# comment\nBOTIFIED_ASR_BASE_URL=https://asr.example\n"
        b"BOTIFIED_ASR_API_KEY=token\n",
        b"export BOTIFIED_ASR_BASE_URL=https://asr.example\n"
        b"BOTIFIED_ASR_API_KEY=token\n",
        b" BOTIFIED_ASR_BASE_URL=https://asr.example\n"
        b"BOTIFIED_ASR_API_KEY=token\n",
        b"BOTIFIED_ASR_BASE_URL =https://asr.example\n"
        b"BOTIFIED_ASR_API_KEY=token\n",
        b"BOTIFIED_ASR_BASE_URL=https://asr.example\n",
        b"BOTIFIED_ASR_BASE_URL=https://one.example\n"
        b"BOTIFIED_ASR_BASE_URL=https://two.example\n"
        b"BOTIFIED_ASR_API_KEY=token\n",
        b"BOTIFIED_ASR_BASE_URL=https://asr.example\n"
        b"BOTIFIED_ASR_API_KEY=one\n"
        b"BOTIFIED_ASR_API_KEY=two\n",
        b"BOTIFIED_ASR_BASE_URL=https://asr.example\r\n"
        b"BOTIFIED_ASR_API_KEY=token\n",
    ],
)
def test_malformed_client_file_is_rejected_without_curl_or_secret_leak(
    tmp_path: Path,
    content: bytes,
) -> None:
    environment, args_path, _ = _install_fake_curl(tmp_path)
    _write_client_config(environment, content)

    result = _run(environment, "health")

    assert result.returncode == 78
    assert _error_code(result) == "invalid_client_config"
    assert result.stderr == b""
    assert b"one" not in result.stdout
    assert b"two" not in result.stdout
    assert not args_path.exists()


def test_client_file_is_never_executed(tmp_path: Path) -> None:
    environment, args_path, _ = _install_fake_curl(tmp_path)
    sentinel = tmp_path / "executed"
    _write_client_config(
        environment,
        (
            "BOTIFIED_ASR_BASE_URL=https://asr.example\n"
            f"EVIL=$(touch {sentinel})\n"
            "BOTIFIED_ASR_API_KEY=token\n"
        ).encode(),
    )

    result = _run(environment, "health")

    assert result.returncode == 78
    assert _error_code(result) == "invalid_client_config"
    assert not sentinel.exists()
    assert not args_path.exists()


@pytest.mark.parametrize("mode", [0o000, 0o200, 0o400, 0o640])
def test_client_file_mode_must_be_exactly_0600(
    tmp_path: Path,
    mode: int,
) -> None:
    environment, args_path, _ = _install_fake_curl(tmp_path)
    _write_client_config(environment, _valid_config(), mode=mode)

    result = _run(environment, "health")

    assert result.returncode == 78
    assert _error_code(result) == "invalid_client_config"
    assert result.stderr == b""
    assert not args_path.exists()


@pytest.mark.parametrize("unsafe_kind", ["symlink", "directory"])
def test_client_file_must_be_private_regular_and_not_a_symlink(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    environment, args_path, _ = _install_fake_curl(tmp_path)
    path = (
        Path(environment["XDG_CONFIG_HOME"])
        / "botified-asr"
        / "client.env"
    )
    path.parent.mkdir(parents=True)
    if unsafe_kind == "symlink":
        target = tmp_path / "real-client.env"
        target.write_bytes(_valid_config())
        target.chmod(0o600)
        path.symlink_to(target)
    else:
        path.mkdir()

    result = _run(environment, "health")

    assert result.returncode == 78
    assert _error_code(result) == "invalid_client_config"
    assert not args_path.exists()


def test_shell_xtrace_never_exposes_the_environment_override(
    tmp_path: Path,
) -> None:
    environment, _, _ = _install_fake_curl(tmp_path)
    environment.update(
        {
            "BOTIFIED_ASR_BASE_URL": "https://override.example",
            "BOTIFIED_ASR_API_KEY": TOKEN,
        }
    )

    result = subprocess.run(
        ["/bin/sh", "-x", HELPER, "health"],
        env=environment,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert TOKEN.encode() not in result.stderr


def test_paired_environment_override_does_not_read_the_client_file(
    tmp_path: Path,
) -> None:
    environment, args_path, stdin_path = _install_fake_curl(tmp_path)
    _write_client_config(environment, b"this file must not be read\n", mode=0)
    environment.pop("XDG_CONFIG_HOME")
    environment.pop("HOME")
    environment.update(
        {
            "BOTIFIED_ASR_BASE_URL": "https://override.example/",
            "BOTIFIED_ASR_API_KEY": TOKEN,
        }
    )

    result = _run(environment, "health")

    assert result.returncode == 0
    assert result.stdout == b'{"status":"ready"}'
    assert TOKEN.encode() not in args_path.read_bytes()
    assert stdin_path.read_bytes() == (
        f'header = "Authorization: Bearer {TOKEN}"\n'.encode()
    )
    curl_environment = (tmp_path / "curl-env").read_bytes()
    assert b"BOTIFIED_ASR_BASE_URL=" not in curl_environment
    assert b"BOTIFIED_ASR_API_KEY=" not in curl_environment


@pytest.mark.parametrize(
    ("base_url", "api_key"),
    [
        ("https://asr.example", None),
        (None, "token"),
        ("", ""),
        ("https://asr.example", ""),
        ("", "token"),
    ],
)
def test_environment_override_must_be_present_valid_and_paired(
    tmp_path: Path,
    base_url: str | None,
    api_key: str | None,
) -> None:
    environment, args_path, _ = _install_fake_curl(tmp_path)
    _write_client_config(environment, _valid_config())
    if base_url is not None:
        environment["BOTIFIED_ASR_BASE_URL"] = base_url
    if api_key is not None:
        environment["BOTIFIED_ASR_API_KEY"] = api_key

    result = _run(environment, "health")

    assert result.returncode == 78
    assert _error_code(result) == "invalid_client_config"
    assert TOKEN.encode() not in result.stdout + result.stderr
    assert not args_path.exists()


@pytest.mark.parametrize(
    "base_url",
    [
        "ftp://asr.example",
        "http://",
        "https://:443",
        "https://asr.example:0",
        "https://asr.example:",
        "https://asr.example:invalid",
        "https://asr.example:65536",
        "https://asr.example/path",
        "https://user@asr.example",
        "https://asr.example?query=1",
        "https://asr.example#fragment",
        "https://asr example",
        "https://asr.example//",
        "https://asr{one,two}.example",
        "https://[::1",
        "https://[::1]",
        "https://asr.example/\t",
    ],
)
def test_base_url_must_be_a_strict_http_origin(
    tmp_path: Path,
    base_url: str,
) -> None:
    environment, args_path, _ = _install_fake_curl(tmp_path)
    _write_client_config(environment, _valid_config(base_url=base_url))

    result = _run(environment, "health")

    assert result.returncode == 78
    assert _error_code(result) == "invalid_client_config"
    assert not args_path.exists()


@pytest.mark.parametrize(
    "api_key",
    ["", "token with space", "token=middle=value", "令牌", "token\r"],
)
def test_api_key_must_be_an_exact_ascii_rfc6750_b64token(
    tmp_path: Path,
    api_key: str,
) -> None:
    environment, args_path, _ = _install_fake_curl(tmp_path)
    _write_client_config(environment, _valid_config(token=api_key))

    result = _run(environment, "health")

    assert result.returncode == 78
    assert _error_code(result) == "invalid_client_config"
    if api_key:
        assert api_key.encode() not in result.stdout + result.stderr
    assert not args_path.exists()


@pytest.mark.parametrize(
    ("body", "curl_stderr", "curl_exit", "expected_stdout"),
    [
        (b'{"status":"ready"}', b"", 0, b'{"status":"ready"}'),
        (
            b'{"error":{"code":"invalid_api_key"}}',
            b"curl http 401 detail",
            22,
            b'{"error":{"code":"invalid_api_key"}}',
        ),
        (
            b'{"error":{"code":"service_not_ready"}}',
            b"curl http 503 detail",
            22,
            b'{"error":{"code":"service_not_ready"}}',
        ),
        (
            b"partial sensitive transport body",
            f"could not connect using {TOKEN}".encode(),
            7,
            (
                b'{"error":{"message":"Botified ASR request failed",'
                b'"type":"client_error","param":null,"code":"curl_failed"}}\n'
            ),
        ),
        (
            b"timeout detail",
            f"timed out using {TOKEN}".encode(),
            28,
            (
                b'{"error":{"message":"Botified ASR request failed",'
                b'"type":"client_error","param":null,"code":"curl_failed"}}\n'
            ),
        ),
    ],
)
def test_health_curl_boundary_is_private_and_preserves_public_results(
    tmp_path: Path,
    body: bytes,
    curl_stderr: bytes,
    curl_exit: int,
    expected_stdout: bytes,
) -> None:
    environment, args_path, stdin_path = _install_fake_curl(
        tmp_path,
        body=body,
        stderr=curl_stderr,
        exit_code=curl_exit,
    )
    _write_client_config(environment, _valid_config())

    result = _run(environment, "health")

    assert result.returncode == curl_exit
    assert result.stdout == expected_stdout
    assert result.stderr == b""
    arguments = args_path.read_bytes().split(b"\0")
    assert arguments == [
        b"--disable",
        b"--globoff",
        b"--config",
        b"-",
        b"--proto",
        b"=http,https",
        b"--silent",
        b"--fail-with-body",
        b"--connect-timeout",
        b"5",
        b"--max-time",
        b"10",
        b"--request",
        b"GET",
        b"--url",
        b"https://asr.example:17770/health/ready",
        b"",
    ]
    assert b"--location" not in arguments
    assert b"-L" not in arguments
    assert TOKEN.encode() not in args_path.read_bytes()
    assert TOKEN.encode() not in result.stdout + result.stderr
    assert stdin_path.read_bytes() == (
        f'header = "Authorization: Bearer {TOKEN}"\n'.encode()
    )
