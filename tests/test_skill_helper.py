from __future__ import annotations

import gzip
import json
import os
import signal
import subprocess
import tarfile
import time
from pathlib import Path, PurePosixPath

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = PROJECT_ROOT / "skills" / "asr"
HELPER = SKILL_ROOT / "scripts" / "botified-asr"
SKILL_BUILDER = PROJECT_ROOT / "scripts" / "build-skill-tarball"
TOKEN = "test-token+/=="


def _install_fake_curl(
    tmp_path: Path,
    *,
    body: bytes = b'{"status":"ready"}',
    stderr: bytes = b"",
    exit_code: int = 0,
    http_code: str = "200",
) -> tuple[dict[str, str], Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    curl = fake_bin / "curl"
    curl.write_text(
        """#!/bin/sh
: >"$FAKE_CURL_ARGS"
output_path=
capture_output=0
write_out=
capture_write_out=0
for argument do
    printf '%s\\0' "$argument" >>"$FAKE_CURL_ARGS"
    if [ "$capture_output" -eq 1 ]; then
        output_path=$argument
        capture_output=0
    elif [ "$capture_write_out" -eq 1 ]; then
        write_out=$argument
        capture_write_out=0
    elif [ "$argument" = "--output" ]; then
        capture_output=1
    elif [ "$argument" = "--write-out" ]; then
        capture_write_out=1
    fi
done
/bin/cat >"$FAKE_CURL_STDIN"
: >"$FAKE_CURL_UPLOAD"
if [ -r /dev/fd/3 ]; then /bin/cat <&3 >"$FAKE_CURL_UPLOAD"; fi
if [ -r /dev/fd/4 ]; then /bin/cat <&4 >"$FAKE_CURL_UPLOAD.4"; fi
if [ -r /dev/fd/5 ]; then /bin/cat <&5 >"$FAKE_CURL_UPLOAD.5"; fi
if [ -r /dev/fd/6 ]; then /bin/cat <&6 >"$FAKE_CURL_UPLOAD.6"; fi
if [ -r /dev/fd/7 ]; then /bin/cat <&7 >"$FAKE_CURL_UPLOAD.7"; fi
/usr/bin/env >"$FAKE_CURL_ENV"
if [ -n "$output_path" ]; then
    /bin/cat "$FAKE_CURL_BODY" >"$output_path"
else
    /bin/cat "$FAKE_CURL_BODY"
fi
if [ "$write_out" = "%{http_code}" ]; then
    printf '%s' "$FAKE_CURL_HTTP_CODE"
fi
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
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "BOTIFIED_ASR_BASE_URL": "https://asr.example:17770/",
            "BOTIFIED_ASR_API_KEY": TOKEN,
            "FAKE_CURL_ARGS": str(args_path),
            "FAKE_CURL_STDIN": str(stdin_path),
            "FAKE_CURL_BODY": str(body_path),
            "FAKE_CURL_STDERR": str(stderr_path),
            "FAKE_CURL_EXIT": str(exit_code),
            "FAKE_CURL_HTTP_CODE": http_code,
            "FAKE_CURL_ENV": str(tmp_path / "curl-env"),
            "FAKE_CURL_UPLOAD": str(tmp_path / "curl-upload"),
        }
    )
    return environment, args_path, stdin_path


def _install_fake_job_wait_tools(
    tmp_path: Path,
    responses: list[tuple[bytes, str, int, str]],
) -> tuple[dict[str, str], Path, Path]:
    environment, _, _ = _install_fake_curl(tmp_path)
    fake_bin = Path(environment["PATH"].split(":", 1)[0])
    response_root = tmp_path / "job-wait-responses"
    response_root.mkdir()
    for index, (body, http_code, exit_code, delay) in enumerate(
        responses,
        start=1,
    ):
        prefix = response_root / str(index)
        prefix.with_suffix(".body").write_bytes(body)
        prefix.with_suffix(".http").write_text(http_code, encoding="ascii")
        prefix.with_suffix(".exit").write_text(
            str(exit_code),
            encoding="ascii",
        )
        prefix.with_suffix(".delay").write_text(delay, encoding="ascii")

    traces = tmp_path / "job-wait-traces"
    traces.mkdir()
    curl = fake_bin / "curl"
    curl.write_text(
        """#!/bin/sh
if [ -f "$FAKE_JOB_WAIT_TRACES/count" ]; then
    IFS= read -r call <"$FAKE_JOB_WAIT_TRACES/count"
else
    call=0
fi
call=$((call + 1))
printf '%s\\n' "$call" >"$FAKE_JOB_WAIT_TRACES/count"
: >"$FAKE_JOB_WAIT_TRACES/args.$call"
output_path=
capture_output=0
for argument do
    printf '%s\\0' "$argument" >>"$FAKE_JOB_WAIT_TRACES/args.$call"
    if [ "$capture_output" -eq 1 ]; then
        output_path=$argument
        capture_output=0
    elif [ "$argument" = "--output" ]; then
        capture_output=1
    fi
done
/bin/cat >"$FAKE_JOB_WAIT_TRACES/stdin.$call"
/bin/cat <&3 >"$FAKE_JOB_WAIT_TRACES/upload.$call"
/usr/bin/env >"$FAKE_JOB_WAIT_TRACES/env.$call"
prefix=$FAKE_JOB_WAIT_RESPONSES/$call
/bin/cat "$prefix.body" >"$output_path"
IFS= read -r delay <"$prefix.delay"
if [ "$delay" != "0" ]; then
    /usr/bin/sleep "$delay"
fi
/bin/cat "$prefix.http"
IFS= read -r exit_code <"$prefix.exit"
exit "$exit_code"
""",
        encoding="utf-8",
    )
    curl.chmod(0o755)
    sleep_log = traces / "sleep"
    sleep = fake_bin / "sleep"
    sleep.write_text(
        """#!/bin/sh
printf '%s\\n' "$1" >>"$FAKE_JOB_WAIT_SLEEP"
exit 0
""",
        encoding="utf-8",
    )
    sleep.chmod(0o755)
    environment.update(
        {
            "FAKE_JOB_WAIT_RESPONSES": str(response_root),
            "FAKE_JOB_WAIT_TRACES": str(traces),
            "FAKE_JOB_WAIT_SLEEP": str(sleep_log),
        }
    )
    return environment, traces, sleep_log


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


def _without_http_capture_suffix(arguments: list[bytes]) -> list[bytes]:
    assert arguments[-5] == b"--output"
    response_path = Path(os.fsdecode(arguments[-4]))
    assert arguments[-3:] == [b"--write-out", b"%{http_code}", b""]
    assert not response_path.exists()
    return [*arguments[:-5], b""]


def test_skill_tarball_has_exact_safe_ustar_shape_and_contents(
    tmp_path: Path,
) -> None:
    output = tmp_path / "asr-skill.tar.gz"

    subprocess.run([SKILL_BUILDER, output], check=True)

    expected = [
        ("asr", tarfile.DIRTYPE, 0o755, None),
        (
            "asr/SKILL.md",
            tarfile.REGTYPE,
            0o644,
            SKILL_ROOT / "SKILL.md",
        ),
        ("asr/agents", tarfile.DIRTYPE, 0o755, None),
        (
            "asr/agents/openai.yaml",
            tarfile.REGTYPE,
            0o644,
            SKILL_ROOT / "agents" / "openai.yaml",
        ),
        ("asr/references", tarfile.DIRTYPE, 0o755, None),
        (
            "asr/references/api.md",
            tarfile.REGTYPE,
            0o644,
            SKILL_ROOT / "references" / "api.md",
        ),
        ("asr/scripts", tarfile.DIRTYPE, 0o755, None),
        (
            "asr/scripts/botified-asr",
            tarfile.REGTYPE,
            0o755,
            HELPER,
        ),
    ]
    with tarfile.open(output, "r:gz") as archive:
        members = archive.getmembers()
        assert [member.name for member in members] == [
            item[0] for item in expected
        ]
        for member, (name, kind, mode, source) in zip(
            members,
            expected,
            strict=True,
        ):
            path = PurePosixPath(name)
            assert not path.is_absolute()
            assert path.parts[0] == "asr"
            assert ".." not in path.parts
            assert member.type == kind
            assert member.mode == mode
            assert member.uid == member.gid == 0
            assert member.uname == member.gname == ""
            assert member.mtime == 0
            if source is not None:
                extracted = archive.extractfile(member)
                assert extracted is not None
                assert extracted.read() == source.read_bytes()

    payload = output.read_bytes()
    assert payload[3] & 0x08 == 0
    assert payload[4:8] == b"\0\0\0\0"
    with gzip.open(output, "rb") as compressed:
        tar_bytes = compressed.read()
    for member in members:
        assert tar_bytes[member.offset + 257 : member.offset + 263] == (
            b"ustar\0"
        )
    assert output.stat().st_mode & 0o777 == 0o644
    assert SKILL_BUILDER.stat().st_mode & 0o111

    preserved = tmp_path / "preserved"
    preserved.write_bytes(b"keep")
    symlink_output = tmp_path / "linked-output.tar.gz"
    symlink_output.symlink_to(preserved)
    rejected = subprocess.run(
        [SKILL_BUILDER, symlink_output],
        capture_output=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert symlink_output.is_symlink()
    assert preserved.read_bytes() == b"keep"


def test_skill_tarball_is_byte_identical_across_umasks(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"

    subprocess.run(
        [
            "sh",
            "-c",
            'umask 077; exec "$1" "$2"',
            "sh",
            str(SKILL_BUILDER),
            str(first),
        ],
        check=True,
    )
    subprocess.run(
        [
            "sh",
            "-c",
            'umask 022; exec "$1" "$2"',
            "sh",
            str(SKILL_BUILDER),
            str(second),
        ],
        check=True,
    )

    assert first.read_bytes() == second.read_bytes()


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
    metadata = yaml.safe_load(frontmatter)
    assert metadata.keys() == {"name", "description"}
    assert metadata["name"] == "asr"
    assert "relative to this `SKILL.md` (the skill root)" in body
    assert "scripts/botified-asr health" in body
    assert "scripts/botified-asr transcribe AUDIO_FILE" in body
    assert "scripts/botified-asr transcribe-long AUDIO_FILE" in body
    assert (
        "scripts/botified-asr transcribe-meeting AUDIO_FILE "
        "[SPEAKER_ID ...]"
    ) in body
    assert "scripts/botified-asr job-get JOB_ID" in body
    assert "scripts/botified-asr job-wait JOB_ID TIMEOUT_SECONDS" in body
    assert "scripts/botified-asr job-delete JOB_ID" in body
    assert "scripts/botified-asr speaker-list" in body
    assert "scripts/botified-asr speaker-get SPEAKER_ID" in body
    assert (
        "scripts/botified-asr speaker-add NAME SAMPLE_FILE_1 SAMPLE_FILE_2 "
        "[SAMPLE_FILE_3 ... SAMPLE_FILE_5]"
    ) in body
    assert (
        "scripts/botified-asr speaker-put SPEAKER_ID NAME [DESCRIPTION]"
        in body
    )
    assert "scripts/botified-asr speaker-delete SPEAKER_ID" in body
    assert (
        body.index("first run `scripts/botified-asr health`")
        < body.index("Only after it returns ready")
        < body.index("`scripts/botified-asr transcribe AUDIO_FILE`")
    )
    assert "references/api.md" in body
    assert "transcrib" in metadata["description"].lower()
    reference = (SKILL_ROOT / "references" / "api.md").read_text(
        encoding="utf-8"
    )
    assert "POST `/v1/audio/transcriptions`" in reference
    assert "`model=sensevoice`" in reference
    assert "`model=sensevoice-diarize`" in reference
    assert "`response_format=diarized_json`" in reference
    assert "`known_speaker_ids[]`" in reference
    assert "`result.segments`" in reference
    assert "`chunking_strategy=auto`" in reference
    assert "GET `/v1/audio/transcriptions/{job_id}`" in reference
    assert "scripts/botified-asr job-wait JOB_ID TIMEOUT_SECONDS" in reference
    assert "DELETE `/v1/audio/transcriptions/{job_id}`" in reference
    assert "GET `/v1/speakers`" in reference
    assert "POST `/v1/speakers`" in reference
    assert "GET `/v1/speakers/{speaker_id}`" in reference
    assert "PUT `/v1/speakers/{speaker_id}`" in reference
    assert "DELETE `/v1/speakers/{speaker_id}`" in reference

    assert yaml.safe_load(
        (SKILL_ROOT / "agents" / "openai.yaml").read_text(encoding="utf-8")
    ) == {
        "interface": {
            "display_name": "Botified ASR",
            "short_description": (
                "Transcribe audio and register speaker profiles"
            ),
            "default_prompt": (
                "Use $asr to check readiness, transcribe audio, "
                "manage transcription jobs, and register or manage speaker "
                "profiles only when I explicitly ask."
            ),
        }
    }


@pytest.mark.parametrize(
    "arguments",
    [
        (),
        ("unknown",),
        ("health", "extra"),
        ("transcribe",),
        ("transcribe", "audio.wav", "extra"),
        ("transcribe-long",),
        ("transcribe-long", "audio.wav", "extra"),
        ("transcribe-meeting",),
        ("job-get",),
        ("job-get", "7K3M9Q2W", "extra"),
        ("job-wait",),
        ("job-wait", "7K3M9Q2W"),
        ("job-wait", "7K3M9Q2W", "1", "extra"),
        ("job-delete",),
        ("job-delete", "7K3M9Q2W", "extra"),
        ("speaker-list", "extra"),
        ("speaker-get",),
        ("speaker-get", "7K3M9Q2W", "extra"),
        ("speaker-add", "Ada", "one.wav"),
        (
            "speaker-add",
            "Ada",
            "1.wav",
            "2.wav",
            "3.wav",
            "4.wav",
            "5.wav",
            "6.wav",
        ),
        ("speaker-put",),
        ("speaker-put", "7K3M9Q2W"),
        ("speaker-put", "7K3M9Q2W", "Ada", "description", "extra"),
        ("speaker-delete",),
        ("speaker-delete", "7K3M9Q2W", "extra"),
    ],
)
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


def test_helper_requires_paired_process_environment(
    tmp_path: Path,
) -> None:
    environment, args_path, _ = _install_fake_curl(tmp_path)
    config_home = tmp_path / "legacy-config"
    client_file = config_home / "botified-asr" / "client.env"
    client_file.parent.mkdir(parents=True)
    client_file.write_text(
        "BOTIFIED_ASR_BASE_URL=https://legacy.example\n"
        f"BOTIFIED_ASR_API_KEY={TOKEN}\n",
        encoding="utf-8",
    )
    client_file.chmod(0o600)
    environment["XDG_CONFIG_HOME"] = str(config_home)
    environment.pop("BOTIFIED_ASR_BASE_URL")
    environment.pop("BOTIFIED_ASR_API_KEY")

    result = _run(environment, "health")

    assert result.returncode == 78
    assert _error_code(result) == "invalid_client_config"
    assert result.stderr == b""
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


def test_paired_process_environment_is_used_without_child_environment_leak(
    tmp_path: Path,
) -> None:
    environment, args_path, stdin_path = _install_fake_curl(tmp_path)
    environment.update(
        {
            "BOTIFIED_ASR_BASE_URL": "https://asr.example/",
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
def test_process_environment_must_be_present_nonempty_and_paired(
    tmp_path: Path,
    base_url: str | None,
    api_key: str | None,
) -> None:
    environment, args_path, _ = _install_fake_curl(tmp_path)
    environment.pop("BOTIFIED_ASR_BASE_URL")
    environment.pop("BOTIFIED_ASR_API_KEY")
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
    environment["BOTIFIED_ASR_BASE_URL"] = base_url

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
    environment["BOTIFIED_ASR_API_KEY"] = api_key

    result = _run(environment, "health")

    assert result.returncode == 78
    assert _error_code(result) == "invalid_client_config"
    if api_key:
        assert api_key.encode() not in result.stdout + result.stderr
    assert not args_path.exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ("transcribe", "missing.wav"),
        ("transcribe-long", "missing.wav"),
        ("job-get", "malformed/id"),
        ("job-wait", "malformed/id", "invalid-timeout"),
        ("job-delete", "malformed/id"),
        ("speaker-get", "malformed/id"),
        ("speaker-put", "malformed/id", "Ada"),
    ],
)
def test_client_configuration_errors_precede_local_input_validation(
    tmp_path: Path,
    arguments: tuple[str, ...],
) -> None:
    environment, args_path, _ = _install_fake_curl(tmp_path)
    environment.pop("BOTIFIED_ASR_BASE_URL")
    environment.pop("BOTIFIED_ASR_API_KEY")
    local_input = str(tmp_path / arguments[1])
    local_arguments = (arguments[0], local_input, *arguments[2:])

    result = _run(environment, *local_arguments)

    assert result.returncode == 78
    assert _error_code(result) == "invalid_client_config"
    assert result.stderr == b""
    assert local_input.encode() not in result.stdout + result.stderr
    assert not args_path.exists()


@pytest.mark.parametrize(
    "invalid_kind",
    ["missing", "directory", "fifo", "unreadable"],
)
def test_transcribe_requires_a_readable_regular_audio_file(
    tmp_path: Path,
    invalid_kind: str,
) -> None:
    environment, args_path, _ = _install_fake_curl(tmp_path)
    audio_path = tmp_path / f"invalid-{invalid_kind}.wav"
    if invalid_kind == "directory":
        audio_path.mkdir()
    elif invalid_kind == "fifo":
        os.mkfifo(audio_path)
    elif invalid_kind == "unreadable":
        audio_path.write_bytes(b"audio")
        audio_path.chmod(0)

    result = _run(environment, "transcribe", str(audio_path))

    assert result.returncode == 66
    assert _error_code(result) == "invalid_audio_file"
    assert json.loads(result.stdout)["error"]["param"] == "file"
    assert result.stderr == b""
    assert str(audio_path).encode() not in result.stdout + result.stderr
    assert not args_path.exists()


def test_audio_fd_open_failure_is_stable_and_does_not_call_curl(
    tmp_path: Path,
) -> None:
    environment, args_path, _ = _install_fake_curl(tmp_path)
    audio_path = tmp_path / "removed-after-validation.wav"
    audio_path.write_bytes(b"audio")
    fake_bin = Path(environment["PATH"].split(":", 1)[0])
    fake_mktemp = fake_bin / "mktemp"
    fake_mktemp.write_text(
        """#!/bin/sh
/bin/rm -f -- "$DELETE_BEFORE_AUDIO_OPEN"
/usr/bin/mktemp "$@"
""",
        encoding="utf-8",
    )
    fake_mktemp.chmod(0o755)
    environment["DELETE_BEFORE_AUDIO_OPEN"] = str(audio_path)

    result = _run(environment, "transcribe", str(audio_path))

    assert result.returncode == 66
    assert _error_code(result) == "invalid_audio_file"
    assert result.stderr == b""
    assert str(audio_path).encode() not in result.stdout + result.stderr
    assert not args_path.exists()


def test_transcribe_long_reuses_audio_file_validation(
    tmp_path: Path,
) -> None:
    environment, args_path, _ = _install_fake_curl(tmp_path)
    audio_path = tmp_path / "missing-long-audio.wav"

    result = _run(environment, "transcribe-long", str(audio_path))

    assert result.returncode == 66
    assert _error_code(result) == "invalid_audio_file"
    assert result.stderr == b""
    assert str(audio_path).encode() not in result.stdout + result.stderr
    assert not args_path.exists()


@pytest.mark.parametrize(
    "job_id",
    [
        "",
        "7K3M9Q2",
        "7K3M9Q2W0",
        "7k3m9q2w",
        "7K3M9Q2I",
        "7K3M9Q2L",
        "7K3M9Q2O",
        "7K3M9Q2U",
        "../health",
        "7K3M9Q?W",
        "7K3M9Q\n",
    ],
)
def test_job_get_rejects_invalid_job_id_without_request_or_echo(
    tmp_path: Path,
    job_id: str,
) -> None:
    environment, args_path, _ = _install_fake_curl(tmp_path)

    result = _run(environment, "job-get", job_id)

    assert result.returncode == 65
    assert _error_code(result) == "invalid_job_id"
    assert json.loads(result.stdout)["error"]["param"] == "job_id"
    assert result.stderr == b""
    if job_id:
        assert job_id.encode() not in result.stdout + result.stderr
    assert not args_path.exists()


@pytest.mark.parametrize(
    ("body", "curl_stderr", "curl_exit", "http_code", "expected_stdout"),
    [
        (
            b'{"id":"7K3M9Q2W","status":"queued",'
            b'"created_at":"2026-07-27T12:00:00Z"}',
            b"",
            0,
            "202",
            b'{"id":"7K3M9Q2W","status":"queued",'
            b'"created_at":"2026-07-27T12:00:00Z"}',
        ),
        (
            b'{"error":{"code":"audio_too_long"}}',
            b"curl http 413 detail",
            22,
            "413",
            b'{"error":{"code":"audio_too_long"}}',
        ),
        (
            b"partial private upload response",
            f"upload failed using {TOKEN}".encode(),
            28,
            "000",
            (
                b'{"error":{"message":"Botified ASR request failed",'
                b'"type":"client_error","param":null,"code":"curl_failed"}}\n'
            ),
        ),
    ],
)
def test_transcribe_long_submits_one_private_async_request(
    tmp_path: Path,
    body: bytes,
    curl_stderr: bytes,
    curl_exit: int,
    http_code: str,
    expected_stdout: bytes,
) -> None:
    environment, args_path, stdin_path = _install_fake_curl(
        tmp_path,
        body=body,
        stderr=curl_stderr,
        exit_code=curl_exit,
        http_code=http_code,
    )
    target = tmp_path / "real-long-audio.wav"
    audio_bytes = b"long local audio bytes"
    target.write_bytes(audio_bytes)
    audio_path = tmp_path / 'long ,;"{}[] $().wav'
    audio_path.symlink_to(target)

    result = _run(environment, "transcribe-long", str(audio_path))

    assert result.returncode == curl_exit
    assert result.stdout == expected_stdout
    assert result.stderr == b""
    arguments = _without_http_capture_suffix(
        args_path.read_bytes().split(b"\0")
    )
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
        b"--request",
        b"POST",
        b"--header",
        b"Prefer: respond-async",
        b"--form-string",
        b"model=sensevoice",
        b"--form-string",
        b"response_format=json",
        b"--form-string",
        b"chunking_strategy=auto",
        b"--form",
        (
            b"file=@/dev/fd/3;filename=audio;"
            b"type=application/octet-stream"
        ),
        b"--url",
        b"https://asr.example:17770/v1/audio/transcriptions",
        b"",
    ]
    assert b"--max-time" not in arguments
    assert b"--location" not in arguments
    assert b"-L" not in arguments
    assert str(audio_path).encode() not in args_path.read_bytes()
    assert TOKEN.encode() not in args_path.read_bytes()
    assert TOKEN.encode() not in result.stdout + result.stderr
    assert stdin_path.read_bytes() == (
        f'header = "Authorization: Bearer {TOKEN}"\n'.encode()
    )
    assert (tmp_path / "curl-upload").read_bytes() == audio_bytes
    curl_environment = (tmp_path / "curl-env").read_bytes()
    assert b"BOTIFIED_ASR_BASE_URL=" not in curl_environment
    assert b"BOTIFIED_ASR_API_KEY=" not in curl_environment


@pytest.mark.parametrize(
    ("known_speaker_ids", "expected_known_fields"),
    [
        ((), []),
        (
            ("7K3M9Q2W", "4X7K2M9Q"),
            [
                b"--form-string",
                b"known_speaker_ids[]=7K3M9Q2W",
                b"--form-string",
                b"known_speaker_ids[]=4X7K2M9Q",
            ],
        ),
        (
            tuple(f"{index:08d}" for index in range(32)),
            [
                field
                for index in range(32)
                for field in (
                    b"--form-string",
                    f"known_speaker_ids[]={index:08d}".encode(),
                )
            ],
        ),
    ],
    ids=("anonymous", "two-known", "max-known"),
)
def test_transcribe_meeting_submits_exact_private_async_request(
    tmp_path: Path,
    known_speaker_ids: tuple[str, ...],
    expected_known_fields: list[bytes],
) -> None:
    body = (
        b'{"id":"7K3M9Q2W","status":"queued",'
        b'"created_at":"2026-07-27T12:00:00Z"}'
    )
    environment, args_path, stdin_path = _install_fake_curl(
        tmp_path,
        body=body,
        http_code="202",
    )
    target = tmp_path / "real-meeting-audio.wav"
    audio_bytes = b"private meeting audio bytes"
    target.write_bytes(audio_bytes)
    audio_path = tmp_path / 'meeting ,;"{}[] $().wav'
    audio_path.symlink_to(target)

    result = _run(
        environment,
        "transcribe-meeting",
        str(audio_path),
        *known_speaker_ids,
    )

    assert result.returncode == 0
    assert result.stdout == body
    assert result.stderr == b""
    arguments = _without_http_capture_suffix(
        args_path.read_bytes().split(b"\0")
    )
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
        b"--request",
        b"POST",
        b"--header",
        b"Prefer: respond-async",
        b"--form-string",
        b"model=sensevoice-diarize",
        b"--form-string",
        b"response_format=diarized_json",
        b"--form-string",
        b"chunking_strategy=auto",
        *expected_known_fields,
        b"--form",
        (
            b"file=@/dev/fd/3;filename=audio;"
            b"type=application/octet-stream"
        ),
        b"--url",
        b"https://asr.example:17770/v1/audio/transcriptions",
        b"",
    ]
    assert str(audio_path).encode() not in args_path.read_bytes()
    assert TOKEN.encode() not in args_path.read_bytes()
    assert TOKEN.encode() not in result.stdout + result.stderr
    assert stdin_path.read_bytes() == (
        f'header = "Authorization: Bearer {TOKEN}"\n'.encode()
    )
    assert (tmp_path / "curl-upload").read_bytes() == audio_bytes
    curl_environment = (tmp_path / "curl-env").read_bytes()
    assert b"BOTIFIED_ASR_BASE_URL=" not in curl_environment
    assert b"BOTIFIED_ASR_API_KEY=" not in curl_environment


@pytest.mark.parametrize(
    "known_speaker_ids",
    [
        ("",),
        ("malformed/id",),
        ("7K3M9Q2W", "7K3M9Q2W"),
        tuple(f"{index:08d}" for index in range(33)),
    ],
    ids=("empty", "invalid", "duplicate", "too-many"),
)
def test_transcribe_meeting_rejects_invalid_known_speakers_before_curl(
    tmp_path: Path,
    known_speaker_ids: tuple[str, ...],
) -> None:
    environment, args_path, _ = _install_fake_curl(tmp_path)
    audio_path = tmp_path / "meeting.wav"
    audio_path.write_bytes(b"meeting audio")

    result = _run(
        environment,
        "transcribe-meeting",
        str(audio_path),
        *known_speaker_ids,
    )

    assert result.returncode == 65
    assert _error_code(result) == "invalid_known_speaker_ids"
    assert json.loads(result.stdout)["error"]["param"] == (
        "known_speaker_ids[]"
    )
    assert result.stderr == b""
    assert not args_path.exists()


@pytest.mark.parametrize(
    ("body", "curl_stderr", "curl_exit", "http_code", "expected_stdout"),
    [
        (
            b'{"id":"7K3M9Q2W","status":"queued","progress":'
            b'{"processed_audio_secs":0.0,"total_audio_secs":null}}',
            b"",
            0,
            "202",
            b'{"id":"7K3M9Q2W","status":"queued","progress":'
            b'{"processed_audio_secs":0.0,"total_audio_secs":null}}',
        ),
        (
            b'{"id":"7K3M9Q2W","status":"succeeded",'
            b'"result":{"text":"hello"}}',
            b"",
            0,
            "200",
            b'{"id":"7K3M9Q2W","status":"succeeded",'
            b'"result":{"text":"hello"}}',
        ),
        (
            b'{"error":{"code":"job_not_found"}}',
            b"curl http 404 detail",
            22,
            "404",
            b'{"error":{"code":"job_not_found"}}',
        ),
        (
            b"partial private job response",
            f"job lookup failed using {TOKEN}".encode(),
            28,
            "000",
            (
                b'{"error":{"message":"Botified ASR request failed",'
                b'"type":"client_error","param":null,"code":"curl_failed"}}\n'
            ),
        ),
    ],
)
def test_job_get_preserves_responses_and_shared_errors(
    tmp_path: Path,
    body: bytes,
    curl_stderr: bytes,
    curl_exit: int,
    http_code: str,
    expected_stdout: bytes,
) -> None:
    environment, args_path, stdin_path = _install_fake_curl(
        tmp_path,
        body=body,
        stderr=curl_stderr,
        exit_code=curl_exit,
        http_code=http_code,
    )

    result = _run(environment, "job-get", "7K3M9Q2W")

    assert result.returncode == curl_exit
    assert result.stdout == expected_stdout
    assert result.stderr == b""
    arguments = _without_http_capture_suffix(
        args_path.read_bytes().split(b"\0")
    )
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
        b"--request",
        b"GET",
        b"--url",
        (
            b"https://asr.example:17770/v1/audio/"
            b"transcriptions/7K3M9Q2W"
        ),
        b"",
    ]
    assert b"--max-time" not in arguments
    assert b"--header" not in arguments
    assert b"--form" not in arguments
    assert b"--form-string" not in arguments
    assert b"--location" not in arguments
    assert b"-L" not in arguments
    assert TOKEN.encode() not in args_path.read_bytes()
    assert TOKEN.encode() not in result.stdout + result.stderr
    assert stdin_path.read_bytes() == (
        f'header = "Authorization: Bearer {TOKEN}"\n'.encode()
    )
    assert (tmp_path / "curl-upload").read_bytes() == b""
    curl_environment = (tmp_path / "curl-env").read_bytes()
    assert b"BOTIFIED_ASR_BASE_URL=" not in curl_environment
    assert b"BOTIFIED_ASR_API_KEY=" not in curl_environment


def _duration_centiseconds(value: bytes) -> int:
    whole, fraction = value.split(b".", 1)
    assert whole.isdigit()
    assert len(fraction) == 2
    assert fraction.isdigit()
    return int(whole) * 100 + int(fraction)


def test_job_wait_discards_active_responses_and_caps_backoff(
    tmp_path: Path,
) -> None:
    active_bodies = [
        (
            f'{{"id":"7K3M9Q2W","status":"running",'
            f'"private_poll":{index}}}'
        ).encode()
        for index in range(1, 6)
    ]
    terminal = (
        b'{"id":"7K3M9Q2W","status":"succeeded",'
        b'"result":{"text":"final"}}'
    )
    environment, traces, sleep_log = _install_fake_job_wait_tools(
        tmp_path,
        [
            *((body, "202", 0, "0") for body in active_bodies),
            (terminal, "200", 0, "0"),
        ],
    )

    result = _run(
        environment,
        "job-wait",
        "7K3M9Q2W",
        "000999999999",
    )

    assert result.returncode == 0
    assert result.stdout == terminal
    assert result.stderr == b""
    assert sleep_log.read_bytes() == b"1.00\n2.00\n4.00\n8.00\n8.00\n"
    assert (traces / "count").read_text(encoding="ascii").strip() == "6"
    response_paths: set[bytes] = set()
    max_times: list[int] = []
    for call in range(1, 7):
        arguments = (traces / f"args.{call}").read_bytes().split(b"\0")
        assert arguments[:10] == [
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
        ]
        assert arguments[10:14] == [
            b"--request",
            b"GET",
            b"--url",
            (
                b"https://asr.example:17770/v1/audio/"
                b"transcriptions/7K3M9Q2W"
            ),
        ]
        assert arguments[14] == b"--max-time"
        max_times.append(_duration_centiseconds(arguments[15]))
        assert arguments[16] == b"--output"
        response_paths.add(arguments[17])
        assert arguments[18:] == [b"--write-out", b"%{http_code}", b""]
        assert b"--location" not in arguments
        assert b"-L" not in arguments
        assert TOKEN.encode() not in (traces / f"args.{call}").read_bytes()
        assert (traces / f"stdin.{call}").read_bytes() == (
            f'header = "Authorization: Bearer {TOKEN}"\n'.encode()
        )
        assert (traces / f"upload.{call}").read_bytes() == b""
        curl_environment = (traces / f"env.{call}").read_bytes()
        assert b"BOTIFIED_ASR_BASE_URL=" not in curl_environment
        assert b"BOTIFIED_ASR_API_KEY=" not in curl_environment
    assert max_times[0] == 999_999_999 * 100
    assert all(
        0 < later <= earlier
        for earlier, later in zip(max_times, max_times[1:])
    )
    assert len(response_paths) == 1
    assert not Path(os.fsdecode(response_paths.pop())).exists()
    for body in active_bodies:
        assert body not in result.stdout + result.stderr
    assert TOKEN.encode() not in result.stdout + result.stderr


@pytest.mark.parametrize(
    "timeout_seconds",
    [
        "",
        "0",
        "000",
        "1000000000",
        "0001000000000",
        "1.0",
        "+1",
        "-1",
        "1x",
        "1\n",
    ],
)
def test_job_wait_rejects_invalid_timeout_after_configuration(
    tmp_path: Path,
    timeout_seconds: str,
) -> None:
    environment, args_path, _ = _install_fake_curl(tmp_path)

    result = _run(
        environment,
        "job-wait",
        "7K3M9Q2W",
        timeout_seconds,
    )

    assert result.returncode == 65
    assert _error_code(result) == "invalid_timeout_seconds"
    assert json.loads(result.stdout)["error"]["param"] == "timeout_seconds"
    assert result.stderr == b""
    if timeout_seconds:
        assert timeout_seconds.encode() not in result.stdout + result.stderr
    assert not args_path.exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ("job-wait", "../health", "1"),
        ("job-delete", "../health"),
    ],
)
def test_job_commands_reuse_strict_job_id_validation(
    tmp_path: Path,
    arguments: tuple[str, ...],
) -> None:
    environment, args_path, _ = _install_fake_curl(tmp_path)

    result = _run(environment, *arguments)

    assert result.returncode == 65
    assert _error_code(result) == "invalid_job_id"
    assert json.loads(result.stdout)["error"]["param"] == "job_id"
    assert b"../health" not in result.stdout + result.stderr
    assert not args_path.exists()


@pytest.mark.parametrize(
    "command_name",
    ["speaker-get", "speaker-put", "speaker-delete"],
)
def test_speaker_commands_reject_dangerous_id_without_request_or_echo(
    tmp_path: Path,
    command_name: str,
) -> None:
    environment, args_path, _ = _install_fake_curl(tmp_path)

    if command_name == "speaker-put":
        command_arguments = ("../health", "Ada")
    else:
        command_arguments = ("../health",)
    result = _run(environment, command_name, *command_arguments)

    assert result.returncode == 65
    assert _error_code(result) == "invalid_speaker_id"
    assert json.loads(result.stdout)["error"]["param"] == "speaker_id"
    assert result.stderr == b""
    assert b"../health" not in result.stdout + result.stderr
    assert not args_path.exists()


@pytest.mark.parametrize(
    (
        "body",
        "http_code",
        "curl_exit",
        "expected_exit",
        "expected_stdout",
    ),
    [
        (
            b'{"redirect":"private"}',
            "302",
            0,
            76,
            (
                b'{"error":{"message":"Botified ASR received an unexpected '
                b'HTTP response","type":"client_error","param":null,'
                b'"code":"unexpected_http_response"}}\n'
            ),
        ),
        (
            b"",
            "204",
            0,
            76,
            (
                b'{"error":{"message":"Botified ASR received an unexpected '
                b'HTTP response","type":"client_error","param":null,'
                b'"code":"unexpected_http_response"}}\n'
            ),
        ),
        (
            b"private malformed status body",
            "2000",
            0,
            76,
            (
                b'{"error":{"message":"Botified ASR received an unexpected '
                b'HTTP response","type":"client_error","param":null,'
                b'"code":"unexpected_http_response"}}\n'
            ),
        ),
        (
            b'{"error":{"code":"job_not_found"}}',
            "404",
            22,
            22,
            b'{"error":{"code":"job_not_found"}}',
        ),
        (
            b"private transport body",
            "000",
            7,
            7,
            (
                b'{"error":{"message":"Botified ASR request failed",'
                b'"type":"client_error","param":null,"code":"curl_failed"}}\n'
            ),
        ),
        (
            b"private early timeout body",
            "000",
            28,
            28,
            (
                b'{"error":{"message":"Botified ASR request failed",'
                b'"type":"client_error","param":null,"code":"curl_failed"}}\n'
            ),
        ),
    ],
)
def test_job_wait_http_and_transport_precedence(
    tmp_path: Path,
    body: bytes,
    http_code: str,
    curl_exit: int,
    expected_exit: int,
    expected_stdout: bytes,
) -> None:
    environment, traces, sleep_log = _install_fake_job_wait_tools(
        tmp_path,
        [(body, http_code, curl_exit, "0")],
    )

    result = _run(environment, "job-wait", "7K3M9Q2W", "10")

    assert result.returncode == expected_exit
    assert result.stdout == expected_stdout
    assert result.stderr == b""
    assert not sleep_log.exists()
    assert (traces / "count").read_text(encoding="ascii").strip() == "1"
    if body and result.returncode != 22:
        assert body not in result.stdout + result.stderr
    assert TOKEN.encode() not in result.stdout + result.stderr


def test_job_wait_maps_curl_28_to_timeout_only_after_deadline(
    tmp_path: Path,
) -> None:
    private_body = b"private timeout body"
    environment, traces, sleep_log = _install_fake_job_wait_tools(
        tmp_path,
        [(private_body, "000", 28, "1.05")],
    )

    result = _run(environment, "job-wait", "7K3M9Q2W", "1")

    assert result.returncode == 75
    assert _error_code(result) == "job_wait_timeout"
    assert json.loads(result.stdout)["error"]["param"] == "timeout_seconds"
    assert result.stderr == b""
    assert private_body not in result.stdout
    assert not sleep_log.exists()
    assert (traces / "count").read_text(encoding="ascii").strip() == "1"
    assert TOKEN.encode() not in result.stdout + result.stderr


def test_job_wait_term_exits_and_cleans_without_another_request(
    tmp_path: Path,
) -> None:
    active_body = b'{"id":"7K3M9Q2W","status":"running"}'
    environment, traces, sleep_log = _install_fake_job_wait_tools(
        tmp_path,
        [
            (active_body, "202", 0, "0"),
            (
                b'{"id":"7K3M9Q2W","status":"succeeded",'
                b'"result":{"text":"must not be requested"}}',
                "200",
                0,
                "0",
            ),
        ],
    )
    sleep_ready = traces / "sleep-ready"
    sleep_pid_path = traces / "sleep-pid"
    temporary_root = tmp_path / "private-tmp"
    temporary_root.mkdir()
    environment["TMPDIR"] = str(temporary_root)
    fake_bin = Path(environment["PATH"].split(":", 1)[0])
    sleep = fake_bin / "sleep"
    sleep.write_text(
        """#!/bin/sh
printf '%s\\n' "$1" >>"$FAKE_JOB_WAIT_SLEEP"
printf '%s\\n' "$$" >"$FAKE_JOB_WAIT_SLEEP_PID"
: >"$FAKE_JOB_WAIT_SLEEP_READY"
exec /usr/bin/sleep 30
""",
        encoding="utf-8",
    )
    sleep.chmod(0o755)
    environment.update(
        {
            "FAKE_JOB_WAIT_SLEEP_PID": str(sleep_pid_path),
            "FAKE_JOB_WAIT_SLEEP_READY": str(sleep_ready),
        }
    )

    process = subprocess.Popen(
        [HELPER, "job-wait", "7K3M9Q2W", "60"],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    sleep_pid: int | None = None
    child_survived_handler: bool | None = None
    try:
        ready_deadline = time.monotonic() + 2
        while not sleep_ready.exists() and time.monotonic() < ready_deadline:
            time.sleep(0.01)
        assert sleep_ready.exists()
        sleep_pid = int(sleep_pid_path.read_text(encoding="ascii"))
        first_arguments = (traces / "args.1").read_bytes().split(b"\0")
        response_body = Path(os.fsdecode(first_arguments[17]))

        termination_started = time.monotonic()
        os.kill(process.pid, signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=1)
        assert time.monotonic() - termination_started < 1
    finally:
        if sleep_pid is not None:
            child_survived_handler = Path(f"/proc/{sleep_pid}").exists()
        if process.poll() is None:
            os.kill(process.pid, signal.SIGKILL)
            process.wait(timeout=2)
        if sleep_pid is not None and Path(f"/proc/{sleep_pid}").exists():
            os.kill(sleep_pid, signal.SIGKILL)

    assert process.returncode == 143
    assert stdout == b""
    assert stderr == b""
    assert sleep_log.read_bytes()
    assert (traces / "count").read_text(encoding="ascii").strip() == "1"
    assert not (traces / "args.2").exists()
    assert not response_body.exists()
    assert sleep_pid is not None
    assert child_survived_handler is False
    assert list(temporary_root.iterdir()) == []


def test_job_wait_term_reaps_blocked_curl_and_cleans(
    tmp_path: Path,
) -> None:
    environment, traces, _ = _install_fake_job_wait_tools(
        tmp_path,
        [(b"", "000", 0, "0")],
    )
    curl_ready = traces / "curl-ready"
    curl_pid_path = traces / "curl-pid"
    temporary_root = tmp_path / "private-tmp"
    temporary_root.mkdir()
    environment["TMPDIR"] = str(temporary_root)
    fake_bin = Path(environment["PATH"].split(":", 1)[0])
    curl = fake_bin / "curl"
    curl.write_text(
        """#!/bin/sh
: >"$FAKE_JOB_WAIT_TRACES/args.1"
for argument do
    printf '%s\\0' "$argument" >>"$FAKE_JOB_WAIT_TRACES/args.1"
done
/bin/cat >"$FAKE_JOB_WAIT_TRACES/stdin.1"
/bin/cat <&3 >"$FAKE_JOB_WAIT_TRACES/upload.1"
printf '%s\\n' 1 >"$FAKE_JOB_WAIT_TRACES/count"
printf '%s\\n' "$$" >"$FAKE_JOB_WAIT_CURL_PID"
: >"$FAKE_JOB_WAIT_CURL_READY"
exec /usr/bin/sleep 30
""",
        encoding="utf-8",
    )
    curl.chmod(0o755)
    environment.update(
        {
            "FAKE_JOB_WAIT_CURL_PID": str(curl_pid_path),
            "FAKE_JOB_WAIT_CURL_READY": str(curl_ready),
        }
    )

    process = subprocess.Popen(
        [HELPER, "job-wait", "7K3M9Q2W", "60"],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    curl_pid: int | None = None
    child_survived_handler: bool | None = None
    try:
        ready_deadline = time.monotonic() + 2
        while not curl_ready.exists() and time.monotonic() < ready_deadline:
            time.sleep(0.01)
        assert curl_ready.exists()
        curl_pid = int(curl_pid_path.read_text(encoding="ascii"))

        termination_started = time.monotonic()
        os.kill(process.pid, signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=1)
        assert time.monotonic() - termination_started < 1
    finally:
        if curl_pid is not None:
            child_survived_handler = Path(f"/proc/{curl_pid}").exists()
        if process.poll() is None:
            os.kill(process.pid, signal.SIGKILL)
            process.wait(timeout=2)
        if curl_pid is not None and Path(f"/proc/{curl_pid}").exists():
            os.kill(curl_pid, signal.SIGKILL)

    assert process.returncode == 143
    assert stdout == b""
    assert stderr == b""
    assert (traces / "count").read_text(encoding="ascii").strip() == "1"
    assert not (traces / "args.2").exists()
    assert curl_pid is not None
    assert child_survived_handler is False
    assert list(temporary_root.iterdir()) == []


@pytest.mark.parametrize(
    ("command_name", "http_code"),
    [
        ("health", "202"),
        ("transcribe", "202"),
        ("transcribe-long", "200"),
        ("transcribe-meeting", "200"),
        ("job-get", "204"),
        ("job-delete", "302"),
        ("speaker-list", "202"),
        ("speaker-get", "204"),
        ("speaker-add", "200"),
        ("speaker-put", "204"),
        ("speaker-delete", "200"),
    ],
)
def test_one_shot_commands_fail_closed_on_unexpected_http_status(
    tmp_path: Path,
    command_name: str,
    http_code: str,
) -> None:
    private_body = b"private mismatched response body"
    environment, _, _ = _install_fake_curl(
        tmp_path,
        body=private_body,
        http_code=http_code,
    )
    if command_name in {
        "transcribe",
        "transcribe-long",
        "transcribe-meeting",
    }:
        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"audio")
        arguments = (command_name, str(audio_path))
    elif command_name == "speaker-add":
        samples = []
        for index in range(2):
            sample = tmp_path / f"sample-{index}.wav"
            sample.write_bytes(b"voice")
            samples.append(str(sample))
        arguments = (command_name, "Ada", *samples)
    elif command_name == "speaker-put":
        arguments = (command_name, "7K3M9Q2W", "Ada")
    elif command_name in {
        "job-get",
        "job-delete",
        "speaker-get",
        "speaker-delete",
    }:
        arguments = (command_name, "7K3M9Q2W")
    else:
        arguments = (command_name,)

    result = _run(environment, *arguments)

    assert result.returncode == 76
    assert result.stdout == (
        b'{"error":{"message":"Botified ASR received an unexpected HTTP '
        b'response","type":"client_error","param":null,'
        b'"code":"unexpected_http_response"}}\n'
    )
    assert result.stderr == b""
    assert private_body not in result.stdout + result.stderr
    assert TOKEN.encode() not in result.stdout + result.stderr


def test_speaker_add_posts_literal_name_and_five_private_samples(
    tmp_path: Path,
) -> None:
    body = b'{"id":"7K3M9Q2W","name":"Ada Lovelace"}'
    environment, args_path, stdin_path = _install_fake_curl(
        tmp_path,
        body=body,
        http_code="201",
    )
    sample_paths = []
    sample_bytes = []
    for index in range(1, 6):
        target = tmp_path / f"voice-target-{index}.wav"
        content = f"private voice {index}".encode()
        target.write_bytes(content)
        sample = tmp_path / f'voice {index} ,;"[] $().wav'
        sample.symlink_to(target)
        sample_paths.append(sample)
        sample_bytes.append(content)

    result = _run(
        environment,
        "speaker-add",
        "@/literal speaker name",
        *(str(path) for path in sample_paths),
    )

    assert result.returncode == 0
    assert result.stdout == body
    assert result.stderr == b""
    arguments = _without_http_capture_suffix(
        args_path.read_bytes().split(b"\0")
    )
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
        b"--request",
        b"POST",
        b"--form-string",
        b"name=@/literal speaker name",
        b"--form",
        (
            b"samples[]=@/dev/fd/3;filename=sample-1;"
            b"type=application/octet-stream"
        ),
        b"--form",
        (
            b"samples[]=@/dev/fd/4;filename=sample-2;"
            b"type=application/octet-stream"
        ),
        b"--form",
        (
            b"samples[]=@/dev/fd/5;filename=sample-3;"
            b"type=application/octet-stream"
        ),
        b"--form",
        (
            b"samples[]=@/dev/fd/6;filename=sample-4;"
            b"type=application/octet-stream"
        ),
        b"--form",
        (
            b"samples[]=@/dev/fd/7;filename=sample-5;"
            b"type=application/octet-stream"
        ),
        b"--url",
        b"https://asr.example:17770/v1/speakers",
        b"",
    ]
    assert stdin_path.read_bytes() == (
        f'header = "Authorization: Bearer {TOKEN}"\n'.encode()
    )
    for descriptor, content in zip(range(3, 8), sample_bytes, strict=True):
        suffix = "" if descriptor == 3 else f".{descriptor}"
        assert (tmp_path / f"curl-upload{suffix}").read_bytes() == content
    curl_environment = (tmp_path / "curl-env").read_bytes()
    for sample_path in sample_paths:
        encoded_path = str(sample_path).encode()
        assert encoded_path not in args_path.read_bytes()
        assert encoded_path not in curl_environment
        assert encoded_path not in result.stdout + result.stderr
    assert TOKEN.encode() not in args_path.read_bytes()
    assert TOKEN.encode() not in curl_environment
    assert TOKEN.encode() not in result.stdout + result.stderr


def test_speaker_add_rejects_an_empty_sample_without_echoing_its_path(
    tmp_path: Path,
) -> None:
    environment, args_path, _ = _install_fake_curl(tmp_path)
    valid_sample = tmp_path / "valid.wav"
    valid_sample.write_bytes(b"voice")
    invalid_sample = tmp_path / "private-empty.wav"
    invalid_sample.touch()

    result = _run(
        environment,
        "speaker-add",
        "Ada",
        str(valid_sample),
        str(invalid_sample),
    )

    assert result.returncode == 66
    assert _error_code(result) == "invalid_speaker_sample"
    assert json.loads(result.stdout)["error"]["param"] == "samples[]"
    assert result.stderr == b""
    assert str(invalid_sample).encode() not in result.stdout + result.stderr
    assert not args_path.exists()


def test_speaker_add_fd_open_failure_does_not_echo_the_sample_path(
    tmp_path: Path,
) -> None:
    environment, args_path, _ = _install_fake_curl(tmp_path)
    first_sample = tmp_path / "first.wav"
    first_sample.write_bytes(b"voice")
    removed_sample = tmp_path / "removed-after-validation.wav"
    removed_sample.write_bytes(b"voice")
    fake_bin = Path(environment["PATH"].split(":", 1)[0])
    fake_mktemp = fake_bin / "mktemp"
    fake_mktemp.write_text(
        """#!/bin/sh
/bin/rm -f -- "$DELETE_BEFORE_SAMPLE_OPEN"
/usr/bin/mktemp "$@"
""",
        encoding="utf-8",
    )
    fake_mktemp.chmod(0o755)
    environment["DELETE_BEFORE_SAMPLE_OPEN"] = str(removed_sample)

    result = _run(
        environment,
        "speaker-add",
        "Ada",
        str(first_sample),
        str(removed_sample),
    )

    assert result.returncode == 66
    assert _error_code(result) == "invalid_speaker_sample"
    assert json.loads(result.stdout)["error"]["param"] == "samples[]"
    assert str(removed_sample).encode() not in result.stdout + result.stderr
    assert not args_path.exists()


@pytest.mark.parametrize(
    ("description_arguments", "description_form"),
    [
        ((), []),
        (("",), [b"--form-string", b"description="]),
        (
            ("@/private/voice.wav;type=audio/wav",),
            [
                b"--form-string",
                b"description=@/private/voice.wav;type=audio/wav",
            ],
        ),
    ],
)
def test_speaker_put_preserves_literal_description_tristate(
    tmp_path: Path,
    description_arguments: tuple[str, ...],
    description_form: list[bytes],
) -> None:
    body = b'{"id":"7K3M9Q2W","name":"Ada Lovelace"}'
    environment, args_path, stdin_path = _install_fake_curl(
        tmp_path,
        body=body,
        http_code="200",
    )

    result = _run(
        environment,
        "speaker-put",
        "7K3M9Q2W",
        "Ada Lovelace",
        *description_arguments,
    )

    assert result.returncode == 0
    assert result.stdout == body
    assert result.stderr == b""
    arguments = args_path.read_bytes().split(b"\0")
    request_arguments = [
        b"--request",
        b"PUT",
        b"--form-string",
        b"name=Ada Lovelace",
        *description_form,
        b"--url",
        b"https://asr.example:17770/v1/speakers/7K3M9Q2W",
    ]
    assert arguments[10 : 10 + len(request_arguments)] == request_arguments
    capture_index = 10 + len(request_arguments)
    assert arguments[capture_index] == b"--output"
    if not description_arguments:
        assert arguments[:10] == [
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
        ]
        assert arguments[capture_index + 2 :] == [
            b"--write-out",
            b"%{http_code}",
            b"",
        ]
        assert b"--max-time" not in arguments
        assert b"--header" not in arguments
        assert b"--form" not in arguments
        assert b"--location" not in arguments
        assert b"-L" not in arguments
        assert TOKEN.encode() not in args_path.read_bytes()
        assert TOKEN.encode() not in result.stdout + result.stderr
        assert stdin_path.read_bytes() == (
            f'header = "Authorization: Bearer {TOKEN}"\n'.encode()
        )
        assert (tmp_path / "curl-upload").read_bytes() == b""
        curl_environment = (tmp_path / "curl-env").read_bytes()
        assert b"BOTIFIED_ASR_BASE_URL=" not in curl_environment
        assert b"BOTIFIED_ASR_API_KEY=" not in curl_environment


@pytest.mark.parametrize(
    (
        "command_name",
        "command_arguments",
        "body",
        "http_code",
        "method",
        "path",
        "expected_stdout",
    ),
    [
        (
            "speaker-list",
            (),
            b'{"data":[{"id":"7K3M9Q2W","name":"Ada"}]}',
            "200",
            "GET",
            "/v1/speakers",
            b'{"data":[{"id":"7K3M9Q2W","name":"Ada"}]}',
        ),
        (
            "speaker-get",
            ("7K3M9Q2W",),
            b'{"id":"7K3M9Q2W","name":"Ada"}',
            "200",
            "GET",
            "/v1/speakers/7K3M9Q2W",
            b'{"id":"7K3M9Q2W","name":"Ada"}',
        ),
        (
            "speaker-delete",
            ("7K3M9Q2W",),
            b"must be discarded",
            "204",
            "DELETE",
            "/v1/speakers/7K3M9Q2W",
            b"",
        ),
    ],
)
def test_speaker_commands_use_the_existing_profile_endpoints(
    tmp_path: Path,
    command_name: str,
    command_arguments: tuple[str, ...],
    body: bytes,
    http_code: str,
    method: str,
    path: str,
    expected_stdout: bytes,
) -> None:
    environment, args_path, stdin_path = _install_fake_curl(
        tmp_path,
        body=body,
        http_code=http_code,
    )

    result = _run(environment, command_name, *command_arguments)

    assert result.returncode == 0
    assert result.stdout == expected_stdout
    assert result.stderr == b""
    arguments = args_path.read_bytes().split(b"\0")
    assert arguments[10:14] == [
        b"--request",
        method.encode(),
        b"--url",
        f"https://asr.example:17770{path}".encode(),
    ]
    if command_name == "speaker-list":
        assert arguments[:10] == [
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
        ]
        assert arguments[14] == b"--output"
        assert arguments[16:] == [b"--write-out", b"%{http_code}", b""]
        assert b"--max-time" not in arguments
        assert b"--header" not in arguments
        assert b"--form" not in arguments
        assert b"--form-string" not in arguments
        assert b"--location" not in arguments
        assert b"-L" not in arguments
        assert TOKEN.encode() not in args_path.read_bytes()
        assert TOKEN.encode() not in result.stdout + result.stderr
        assert stdin_path.read_bytes() == (
            f'header = "Authorization: Bearer {TOKEN}"\n'.encode()
        )
        assert (tmp_path / "curl-upload").read_bytes() == b""
        curl_environment = (tmp_path / "curl-env").read_bytes()
        assert b"BOTIFIED_ASR_BASE_URL=" not in curl_environment
        assert b"BOTIFIED_ASR_API_KEY=" not in curl_environment


def test_speaker_get_preserves_one_service_error_body(tmp_path: Path) -> None:
    body = b'{"error":{"code":"speaker_not_found"}}'
    environment, _, _ = _install_fake_curl(
        tmp_path,
        body=body,
        exit_code=22,
        http_code="404",
    )

    result = _run(environment, "speaker-get", "7K3M9Q2W")

    assert result.returncode == 22
    assert result.stdout == body
    assert result.stderr == b""


@pytest.mark.parametrize(
    (
        "body",
        "http_code",
        "curl_exit",
        "expected_exit",
        "expected_stdout",
    ),
    [
        (
            b'{"id":"7K3M9Q2W","status":"running"}',
            "202",
            0,
            0,
            b'{"id":"7K3M9Q2W","status":"running"}',
        ),
        (b"must be discarded", "204", 0, 0, b""),
        (
            b'{"error":{"code":"job_not_found"}}',
            "404",
            22,
            22,
            b'{"error":{"code":"job_not_found"}}',
        ),
        (
            b"private transport body",
            "000",
            7,
            7,
            (
                b'{"error":{"message":"Botified ASR request failed",'
                b'"type":"client_error","param":null,"code":"curl_failed"}}\n'
            ),
        ),
    ],
)
def test_job_delete_uses_one_delete_request_and_shared_boundaries(
    tmp_path: Path,
    body: bytes,
    http_code: str,
    curl_exit: int,
    expected_exit: int,
    expected_stdout: bytes,
) -> None:
    environment, args_path, stdin_path = _install_fake_curl(
        tmp_path,
        body=body,
        exit_code=curl_exit,
        http_code=http_code,
    )

    result = _run(environment, "job-delete", "7K3M9Q2W")

    assert result.returncode == expected_exit
    assert result.stdout == expected_stdout
    assert result.stderr == b""
    if http_code == "202":
        arguments = args_path.read_bytes().split(b"\0")
        assert arguments[:14] == [
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
            b"--request",
            b"DELETE",
            b"--url",
            (
                b"https://asr.example:17770/v1/audio/"
                b"transcriptions/7K3M9Q2W"
            ),
        ]
        assert arguments[14] == b"--output"
        assert arguments[16:] == [b"--write-out", b"%{http_code}", b""]
        assert b"--max-time" not in arguments
        assert b"--header" not in arguments
        assert b"--form" not in arguments
        assert b"--form-string" not in arguments
        assert b"--location" not in arguments
        assert b"-L" not in arguments
        assert TOKEN.encode() not in args_path.read_bytes()
        assert TOKEN.encode() not in result.stdout + result.stderr
        assert stdin_path.read_bytes() == (
            f'header = "Authorization: Bearer {TOKEN}"\n'.encode()
        )
        assert (tmp_path / "curl-upload").read_bytes() == b""
        curl_environment = (tmp_path / "curl-env").read_bytes()
        assert b"BOTIFIED_ASR_BASE_URL=" not in curl_environment
        assert b"BOTIFIED_ASR_API_KEY=" not in curl_environment


@pytest.mark.parametrize(
    ("body", "curl_stderr", "curl_exit", "http_code", "expected_stdout"),
    [
        (b'{"text":"hello"}', b"", 0, "200", b'{"text":"hello"}'),
        (
            b'{"error":{"code":"invalid_audio"}}',
            b"curl http 400 detail",
            22,
            "400",
            b'{"error":{"code":"invalid_audio"}}',
        ),
        (
            b"partial sensitive transport body",
            f"timed out using {TOKEN}".encode(),
            28,
            "000",
            (
                b'{"error":{"message":"Botified ASR request failed",'
                b'"type":"client_error","param":null,"code":"curl_failed"}}\n'
            ),
        ),
    ],
)
def test_transcribe_curl_boundary_uses_fd3_and_shared_errors(
    tmp_path: Path,
    body: bytes,
    curl_stderr: bytes,
    curl_exit: int,
    http_code: str,
    expected_stdout: bytes,
) -> None:
    environment, args_path, stdin_path = _install_fake_curl(
        tmp_path,
        body=body,
        stderr=curl_stderr,
        exit_code=curl_exit,
        http_code=http_code,
    )
    target = tmp_path / "real-audio.wav"
    audio_bytes = b"dangerous local audio bytes"
    target.write_bytes(audio_bytes)
    audio_path = tmp_path / 'audio ,;"{}[] $().wav'
    audio_path.symlink_to(target)

    result = _run(environment, "transcribe", str(audio_path))

    assert result.returncode == curl_exit
    assert result.stdout == expected_stdout
    assert result.stderr == b""
    arguments = _without_http_capture_suffix(
        args_path.read_bytes().split(b"\0")
    )
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
        b"--request",
        b"POST",
        b"--form-string",
        b"model=sensevoice",
        b"--form-string",
        b"response_format=json",
        b"--form",
        (
            b"file=@/dev/fd/3;filename=audio;"
            b"type=application/octet-stream"
        ),
        b"--url",
        b"https://asr.example:17770/v1/audio/transcriptions",
        b"",
    ]
    assert b"--max-time" not in arguments
    assert b"--location" not in arguments
    assert b"-L" not in arguments
    assert str(audio_path).encode() not in args_path.read_bytes()
    assert TOKEN.encode() not in args_path.read_bytes()
    assert TOKEN.encode() not in result.stdout + result.stderr
    assert stdin_path.read_bytes() == (
        f'header = "Authorization: Bearer {TOKEN}"\n'.encode()
    )
    assert (tmp_path / "curl-upload").read_bytes() == audio_bytes
    curl_environment = (tmp_path / "curl-env").read_bytes()
    assert b"BOTIFIED_ASR_BASE_URL=" not in curl_environment
    assert b"BOTIFIED_ASR_API_KEY=" not in curl_environment


@pytest.mark.parametrize(
    ("body", "curl_stderr", "curl_exit", "http_code", "expected_stdout"),
    [
        (
            b'{"status":"ready"}',
            b"",
            0,
            "200",
            b'{"status":"ready"}',
        ),
        (
            b'{"error":{"code":"invalid_api_key"}}',
            b"curl http 401 detail",
            22,
            "401",
            b'{"error":{"code":"invalid_api_key"}}',
        ),
        (
            b'{"error":{"code":"service_not_ready"}}',
            b"curl http 503 detail",
            22,
            "503",
            b'{"error":{"code":"service_not_ready"}}',
        ),
        (
            b"partial sensitive transport body",
            f"could not connect using {TOKEN}".encode(),
            7,
            "000",
            (
                b'{"error":{"message":"Botified ASR request failed",'
                b'"type":"client_error","param":null,"code":"curl_failed"}}\n'
            ),
        ),
        (
            b"timeout detail",
            f"timed out using {TOKEN}".encode(),
            28,
            "000",
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
    http_code: str,
    expected_stdout: bytes,
) -> None:
    environment, args_path, stdin_path = _install_fake_curl(
        tmp_path,
        body=body,
        stderr=curl_stderr,
        exit_code=curl_exit,
        http_code=http_code,
    )

    result = _run(environment, "health")

    assert result.returncode == curl_exit
    assert result.stdout == expected_stdout
    assert result.stderr == b""
    arguments = _without_http_capture_suffix(
        args_path.read_bytes().split(b"\0")
    )
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
