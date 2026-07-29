from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import tomllib
from pathlib import Path

from botified_asr.model_artifacts import (
    CAMPLUS_SPEC,
    FSMN_VAD_SPEC,
    SENSEVOICE_SPEC,
    ResolvedModelSnapshot,
)
from botified_asr.model_loader import _build_processor_fingerprint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = PROJECT_ROOT / "scripts" / "generate-release-manifest"
DIGESTS = {
    "index": f"sha256:{'a' * 64}",
    "amd64": f"sha256:{'b' * 64}",
    "arm64": f"sha256:{'c' * 64}",
}
ARTIFACT_CONTENTS = {
    "LICENSE": b"project license\n",
    "THIRD_PARTY_NOTICES": b"third-party notices\n",
    "botified-asr-openapi.json": b'{"openapi":"3.1.0"}\n',
    "botified-asr-skill.tar.gz": b"deterministic skill archive",
    "botified-asr-smoke.flac": b"fLaCdeterministic smoke audio",
}


def _release_version() -> str:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as source:
        return f"v{tomllib.load(source)['project']['version']}"


def _write_artifacts(directory: Path) -> dict[str, Path]:
    directory.mkdir()
    paths = {}
    for name, content in ARTIFACT_CONTENTS.items():
        path = directory / name
        path.write_bytes(content)
        paths[name] = path
    return paths


def _command(
    artifacts: dict[str, Path],
    output: Path,
    *,
    index_digest: str = DIGESTS["index"],
) -> list[str | Path]:
    return [
        sys.executable,
        GENERATOR,
        "--license",
        artifacts["LICENSE"],
        "--notices",
        artifacts["THIRD_PARTY_NOTICES"],
        "--openapi",
        artifacts["botified-asr-openapi.json"],
        "--skill",
        artifacts["botified-asr-skill.tar.gz"],
        "--smoke",
        artifacts["botified-asr-smoke.flac"],
        "--index-digest",
        index_digest,
        "--amd64-digest",
        DIGESTS["amd64"],
        "--arm64-digest",
        DIGESTS["arm64"],
        "--release-version",
        _release_version(),
        "--output",
        output,
    ]


def _run(arguments: list[str | Path]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        arguments,
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
    )


def test_release_manifest_is_exact_deterministic_and_hashes_real_inputs(
    tmp_path: Path,
) -> None:
    artifacts = _write_artifacts(tmp_path / "artifacts")
    output = tmp_path / "botified-asr-release.json"
    first = subprocess.run(
        _command(artifacts, output),
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        preexec_fn=lambda: os.umask(0o077),
    )
    assert first.returncode == 0, first.stderr
    first_bytes = output.read_bytes()
    second = _run(_command(artifacts, output))
    assert second.returncode == 0, second.stderr

    manifest = json.loads(first_bytes)
    expected_fingerprint = _build_processor_fingerprint(
        ResolvedModelSnapshot(SENSEVOICE_SPEC, Path("/unused/sensevoice")),
        ResolvedModelSnapshot(FSMN_VAD_SPEC, Path("/unused/fsmn-vad")),
        ResolvedModelSnapshot(CAMPLUS_SPEC, Path("/unused/campplus")),
    )
    assert manifest == {
        "artifacts": {
            name: {"sha256": hashlib.sha256(content).hexdigest()}
            for name, content in ARTIFACT_CONTENTS.items()
        },
        "models": [
            {
                "id": SENSEVOICE_SPEC.model_id,
                "license": "FunASR-Model-License-1.1",
                "license_url": (
                    "https://github.com/modelscope/FunASR/blob/"
                    "8a34247dc5ff71bea61b37e57f941680b456753f/"
                    "MODEL_LICENSE"
                ),
                "provider": SENSEVOICE_SPEC.provider,
                "revision": SENSEVOICE_SPEC.revision,
                "source_url": (
                    "https://huggingface.co/"
                    f"{SENSEVOICE_SPEC.model_id}/tree/"
                    f"{SENSEVOICE_SPEC.revision}"
                ),
            },
            {
                "id": FSMN_VAD_SPEC.model_id,
                "license": "Apache-2.0",
                "license_url": (
                    "https://www.apache.org/licenses/LICENSE-2.0"
                ),
                "provider": FSMN_VAD_SPEC.provider,
                "revision": FSMN_VAD_SPEC.revision,
                "source_url": (
                    "https://huggingface.co/"
                    f"{FSMN_VAD_SPEC.model_id}/tree/"
                    f"{FSMN_VAD_SPEC.revision}"
                ),
            },
            {
                "id": CAMPLUS_SPEC.model_id,
                "license": "Apache-2.0",
                "license_url": (
                    "https://www.apache.org/licenses/LICENSE-2.0"
                ),
                "provider": CAMPLUS_SPEC.provider,
                "revision": CAMPLUS_SPEC.revision,
                "source_url": (
                    "https://huggingface.co/"
                    f"{CAMPLUS_SPEC.model_id}/tree/"
                    f"{CAMPLUS_SPEC.revision}"
                ),
            },
        ],
        "oci_image": {
            "index_digest": DIGESTS["index"],
            "platforms": {
                "linux/amd64": {
                    "device": "cpu",
                    "digest": DIGESTS["amd64"],
                },
                "linux/arm64": {
                    "device": "cpu",
                    "digest": DIGESTS["arm64"],
                },
            },
            "repository": "ghcr.io/lzjever/botified-asr",
        },
        "processor_fingerprint": expected_fingerprint,
        "release_version": _release_version(),
        "runtime": {
            "container_engines": ["docker", "podman"],
            "container_user": "10001:10001",
        },
        "schema_version": 1,
    }
    assert output.read_bytes() == first_bytes
    assert first_bytes == (
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    assert stat.S_IMODE(output.stat().st_mode) == 0o644
    assert not list(tmp_path.glob(".botified-asr-release.json.*.tmp"))


def test_release_manifest_rejects_malformed_digest_without_replacing_output(
    tmp_path: Path,
) -> None:
    artifacts = _write_artifacts(tmp_path / "artifacts")
    output = tmp_path / "botified-asr-release.json"
    output.write_bytes(b"keep\n")

    result = _run(
        _command(
            artifacts,
            output,
            index_digest=f"sha256:{'a' * 63}",
        )
    )

    assert result.returncode != 0
    assert output.read_bytes() == b"keep\n"

    duplicate = _run(
        _command(
            artifacts,
            output,
            index_digest=DIGESTS["amd64"],
        )
    )

    assert duplicate.returncode != 0
    assert output.read_bytes() == b"keep\n"


def test_release_manifest_rejects_unsafe_paths_and_wrong_input_name(
    tmp_path: Path,
) -> None:
    artifacts = _write_artifacts(tmp_path / "artifacts")
    preserved = tmp_path / "preserved"
    preserved.write_bytes(b"keep\n")
    linked_output = tmp_path / "botified-asr-release.json"
    linked_output.symlink_to(preserved)

    rejected_output = _run(_command(artifacts, linked_output))

    assert rejected_output.returncode != 0
    assert linked_output.is_symlink()
    assert preserved.read_bytes() == b"keep\n"

    wrong_name = tmp_path / "COPYING"
    wrong_name.write_bytes(ARTIFACT_CONTENTS["LICENSE"])
    wrong_artifacts = {**artifacts, "LICENSE": wrong_name}
    wrong_name_output_dir = tmp_path / "wrong-name-output"
    wrong_name_output_dir.mkdir()
    rejected_name = _run(
        _command(
            wrong_artifacts,
            wrong_name_output_dir / "botified-asr-release.json",
        )
    )
    assert rejected_name.returncode != 0

    linked_inputs = _write_artifacts(tmp_path / "linked-inputs")
    linked_inputs["LICENSE"].unlink()
    linked_inputs["LICENSE"].symlink_to(artifacts["LICENSE"])
    linked_input_output_dir = tmp_path / "linked-input-output"
    linked_input_output_dir.mkdir()
    rejected_input = _run(
        _command(
            linked_inputs,
            linked_input_output_dir / "botified-asr-release.json",
        )
    )
    assert rejected_input.returncode != 0
