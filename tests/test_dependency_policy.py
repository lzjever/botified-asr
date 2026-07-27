from __future__ import annotations

import hashlib
import importlib.metadata
import re
import shutil
import subprocess
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_VERSION = "3.11.13"
PYTHON_REQUIREMENT = f"=={PYTHON_VERSION}"
UV_VERSION = "0.9.26"
FUNASR_VERSION = "1.3.29"
FUNASR_WHEEL = "funasr-1.3.29-py3-none-any.whl"
FUNASR_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/9c/10/"
    "0a43f6233db074e263c025718afff7e7960976ef5e545c40c92c5f59f1c9/"
    "funasr-1.3.29-py3-none-any.whl"
)
FUNASR_WHEEL_SHA256 = "bc022d3f80cab635227841a401cc872e5b863a207f8fa01262f15c42ed630137"
FUNASR_WHEEL_BYTES = 956_044
FUNASR_METADATA_SHA256 = (
    "482f36e237f46b5fdf8dd94f85f5abd4bef1560d3d05cf5f808dbf9b67335e37"
)
PYPI_INDEX = "https://pypi.org/simple"
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
PYTORCH_VERSION = "2.11.0+cpu"
PROJECT_ENVIRONMENTS = [
    "sys_platform == 'linux' and platform_machine == 'x86_64'",
    "sys_platform == 'linux' and platform_machine == 'aarch64'",
]
LOCK_ENVIRONMENTS = [
    "platform_machine == 'x86_64' and sys_platform == 'linux'",
    "platform_machine == 'aarch64' and sys_platform == 'linux'",
]
CRITICAL_FUNASR_SHA256 = {
    "funasr/auto/auto_model.py": (
        "2e4efb19f1444daed54d254f0a5384f3a6ed24b436739206d5349e3d9fe98526"
    ),
    "funasr/download/download_model_from_hub.py": (
        "d88fd45c1ed90a3ec2a46d9630a57e81941472d8b791df174de22be573f1b340"
    ),
    "funasr/models/campplus/model.py": (
        "7e80450f1221202d516dc2e05d4e965b15f0f80297249765eee6b81b9e551c91"
    ),
    "funasr/models/fsmn_vad_streaming/model.py": (
        "48c531c6dcab53475983554cb580c7ff26ba616c6c4ca5da32f825038f8ba661"
    ),
    "funasr/models/sense_voice/model.py": (
        "3d5fd2e7f734c280b85856184eebb06b7aece9fed29d677fcfc59987fe12baa8"
    ),
    "funasr/register.py": (
        "996e9043491bbc8b7fd2047b01f8fd2274c48cfa999f79c5c5273daa345606d8"
    ),
    "funasr/utils/load_utils.py": (
        "293a8e9b3173f4c73b705878c18d05f4b1d61336729f73e2e9a0e9d05c664c01"
    ),
    "funasr/utils/postprocess_utils.py": (
        "bce917230a25d0602843220be80ee1223d1b9792a984d6d192ba2c4d8a7167a2"
    ),
}


def _load_toml(filename: str) -> dict[str, object]:
    path = PROJECT_ROOT / filename
    assert path.is_file(), f"{filename} must exist"
    try:
        with path.open("rb") as source:
            return tomllib.load(source)
    except tomllib.TOMLDecodeError as error:
        pytest.fail(f"{filename} must contain valid TOML: {error}")


def _normalized_requirement_name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
    assert match is not None, f"invalid requirement: {requirement!r}"
    return re.sub(r"[-_.]+", "-", match.group()).lower()


def _requirements(project: dict[str, object]) -> dict[str, list[str]]:
    project_metadata = project["project"]
    assert isinstance(project_metadata, dict)
    declared = project_metadata["dependencies"]
    assert isinstance(declared, list)
    by_name: dict[str, list[str]] = {}
    for requirement in declared:
        assert isinstance(requirement, str)
        by_name.setdefault(_normalized_requirement_name(requirement), []).append(
            requirement
        )
    return by_name


def _one_locked_package(
    lock: dict[str, object],
    name: str,
) -> dict[str, object]:
    packages = lock.get("package")
    assert isinstance(packages, list), "uv.lock must contain a package array"
    matches = [
        package
        for package in packages
        if isinstance(package, dict) and package.get("name") == name
    ]
    assert len(matches) == 1, f"uv.lock must contain exactly one {name} package"
    return matches[0]


def _wheel_filename(wheel: dict[str, object]) -> str:
    url = wheel.get("url")
    assert isinstance(url, str)
    return Path(unquote(urlsplit(url).path)).name


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_project_declares_exact_runtime_and_dependency_policy() -> None:
    project = _load_toml("pyproject.toml")
    lock = _load_toml("uv.lock")
    project_metadata = project["project"]
    assert isinstance(project_metadata, dict)

    assert project_metadata.get("requires-python") == PYTHON_REQUIREMENT
    assert lock.get("requires-python") == PYTHON_REQUIREMENT
    assert (PROJECT_ROOT / ".python-version").read_text().strip() == PYTHON_VERSION

    requirements = _requirements(project)
    assert requirements.get("funasr") == [f"funasr=={FUNASR_VERSION}"]
    assert requirements.get("torch") == [f"torch=={PYTORCH_VERSION}"]
    assert requirements.get("torchaudio") == [f"torchaudio=={PYTORCH_VERSION}"]

    uv_policy = project.get("tool")
    assert isinstance(uv_policy, dict)
    uv_policy = uv_policy.get("uv")
    assert isinstance(uv_policy, dict)
    assert uv_policy.get("required-version") == f"=={UV_VERSION}"
    assert uv_policy.get("environments") == PROJECT_ENVIRONMENTS
    assert uv_policy.get("required-environments") == PROJECT_ENVIRONMENTS
    assert lock.get("resolution-markers") == LOCK_ENVIRONMENTS
    assert lock.get("supported-markers") == LOCK_ENVIRONMENTS
    assert lock.get("required-markers") == LOCK_ENVIRONMENTS

    sources = uv_policy.get("sources", {})
    assert isinstance(sources, dict)
    assert "funasr" not in sources, "FunASR must be resolved from PyPI, not Git"
    no_build = uv_policy.get("no-build-package")
    assert isinstance(no_build, list)
    assert "funasr" in no_build

    assert sources.get("torch") == {"index": "pytorch-cpu"}
    assert sources.get("torchaudio") == {"index": "pytorch-cpu"}
    cpu_index_consumers = {
        name
        for name, source in sources.items()
        if (isinstance(source, dict) and source.get("index") == "pytorch-cpu")
        or (
            isinstance(source, list)
            and any(
                isinstance(item, dict) and item.get("index") == "pytorch-cpu"
                for item in source
            )
        )
    }
    assert cpu_index_consumers == {"torch", "torchaudio"}
    indexes = uv_policy.get("index")
    assert isinstance(indexes, list)
    assert all(isinstance(index, dict) for index in indexes)
    indexes_by_name = {index.get("name"): index for index in indexes}
    assert len(indexes_by_name) == len(indexes)
    assert indexes_by_name == {
        "pypi": {
            "name": "pypi",
            "url": PYPI_INDEX,
            "default": True,
        },
        "pytorch-cpu": {
            "name": "pytorch-cpu",
            "url": PYTORCH_CPU_INDEX,
            "explicit": True,
        },
    }


def test_lock_uses_only_approved_package_sources() -> None:
    lock = _load_toml("uv.lock")
    packages = lock.get("package")
    assert isinstance(packages, list)
    approved_sources = {
        ("editable", "."),
        ("registry", PYPI_INDEX),
        ("registry", PYTORCH_CPU_INDEX),
    }
    package_sources: dict[str, tuple[str, str]] = {}
    for package in packages:
        assert isinstance(package, dict)
        name = package.get("name")
        source = package.get("source")
        assert isinstance(name, str)
        assert isinstance(source, dict)
        assert len(source) == 1
        source_item = next(iter(source.items()))
        assert source_item in approved_sources, (
            f"unapproved source for {name}: {source!r}"
        )
        package_sources[name] = source_item

    assert {
        name
        for name, source in package_sources.items()
        if source == ("registry", PYTORCH_CPU_INDEX)
    } == {"torch", "torchaudio"}
    assert {
        name for name, source in package_sources.items() if source == ("editable", ".")
    } == {"botified-asr"}


def test_uv_lock_is_fresh() -> None:
    uv = shutil.which("uv")
    assert uv is not None, "uv must be installed to validate uv.lock freshness"
    result = subprocess.run(
        [uv, "lock", "--check"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"uv.lock must be fresh for pyproject.toml:\n{result.stdout}{result.stderr}"
    )


def test_lock_pins_the_audited_funasr_wheel() -> None:
    lock = _load_toml("uv.lock")
    package = _one_locked_package(lock, "funasr")

    assert package.get("version") == FUNASR_VERSION
    assert package.get("source") == {"registry": PYPI_INDEX}
    wheels = package.get("wheels")
    assert isinstance(wheels, list)
    assert len(wheels) == 1
    matching = [
        wheel
        for wheel in wheels
        if isinstance(wheel, dict) and _wheel_filename(wheel) == FUNASR_WHEEL
    ]
    assert len(matching) == 1
    assert matching[0].get("url") == FUNASR_WHEEL_URL
    assert matching[0].get("hash") == f"sha256:{FUNASR_WHEEL_SHA256}"
    assert matching[0].get("size") == FUNASR_WHEEL_BYTES


def test_lock_pins_cpu_only_pytorch_wheels_for_supported_linux_arches() -> None:
    lock = _load_toml("uv.lock")
    for name in ("torch", "torchaudio"):
        package = _one_locked_package(lock, name)
        assert package.get("version") == PYTORCH_VERSION
        assert package.get("source") == {"registry": PYTORCH_CPU_INDEX}
        wheels = package.get("wheels")
        assert isinstance(wheels, list)
        filenames = {
            _wheel_filename(wheel) for wheel in wheels if isinstance(wheel, dict)
        }
        expected_prefix = f"{name}-2.11.0+cpu-cp311-cp311"
        assert f"{expected_prefix}-manylinux_2_28_x86_64.whl" in filenames
        assert f"{expected_prefix}-manylinux_2_28_aarch64.whl" in filenames

    packages = lock.get("package")
    assert isinstance(packages, list)
    package_names = {
        package["name"]
        for package in packages
        if isinstance(package, dict) and isinstance(package.get("name"), str)
    }
    assert not any(name.startswith("nvidia-") for name in package_names)
    assert not any("cuda" in name for name in package_names)


def test_installed_funasr_distribution_matches_audited_runtime_surface() -> None:
    try:
        distribution = importlib.metadata.distribution("funasr")
    except importlib.metadata.PackageNotFoundError:
        pytest.fail("the funasr distribution must be installed")

    assert distribution.metadata["Name"] == "funasr"
    assert distribution.version == FUNASR_VERSION
    files = tuple(distribution.files or ())
    metadata_files = [
        file
        for file in files
        if file.name == "METADATA" and file.parent.name.endswith(".dist-info")
    ]
    assert len(metadata_files) == 1
    metadata_path = Path(distribution.locate_file(metadata_files[0]))
    assert _sha256(metadata_path) == FUNASR_METADATA_SHA256

    files_by_path = {file.as_posix(): file for file in files}
    for relative_path, expected_sha256 in CRITICAL_FUNASR_SHA256.items():
        assert relative_path in files_by_path, (
            f"installed funasr is missing {relative_path}"
        )
        installed_path = Path(distribution.locate_file(files_by_path[relative_path]))
        assert _sha256(installed_path) == expected_sha256
