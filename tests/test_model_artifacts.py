from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import pytest

from botified_asr import model_artifacts


SENSEVOICE_REVISION = "3847d57b6bdf2dd8875cb1508d2af43d80a16bf7"
SENSEVOICE_SHA256 = "833ca2dcfdf8ec91bd4f31cfac36d6124e0c459074d5e909aec9cabe6204a3ea"
SENSEVOICE_CONFIGURATION_SHA256 = (
    "02810a7f8e9e8aee10370a265f7e799728ce25b4c00cdbf4602b303ee395a38e"
)
SENSEVOICE_CONFIG_SHA256 = (
    "f71e239ba36705564b5bf2d2ffd07eece07b8e3f2bbf6d2c99d8df856339ac19"
)
SENSEVOICE_AM_MVN_SHA256 = (
    "29b3c740a2c0cfc6b308126d31d7f265fa2be74f3bb095cd2f143ea970896ae5"
)
SENSEVOICE_BPE_SHA256 = (
    "aa87f86064c3730d799ddf7af3c04659151102cba548bce325cf06ba4da4e6a8"
)
FSMN_VAD_REVISION = "df20e6b30c653645fa4ff125cacfcabd1020a669"
FSMN_VAD_SHA256 = "b3be75be477f0780277f3bae0fe489f48718f585f3a6e45d7dd1fbb1a4255fc5"
FSMN_VAD_CONFIGURATION_SHA256 = (
    "7bce8867e37d55c3dd8f672695ced18077a2be199ea529a5d432d5350fc0acba"
)
FSMN_VAD_CONFIG_SHA256 = (
    "486861ca26ddb79081663b6179cb204c6bfae71c52f04aafc48a9e9d8dde1e93"
)
FSMN_VAD_AM_MVN_SHA256 = (
    "df189fd5f4352df84a0fd464eeab4e450a5e645665d6b38f13c832492261a739"
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _spec(
    *,
    provider: str = "huggingface",
    model_id: str = "example/model",
    revision: str = "0123456789abcdef0123456789abcdef01234567",
    files: tuple[model_artifacts.ModelArtifactFile, ...] | None = None,
) -> model_artifacts.ModelArtifactSpec:
    return model_artifacts.ModelArtifactSpec(
        provider=provider,
        model_id=model_id,
        revision=revision,
        files=files
        or (
            model_artifacts.ModelArtifactFile(
                relative_path="model.bin",
                sha256=_sha256(b"model"),
            ),
        ),
    )


def _target(cache_root: Path, spec: model_artifacts.ModelArtifactSpec) -> Path:
    return cache_root.joinpath(*spec.model_id.split("/"), spec.revision)


def _write_files(
    root: Path,
    payloads: dict[str, bytes],
) -> None:
    for relative_path, payload in payloads.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


class FakeSnapshotFetcher:
    def __init__(
        self,
        action: Callable[
            [model_artifacts.ModelArtifactSpec, Path],
            None,
        ],
    ) -> None:
        self._action = action
        self.calls: list[tuple[model_artifacts.ModelArtifactSpec, Path]] = []

    def fetch(
        self,
        spec: model_artifacts.ModelArtifactSpec,
        destination: Path,
    ) -> None:
        self.calls.append((spec, destination))
        self._action(spec, destination)


def test_pinned_model_manifests_are_exact_and_immutable() -> None:
    assert model_artifacts.SENSEVOICE_SPEC == model_artifacts.ModelArtifactSpec(
        provider="huggingface",
        model_id="FunAudioLLM/SenseVoiceSmall",
        revision=SENSEVOICE_REVISION,
        files=(
            model_artifacts.ModelArtifactFile(
                relative_path="model.pt",
                sha256=SENSEVOICE_SHA256,
            ),
            model_artifacts.ModelArtifactFile(
                relative_path="configuration.json",
                sha256=SENSEVOICE_CONFIGURATION_SHA256,
            ),
            model_artifacts.ModelArtifactFile(
                relative_path="config.yaml",
                sha256=SENSEVOICE_CONFIG_SHA256,
            ),
            model_artifacts.ModelArtifactFile(
                relative_path="am.mvn",
                sha256=SENSEVOICE_AM_MVN_SHA256,
            ),
            model_artifacts.ModelArtifactFile(
                relative_path="chn_jpn_yue_eng_ko_spectok.bpe.model",
                sha256=SENSEVOICE_BPE_SHA256,
            ),
        ),
    )
    assert model_artifacts.FSMN_VAD_SPEC == model_artifacts.ModelArtifactSpec(
        provider="huggingface",
        model_id="funasr/fsmn-vad",
        revision=FSMN_VAD_REVISION,
        files=(
            model_artifacts.ModelArtifactFile(
                relative_path="model.pt",
                sha256=FSMN_VAD_SHA256,
            ),
            model_artifacts.ModelArtifactFile(
                relative_path="configuration.json",
                sha256=FSMN_VAD_CONFIGURATION_SHA256,
            ),
            model_artifacts.ModelArtifactFile(
                relative_path="config.yaml",
                sha256=FSMN_VAD_CONFIG_SHA256,
            ),
            model_artifacts.ModelArtifactFile(
                relative_path="am.mvn",
                sha256=FSMN_VAD_AM_MVN_SHA256,
            ),
        ),
    )
    fetcher_protocol = model_artifacts.SnapshotFetcher
    fake_fetcher = FakeSnapshotFetcher(lambda _spec, _destination: None)
    assert fetcher_protocol is not None
    assert callable(fake_fetcher.fetch)


@pytest.mark.parametrize(
    "factory",
    (
        lambda: _spec(revision="master"),
        lambda: _spec(revision="3847d57"),
        lambda: _spec(provider="modelscope"),
        lambda: _spec(provider=""),
        lambda: _spec(
            files=(
                model_artifacts.ModelArtifactFile(
                    relative_path="model.bin",
                    sha256="z" * 64,
                ),
            )
        ),
        lambda: _spec(
            files=(
                model_artifacts.ModelArtifactFile(
                    relative_path="/model.bin",
                    sha256=_sha256(b"model"),
                ),
            )
        ),
        lambda: _spec(
            files=(
                model_artifacts.ModelArtifactFile(
                    relative_path="../model.bin",
                    sha256=_sha256(b"model"),
                ),
            )
        ),
        lambda: _spec(
            files=(
                model_artifacts.ModelArtifactFile(
                    relative_path="config",
                    sha256=_sha256(b"config"),
                ),
                model_artifacts.ModelArtifactFile(
                    relative_path="config/x.json",
                    sha256=_sha256(b"nested"),
                ),
            )
        ),
    ),
    ids=(
        "revision_alias",
        "short_revision",
        "modelscope_provider",
        "empty_provider",
        "invalid_sha256",
        "absolute_artifact",
        "parent_traversal",
        "artifact_path_prefix_conflict",
    ),
)
def test_model_artifact_spec_rejects_mutable_or_unsafe_identity(
    factory: Callable[[], model_artifacts.ModelArtifactSpec],
) -> None:
    with pytest.raises(model_artifacts.ModelManifestError) as caught:
        factory()

    assert type(caught.value) is model_artifacts.ModelManifestError


def test_revision_is_part_of_the_isolated_snapshot_path(tmp_path: Path) -> None:
    first_payload = b"first revision"
    second_payload = b"second revision"
    first = _spec(
        revision="1111111111111111111111111111111111111111",
        files=(
            model_artifacts.ModelArtifactFile(
                "model.bin",
                _sha256(first_payload),
            ),
        ),
    )
    second = _spec(
        revision="2222222222222222222222222222222222222222",
        files=(
            model_artifacts.ModelArtifactFile(
                "model.bin",
                _sha256(second_payload),
            ),
        ),
    )

    def fetch(
        spec: model_artifacts.ModelArtifactSpec,
        destination: Path,
    ) -> None:
        payload = first_payload if spec == first else second_payload
        _write_files(destination, {"model.bin": payload})

    resolver = model_artifacts.ModelArtifactResolver(
        tmp_path,
        FakeSnapshotFetcher(fetch),
    )

    first_snapshot = resolver.resolve(first)
    second_snapshot = resolver.resolve(second)

    assert first_snapshot.root == _target(tmp_path, first)
    assert second_snapshot.root == _target(tmp_path, second)
    assert first_snapshot.root != second_snapshot.root
    assert first.revision in first_snapshot.root.parts
    assert second.revision in second_snapshot.root.parts
    assert (first_snapshot.root / "model.bin").read_bytes() == first_payload
    assert (second_snapshot.root / "model.bin").read_bytes() == second_payload


def test_invalid_existing_cache_is_integrity_error_without_fetch_or_repair(
    tmp_path: Path,
) -> None:
    payloads = {
        "model.bin": b"model",
        "config/config.json": b'{"fixed":true}',
    }
    spec = _spec(
        files=tuple(
            model_artifacts.ModelArtifactFile(path, _sha256(payload))
            for path, payload in payloads.items()
        )
    )
    target = _target(tmp_path, spec)
    _write_files(target, payloads)
    fetcher = FakeSnapshotFetcher(
        lambda _spec, destination: _write_files(destination, payloads)
    )
    resolver = model_artifacts.ModelArtifactResolver(tmp_path, fetcher)

    snapshot = resolver.resolve(spec)

    assert isinstance(snapshot, model_artifacts.ResolvedModelSnapshot)
    assert snapshot.spec == spec
    assert snapshot.root == target
    assert fetcher.calls == []

    external = tmp_path / "external-config.json"
    external.write_bytes(payloads["config/config.json"])
    config_path = target / "config/config.json"
    config_path.unlink()
    config_path.symlink_to(external)

    with pytest.raises(model_artifacts.ModelArtifactIntegrityError) as symlink:
        resolver.resolve(spec)

    assert type(symlink.value) is model_artifacts.ModelArtifactIntegrityError
    assert fetcher.calls == []
    assert config_path.is_symlink()

    config_path.unlink()
    with pytest.raises(model_artifacts.ModelArtifactIntegrityError) as missing:
        resolver.resolve(spec)

    assert type(missing.value) is model_artifacts.ModelArtifactIntegrityError
    assert fetcher.calls == []
    assert not config_path.exists()

    config_path.write_bytes(b"wrong")
    with pytest.raises(model_artifacts.ModelArtifactIntegrityError) as mismatch:
        resolver.resolve(spec)

    assert type(mismatch.value) is model_artifacts.ModelArtifactIntegrityError
    assert fetcher.calls == []
    assert config_path.read_bytes() == b"wrong"


def test_miss_fetches_to_same_parent_partial_then_atomically_publishes(
    tmp_path: Path,
) -> None:
    payloads = {"model.bin": b"model"}
    spec = _spec()
    target = _target(tmp_path, spec)

    def fetch(
        _spec: model_artifacts.ModelArtifactSpec,
        destination: Path,
    ) -> None:
        assert destination != target
        assert destination.parent == target.parent
        assert not target.exists()
        _write_files(destination, payloads)

    fetcher = FakeSnapshotFetcher(fetch)
    resolver = model_artifacts.ModelArtifactResolver(tmp_path, fetcher)

    snapshot = resolver.resolve(spec)

    assert snapshot == model_artifacts.ResolvedModelSnapshot(spec=spec, root=target)
    assert target.is_dir()
    assert (target / "model.bin").read_bytes() == payloads["model.bin"]
    assert len(fetcher.calls) == 1
    partial = fetcher.calls[0][1]
    assert partial != target
    assert not partial.exists()


@pytest.mark.parametrize(
    ("mode", "expected_error"),
    (
        ("missing", model_artifacts.ModelArtifactUnavailable),
        ("hash_mismatch", model_artifacts.ModelArtifactIntegrityError),
        ("removed_root", model_artifacts.ModelArtifactUnavailable),
    ),
)
def test_invalid_fetch_cleans_partial_and_never_publishes(
    tmp_path: Path,
    mode: str,
    expected_error: type[Exception],
) -> None:
    spec = _spec()
    target = _target(tmp_path, spec)

    def fetch(
        _spec: model_artifacts.ModelArtifactSpec,
        destination: Path,
    ) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        if mode == "hash_mismatch":
            (destination / "model.bin").write_bytes(b"wrong")
        elif mode == "removed_root":
            destination.rmdir()

    fetcher = FakeSnapshotFetcher(fetch)
    resolver = model_artifacts.ModelArtifactResolver(tmp_path, fetcher)

    with pytest.raises(expected_error) as caught:
        resolver.resolve(spec)

    assert type(caught.value) is expected_error
    assert not target.exists()
    assert len(fetcher.calls) == 1
    assert not fetcher.calls[0][1].exists()


def test_fetch_exception_maps_to_unavailable_and_cleans_unique_partial(
    tmp_path: Path,
) -> None:
    spec = _spec()

    def fail(
        _spec: model_artifacts.ModelArtifactSpec,
        _destination: Path,
    ) -> None:
        raise OSError("upstream unavailable")

    first_fetcher = FakeSnapshotFetcher(fail)
    second_fetcher = FakeSnapshotFetcher(fail)

    for fetcher in (first_fetcher, second_fetcher):
        resolver = model_artifacts.ModelArtifactResolver(tmp_path, fetcher)
        with pytest.raises(model_artifacts.ModelArtifactUnavailable) as caught:
            resolver.resolve(spec)
        assert isinstance(caught.value.__cause__, OSError)
        assert not fetcher.calls[0][1].exists()

    assert first_fetcher.calls[0][1] != second_fetcher.calls[0][1]
    assert not _target(tmp_path, spec).exists()


def test_publish_race_keeps_and_revalidates_existing_valid_target(
    tmp_path: Path,
) -> None:
    payloads = {"model.bin": b"model"}
    spec = _spec()
    target = _target(tmp_path, spec)

    def race(
        _spec: model_artifacts.ModelArtifactSpec,
        destination: Path,
    ) -> None:
        _write_files(destination, {**payloads, "loser": b"discard me"})
        _write_files(target, {**payloads, "winner": b"keep me"})

    fetcher = FakeSnapshotFetcher(race)
    resolver = model_artifacts.ModelArtifactResolver(tmp_path, fetcher)

    snapshot = resolver.resolve(spec)

    assert snapshot.root == target
    assert (target / "winner").read_bytes() == b"keep me"
    assert not (target / "loser").exists()
    assert not fetcher.calls[0][1].exists()
