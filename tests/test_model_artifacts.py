from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from botified_asr import model_artifacts


SENSEVOICE_REVISION = "3847d57b6bdf2dd8875cb1508d2af43d80a16bf7"
SENSEVOICE_SHA256 = "833ca2dcfdf8ec91bd4f31cfac36d6124e0c459074d5e909aec9cabe6204a3ea"
SENSEVOICE_BYTES = 936_291_369
SENSEVOICE_CONFIGURATION_SHA256 = (
    "02810a7f8e9e8aee10370a265f7e799728ce25b4c00cdbf4602b303ee395a38e"
)
SENSEVOICE_CONFIGURATION_BYTES = 396
SENSEVOICE_CONFIG_SHA256 = (
    "f71e239ba36705564b5bf2d2ffd07eece07b8e3f2bbf6d2c99d8df856339ac19"
)
SENSEVOICE_CONFIG_BYTES = 1_855
SENSEVOICE_AM_MVN_SHA256 = (
    "29b3c740a2c0cfc6b308126d31d7f265fa2be74f3bb095cd2f143ea970896ae5"
)
SENSEVOICE_AM_MVN_BYTES = 11_203
SENSEVOICE_BPE_SHA256 = (
    "aa87f86064c3730d799ddf7af3c04659151102cba548bce325cf06ba4da4e6a8"
)
SENSEVOICE_BPE_BYTES = 377_341
FSMN_VAD_REVISION = "df20e6b30c653645fa4ff125cacfcabd1020a669"
FSMN_VAD_SHA256 = "b3be75be477f0780277f3bae0fe489f48718f585f3a6e45d7dd1fbb1a4255fc5"
FSMN_VAD_BYTES = 1_721_366
FSMN_VAD_CONFIGURATION_SHA256 = (
    "7bce8867e37d55c3dd8f672695ced18077a2be199ea529a5d432d5350fc0acba"
)
FSMN_VAD_CONFIGURATION_BYTES = 365
FSMN_VAD_CONFIG_SHA256 = (
    "486861ca26ddb79081663b6179cb204c6bfae71c52f04aafc48a9e9d8dde1e93"
)
FSMN_VAD_CONFIG_BYTES = 1_215
FSMN_VAD_AM_MVN_SHA256 = (
    "df189fd5f4352df84a0fd464eeab4e450a5e645665d6b38f13c832492261a739"
)
FSMN_VAD_AM_MVN_BYTES = 8_033
CAMPLUS_REVISION = "e4b6ede7ce16997aff4ae69fbca1f0175e2afede"
CAMPLUS_SHA256 = "3388cf5fd3493c9ac9c69851d8e7a8badcfb4f3dc631020c4961371646d5ada8"
CAMPLUS_BYTES = 28_036_335
CAMPLUS_CONFIGURATION_SHA256 = (
    "6f7acaf1e81ca121f4a3c71b6ddb66beec24350a3ef330e2c846f17829176a8f"
)
CAMPLUS_CONFIGURATION_BYTES = 581
CAMPLUS_CONFIG_SHA256 = (
    "17342041bd5b22f6fd7e32f6e7a267b0bf65f018c0a721bada6547e3d28fbfc9"
)
CAMPLUS_CONFIG_BYTES = 537


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


_MODEL_MANIFEST_BYTES = (
    b'{"files":[{"byte_size":6,"relative_path":"config.json",'
    b'"sha256":"b79606fb3afea5bd1609ed40b622142f1c98125abcfe89a76a661b0e8e343910"},'
    b'{"byte_size":5,"relative_path":"weights/model.bin",'
    b'"sha256":"9372c470eeadd5ecd9c3c74c2b3cb633f8e2f2fad799250a0f70d652b6b825e4"}],'
    b'"immutable_revision":"0123456789abcdef0123456789abcdef01234567",'
    b'"model_id":"example/model","provider":"huggingface","version":1}'
)
_MODEL_MANIFEST_SHA256 = (
    "bed3f90fd2ad7a4e8f9222b837fff2e6413899cd2394093133bcb1611bdfcd2b"
)


def _manifest_files() -> tuple[model_artifacts.ModelArtifactFile, ...]:
    return (
        model_artifacts.ModelArtifactFile(
            relative_path="weights/model.bin",
            sha256=_sha256(b"model"),
            expected_bytes=len(b"model"),
        ),
        model_artifacts.ModelArtifactFile(
            relative_path="config.json",
            sha256=_sha256(b"config"),
            expected_bytes=len(b"config"),
        ),
    )


def _resolved_snapshot(
    *,
    files: tuple[model_artifacts.ModelArtifactFile, ...] | None = None,
    root: Path = Path("/missing/model/snapshot"),
) -> model_artifacts.ResolvedModelSnapshot:
    return model_artifacts.ResolvedModelSnapshot(
        spec=_spec(files=files or _manifest_files()),
        root=root,
    )


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
                expected_bytes=len(b"model"),
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
                expected_bytes=SENSEVOICE_BYTES,
            ),
            model_artifacts.ModelArtifactFile(
                relative_path="configuration.json",
                sha256=SENSEVOICE_CONFIGURATION_SHA256,
                expected_bytes=SENSEVOICE_CONFIGURATION_BYTES,
            ),
            model_artifacts.ModelArtifactFile(
                relative_path="config.yaml",
                sha256=SENSEVOICE_CONFIG_SHA256,
                expected_bytes=SENSEVOICE_CONFIG_BYTES,
            ),
            model_artifacts.ModelArtifactFile(
                relative_path="am.mvn",
                sha256=SENSEVOICE_AM_MVN_SHA256,
                expected_bytes=SENSEVOICE_AM_MVN_BYTES,
            ),
            model_artifacts.ModelArtifactFile(
                relative_path="chn_jpn_yue_eng_ko_spectok.bpe.model",
                sha256=SENSEVOICE_BPE_SHA256,
                expected_bytes=SENSEVOICE_BPE_BYTES,
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
                expected_bytes=FSMN_VAD_BYTES,
            ),
            model_artifacts.ModelArtifactFile(
                relative_path="configuration.json",
                sha256=FSMN_VAD_CONFIGURATION_SHA256,
                expected_bytes=FSMN_VAD_CONFIGURATION_BYTES,
            ),
            model_artifacts.ModelArtifactFile(
                relative_path="config.yaml",
                sha256=FSMN_VAD_CONFIG_SHA256,
                expected_bytes=FSMN_VAD_CONFIG_BYTES,
            ),
            model_artifacts.ModelArtifactFile(
                relative_path="am.mvn",
                sha256=FSMN_VAD_AM_MVN_SHA256,
                expected_bytes=FSMN_VAD_AM_MVN_BYTES,
            ),
        ),
    )
    assert model_artifacts.CAMPLUS_SPEC == model_artifacts.ModelArtifactSpec(
        provider="huggingface",
        model_id="funasr/campplus",
        revision=CAMPLUS_REVISION,
        files=(
            model_artifacts.ModelArtifactFile(
                relative_path="campplus_cn_common.bin",
                sha256=CAMPLUS_SHA256,
                expected_bytes=CAMPLUS_BYTES,
            ),
            model_artifacts.ModelArtifactFile(
                relative_path="configuration.json",
                sha256=CAMPLUS_CONFIGURATION_SHA256,
                expected_bytes=CAMPLUS_CONFIGURATION_BYTES,
            ),
            model_artifacts.ModelArtifactFile(
                relative_path="config.yaml",
                sha256=CAMPLUS_CONFIG_SHA256,
                expected_bytes=CAMPLUS_CONFIG_BYTES,
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
                    sha256=_sha256(b"model"),
                    expected_bytes=0,
                ),
            )
        ),
        lambda: _spec(
            files=(
                model_artifacts.ModelArtifactFile(
                    relative_path="model.bin",
                    sha256=_sha256(b"model"),
                    expected_bytes=-1,
                ),
            )
        ),
        lambda: _spec(
            files=(
                model_artifacts.ModelArtifactFile(
                    relative_path="model.bin",
                    sha256=_sha256(b"model"),
                    expected_bytes=True,
                ),
            )
        ),
        lambda: _spec(
            files=(
                model_artifacts.ModelArtifactFile(
                    relative_path="model.bin",
                    sha256="z" * 64,
                    expected_bytes=len(b"model"),
                ),
            )
        ),
        lambda: _spec(
            files=(
                model_artifacts.ModelArtifactFile(
                    relative_path="/model.bin",
                    sha256=_sha256(b"model"),
                    expected_bytes=len(b"model"),
                ),
            )
        ),
        lambda: _spec(
            files=(
                model_artifacts.ModelArtifactFile(
                    relative_path="../model.bin",
                    sha256=_sha256(b"model"),
                    expected_bytes=len(b"model"),
                ),
            )
        ),
        lambda: _spec(
            files=(
                model_artifacts.ModelArtifactFile(
                    relative_path="config",
                    sha256=_sha256(b"config"),
                    expected_bytes=len(b"config"),
                ),
                model_artifacts.ModelArtifactFile(
                    relative_path="config/x.json",
                    sha256=_sha256(b"nested"),
                    expected_bytes=len(b"nested"),
                ),
            )
        ),
    ),
    ids=(
        "revision_alias",
        "short_revision",
        "modelscope_provider",
        "empty_provider",
        "zero_expected_bytes",
        "negative_expected_bytes",
        "bool_expected_bytes",
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


def test_model_snapshot_manifest_digest_is_exact_sorted_and_file_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_root = tmp_path / "does-not-exist"
    declared = _manifest_files()
    snapshot = _resolved_snapshot(files=declared, root=missing_root)
    reversed_snapshot = _resolved_snapshot(
        files=tuple(reversed(declared)),
        root=tmp_path / "also-missing",
    )

    def fail_open(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("manifest digest must not open model files")

    monkeypatch.setattr(Path, "open", fail_open)

    assert not missing_root.exists()
    assert _sha256(_MODEL_MANIFEST_BYTES) == _MODEL_MANIFEST_SHA256
    assert (
        model_artifacts.model_snapshot_manifest_digest(snapshot)
        == _MODEL_MANIFEST_SHA256
    )
    assert (
        model_artifacts.model_snapshot_manifest_digest(reversed_snapshot)
        == _MODEL_MANIFEST_SHA256
    )


def _change_snapshot_manifest_field(
    snapshot: model_artifacts.ResolvedModelSnapshot,
    field: str,
) -> model_artifacts.ResolvedModelSnapshot:
    spec = snapshot.spec
    if field == "provider":
        spec = replace(spec)
        object.__setattr__(spec, "provider", "other-provider")
    elif field == "model_id":
        spec = replace(spec, model_id="example/other-model")
    elif field == "revision":
        spec = replace(spec, revision="1" * 40)
    elif field == "file_added":
        spec = replace(
            spec,
            files=(
                *spec.files,
                model_artifacts.ModelArtifactFile(
                    relative_path="tokenizer.json",
                    sha256=_sha256(b"tokenizer"),
                    expected_bytes=len(b"tokenizer"),
                ),
            ),
        )
    elif field == "file_removed":
        spec = replace(spec, files=spec.files[:-1])
    else:
        first, *remaining = spec.files
        file_changes = {
            "path": {"relative_path": "weights/other-model.bin"},
            "size": {"expected_bytes": first.expected_bytes + 1},
            "hash": {"sha256": _sha256(b"other-model")},
        }
        spec = replace(
            spec,
            files=(replace(first, **file_changes[field]), *remaining),
        )
    return replace(snapshot, spec=spec)


@pytest.mark.parametrize(
    "field",
    (
        "provider",
        "model_id",
        "revision",
        "path",
        "size",
        "hash",
        "file_added",
        "file_removed",
    ),
)
def test_each_model_snapshot_manifest_field_changes_the_digest(
    field: str,
) -> None:
    snapshot = _resolved_snapshot()
    changed = _change_snapshot_manifest_field(snapshot, field)

    assert model_artifacts.model_snapshot_manifest_digest(changed) != (
        model_artifacts.model_snapshot_manifest_digest(snapshot)
    )


def test_unicode_artifact_path_uses_canonical_utf8_metadata() -> None:
    snapshot = _resolved_snapshot(
        files=(
            model_artifacts.ModelArtifactFile(
                relative_path="配置.json",
                sha256=_sha256(b"config"),
                expected_bytes=len(b"config"),
            ),
        )
    )
    utf8_metadata = (
        '{"files":[{"byte_size":6,"relative_path":"配置.json",'
        '"sha256":"b79606fb3afea5bd1609ed40b622142f1c98125abcfe89a76a661b0e8e343910"}],'
        '"immutable_revision":"0123456789abcdef0123456789abcdef01234567",'
        '"model_id":"example/model","provider":"huggingface","version":1}'
    ).encode("utf-8")
    ascii_escaped_metadata = (
        b'{"files":[{"byte_size":6,"relative_path":"\\u914d\\u7f6e.json",'
        b'"sha256":"b79606fb3afea5bd1609ed40b622142f1c98125abcfe89a76a661b0e8e343910"}],'
        b'"immutable_revision":"0123456789abcdef0123456789abcdef01234567",'
        b'"model_id":"example/model","provider":"huggingface","version":1}'
    )

    digest = model_artifacts.model_snapshot_manifest_digest(snapshot)

    assert digest == _sha256(utf8_metadata)
    assert digest != _sha256(ascii_escaped_metadata)


@pytest.mark.parametrize("invalid", (None, object(), _spec()))
def test_model_snapshot_manifest_digest_rejects_invalid_types(
    invalid: object,
) -> None:
    with pytest.raises(TypeError):
        model_artifacts.model_snapshot_manifest_digest(invalid)  # type: ignore[arg-type]


def test_revision_is_part_of_the_isolated_snapshot_path(tmp_path: Path) -> None:
    first_payload = b"first revision"
    second_payload = b"second revision"
    first = _spec(
        revision="1111111111111111111111111111111111111111",
        files=(
            model_artifacts.ModelArtifactFile(
                relative_path="model.bin",
                sha256=_sha256(first_payload),
                expected_bytes=len(first_payload),
            ),
        ),
    )
    second = _spec(
        revision="2222222222222222222222222222222222222222",
        files=(
            model_artifacts.ModelArtifactFile(
                relative_path="model.bin",
                sha256=_sha256(second_payload),
                expected_bytes=len(second_payload),
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
            model_artifacts.ModelArtifactFile(
                relative_path=path,
                sha256=_sha256(payload),
                expected_bytes=len(payload),
            )
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

    config_path.write_bytes(b"x")
    with pytest.raises(model_artifacts.ModelArtifactIntegrityError) as wrong_size:
        resolver.resolve(spec)

    assert type(wrong_size.value) is model_artifacts.ModelArtifactIntegrityError
    assert fetcher.calls == []
    assert config_path.read_bytes() == b"x"

    wrong_hash = b"x" * len(payloads["config/config.json"])
    config_path.write_bytes(wrong_hash)
    with pytest.raises(model_artifacts.ModelArtifactIntegrityError) as mismatch:
        resolver.resolve(spec)

    assert type(mismatch.value) is model_artifacts.ModelArtifactIntegrityError
    assert fetcher.calls == []
    assert config_path.read_bytes() == wrong_hash


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
        ("size_short", model_artifacts.ModelArtifactUnavailable),
        ("size_long", model_artifacts.ModelArtifactUnavailable),
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
        elif mode == "size_short":
            (destination / "model.bin").write_bytes(b"x")
        elif mode == "size_long":
            (destination / "model.bin").write_bytes(b"model!")
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
