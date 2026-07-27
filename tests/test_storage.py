from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path

import pytest

from botified_asr.config import LimitsConfig
from botified_asr.storage import (
    ArtifactRef,
    InputRef,
    RESERVATION_QUANTUM,
    ReservedByteWriter,
    Storage,
    StorageAdmissionError,
    StorageSchemaError,
    UploadLease,
)


def limits(**overrides: int) -> LimitsConfig:
    values = {
        "max_upload_bytes": RESERVATION_QUANTUM,
        "sync_max_upload_bytes": RESERVATION_QUANTUM,
        "max_active_uploads": 2,
        "max_job_storage_bytes": RESERVATION_QUANTUM,
        "min_filesystem_free_bytes": 1,
    }
    values.update(overrides)
    return LimitsConfig(**values)


def create_v1_database(
    data_dir: Path,
    *,
    lease_id: str = "a" * 32,
    suffix: str = "ready",
    reserved_bytes: int = RESERVATION_QUANTUM,
    actual_bytes: int = 4,
) -> Path:
    staging = data_dir / "staging"
    staging.mkdir(parents=True)
    controlled_path = staging / f"{lease_id}.{suffix}"
    controlled_path.write_bytes(b"data")
    connection = sqlite3.connect(data_dir / "botified-asr.sqlite3")
    connection.executescript(
        """
        CREATE TABLE schema_meta (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            version INTEGER NOT NULL
        );
        INSERT INTO schema_meta(singleton, version) VALUES (1, 1);
        CREATE TABLE upload_leases (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            phase TEXT NOT NULL CHECK (phase = 'receiving'),
            staging_path TEXT NOT NULL UNIQUE,
            reserved_bytes INTEGER NOT NULL CHECK (reserved_bytes >= 0),
            actual_bytes INTEGER NOT NULL CHECK (actual_bytes >= 0),
            content_sha256 TEXT,
            created_at TEXT NOT NULL
        );
        """
    )
    connection.execute(
        """
        INSERT INTO upload_leases(
            id, kind, phase, staging_path, reserved_bytes, actual_bytes,
            content_sha256, created_at
        ) VALUES (?, 'transcription', 'receiving', ?, ?, ?, NULL, ?)
        """,
        (
            lease_id,
            str(controlled_path),
            reserved_bytes,
            actual_bytes,
            "2026-07-26T00:00:00.000Z",
        ),
    )
    connection.commit()
    connection.close()
    return controlled_path


def test_fresh_schema_is_explicit_v2_generic_ledger(tmp_path: Path) -> None:
    storage = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    try:
        version = storage._connection.execute(
            "SELECT version FROM schema_meta WHERE singleton = 1"
        ).fetchone()[0]
        columns = {
            row["name"]
            for row in storage._connection.execute(
                "PRAGMA table_info(storage_leases)"
            )
        }
        indexes = {
            row["name"]
            for row in storage._connection.execute(
                "PRAGMA index_list(storage_leases)"
            )
        }

        assert version == 2
        assert columns == {
            "id",
            "lease_type",
            "resource_kind",
            "owner_kind",
            "owner_id",
            "phase",
            "controlled_path",
            "reserved_bytes",
            "actual_bytes",
            "content_sha256",
            "created_at",
        }
        assert {
            "storage_leases_type_phase_idx",
            "storage_leases_owner_idx",
        } <= indexes
    finally:
        storage.close()

    connection = sqlite3.connect(tmp_path / "botified-asr.sqlite3")
    connection.execute("CREATE TABLE upload_leases (id TEXT PRIMARY KEY)")
    connection.close()
    with pytest.raises(
        StorageSchemaError, match="unexpected legacy upload ledger"
    ):
        Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)


def test_v1_schema_migrates_transactionally_and_reconciles_legacy(
    tmp_path: Path,
) -> None:
    legacy_path = create_v1_database(tmp_path)

    storage = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    try:
        assert storage._connection.execute(
            "SELECT version FROM schema_meta WHERE singleton = 1"
        ).fetchone()[0] == 2
        assert storage._connection.execute(
            "SELECT COUNT(*) FROM storage_leases"
        ).fetchone()[0] == 0
        assert storage._connection.execute(
            """
            SELECT COUNT(*) FROM sqlite_master
            WHERE type = 'table' AND name = 'upload_leases'
            """
        ).fetchone()[0] == 0
        assert not legacy_path.exists()
    finally:
        storage.close()


def test_unknown_schema_version_fails_closed(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / "botified-asr.sqlite3")
    connection.executescript(
        """
        CREATE TABLE schema_meta (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            version INTEGER NOT NULL
        );
        INSERT INTO schema_meta(singleton, version) VALUES (1, 99);
        """
    )
    connection.close()

    with pytest.raises(StorageSchemaError, match="unsupported storage schema"):
        Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)

    unversioned = tmp_path / "unversioned"
    unversioned.mkdir()
    sentinel = unversioned / "keep.txt"
    sentinel.write_bytes(b"keep")
    connection = sqlite3.connect(unversioned / "botified-asr.sqlite3")
    connection.execute("CREATE TABLE application_data (value TEXT)")
    connection.close()

    with pytest.raises(StorageSchemaError, match="unversioned"):
        Storage(unversioned, limits(), free_bytes=lambda _: 1 << 40)
    assert sentinel.read_bytes() == b"keep"


def test_failed_v1_migration_rolls_back_structure_and_version(
    tmp_path: Path,
) -> None:
    legacy_path = create_v1_database(tmp_path)
    connection = sqlite3.connect(tmp_path / "botified-asr.sqlite3")
    connection.execute(
        "UPDATE upload_leases SET staging_path = ?",
        (str(tmp_path.parent / legacy_path.name),),
    )
    connection.commit()
    connection.close()

    with pytest.raises(StorageSchemaError, match="invalid v1 upload lease"):
        Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)

    connection = sqlite3.connect(tmp_path / "botified-asr.sqlite3")
    try:
        assert connection.execute(
            "SELECT version FROM schema_meta WHERE singleton = 1"
        ).fetchone()[0] == 1
        assert connection.execute(
            """
            SELECT COUNT(*) FROM sqlite_master
            WHERE type = 'table' AND name = 'upload_leases'
            """
        ).fetchone()[0] == 1
        assert connection.execute(
            """
            SELECT COUNT(*) FROM sqlite_master
            WHERE type = 'table' AND name = 'storage_leases'
            """
        ).fetchone()[0] == 0
        assert legacy_path.read_bytes() == b"data"
    finally:
        connection.close()


def test_storage_rejects_preexisting_control_directory_symlink(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    outside = tmp_path / "outside"
    data_dir.mkdir()
    outside.mkdir()
    (outside / "keep.txt").write_bytes(b"keep")
    (data_dir / "staging").symlink_to(outside, target_is_directory=True)

    with pytest.raises(StorageSchemaError, match="staging"):
        Storage(data_dir, limits(), free_bytes=lambda _: 1 << 40)

    assert (outside / "keep.txt").read_bytes() == b"keep"
    assert not (data_dir / "botified-asr.sqlite3").exists()


def test_controlled_file_symlink_fails_closed_on_startup_and_resolve(
    tmp_path: Path,
) -> None:
    startup_dir = tmp_path / "startup"
    first = Storage(startup_dir, limits(), free_bytes=lambda _: 1 << 40)
    lease = first.begin_upload("transcription")
    first.close()
    target = first.staging_dir / f"{'b' * 32}.partial"
    target.write_bytes(b"target")
    lease.path.unlink()
    lease.path.symlink_to(target.name)

    with pytest.raises(StorageSchemaError, match="corrupt storage lease"):
        Storage(startup_dir, limits(), free_bytes=lambda _: 1 << 40)
    assert lease.path.is_symlink()
    assert target.read_bytes() == b"target"
    connection = sqlite3.connect(startup_dir / "botified-asr.sqlite3")
    try:
        assert connection.execute(
            "SELECT COUNT(*) FROM storage_leases WHERE id = ?",
            (lease.id,),
        ).fetchone()[0] == 1
    finally:
        connection.close()

    resolve_dir = tmp_path / "resolve"
    storage = Storage(resolve_dir, limits(), free_bytes=lambda _: 1 << 40)
    try:
        upload = storage.begin_upload("transcription")
        storage.append(upload, b"input")
        input_ref = storage.seal_upload(upload)
        resolve_target = storage.staging_dir / f"{'c' * 32}.ready"
        resolve_target.write_bytes(b"other")
        input_ref.path.unlink()
        input_ref.path.symlink_to(resolve_target.name)

        with pytest.raises(RuntimeError, match="not controlled"):
            storage.resolve_input(input_ref)
        assert input_ref.path.is_symlink()
        assert resolve_target.read_bytes() == b"other"
    finally:
        storage.close()


def test_transaction_releases_lock_on_begin_and_commit_failures() -> None:
    class Connection:
        def __init__(self, fail_on: str) -> None:
            self.fail_on = fail_on
            self.statements: list[str] = []

        def execute(self, statement: str) -> None:
            self.statements.append(statement)
            if statement == self.fail_on:
                raise sqlite3.OperationalError(f"{statement} failed")

    storage = Storage.__new__(Storage)
    storage._lock = threading.Lock()

    for failure in ("BEGIN IMMEDIATE", "COMMIT"):
        connection = Connection(failure)
        storage._connection = connection  # type: ignore[assignment]
        with pytest.raises(sqlite3.OperationalError, match=failure):
            with storage._transaction():
                pass
        assert not storage._lock.locked()
        if failure == "COMMIT":
            assert connection.statements == [
                "BEGIN IMMEDIATE",
                "COMMIT",
                "ROLLBACK",
            ]


def test_upload_lease_and_initial_reservation_are_atomic(tmp_path: Path) -> None:
    storage = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    try:
        lease = storage.begin_upload("transcription")

        assert lease.path.name.endswith(".partial")
        assert storage.total_reserved_bytes() == RESERVATION_QUANTUM
        assert storage.active_upload_count() == 1
    finally:
        storage.close()


def test_typed_upload_and_artifact_lifecycles_reject_handle_misuse(
    tmp_path: Path,
) -> None:
    storage = Storage(
        tmp_path,
        limits(max_job_storage_bytes=3 * RESERVATION_QUANTUM),
        free_bytes=lambda _: 1 << 40,
    )
    try:
        upload = storage.begin_upload("transcription")
        assert isinstance(upload, UploadLease)
        storage.append(upload, b"input")
        input_ref = storage.seal_upload(upload)
        assert isinstance(input_ref, InputRef)
        assert storage.resolve_input(input_ref).read_bytes() == b"input"

        writer = storage.begin_artifact(
            "segment_jsonl",
            owner_kind="sync",
            owner_id="request-1",
        )
        assert isinstance(writer, ReservedByteWriter)
        storage.append_artifact(writer, b'{"text":"ok"}\n')
        artifact_ref = storage.seal_artifact(writer)
        assert isinstance(artifact_ref, ArtifactRef)
        assert storage.resolve_artifact(artifact_ref).read_bytes() == (
            b'{"text":"ok"}\n'
        )
        with pytest.raises(ValueError, match="unsupported storage owner"):
            storage.begin_artifact(
                "segment_jsonl",
                owner_kind="job",
                owner_id="job-1",
            )
        for operation in (
            lambda: storage.append(upload, b"late"),
            lambda: storage.seal_upload(upload),
            lambda: storage.abort_upload(upload),
            lambda: storage.append_artifact(writer, b"late"),
            lambda: storage.seal_artifact(writer),
            lambda: storage.abort_artifact(writer),
        ):
            with pytest.raises(RuntimeError, match="already sealed"):
                operation()
        assert input_ref.path.read_bytes() == b"input"
        assert artifact_ref.path.read_bytes() == b'{"text":"ok"}\n'

        with pytest.raises(TypeError):
            storage.append(writer, b"wrong")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            storage.append_artifact(upload, b"wrong")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            storage.release_input(artifact_ref)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            storage.release_artifact(input_ref)  # type: ignore[arg-type]

        storage.release_artifact(artifact_ref)
        storage.release_input(input_ref)
        assert storage.total_reserved_bytes() == 0
    finally:
        storage.close()


def test_sealed_upload_releases_slot_artifact_never_consumes_slot(
    tmp_path: Path,
) -> None:
    storage = Storage(
        tmp_path,
        limits(
            max_active_uploads=1,
            max_job_storage_bytes=3 * RESERVATION_QUANTUM,
        ),
        free_bytes=lambda _: 1 << 40,
    )
    try:
        first = storage.begin_upload("transcription")
        storage.append(first, b"x")
        first_ref = storage.seal_upload(first)
        assert storage.active_upload_count() == 0
        assert storage.total_reserved_bytes() == 1

        artifact = storage.begin_artifact(
            "segment_jsonl",
            owner_kind="sync",
            owner_id="request-1",
        )
        assert storage.active_upload_count() == 0

        second = storage.begin_upload("transcription")
        assert storage.active_upload_count() == 1
        with pytest.raises(StorageAdmissionError) as caught:
            storage.begin_upload("transcription")
        assert caught.value.code == "too_many_active_uploads"

        storage.abort_upload(second)
        storage.abort_artifact(artifact)
        storage.release_input(first_ref)
    finally:
        storage.close()


@pytest.mark.parametrize("lease_type", ["upload", "artifact"])
def test_begin_capacity_and_free_floor_share_public_error_code(
    tmp_path: Path,
    lease_type: str,
) -> None:
    storage = Storage(
        tmp_path,
        limits(),
        free_bytes=lambda _: RESERVATION_QUANTUM,
    )
    try:
        with pytest.raises(StorageAdmissionError) as caught:
            if lease_type == "upload":
                storage.begin_upload("transcription")
            else:
                storage.begin_artifact(
                    "segment_jsonl",
                    owner_kind="sync",
                    owner_id="request-1",
                )
        assert caught.value.code == "storage_capacity_exceeded"
    finally:
        storage.close()


def test_artifact_expands_reservation_before_writing_next_byte(
    tmp_path: Path,
) -> None:
    storage = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    try:
        writer = storage.begin_artifact(
            "segment_jsonl",
            owner_kind="sync",
            owner_id="request-1",
        )
        storage.append_artifact(writer, b"x" * RESERVATION_QUANTUM)
        assert writer.path.stat().st_size == RESERVATION_QUANTUM

        with pytest.raises(StorageAdmissionError) as caught:
            storage.append_artifact(writer, b"x")

        assert caught.value.code == "storage_capacity_exceeded"
        assert writer.path.stat().st_size == RESERVATION_QUANTUM
        assert writer.actual_bytes == RESERVATION_QUANTUM
    finally:
        storage.abort_artifact(writer)
        storage.close()


def test_actual_write_failure_is_not_disguised_as_capacity_error(
    tmp_path: Path,
) -> None:
    storage = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    writer = storage.begin_artifact(
        "segment_jsonl",
        owner_kind="sync",
        owner_id="request-1",
    )
    original = storage._files.pop(writer.id)
    original.close()

    class BrokenWriter:
        def write(self, _: bytes) -> int:
            raise OSError("simulated disk failure")

        def close(self) -> None:
            pass

    storage._files[writer.id] = BrokenWriter()  # type: ignore[assignment]
    try:
        with pytest.raises(OSError, match="simulated disk failure"):
            storage.append_artifact(writer, b"x")
        assert writer.actual_bytes == 0
        assert storage.total_reserved_bytes() == RESERVATION_QUANTUM
    finally:
        storage.abort_artifact(writer)
        storage.close()


def test_seal_and_release_keep_ledger_until_filesystem_is_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import botified_asr.storage as storage_module

    storage = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    writer = storage.begin_artifact(
        "segment_jsonl",
        owner_kind="sync",
        owner_id="request-1",
    )
    storage.append_artifact(writer, b"result")
    observed: list[tuple[str, bool, str | None]] = []
    original_fsync_directory = storage_module._fsync_directory

    def observe_directory_fsync(path: Path) -> None:
        row = storage._connection.execute(
            "SELECT phase FROM storage_leases WHERE id = ?",
            (writer.id,),
        ).fetchone()
        observed.append(
            (
                path.name,
                (storage.artifact_dir / f"{writer.id}.complete").exists(),
                None if row is None else row["phase"],
            )
        )
        original_fsync_directory(path)

    monkeypatch.setattr(
        storage_module, "_fsync_directory", observe_directory_fsync
    )
    try:
        artifact_ref = storage.seal_artifact(writer)
        assert observed[-1] == ("artifacts", True, "writing")

        storage.release_artifact(artifact_ref)
        assert observed[-1] == ("artifacts", False, "sealed")
        assert storage.total_reserved_bytes() == 0
    finally:
        storage.close()


@pytest.mark.parametrize("lease_type", ["artifact", "upload"])
@pytest.mark.parametrize(
    ("fault", "expected_type", "message"),
    [
        ("rename", OSError, "rename failed"),
        ("directory_fsync", OSError, "directory fsync failed"),
        ("database_update", sqlite3.IntegrityError, "seal update failed"),
        ("commit_ack", OSError, "commit acknowledgement failed"),
    ],
)
def test_seal_failure_windows_compensate_before_returning_public_ref(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lease_type: str,
    fault: str,
    expected_type: type[Exception],
    message: str,
) -> None:
    import botified_asr.storage as storage_module

    storage = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    if lease_type == "artifact":
        lease = storage.begin_artifact(
            "segment_jsonl",
            owner_kind="sync",
            owner_id="request-1",
        )
        storage.append_artifact(lease, b"result")

        def seal() -> object:
            return storage.seal_artifact(lease)

        sealed_path = storage.artifact_dir / f"{lease.id}.complete"
    else:
        lease = storage.begin_upload("transcription")
        storage.append(lease, b"input")

        def seal() -> object:
            return storage.seal_upload(lease)

        sealed_path = storage.staging_dir / f"{lease.id}.ready"

    if fault == "rename":
        original_replace = storage_module.os.replace
        failed = False

        def fail_replace(source: Path, target: Path) -> None:
            nonlocal failed
            if not failed:
                failed = True
                raise OSError(message)
            original_replace(source, target)

        monkeypatch.setattr(storage_module.os, "replace", fail_replace)
    elif fault == "directory_fsync":
        original_fsync_directory = storage_module._fsync_directory
        failed = False

        def fail_fsync(path: Path) -> None:
            nonlocal failed
            if not failed:
                failed = True
                raise OSError(message)
            original_fsync_directory(path)

        monkeypatch.setattr(
            storage_module,
            "_fsync_directory",
            fail_fsync,
        )
    elif fault == "database_update":
        storage._connection.execute(
            """
            CREATE TRIGGER reject_seal_update
            BEFORE UPDATE OF phase ON storage_leases
            WHEN NEW.phase = 'sealed'
            BEGIN
                SELECT RAISE(FAIL, 'seal update failed');
            END
            """
        )
    else:
        original_exit = storage_module._Transaction.__exit__
        failed = False

        def fail_after_commit(
            transaction,
            exc_type,
            exc,
            traceback,
        ) -> bool:
            nonlocal failed
            result = original_exit(transaction, exc_type, exc, traceback)
            if exc_type is None and not failed:
                failed = True
                raise OSError(message)
            return result

        monkeypatch.setattr(
            storage_module._Transaction,
            "__exit__",
            fail_after_commit,
        )

    try:
        with pytest.raises(expected_type, match=message):
            seal()

        assert lease._state == "aborted"
        assert lease.id not in storage._files
        assert not lease.path.exists()
        assert not sealed_path.exists()
        assert (
            storage._connection.execute(
                "SELECT 1 FROM storage_leases WHERE id = ?",
                (lease.id,),
            ).fetchone()
            is None
        )
        assert storage.total_reserved_bytes() == 0
    finally:
        storage.close()


@pytest.mark.parametrize("lease_type", ["artifact", "upload"])
@pytest.mark.parametrize(
    "ledger_state",
    ["writing", "sealed", "missing"],
)
def test_failed_seal_compensation_remains_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lease_type: str,
    ledger_state: str,
) -> None:
    import botified_asr.storage as storage_module

    storage = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    if lease_type == "artifact":
        writer = storage.begin_artifact(
            "segment_jsonl",
            owner_kind="sync",
            owner_id="request-1",
        )
        storage.append_artifact(writer, b"result")

        def seal() -> object:
            return storage.seal_artifact(writer)

        def abort() -> None:
            storage.abort_artifact(writer)

        directory = storage.artifact_dir
    else:
        writer = storage.begin_upload("transcription")
        storage.append(writer, b"input")

        def seal() -> object:
            return storage.seal_upload(writer)

        def abort() -> None:
            storage.abort_upload(writer)

        directory = storage.staging_dir

    original_fsync_directory = storage_module._fsync_directory
    fsync_calls = 0

    def fail_compensation_fsync(path: Path) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if (
            ledger_state == "writing"
            and fsync_calls in {1, 2}
        ) or (
            ledger_state == "sealed"
            and fsync_calls == 2
        ) or (
            ledger_state == "missing"
            and fsync_calls == 1
        ):
            raise OSError("directory fsync failed")
        original_fsync_directory(path)

    monkeypatch.setattr(
        storage_module,
        "_fsync_directory",
        fail_compensation_fsync,
    )
    if ledger_state in {"sealed", "missing"}:
        original_exit = storage_module._Transaction.__exit__
        failed = False

        def fail_after_commit(
            transaction,
            exc_type,
            exc,
            traceback,
        ) -> bool:
            nonlocal failed
            result = original_exit(transaction, exc_type, exc, traceback)
            if exc_type is None and not failed:
                failed = True
                raise OSError("commit acknowledgement failed")
            return result

        monkeypatch.setattr(
            storage_module._Transaction,
            "__exit__",
            fail_after_commit,
        )
    try:
        expected_error = (
            "commit acknowledgement failed"
            if ledger_state == "missing"
            else "directory fsync failed"
        )
        with pytest.raises(OSError, match=expected_error):
            seal()

        assert writer._state == "writing"
        row = storage._connection.execute(
            "SELECT phase FROM storage_leases WHERE id = ?",
            (writer.id,),
        ).fetchone()
        if ledger_state == "missing":
            assert row is None
        else:
            assert row["phase"] == ledger_state
        abort()
        assert writer._state == "aborted"
        assert storage.total_reserved_bytes() == 0
        assert not list(directory.iterdir())
    finally:
        storage.close()


def test_competing_reservations_cannot_overcommit(tmp_path: Path) -> None:
    storage = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    try:
        first = storage.begin_upload("transcription")
        with pytest.raises(StorageAdmissionError) as caught:
            storage.begin_upload("enrollment")

        assert caught.value.code == "storage_capacity_exceeded"
        assert storage.total_reserved_bytes() == RESERVATION_QUANTUM
        storage.abort_upload(first)
        assert storage.total_reserved_bytes() == 0
    finally:
        storage.close()


def test_startup_cleans_receiving_files_and_reservations(tmp_path: Path) -> None:
    first = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    lease = first.begin_upload("transcription")
    first.append(lease, b"unfinished")
    assert lease.path.exists()
    first.close()

    recovered = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    try:
        assert recovered.active_upload_count() == 0
        assert recovered.total_reserved_bytes() == 0
        assert not lease.path.exists()
    finally:
        recovered.close()


def test_abort_is_idempotent_and_removes_file(tmp_path: Path) -> None:
    storage = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    try:
        lease = storage.begin_upload("transcription")
        storage.append(lease, b"content")
        lease.resource_kind = "tampered"
        with pytest.raises(RuntimeError, match="does not match ledger"):
            storage.abort_upload(lease)
        assert lease.path.exists()
        lease.resource_kind = "transcription"

        storage._connection.execute(
            """
            CREATE TRIGGER reject_seal
            BEFORE UPDATE OF phase ON storage_leases
            WHEN NEW.phase = 'sealed'
            BEGIN
                SELECT RAISE(FAIL, 'seal failed');
            END
            """
        )
        with pytest.raises(sqlite3.IntegrityError, match="seal failed"):
            storage.seal_upload(lease)

        storage.abort_upload(lease)
        storage.abort_upload(lease)

        assert not lease.path.exists()
        assert not (
            storage.staging_dir / f"{lease.id}.ready"
        ).exists()
        assert storage.total_reserved_bytes() == 0
    finally:
        storage.close()


def test_reservation_expands_before_crossing_quantum(tmp_path: Path) -> None:
    storage = Storage(
        tmp_path,
        limits(
            max_upload_bytes=RESERVATION_QUANTUM + 1,
            sync_max_upload_bytes=RESERVATION_QUANTUM + 1,
            max_job_storage_bytes=2 * RESERVATION_QUANTUM,
        ),
        free_bytes=lambda _: 1 << 40,
    )
    try:
        lease = storage.begin_upload("transcription")
        storage.append(lease, b"x" * RESERVATION_QUANTUM)
        assert storage.total_reserved_bytes() == RESERVATION_QUANTUM

        storage.append(lease, b"x")

        assert storage.total_reserved_bytes() == 2 * RESERVATION_QUANTUM
    finally:
        storage.close()


def test_startup_cleans_sealed_sync_input_and_reservation(
    tmp_path: Path,
) -> None:
    first = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    lease = first.begin_upload("transcription")
    first.append(lease, b"complete-but-not-promoted")
    input_ref = first.seal_upload(lease)
    assert input_ref.path.exists()
    first.close()

    recovered = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    try:
        assert recovered.active_upload_count() == 0
        assert recovered.total_reserved_bytes() == 0
        assert not input_ref.path.exists()
    finally:
        recovered.close()


def test_completion_records_incremental_content_hash(tmp_path: Path) -> None:
    storage = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    input_ref = None
    try:
        lease = storage.begin_upload("transcription")
        storage.append(lease, b"first-")
        storage.append(lease, b"second")

        input_ref = storage.seal_upload(lease)

        assert input_ref.content_sha256 == hashlib.sha256(
            b"first-second"
        ).hexdigest()
    finally:
        if input_ref is None:
            storage.abort_upload(lease)
        else:
            storage.release_input(input_ref)
        storage.close()


def test_two_connections_cannot_race_past_storage_capacity(
    tmp_path: Path,
) -> None:
    first = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    second = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    barrier = threading.Barrier(2)
    outcomes: list[tuple[str, object]] = []

    def acquire(storage: Storage) -> None:
        barrier.wait()
        try:
            outcomes.append(("ok", storage.begin_upload("transcription")))
        except StorageAdmissionError as exc:
            outcomes.append((exc.code, storage))

    threads = [
        threading.Thread(target=acquire, args=(first,)),
        threading.Thread(target=acquire, args=(second,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    try:
        assert [kind for kind, _ in outcomes].count("ok") == 1
        assert [kind for kind, _ in outcomes].count(
            "storage_capacity_exceeded"
        ) == 1
    finally:
        for kind, value in outcomes:
            if kind == "ok":
                owner = first if value.id in first._files else second
                owner.abort_upload(value)
        first.close()
        second.close()


def test_free_space_admission_includes_outstanding_reservations(
    tmp_path: Path,
) -> None:
    configured = limits(max_job_storage_bytes=2 * RESERVATION_QUANTUM)
    storage = Storage(
        tmp_path,
        configured,
        free_bytes=lambda _: 2 * RESERVATION_QUANTUM,
    )
    try:
        first = storage.begin_upload("transcription")
        with pytest.raises(StorageAdmissionError) as caught:
            storage.begin_upload("transcription")

        assert caught.value.code == "storage_capacity_exceeded"
        storage.abort_upload(first)
    finally:
        storage.close()


def test_actual_size_checkpoint_transactions_do_not_follow_network_chunks(
    tmp_path: Path,
) -> None:
    configured = limits(
        max_upload_bytes=RESERVATION_QUANTUM + 1,
        sync_max_upload_bytes=RESERVATION_QUANTUM + 1,
        max_job_storage_bytes=2 * RESERVATION_QUANTUM,
    )

    def count_updates(chunk_size: int, directory: Path) -> int:
        storage = Storage(directory, configured, free_bytes=lambda _: 1 << 40)
        updates: list[str] = []
        storage._connection.set_trace_callback(
            lambda sql: updates.append(sql)
            if "UPDATE storage_leases" in sql
            else None
        )
        lease = storage.begin_upload("transcription")
        remaining = RESERVATION_QUANTUM + 1
        while remaining:
            size = min(chunk_size, remaining)
            storage.append(lease, b"x" * size)
            remaining -= size
        input_ref = storage.seal_upload(lease)
        storage.release_input(input_ref)
        storage.close()
        return len(updates)

    small = count_updates(64 * 1024, tmp_path / "small")
    large = count_updates(1024 * 1024, tmp_path / "large")

    assert small == large
    assert small <= 3


def test_cleanup_removes_strict_orphans_but_never_escapes_data_dir(
    tmp_path: Path,
) -> None:
    storage = Storage(
        tmp_path,
        limits(max_job_storage_bytes=4 * RESERVATION_QUANTUM),
        free_bytes=lambda _: 1 << 40,
    )
    sync_upload = storage.begin_upload("transcription")
    storage.append(sync_upload, b"sync input")
    sync_input = storage.seal_upload(sync_upload)
    sync_writer = storage.begin_artifact(
        "segment_jsonl",
        owner_kind="sync",
        owner_id="request-1",
    )
    storage.append_artifact(sync_writer, b"sync artifact")
    sync_artifact = storage.seal_artifact(sync_writer)
    storage.close()

    outside = tmp_path.parent / "outside-must-stay"
    outside.write_bytes(b"safe")
    orphan = tmp_path / "staging" / f"{'a' * 32}.partial"
    orphan.write_bytes(b"orphan")
    unrelated = tmp_path / "staging" / "notes.txt"
    unrelated.write_bytes(b"keep")
    artifact_orphan = (
        tmp_path / "artifacts" / f"{'b' * 32}.complete"
    )
    artifact_orphan.write_bytes(b"orphan")
    outside_link = (
        tmp_path / "artifacts" / f"{'c' * 32}.partial"
    )
    outside_link.symlink_to(outside)
    connection = sqlite3.connect(tmp_path / "botified-asr.sqlite3")
    connection.execute(
        """
        INSERT INTO storage_leases(
            id, lease_type, resource_kind, owner_kind, owner_id, phase,
            controlled_path, reserved_bytes, actual_bytes, content_sha256,
            created_at
        ) VALUES (
            ?, 'artifact', 'segment_jsonl', 'sync', 'malicious', 'writing',
            ?, ?, 0, NULL, '2026-07-26T00:00:00.000Z'
        )
        """,
        ("../outside-must-stay", str(outside), RESERVATION_QUANTUM),
    )
    connection.commit()
    connection.close()

    configured = limits(max_job_storage_bytes=4 * RESERVATION_QUANTUM)
    with pytest.raises(StorageSchemaError, match="corrupt storage lease"):
        Storage(tmp_path, configured, free_bytes=lambda _: 1 << 40)
    assert sync_input.path.read_bytes() == b"sync input"
    assert sync_artifact.path.read_bytes() == b"sync artifact"
    assert orphan.read_bytes() == b"orphan"
    assert artifact_orphan.read_bytes() == b"orphan"
    assert outside_link.is_symlink()
    assert outside.read_bytes() == b"safe"
    assert unrelated.read_bytes() == b"keep"

    connection = sqlite3.connect(tmp_path / "botified-asr.sqlite3")
    connection.execute(
        "DELETE FROM storage_leases WHERE id = ?",
        ("../outside-must-stay",),
    )
    connection.commit()
    connection.close()

    recovered = Storage(tmp_path, configured, free_bytes=lambda _: 1 << 40)
    try:
        assert not sync_input.path.exists()
        assert not sync_artifact.path.exists()
        assert recovered.total_reserved_bytes() == 0
        assert not orphan.exists()
        assert not artifact_orphan.exists()
        assert not outside_link.exists()
        assert outside.read_bytes() == b"safe"
        assert unrelated.read_bytes() == b"keep"
    finally:
        recovered.close()
    recovered_again = Storage(
        tmp_path, configured, free_bytes=lambda _: 1 << 40
    )
    recovered_again.close()

    assert outside.read_bytes() == b"safe"
    assert unrelated.read_bytes() == b"keep"
