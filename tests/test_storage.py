from __future__ import annotations

import hashlib
import sqlite3
import struct
import threading
from pathlib import Path

import pytest

import botified_asr.storage as storage_module
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

PROCESSOR_FINGERPRINT = "3" * 64


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


def create_v2_database(
    data_dir: Path,
    *,
    with_lease: bool = False,
) -> Path | None:
    data_dir.mkdir(parents=True, exist_ok=True)
    staging = data_dir / "staging"
    staging.mkdir()
    lease_id = "b" * 32
    controlled_path = staging / f"{lease_id}.ready"

    connection = sqlite3.connect(data_dir / "botified-asr.sqlite3")
    connection.executescript(
        """
        CREATE TABLE schema_meta (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            version INTEGER NOT NULL
        );
        INSERT INTO schema_meta(singleton, version) VALUES (1, 2);
        CREATE TABLE storage_leases (
            id TEXT PRIMARY KEY,
            lease_type TEXT NOT NULL
                CHECK (lease_type IN ('upload', 'artifact')),
            resource_kind TEXT NOT NULL,
            owner_kind TEXT NOT NULL
                CHECK (owner_kind IN ('sync', 'job', 'legacy')),
            owner_id TEXT NOT NULL,
            phase TEXT NOT NULL
                CHECK (phase IN ('writing', 'sealed')),
            controlled_path TEXT NOT NULL UNIQUE,
            reserved_bytes INTEGER NOT NULL CHECK (reserved_bytes >= 0),
            actual_bytes INTEGER NOT NULL CHECK (
                actual_bytes >= 0 AND actual_bytes <= reserved_bytes
            ),
            content_sha256 TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX storage_leases_type_phase_idx
            ON storage_leases(lease_type, phase);
        CREATE INDEX storage_leases_owner_idx
            ON storage_leases(owner_kind, owner_id);
        """
    )
    if with_lease:
        payload = b"data"
        controlled_path.write_bytes(payload)
        connection.execute(
            """
            INSERT INTO storage_leases(
                id, lease_type, resource_kind, owner_kind, owner_id,
                phase, controlled_path, reserved_bytes, actual_bytes,
                content_sha256, created_at
            ) VALUES (?, 'upload', 'transcription', 'legacy', ?, 'sealed',
                      ?, ?, ?, ?, ?)
            """,
            (
                lease_id,
                lease_id,
                str(controlled_path),
                len(payload),
                len(payload),
                hashlib.sha256(payload).hexdigest(),
                "2026-07-26T00:00:00.000Z",
            ),
        )
    connection.commit()
    connection.close()
    return controlled_path if with_lease else None


def create_v3_database(data_dir: Path) -> None:
    create_v2_database(data_dir, with_lease=True)
    connection = sqlite3.connect(data_dir / "botified-asr.sqlite3")
    try:
        connection.executescript(
            """
            CREATE TABLE speaker_profiles (
                id TEXT PRIMARY KEY NOT NULL,
                name TEXT NOT NULL,
                name_key TEXT NOT NULL,
                description TEXT,
                embedding BLOB NOT NULL
                    CHECK (
                        typeof(embedding) = 'blob'
                        AND length(embedding) = 4 * embedding_dimension
                    ),
                embedding_model_id TEXT NOT NULL,
                embedding_model_revision TEXT NOT NULL,
                embedding_dimension INTEGER NOT NULL
                    CHECK (embedding_dimension > 0),
                embedding_policy_fingerprint TEXT NOT NULL,
                sample_count INTEGER NOT NULL
                    CHECK (sample_count BETWEEN 2 AND 5),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE UNIQUE INDEX speaker_profiles_name_key_uq
                ON speaker_profiles(name_key);
            UPDATE schema_meta SET version = 3 WHERE singleton = 1;
            """
        )
        _insert_speaker_profile(connection)
        connection.commit()
    finally:
        connection.close()


def create_v4_database(
    data_dir: Path,
    *,
    with_receiving_job: bool = False,
) -> None:
    create_v3_database(data_dir)
    connection = sqlite3.connect(data_dir / "botified-asr.sqlite3")
    try:
        connection.execute(storage_module._V4_TRANSCRIPTION_JOBS_DDL)
        connection.executescript(
            """
            CREATE INDEX transcription_jobs_fifo_idx
                ON transcription_jobs(phase, status, created_at, id);
            CREATE INDEX transcription_jobs_retention_idx
                ON transcription_jobs(phase, status, finished_at, id);
            """
        )
        connection.execute(storage_module._V4_SHUTDOWN_MARKER_DDL)
        if with_receiving_job:
            connection.execute(
                """
                INSERT INTO transcription_jobs(
                    id, phase, status, input_lease_id, processed_samples,
                    attempt_no, crash_recoveries, cancel_requested,
                    input_cleanup_pending, created_at
                ) VALUES (
                    '7K3M9Q2W', 'receiving', NULL, '7K3M9Q2W', 0,
                    0, 0, 0, 0, '2026-07-27T12:00:00Z'
                )
                """
            )
        connection.execute(
            "UPDATE schema_meta SET version = 4 WHERE singleton = 1"
        )
        connection.commit()
    finally:
        connection.close()


def _speaker_profile_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": "01234567",
        "name": "Alice",
        "name_key": "alice",
        "description": None,
        "embedding": struct.pack("<ff", 1.0, 0.0),
        "embedding_model_id": "funasr/campplus",
        "embedding_model_revision": "1" * 40,
        "embedding_dimension": 2,
        "embedding_policy_fingerprint": "a" * 64,
        "sample_count": 2,
        "created_at": "2026-07-27T12:00:00Z",
        "updated_at": "2026-07-27T12:00:00Z",
    }
    row.update(overrides)
    return row


def _insert_speaker_profile(
    connection: sqlite3.Connection,
    **overrides: object,
) -> None:
    connection.execute(
        """
        INSERT INTO speaker_profiles(
            id, name, name_key, description, embedding,
            embedding_model_id, embedding_model_revision,
            embedding_dimension, embedding_policy_fingerprint,
            sample_count, created_at, updated_at
        ) VALUES (
            :id, :name, :name_key, :description, :embedding,
            :embedding_model_id, :embedding_model_revision,
            :embedding_dimension, :embedding_policy_fingerprint,
            :sample_count, :created_at, :updated_at
        )
        """,
        _speaker_profile_row(**overrides),
    )


JOB_COLUMNS = (
    ("id", "TEXT", 1, None, 1),
    ("phase", "TEXT", 1, None, 0),
    ("status", "TEXT", 0, None, 0),
    ("input_lease_id", "TEXT", 0, None, 0),
    ("canonical_options_json", "TEXT", 0, None, 0),
    ("selected_speaker_snapshot", "BLOB", 0, None, 0),
    ("snapshot_sha256", "TEXT", 0, None, 0),
    ("input_size_bytes", "INTEGER", 0, None, 0),
    ("effective_max_audio_samples", "INTEGER", 0, None, 0),
    ("effective_direct_max_audio_samples", "INTEGER", 0, None, 0),
    ("total_samples", "INTEGER", 0, None, 0),
    ("processed_samples", "INTEGER", 1, None, 0),
    ("request_fingerprint", "TEXT", 0, None, 0),
    ("processor_fingerprint", "TEXT", 0, None, 0),
    ("attempt_no", "INTEGER", 1, None, 0),
    ("attempt_token", "TEXT", 0, None, 0),
    ("owner_generation", "TEXT", 0, None, 0),
    ("crash_recoveries", "INTEGER", 1, None, 0),
    ("cancel_requested", "INTEGER", 1, None, 0),
    ("result_lease_id", "TEXT", 0, None, 0),
    ("error_code", "TEXT", 0, None, 0),
    ("input_cleanup_pending", "INTEGER", 1, None, 0),
    ("created_at", "TEXT", 1, None, 0),
    ("started_at", "TEXT", 0, None, 0),
    ("finished_at", "TEXT", 0, None, 0),
)
MARKER_COLUMNS = (
    ("singleton", "INTEGER", 0, None, 1),
    ("generation", "TEXT", 1, None, 0),
    ("created_at", "TEXT", 1, None, 0),
)


def _job_schema_snapshot(database: Path) -> tuple[object, ...]:
    connection = sqlite3.connect(database)
    try:
        return (
            connection.execute(
                "SELECT version FROM schema_meta WHERE singleton = 1"
            ).fetchone()[0],
            connection.execute(
                """
                SELECT sql FROM sqlite_master
                WHERE type = 'table' AND name = 'transcription_jobs'
                """
            ).fetchone()[0],
            tuple(connection.execute("PRAGMA table_info(transcription_jobs)")),
            tuple(connection.execute("SELECT * FROM transcription_jobs")),
        )
    finally:
        connection.close()


def _transcription_job_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": "7K3M9Q2W",
        "phase": "visible",
        "status": "queued",
        "input_lease_id": "a" * 32,
        "canonical_options_json": (
            '{"chunking_strategy":null,"include":[],"known_speaker_ids":[],'
            '"language":"auto","model":"sensevoice","response_format":"json"}'
        ),
        "selected_speaker_snapshot": b'{"speakers":[]}',
        "snapshot_sha256": "1" * 64,
        "input_size_bytes": 4,
        "effective_max_audio_samples": 32_000,
        "effective_direct_max_audio_samples": 16_000,
        "total_samples": 32_000,
        "processed_samples": 0,
        "request_fingerprint": "2" * 64,
        "processor_fingerprint": "3" * 64,
        "attempt_no": 0,
        "attempt_token": None,
        "owner_generation": None,
        "crash_recoveries": 0,
        "cancel_requested": 0,
        "result_lease_id": None,
        "error_code": None,
        "input_cleanup_pending": 0,
        "created_at": "2026-07-27T12:00:00Z",
        "started_at": None,
        "finished_at": None,
    }
    row.update(overrides)
    return row


def _insert_transcription_job(
    connection: sqlite3.Connection,
    **overrides: object,
) -> None:
    columns = ", ".join(name for name, *_ in JOB_COLUMNS)
    values = ", ".join(f":{name}" for name, *_ in JOB_COLUMNS)
    connection.execute(
        f"INSERT INTO transcription_jobs({columns}) VALUES ({values})",
        _transcription_job_row(**overrides),
    )


def _recreate_v5_table_with_column_drift(
    connection: sqlite3.Connection,
    *,
    table: str,
    column: str | None = None,
    replacement: str | None = None,
    include_critical_check: bool = True,
) -> None:
    columns = JOB_COLUMNS if table == "transcription_jobs" else MARKER_COLUMNS
    old_table = f"old_{table}"
    connection.execute(f"ALTER TABLE {table} RENAME TO {old_table}")
    if table == "transcription_jobs":
        connection.execute("DROP INDEX transcription_jobs_fifo_idx")
        connection.execute("DROP INDEX transcription_jobs_retention_idx")

    definitions = []
    for name, data_type, not_null, default, primary_key in columns:
        if name == column:
            assert replacement is not None
            definitions.append(replacement)
            continue
        definition = f"{name} {data_type}"
        if primary_key:
            definition += " PRIMARY KEY"
        if not_null:
            definition += " NOT NULL"
        if default is not None:
            definition += f" DEFAULT {default}"
        definitions.append(definition)
    if table == "transcription_jobs":
        definitions.extend(
            (
                "CHECK (length(id) = 8)",
                "CHECK (id NOT GLOB '*[^0-9A-HJKMNP-TV-Z]*')",
                "CHECK (phase IN ('receiving', 'visible', 'deleting'))",
                "CHECK (status IS NULL OR status IN "
                "('queued', 'running', 'succeeded', 'failed', 'cancelled'))",
                "CHECK (snapshot_sha256 IS NULL OR "
                "(length(snapshot_sha256) = 64 AND "
                "snapshot_sha256 NOT GLOB '*[^0-9a-f]*'))",
                "CHECK (input_size_bytes IS NULL OR input_size_bytes >= 0)",
                "CHECK (total_samples IS NULL OR total_samples >= 0)",
                "CHECK (processed_samples >= 0)",
                "CHECK (request_fingerprint IS NULL OR "
                "(length(request_fingerprint) = 64 AND "
                "request_fingerprint NOT GLOB '*[^0-9a-f]*'))",
                "CHECK (processor_fingerprint IS NULL OR "
                "(length(processor_fingerprint) = 64 AND "
                "processor_fingerprint NOT GLOB '*[^0-9a-f]*'))",
                "CHECK (attempt_no >= 0)",
                "CHECK (crash_recoveries BETWEEN 0 AND 1)",
                "CHECK (cancel_requested IN (0, 1))",
                "CHECK (input_cleanup_pending IN (0, 1))",
            )
        )
        if include_critical_check:
            definitions.append(
                "CHECK ("
                "(phase = 'receiving' AND status IS NULL) OR "
                "(phase = 'visible' AND status IS NOT NULL) OR "
                "(phase = 'deleting' AND (status IS NULL OR status IN "
                "('succeeded', 'failed', 'cancelled')))"
                ")"
            )
    else:
        definitions.append("CHECK (singleton = 1)")
        if include_critical_check:
            definitions.append("CHECK (length(generation) > 0)")
    connection.execute(f"CREATE TABLE {table} ({', '.join(definitions)})")
    connection.execute(f"DROP TABLE {old_table}")

    if table == "transcription_jobs":
        connection.execute(
            """
            CREATE INDEX transcription_jobs_fifo_idx
            ON transcription_jobs(phase, status, created_at, id)
            """
        )
        connection.execute(
            """
            CREATE INDEX transcription_jobs_retention_idx
            ON transcription_jobs(phase, status, finished_at, id)
            """
        )


def _recreate_speaker_profiles_with_column_drift(
    connection: sqlite3.Connection,
    *,
    drift: str,
) -> None:
    description = "TEXT NOT NULL" if drift == "description-nullability" else "TEXT"
    embedding = "TEXT NOT NULL" if drift == "embedding-type" else "BLOB NOT NULL"
    connection.executescript(
        f"""
        DROP INDEX speaker_profiles_name_key_uq;
        DROP TABLE speaker_profiles;
        CREATE TABLE speaker_profiles (
            id TEXT PRIMARY KEY NOT NULL,
            name TEXT NOT NULL,
            name_key TEXT NOT NULL,
            description {description},
            embedding {embedding}
                CHECK (
                    typeof(embedding) = 'blob'
                    AND length(embedding) = 4 * embedding_dimension
                ),
            embedding_model_id TEXT NOT NULL,
            embedding_model_revision TEXT NOT NULL,
            embedding_dimension INTEGER NOT NULL
                CHECK (embedding_dimension > 0),
            embedding_policy_fingerprint TEXT NOT NULL,
            sample_count INTEGER NOT NULL
                CHECK (sample_count BETWEEN 2 AND 5),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX speaker_profiles_name_key_uq
            ON speaker_profiles(name_key);
        """
    )


def test_fresh_schema_is_explicit_v5_with_job_foundation(tmp_path: Path) -> None:
    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    try:
        version = storage._connection.execute(
            "SELECT version FROM schema_meta WHERE singleton = 1"
        ).fetchone()[0]
        profile_columns = tuple(
            storage._connection.execute("PRAGMA table_info(speaker_profiles)")
        )
        profile_indexes = {
            row["name"]: row
            for row in storage._connection.execute(
                "PRAGMA index_list(speaker_profiles)"
            )
        }
        job_columns = tuple(
            (
                row["name"],
                row["type"],
                row["notnull"],
                row["dflt_value"],
                row["pk"],
            )
            for row in storage._connection.execute(
                "PRAGMA table_info(transcription_jobs)"
            )
        )
        job_indexes = {
            row["name"]: row
            for row in storage._connection.execute(
                "PRAGMA index_list(transcription_jobs)"
            )
        }
        marker_columns = tuple(
            (
                row["name"],
                row["type"],
                row["notnull"],
                row["dflt_value"],
                row["pk"],
            )
            for row in storage._connection.execute(
                "PRAGMA table_info(shutdown_marker)"
            )
        )

        assert version == 5
        assert {
            row["name"]
            for row in storage._connection.execute(
                "PRAGMA table_info(storage_leases)"
            )
        } == {
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
        assert tuple(
            (
                row["name"],
                row["type"],
                row["notnull"],
                row["dflt_value"],
                row["pk"],
            )
            for row in profile_columns
        ) == (
            ("id", "TEXT", 1, None, 1),
            ("name", "TEXT", 1, None, 0),
            ("name_key", "TEXT", 1, None, 0),
            ("description", "TEXT", 0, None, 0),
            ("embedding", "BLOB", 1, None, 0),
            ("embedding_model_id", "TEXT", 1, None, 0),
            ("embedding_model_revision", "TEXT", 1, None, 0),
            ("embedding_dimension", "INTEGER", 1, None, 0),
            ("embedding_policy_fingerprint", "TEXT", 1, None, 0),
            ("sample_count", "INTEGER", 1, None, 0),
            ("created_at", "TEXT", 1, None, 0),
            ("updated_at", "TEXT", 1, None, 0),
        )
        unique_name_key = profile_indexes["speaker_profiles_name_key_uq"]
        assert unique_name_key["unique"] == 1
        assert unique_name_key["partial"] == 0
        assert tuple(
            row["name"]
            for row in storage._connection.execute(
                "PRAGMA index_info(speaker_profiles_name_key_uq)"
            )
        ) == ("name_key",)
        assert not any("created" in name for name in profile_indexes)
        assert {
            "storage_leases_type_phase_idx",
            "storage_leases_owner_idx",
        } <= {
            row["name"]
            for row in storage._connection.execute(
                "PRAGMA index_list(storage_leases)"
            )
        }
        assert job_columns == JOB_COLUMNS
        assert marker_columns == MARKER_COLUMNS
        assert (
            storage._connection.execute(
                "SELECT COUNT(*) FROM shutdown_marker"
            ).fetchone()[0]
            == 0
        )
        assert {
            "transcription_jobs_fifo_idx",
            "transcription_jobs_retention_idx",
        } <= set(job_indexes)
        assert all(
            job_indexes[name]["unique"] == 0
            and job_indexes[name]["partial"] == 0
            for name in (
                "transcription_jobs_fifo_idx",
                "transcription_jobs_retention_idx",
            )
        )
        assert tuple(
            row["name"]
            for row in storage._connection.execute(
                "PRAGMA index_info(transcription_jobs_fifo_idx)"
            )
        ) == ("phase", "status", "created_at", "id")
        assert tuple(
            row["name"]
            for row in storage._connection.execute(
                "PRAGMA index_info(transcription_jobs_retention_idx)"
            )
        ) == ("phase", "status", "finished_at", "id")

        _insert_speaker_profile(storage._connection)
        with pytest.raises(sqlite3.IntegrityError):
            _insert_speaker_profile(
                storage._connection,
                id="ABCDEFGH",
                name="ALICE",
            )
    finally:
        storage.close()

    connection = sqlite3.connect(tmp_path / "botified-asr.sqlite3")
    connection.execute("CREATE TABLE upload_leases (id TEXT PRIMARY KEY)")
    connection.close()
    with pytest.raises(
        StorageSchemaError, match="unexpected legacy upload ledger"
    ):
        Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)


def test_job_tables_enforce_local_values_without_encoding_state_machine(
    tmp_path: Path,
) -> None:
    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    try:
        for changes in (
            {"phase": "receiving", "status": "queued"},
            {"phase": "visible", "status": None},
            {"phase": "deleting", "status": "queued"},
            {"phase": "deleting", "status": "running"},
        ):
            with pytest.raises(sqlite3.IntegrityError):
                _insert_transcription_job(storage._connection, **changes)

        for status in (None, "succeeded"):
            _insert_transcription_job(
                storage._connection,
                phase="deleting",
                status=status,
            )
            storage._connection.execute("DELETE FROM transcription_jobs")

        invalid_rows = (
            {"id": "ABCDEFGI"},
            {"phase": "hidden"},
            {"status": "retrying"},
            {"snapshot_sha256": "x" * 64},
            {"input_size_bytes": -1},
            {"total_samples": -1},
            {"processed_samples": -1},
            {"request_fingerprint": "2" * 63},
            {"processor_fingerprint": "G" * 64},
            {"attempt_no": -1},
            {"crash_recoveries": 2},
            {"cancel_requested": 2},
            {"input_cleanup_pending": 2},
        )
        for changes in invalid_rows:
            with pytest.raises(sqlite3.IntegrityError):
                _insert_transcription_job(storage._connection, **changes)

        with pytest.raises(sqlite3.IntegrityError):
            storage._connection.execute(
                """
                INSERT INTO shutdown_marker(singleton, generation, created_at)
                VALUES (1, '', '2026-07-27T12:00:00Z')
                """
            )
        storage._connection.execute(
            """
            INSERT INTO shutdown_marker(singleton, generation, created_at)
            VALUES (1, 'generation-1', '2026-07-27T12:00:00Z')
            """
        )
        with pytest.raises(sqlite3.IntegrityError):
            storage._connection.execute(
                """
                INSERT INTO shutdown_marker(singleton, generation, created_at)
                VALUES (2, 'generation-2', '2026-07-27T12:00:01Z')
                """
            )
    finally:
        storage.close()


def test_v3_schema_migrates_through_v5_atomically_preserving_existing_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_v3_database(tmp_path)
    connection = sqlite3.connect(tmp_path / "botified-asr.sqlite3")
    try:
        lease_before = tuple(
            connection.execute("SELECT * FROM storage_leases").fetchone()
        )
        profile_before = tuple(
            connection.execute("SELECT * FROM speaker_profiles").fetchone()
        )
    finally:
        connection.close()
    monkeypatch.setattr(Storage, "_reconcile_startup", lambda _self: None)

    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    try:
        assert storage._connection.execute(
            "SELECT version FROM schema_meta WHERE singleton = 1"
        ).fetchone()[0] == 5
        assert tuple(
            storage._connection.execute(
                "SELECT * FROM storage_leases"
            ).fetchone()
        ) == lease_before
        assert tuple(
            storage._connection.execute(
                "SELECT * FROM speaker_profiles"
            ).fetchone()
        ) == profile_before
        assert (
            storage._connection.execute(
                "SELECT COUNT(*) FROM transcription_jobs"
            ).fetchone()[0]
            == 0
        )
        assert (
            storage._connection.execute(
                "SELECT COUNT(*) FROM shutdown_marker"
            ).fetchone()[0]
            == 0
        )
    finally:
        storage.close()


def test_failed_v3_to_v4_migration_rolls_back_then_retries_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_v3_database(tmp_path)
    database = tmp_path / "botified-asr.sqlite3"
    connection = sqlite3.connect(database)
    try:
        connection.executescript(
            """
            CREATE TRIGGER reject_v4_schema_version
            BEFORE UPDATE OF version ON schema_meta
            WHEN NEW.version = 4
            BEGIN
                SELECT RAISE(ABORT, 'injected v4 migration failure');
            END;
            """
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(
        sqlite3.IntegrityError,
        match="injected v4 migration failure",
    ):
        Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)

    connection = sqlite3.connect(database)
    try:
        assert connection.execute(
            "SELECT version FROM schema_meta WHERE singleton = 1"
        ).fetchone()[0] == 3
        assert {
            row[0]
            for row in connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE name IN (
                    'transcription_jobs',
                    'transcription_jobs_fifo_idx',
                    'transcription_jobs_retention_idx',
                    'shutdown_marker'
                )
                """
            )
        } == set()
        assert connection.execute(
            "SELECT COUNT(*) FROM storage_leases"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM speaker_profiles"
        ).fetchone()[0] == 1
        connection.execute("DROP TRIGGER reject_v4_schema_version")
        connection.commit()
    finally:
        connection.close()

    with monkeypatch.context() as retry:
        retry.setattr(Storage, "_reconcile_startup", lambda _self: None)
        storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    try:
        assert storage._connection.execute(
            "SELECT version FROM schema_meta WHERE singleton = 1"
        ).fetchone()[0] == 5
        assert storage._connection.execute(
            "SELECT COUNT(*) FROM storage_leases"
        ).fetchone()[0] == 1
        assert storage._connection.execute(
            "SELECT COUNT(*) FROM speaker_profiles"
        ).fetchone()[0] == 1
    finally:
        storage.close()


def test_empty_exact_v4_schema_migrates_to_v5_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_v4_database(tmp_path)
    monkeypatch.setattr(Storage, "_reconcile_startup", lambda _self: None)

    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    try:
        assert storage._connection.execute(
            "SELECT version FROM schema_meta WHERE singleton = 1"
        ).fetchone()[0] == 5
        assert tuple(
            row[1:]
            for row in storage._connection.execute(
                "PRAGMA table_info(transcription_jobs)"
            )
        ) == JOB_COLUMNS
        assert storage._connection.execute(
            "SELECT COUNT(*) FROM transcription_jobs"
        ).fetchone()[0] == 0
    finally:
        storage.close()


def test_v4_schema_with_any_job_fails_closed_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_v4_database(tmp_path, with_receiving_job=True)
    database = tmp_path / "botified-asr.sqlite3"
    before = _job_schema_snapshot(database)

    def unexpected_reconcile(_storage: Storage) -> None:
        raise AssertionError("startup reconciliation must not run")

    monkeypatch.setattr(Storage, "_reconcile_startup", unexpected_reconcile)
    with pytest.raises(StorageSchemaError):
        Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)

    assert before[0] == 4
    assert before[1] == storage_module._V4_TRANSCRIPTION_JOBS_DDL
    assert len(before[3]) == 1
    assert _job_schema_snapshot(database) == before


def test_failed_v4_to_v5_version_update_rolls_back_then_retries_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_v4_database(tmp_path)
    database = tmp_path / "botified-asr.sqlite3"
    before = _job_schema_snapshot(database)
    connection = sqlite3.connect(database)
    try:
        connection.executescript(
            """
            CREATE TRIGGER reject_v5_schema_version
            BEFORE UPDATE OF version ON schema_meta
            WHEN NEW.version = 5
            BEGIN
                SELECT RAISE(ABORT, 'injected v5 migration failure');
            END;
            """
        )
        connection.commit()
    finally:
        connection.close()
    monkeypatch.setattr(Storage, "_reconcile_startup", lambda _self: None)

    with pytest.raises(
        sqlite3.IntegrityError,
        match="injected v5 migration failure",
    ):
        Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)

    assert _job_schema_snapshot(database) == before
    connection = sqlite3.connect(database)
    try:
        connection.execute("DROP TRIGGER reject_v5_schema_version")
        connection.commit()
    finally:
        connection.close()

    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    try:
        assert storage._connection.execute(
            "SELECT version FROM schema_meta WHERE singleton = 1"
        ).fetchone()[0] == 5
        assert tuple(
            row[1:]
            for row in storage._connection.execute(
                "PRAGMA table_info(transcription_jobs)"
            )
        ) == JOB_COLUMNS
    finally:
        storage.close()


@pytest.mark.parametrize(
    "mutation",
    (
        "missing-job-table",
        "wrong-job-column",
        "missing-fifo-index",
        "wrong-retention-index",
        "missing-marker-table",
        "wrong-marker-column",
        "job-column-type",
        "job-column-nullability",
        "job-column-default",
        "marker-column-type",
        "marker-column-nullability",
        "marker-column-default",
        "job-phase-status-check",
        "marker-generation-check",
    ),
)
def test_v5_schema_verifier_rejects_critical_job_foundation_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40).close()
    connection = sqlite3.connect(tmp_path / "botified-asr.sqlite3")
    try:
        column_drifts = {
            "job-column-type": (
                "transcription_jobs",
                "processor_fingerprint",
                "processor_fingerprint BLOB",
            ),
            "job-column-nullability": (
                "transcription_jobs",
                "status",
                "status TEXT NOT NULL",
            ),
            "job-column-default": (
                "transcription_jobs",
                "attempt_no",
                "attempt_no INTEGER NOT NULL DEFAULT 0",
            ),
            "marker-column-type": (
                "shutdown_marker",
                "generation",
                "generation BLOB NOT NULL",
            ),
            "marker-column-nullability": (
                "shutdown_marker",
                "generation",
                "generation TEXT",
            ),
            "marker-column-default": (
                "shutdown_marker",
                "created_at",
                "created_at TEXT NOT NULL DEFAULT 'unexpected'",
            ),
        }
        check_drifts = {
            "job-phase-status-check": "transcription_jobs",
            "marker-generation-check": "shutdown_marker",
        }
        if mutation in check_drifts:
            _recreate_v5_table_with_column_drift(
                connection,
                table=check_drifts[mutation],
                include_critical_check=False,
            )
            connection.commit()
        elif mutation in column_drifts:
            table, column, replacement = column_drifts[mutation]
            _recreate_v5_table_with_column_drift(
                connection,
                table=table,
                column=column,
                replacement=replacement,
            )
            connection.commit()
        else:
            statements = {
                "missing-job-table": ("DROP TABLE transcription_jobs",),
                "wrong-job-column": (
                    "ALTER TABLE transcription_jobs "
                    "RENAME COLUMN finished_at TO completed_at",
                ),
                "missing-fifo-index": (
                    "DROP INDEX transcription_jobs_fifo_idx",
                ),
                "wrong-retention-index": (
                    "DROP INDEX transcription_jobs_retention_idx",
                    "CREATE INDEX transcription_jobs_retention_idx "
                    "ON transcription_jobs(status, phase, finished_at, id)",
                ),
                "missing-marker-table": ("DROP TABLE shutdown_marker",),
                "wrong-marker-column": (
                    "ALTER TABLE shutdown_marker "
                    "RENAME COLUMN created_at TO marked_at",
                ),
            }[mutation]
            for statement in statements:
                connection.execute(statement)
            connection.commit()
    finally:
        connection.close()

    with pytest.raises(StorageSchemaError):
        Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)


def test_v1_schema_migrates_transactionally_and_reconciles_legacy(
    tmp_path: Path,
) -> None:
    legacy_path = create_v1_database(tmp_path)

    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    try:
        assert storage._connection.execute(
            "SELECT version FROM schema_meta WHERE singleton = 1"
        ).fetchone()[0] == 5
        assert storage._connection.execute(
            "SELECT COUNT(*) FROM storage_leases"
        ).fetchone()[0] == 0
        assert storage._connection.execute(
            """
            SELECT COUNT(*) FROM sqlite_master
            WHERE type = 'table' AND name = 'speaker_profiles'
            """
        ).fetchone()[0] == 1
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
        Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)

    unversioned = tmp_path / "unversioned"
    unversioned.mkdir()
    sentinel = unversioned / "keep.txt"
    sentinel.write_bytes(b"keep")
    connection = sqlite3.connect(unversioned / "botified-asr.sqlite3")
    connection.execute("CREATE TABLE application_data (value TEXT)")
    connection.close()

    with pytest.raises(StorageSchemaError, match="unversioned"):
        Storage(unversioned, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
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
        Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)

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


def test_v2_schema_migrates_through_v5_without_rewriting_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controlled_path = create_v2_database(tmp_path, with_lease=True)
    monkeypatch.setattr(Storage, "_reconcile_startup", lambda _self: None)

    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    try:
        assert (
            storage._connection.execute(
                "SELECT version FROM schema_meta WHERE singleton = 1"
            ).fetchone()[0]
            == 5
        )
        lease = storage._connection.execute(
            """
            SELECT id, lease_type, resource_kind, owner_kind, owner_id,
                   phase, controlled_path, reserved_bytes, actual_bytes,
                   content_sha256, created_at
            FROM storage_leases
            """
        ).fetchone()
        assert tuple(lease) == (
            "b" * 32,
            "upload",
            "transcription",
            "legacy",
            "b" * 32,
            "sealed",
            str(controlled_path),
            4,
            4,
            hashlib.sha256(b"data").hexdigest(),
            "2026-07-26T00:00:00.000Z",
        )
        assert controlled_path is not None
        assert controlled_path.read_bytes() == b"data"
        assert (
            storage._connection.execute(
                "SELECT COUNT(*) FROM speaker_profiles"
            ).fetchone()[0]
            == 0
        )
    finally:
        storage.close()


def test_failed_v2_migration_rolls_back_then_retries_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controlled_path = create_v2_database(tmp_path, with_lease=True)
    connection = sqlite3.connect(tmp_path / "botified-asr.sqlite3")
    try:
        connection.executescript(
            """
            CREATE TRIGGER reject_v3_schema_version
            BEFORE UPDATE OF version ON schema_meta
            WHEN NEW.version = 3
            BEGIN
                SELECT RAISE(ABORT, 'injected v3 migration failure');
            END;
            """
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(
        sqlite3.IntegrityError,
        match="injected v3 migration failure",
    ):
        Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)

    connection = sqlite3.connect(tmp_path / "botified-asr.sqlite3")
    try:
        assert (
            connection.execute(
                "SELECT version FROM schema_meta WHERE singleton = 1"
            ).fetchone()[0]
            == 2
        )
        assert (
            connection.execute(
                """
            SELECT COUNT(*) FROM sqlite_master
            WHERE name IN (
                'speaker_profiles',
                'speaker_profiles_name_key_uq'
            )
            """
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM storage_leases").fetchone()[0] == 1
        )
        assert controlled_path is not None
        assert controlled_path.read_bytes() == b"data"
        connection.execute("DROP TRIGGER reject_v3_schema_version")
        connection.commit()
    finally:
        connection.close()

    with monkeypatch.context() as retry:
        retry.setattr(Storage, "_reconcile_startup", lambda _self: None)
        storage = Storage(
            tmp_path,
            limits(),
            current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40,
        )
    try:
        assert (
            storage._connection.execute(
                "SELECT version FROM schema_meta WHERE singleton = 1"
            ).fetchone()[0]
            == 5
        )
        assert (
            storage._connection.execute(
                "SELECT COUNT(*) FROM storage_leases"
            ).fetchone()[0]
            == 1
        )
    finally:
        storage.close()


def test_failed_v3_verification_rolls_back_then_retries_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_v2_database(tmp_path)
    create_speaker_profiles = Storage._create_v3_speaker_profiles

    def create_invalid_speaker_profiles(storage: Storage) -> None:
        create_speaker_profiles(storage)
        storage._connection.execute(
            "DROP INDEX speaker_profiles_name_key_uq"
        )

    with monkeypatch.context() as failure:
        failure.setattr(
            Storage,
            "_create_v3_speaker_profiles",
            create_invalid_speaker_profiles,
        )
        with pytest.raises(StorageSchemaError, match="name index"):
            Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)

    connection = sqlite3.connect(tmp_path / "botified-asr.sqlite3")
    try:
        assert (
            connection.execute(
                "SELECT version FROM schema_meta WHERE singleton = 1"
            ).fetchone()[0]
            == 2
        )
        assert (
            connection.execute(
                """
            SELECT COUNT(*) FROM sqlite_master
            WHERE name IN (
                'speaker_profiles',
                'speaker_profiles_name_key_uq'
            )
            """
            ).fetchone()[0]
            == 0
        )
    finally:
        connection.close()

    Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40).close()
    connection = sqlite3.connect(tmp_path / "botified-asr.sqlite3")
    try:
        assert (
            connection.execute(
                "SELECT version FROM schema_meta WHERE singleton = 1"
            ).fetchone()[0]
            == 5
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM speaker_profiles"
            ).fetchone()[0]
            == 0
        )
    finally:
        connection.close()


@pytest.mark.parametrize(
    "mutation",
    (
        "missing-table",
        "wrong-table",
        "wrong-column",
        "missing-index",
        "wrong-index-column",
        "wrong-column-type",
        "wrong-column-nullability",
    ),
)
def test_v3_schema_verifier_rejects_missing_or_wrong_profile_structure(
    tmp_path: Path,
    mutation: str,
) -> None:
    Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40).close()
    connection = sqlite3.connect(tmp_path / "botified-asr.sqlite3")
    try:
        if mutation == "wrong-column-type":
            _recreate_speaker_profiles_with_column_drift(
                connection,
                drift="embedding-type",
            )
        elif mutation == "wrong-column-nullability":
            _recreate_speaker_profiles_with_column_drift(
                connection,
                drift="description-nullability",
            )
        else:
            statements = {
                "missing-table": ("DROP TABLE speaker_profiles",),
                "wrong-table": (
                    "ALTER TABLE speaker_profiles RENAME TO wrong_speaker_profiles",
                ),
                "wrong-column": (
                    "ALTER TABLE speaker_profiles "
                    "RENAME COLUMN updated_at TO wrong_updated_at",
                ),
                "missing-index": ("DROP INDEX speaker_profiles_name_key_uq",),
                "wrong-index-column": (
                    "DROP INDEX speaker_profiles_name_key_uq",
                    "CREATE UNIQUE INDEX speaker_profiles_name_key_uq "
                    "ON speaker_profiles(name)",
                ),
            }[mutation]
            for statement in statements:
                connection.execute(statement)
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(StorageSchemaError):
        Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)


def test_v3_schema_verifier_allows_unrelated_tables_and_indexes(
    tmp_path: Path,
) -> None:
    Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40).close()
    connection = sqlite3.connect(tmp_path / "botified-asr.sqlite3")
    try:
        connection.execute("CREATE TABLE extension_data (key TEXT PRIMARY KEY)")
        connection.execute(
            "CREATE INDEX extension_speaker_name_idx ON speaker_profiles(name)"
        )
        connection.commit()
    finally:
        connection.close()

    Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40).close()


@pytest.mark.parametrize(
    "constraint",
    ("dimension", "embedding", "sample-count"),
)
def test_speaker_profile_table_enforces_only_local_storage_checks(
    tmp_path: Path,
    constraint: str,
) -> None:
    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    try:
        _insert_speaker_profile(
            storage._connection,
            id="bad",
            name=" A ",
            name_key="a",
            embedding_model_id="bad",
            embedding_model_revision="short",
            embedding_policy_fingerprint="opaque",
            created_at="later",
            updated_at="earlier",
        )
        assert (
            storage._connection.execute(
                "SELECT COUNT(*) FROM speaker_profiles"
            ).fetchone()[0]
            == 1
        )

        if constraint == "dimension":
            invalid_rows = ({"embedding_dimension": 0},)
        elif constraint == "embedding":
            invalid_rows = (
                {"embedding": "12345678"},
                {"embedding": b"\x00" * 7},
            )
        else:
            invalid_rows = (
                {"sample_count": 1},
                {"sample_count": 6},
            )

        for invalid in invalid_rows:
            with pytest.raises(sqlite3.IntegrityError):
                _insert_speaker_profile(storage._connection, **invalid)
    finally:
        storage.close()


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
        Storage(data_dir, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)

    assert (outside / "keep.txt").read_bytes() == b"keep"
    assert not (data_dir / "botified-asr.sqlite3").exists()


def test_controlled_file_symlink_fails_closed_on_startup_and_resolve(
    tmp_path: Path,
) -> None:
    startup_dir = tmp_path / "startup"
    first = Storage(startup_dir, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    lease = first.begin_upload("transcription")
    first.close()
    target = first.staging_dir / f"{'b' * 32}.partial"
    target.write_bytes(b"target")
    lease.path.unlink()
    lease.path.symlink_to(target.name)

    with pytest.raises(StorageSchemaError, match="corrupt storage lease"):
        Storage(startup_dir, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
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
    storage = Storage(resolve_dir, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
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
    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
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
        current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40,
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
        current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40,
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
        current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: RESERVATION_QUANTUM,
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
    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
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
    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
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

    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
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

    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
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

    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
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
    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
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
    first = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    lease = first.begin_upload("transcription")
    first.append(lease, b"unfinished")
    assert lease.path.exists()
    first.close()

    recovered = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    try:
        assert recovered.active_upload_count() == 0
        assert recovered.total_reserved_bytes() == 0
        assert not lease.path.exists()
    finally:
        recovered.close()


def test_abort_is_idempotent_and_removes_file(tmp_path: Path) -> None:
    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
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
        current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40,
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
    first = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    lease = first.begin_upload("transcription")
    first.append(lease, b"complete-but-not-promoted")
    input_ref = first.seal_upload(lease)
    assert input_ref.path.exists()
    first.close()

    recovered = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    try:
        assert recovered.active_upload_count() == 0
        assert recovered.total_reserved_bytes() == 0
        assert not input_ref.path.exists()
    finally:
        recovered.close()


def test_completion_records_incremental_content_hash(tmp_path: Path) -> None:
    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
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
    first = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    second = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
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
        current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 2 * RESERVATION_QUANTUM,
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
        storage = Storage(directory, configured, current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
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
        current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40,
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
        Storage(tmp_path, configured, current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
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

    recovered = Storage(tmp_path, configured, current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
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
        tmp_path, configured, current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40
    )
    recovered_again.close()

    assert outside.read_bytes() == b"safe"
    assert unrelated.read_bytes() == b"keep"
