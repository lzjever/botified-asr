from __future__ import annotations

import hashlib
import os
import re
import shutil
import sqlite3
import stat
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Callable, Iterator

from botified_asr.canonical_options import parse_canonical_options_json
from botified_asr.config import RESERVATION_QUANTUM, LimitsConfig
from botified_asr.job_fingerprints import build_request_fingerprints
from botified_asr.jobs import (
    DurableJob,
    JobCancellationRequestedError,
    JobPhase,
    JobProgressOutcome,
    JobSuccessOutcome,
    JobStatus,
    JobTerminalOutcome,
    QueuedJobSpec,
    StaleJobAttemptError,
    generate_attempt_token,
    generate_job_id,
    validate_job_id,
)
from botified_asr.result_artifact import (
    CanonicalArtifactError,
    ResultEnvelopeManifest,
    ResultEnvelopeReader,
)
from botified_asr.speaker_profiles import (
    KEEP_EXISTING,
    SpeakerEmbedding,
    SpeakerEmbeddingReplacement,
    SpeakerProfile,
    SpeakerProfileUpdate,
)
from botified_asr.speaker_snapshot import (
    resolve_selected_speaker_snapshot,
    serialize_selected_speaker_snapshot,
)
from botified_asr.speakers import SpeakerEmbeddingPolicy

SCHEMA_VERSION = 5
MAX_SPEAKER_PROFILES = 256
LEASE_ID_PATTERN = re.compile(
    r"^(?:[0-9a-f]{32}|[0-9A-HJKMNP-TV-Z]{8})$"
)
STAGING_NAME_PATTERN = re.compile(
    r"^(?:[0-9a-f]{32}|[0-9A-HJKMNP-TV-Z]{8})\.(?:partial|ready)$"
)
ARTIFACT_NAME_PATTERN = re.compile(
    r"^(?:[0-9a-f]{32}|[0-9A-HJKMNP-TV-Z]{8})\.(?:partial|complete)$"
)
LEASE_TYPES = {"upload", "artifact"}
ARTIFACT_KINDS = {"segment_jsonl", "result_complete"}
_RUNTIME_JOB_FAILURE_CODES = {
    "invalid_audio",
    "audio_too_long",
    "long_audio_requires_vad",
    "too_many_speakers",
    "invalid_model_output",
    "pipeline_not_ready",
    "internal_error",
}
_SPEAKER_PROFILE_COLUMNS = """
    id, name, name_key, description, embedding,
    embedding_model_id, embedding_model_revision,
    embedding_dimension, embedding_policy_fingerprint,
    sample_count, created_at, updated_at
"""
_TRANSCRIPTION_JOB_COLUMNS = """
    id, phase, status, input_lease_id, canonical_options_json,
    selected_speaker_snapshot, snapshot_sha256, input_size_bytes,
    effective_max_audio_samples, effective_direct_max_audio_samples,
    total_samples, processed_samples, request_fingerprint,
    processor_fingerprint, attempt_no, attempt_token, owner_generation,
    crash_recoveries, cancel_requested, result_lease_id, error_code,
    input_cleanup_pending, created_at, started_at, finished_at
"""
_EXACT_SEALED_JOB_RESULT_WHERE = """
    id = ? AND lease_type = 'artifact'
    AND resource_kind = 'result_complete'
    AND owner_kind = 'job' AND owner_id = ?
    AND phase = 'sealed' AND controlled_path = ?
    AND reserved_bytes = ? AND actual_bytes = ?
    AND content_sha256 = ?
"""
_V4_TRANSCRIPTION_JOBS_DDL = """CREATE TABLE transcription_jobs (
                id TEXT PRIMARY KEY NOT NULL,
                phase TEXT NOT NULL,
                status TEXT,
                input_lease_id TEXT,
                canonical_options_json TEXT,
                selected_speaker_snapshot BLOB,
                snapshot_sha256 TEXT,
                input_size_bytes INTEGER,
                total_samples INTEGER,
                processed_samples INTEGER NOT NULL,
                request_fingerprint TEXT,
                processor_fingerprint TEXT,
                attempt_no INTEGER NOT NULL,
                attempt_token TEXT,
                owner_generation TEXT,
                crash_recoveries INTEGER NOT NULL,
                cancel_requested INTEGER NOT NULL,
                result_lease_id TEXT,
                error_code TEXT,
                input_cleanup_pending INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                CHECK (length(id) = 8),
                CHECK (id NOT GLOB '*[^0-9A-HJKMNP-TV-Z]*'),
                CHECK (phase IN ('receiving', 'visible', 'deleting')),
                CHECK (
                    status IS NULL
                    OR status IN (
                        'queued', 'running', 'succeeded', 'failed', 'cancelled'
                    )
                ),
                CHECK (
                    (phase = 'receiving' AND status IS NULL)
                    OR (phase = 'visible' AND status IS NOT NULL)
                    OR (
                        phase = 'deleting'
                        AND (
                            status IS NULL
                            OR status IN ('succeeded', 'failed', 'cancelled')
                        )
                    )
                ),
                CHECK (
                    snapshot_sha256 IS NULL
                    OR (
                        length(snapshot_sha256) = 64
                        AND snapshot_sha256 NOT GLOB '*[^0-9a-f]*'
                    )
                ),
                CHECK (input_size_bytes IS NULL OR input_size_bytes >= 0),
                CHECK (total_samples IS NULL OR total_samples >= 0),
                CHECK (processed_samples >= 0),
                CHECK (
                    request_fingerprint IS NULL
                    OR (
                        length(request_fingerprint) = 64
                        AND request_fingerprint NOT GLOB '*[^0-9a-f]*'
                    )
                ),
                CHECK (
                    processor_fingerprint IS NULL
                    OR (
                        length(processor_fingerprint) = 64
                        AND processor_fingerprint NOT GLOB '*[^0-9a-f]*'
                    )
                ),
                CHECK (attempt_no >= 0),
                CHECK (crash_recoveries BETWEEN 0 AND 1),
                CHECK (cancel_requested IN (0, 1)),
                CHECK (input_cleanup_pending IN (0, 1))
            )"""
_V4_SHUTDOWN_MARKER_DDL = """CREATE TABLE shutdown_marker (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                generation TEXT NOT NULL CHECK (length(generation) > 0),
                created_at TEXT NOT NULL
            )"""
_V5_TRANSCRIPTION_JOBS_DDL = _V4_TRANSCRIPTION_JOBS_DDL.replace(
    """                input_size_bytes INTEGER,
                total_samples INTEGER,""",
    """                input_size_bytes INTEGER,
                effective_max_audio_samples INTEGER,
                effective_direct_max_audio_samples INTEGER,
                total_samples INTEGER,""",
).replace(
    """                CHECK (input_size_bytes IS NULL OR input_size_bytes >= 0),
                CHECK (total_samples IS NULL OR total_samples >= 0),""",
    """                CHECK (input_size_bytes IS NULL OR input_size_bytes >= 0),
                CHECK (
                    effective_max_audio_samples IS NULL
                    OR effective_max_audio_samples BETWEEN 1 AND 691200000
                ),
                CHECK (
                    effective_direct_max_audio_samples IS NULL
                    OR effective_direct_max_audio_samples BETWEEN 1 AND 480000
                ),
                CHECK (
                    (
                        effective_max_audio_samples IS NULL
                        AND effective_direct_max_audio_samples IS NULL
                    )
                    OR (
                        effective_max_audio_samples IS NOT NULL
                        AND effective_direct_max_audio_samples IS NOT NULL
                        AND effective_direct_max_audio_samples
                            <= effective_max_audio_samples
                    )
                ),
                CHECK (total_samples IS NULL OR total_samples >= 0),""",
)
_V4_TRANSCRIPTION_JOB_COLUMNS = (
    ("id", "TEXT", 1, None, 1),
    ("phase", "TEXT", 1, None, 0),
    ("status", "TEXT", 0, None, 0),
    ("input_lease_id", "TEXT", 0, None, 0),
    ("canonical_options_json", "TEXT", 0, None, 0),
    ("selected_speaker_snapshot", "BLOB", 0, None, 0),
    ("snapshot_sha256", "TEXT", 0, None, 0),
    ("input_size_bytes", "INTEGER", 0, None, 0),
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
_V5_TRANSCRIPTION_JOB_COLUMNS = (
    *_V4_TRANSCRIPTION_JOB_COLUMNS[:8],
    ("effective_max_audio_samples", "INTEGER", 0, None, 0),
    ("effective_direct_max_audio_samples", "INTEGER", 0, None, 0),
    *_V4_TRANSCRIPTION_JOB_COLUMNS[8:],
)
_SHUTDOWN_MARKER_COLUMNS = (
    ("singleton", "INTEGER", 0, None, 1),
    ("generation", "TEXT", 1, None, 0),
    ("created_at", "TEXT", 1, None, 0),
)


class StorageSchemaError(RuntimeError):
    pass


class StorageAdmissionError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class SpeakerProfileStorageError(RuntimeError):
    pass


class SpeakerProfileNameConflictError(SpeakerProfileStorageError):
    def __init__(self) -> None:
        super().__init__("speaker profile name already exists")


class SpeakerProfileLimitReachedError(SpeakerProfileStorageError):
    def __init__(self) -> None:
        super().__init__("speaker profile limit reached")


class SpeakerProfileIdCollisionError(SpeakerProfileStorageError):
    def __init__(self) -> None:
        super().__init__("speaker profile ID already exists")


@dataclass
class UploadLease:
    id: str
    resource_kind: str
    owner_kind: str
    owner_id: str
    path: Path
    reserved_bytes: int
    actual_bytes: int = 0
    content_sha256: str | None = None
    _hasher: object = field(default_factory=hashlib.sha256, repr=False)
    _checkpointed_bytes: int = field(default=0, repr=False)
    _state: str = field(default="writing", repr=False)


@dataclass(frozen=True)
class InputRef:
    id: str
    resource_kind: str
    owner_kind: str
    owner_id: str
    path: Path
    actual_bytes: int
    content_sha256: str


@dataclass
class JobUploadLease:
    id: str
    resource_kind: str
    owner_kind: str
    owner_id: str
    path: Path
    reserved_bytes: int
    actual_bytes: int = 0
    content_sha256: str | None = None
    _hasher: object = field(default_factory=hashlib.sha256, repr=False)
    _checkpointed_bytes: int = field(default=0, repr=False)
    _state: str = field(default="writing", repr=False)


@dataclass(frozen=True)
class JobInputRef:
    id: str
    resource_kind: str
    owner_kind: str
    owner_id: str
    path: Path
    actual_bytes: int
    content_sha256: str


@dataclass
class ReservedByteWriter:
    id: str
    resource_kind: str
    owner_kind: str
    owner_id: str
    path: Path
    reserved_bytes: int
    actual_bytes: int = 0
    content_sha256: str | None = None
    _hasher: object = field(default_factory=hashlib.sha256, repr=False)
    _checkpointed_bytes: int = field(default=0, repr=False)
    _state: str = field(default="writing", repr=False)


@dataclass(frozen=True)
class ArtifactRef:
    id: str
    resource_kind: str
    owner_kind: str
    owner_id: str
    path: Path
    actual_bytes: int
    content_sha256: str


class StoredJobResult:
    def __init__(
        self,
        reader: ResultEnvelopeReader,
        handle: BinaryIO,
    ) -> None:
        self._reader = reader
        self._handle = handle
        self._lock = threading.Lock()
        self._claimed = False
        self._closed = False

    def iter_body(self) -> Iterator[bytes]:
        with self._lock:
            if self._closed:
                raise RuntimeError("stored job result is closed")
            if self._claimed:
                raise RuntimeError(
                    "stored job result body was already requested"
                )
            self._claimed = True
        iterator = self._reader.iter_body()

        def chunks() -> Iterator[bytes]:
            try:
                yield from iterator
            finally:
                try:
                    close = getattr(iterator, "close", None)
                    if close is not None:
                        close()
                finally:
                    self.close()

        return chunks()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._handle.close()
            self._closed = True


class Storage:
    """SQLite-owned generic storage leases and controlled files."""

    def __init__(
        self,
        data_dir: str | Path,
        limits: LimitsConfig,
        *,
        current_processor_fingerprint: str,
        free_bytes: Callable[[Path], int] | None = None,
    ) -> None:
        if type(current_processor_fingerprint) is not str:
            raise TypeError("current processor fingerprint must be a string")
        if re.fullmatch(r"[0-9a-f]{64}", current_processor_fingerprint) is None:
            raise ValueError("current processor fingerprint is invalid")
        self._current_processor_fingerprint = current_processor_fingerprint
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.staging_dir = self.data_dir / "staging"
        self.artifact_dir = self.data_dir / "artifacts"
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._prepare_control_directory(self.staging_dir)
        self._prepare_control_directory(self.artifact_dir)
        self.data_dir.chmod(0o700)
        self.limits = limits
        self._free_bytes = free_bytes or (
            lambda path: shutil.disk_usage(path).free
        )
        self._lock = threading.RLock()
        self._closed = False
        self._files: dict[str, BinaryIO] = {}
        self._connection = sqlite3.connect(
            self.data_dir / "botified-asr.sqlite3",
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        try:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            Path(
                self._connection.execute("PRAGMA database_list").fetchone()[2]
            ).chmod(0o600)
            self._initialize_schema()
            self._verify_active_processor_fingerprints()
            self._reconcile_startup()
        except BaseException:
            self._connection.close()
            raise

    def _verify_active_processor_fingerprints(self) -> None:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT processor_fingerprint
                FROM transcription_jobs
                WHERE phase = 'visible'
                  AND status IN ('queued', 'running')
                """
            ).fetchall()
        if any(
            row["processor_fingerprint"]
            != self._current_processor_fingerprint
            for row in rows
        ):
            raise StorageSchemaError("processor_fingerprint_mismatch")

    def _prepare_control_directory(self, path: Path) -> None:
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise StorageSchemaError(
                f"storage control path {path.name} must be a real directory"
            )
        if path.resolve(strict=False).parent != self.data_dir:
            raise StorageSchemaError(
                f"storage control path {path.name} escapes data directory"
            )
        path.mkdir(mode=0o700, exist_ok=True)
        path.chmod(0o700)

    def _initialize_schema(self) -> None:
        with self._lock:
            application_tables = {
                row["name"]
                for row in self._connection.execute(
                    """
                    SELECT name FROM sqlite_master
                    WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                    """
                )
            }
        has_meta = "schema_meta" in application_tables
        if not has_meta:
            if application_tables:
                raise StorageSchemaError(
                    "unversioned storage database contains application tables"
                )
            with self._transaction():
                self._connection.execute(
                    """
                    CREATE TABLE schema_meta (
                        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                        version INTEGER NOT NULL
                    )
                    """
                )
                self._create_v2_ledger()
                self._create_v3_speaker_profiles()
                self._create_v5_job_foundation()
                self._connection.execute(
                    "INSERT INTO schema_meta(singleton, version) VALUES (1, ?)",
                    (SCHEMA_VERSION,),
                )
            return

        with self._lock:
            row = self._connection.execute(
                "SELECT version FROM schema_meta WHERE singleton = 1"
            ).fetchone()
        if row is None:
            raise StorageSchemaError("storage schema version is missing")
        version = int(row["version"])
        if version == SCHEMA_VERSION:
            self._verify_v5_schema()
            return
        if version == 1:
            self._migrate_v1_to_v2()
            version = 2
        if version == 2:
            self._migrate_v2_to_v3()
            version = 3
        if version == 3:
            self._migrate_v3_to_v4()
            version = 4
        if version == 4:
            self._migrate_v4_to_v5()
            return
        raise StorageSchemaError(
            f"unsupported storage schema version: {version}"
        )

    def _create_v2_ledger(self) -> None:
        self._connection.execute(
            """
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
                reserved_bytes INTEGER NOT NULL
                    CHECK (reserved_bytes >= 0),
                actual_bytes INTEGER NOT NULL
                    CHECK (
                        actual_bytes >= 0
                        AND actual_bytes <= reserved_bytes
                    ),
                content_sha256 TEXT,
                created_at TEXT NOT NULL DEFAULT (
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                )
            )
            """
        )
        self._connection.execute(
            """
            CREATE INDEX storage_leases_type_phase_idx
            ON storage_leases(lease_type, phase)
            """
        )
        self._connection.execute(
            """
            CREATE INDEX storage_leases_owner_idx
            ON storage_leases(owner_kind, owner_id)
            """
        )

    def _verify_v2_ledger(self) -> None:
        with self._lock:
            legacy = self._connection.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'upload_leases'
                """
            ).fetchone()
            exists = self._connection.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'storage_leases'
                """
            ).fetchone()
        if legacy is not None:
            raise StorageSchemaError(
                "storage schema v2 has unexpected legacy upload ledger"
            )
        if exists is None:
            raise StorageSchemaError("storage schema v2 ledger is missing")
        expected_columns = {
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
        with self._lock:
            columns = {
                row["name"]
                for row in self._connection.execute(
                    "PRAGMA table_info(storage_leases)"
                )
            }
        if columns != expected_columns:
            raise StorageSchemaError(
                "storage schema v2 ledger has unexpected columns"
            )

    def _create_v3_speaker_profiles(self) -> None:
        self._connection.execute(
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
            )
            """
        )
        self._connection.execute(
            """
            CREATE UNIQUE INDEX speaker_profiles_name_key_uq
            ON speaker_profiles(name_key)
            """
        )

    def _verify_v3_schema(self) -> None:
        self._verify_v2_ledger()
        expected_columns = (
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
        with self._lock:
            exists = self._connection.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'speaker_profiles'
                """
            ).fetchone()
            columns = tuple(
                (
                    row["name"],
                    row["type"],
                    row["notnull"],
                    row["dflt_value"],
                    row["pk"],
                )
                for row in self._connection.execute(
                    "PRAGMA table_info(speaker_profiles)"
                )
            )
            indexes = {
                row["name"]: row
                for row in self._connection.execute(
                    "PRAGMA index_list(speaker_profiles)"
                )
            }
        if exists is None:
            raise StorageSchemaError(
                "storage schema v3 speaker profile table is missing"
            )
        if columns != expected_columns:
            raise StorageSchemaError(
                "storage schema v3 speaker profile table has unexpected columns"
            )
        name_key_index = indexes.get("speaker_profiles_name_key_uq")
        if (
            name_key_index is None
            or name_key_index["unique"] != 1
            or name_key_index["partial"] != 0
        ):
            raise StorageSchemaError(
                "storage schema v3 speaker profile name index is invalid"
            )
        with self._lock:
            indexed_columns = tuple(
                row["name"]
                for row in self._connection.execute(
                    "PRAGMA index_info(speaker_profiles_name_key_uq)"
                )
            )
        if indexed_columns != ("name_key",):
            raise StorageSchemaError(
                "storage schema v3 speaker profile name index is invalid"
            )

    def _migrate_v1_to_v2(self) -> None:
        with self._transaction():
            old_table = self._connection.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'upload_leases'
                """
            ).fetchone()
            if old_table is None:
                raise StorageSchemaError(
                    "storage schema v1 upload ledger is missing"
                )
            rows = self._connection.execute(
                "SELECT * FROM upload_leases"
            ).fetchall()
            migrated_rows: list[tuple[object, ...]] = []
            for row in rows:
                lease_id = row["id"]
                staging_path = row["staging_path"]
                suffix = Path(staging_path).suffix
                phase = (
                    "writing"
                    if suffix == ".partial"
                    else "sealed"
                    if suffix == ".ready"
                    else None
                )
                expected_path = (
                    self.staging_dir / f"{lease_id}{suffix}"
                    if phase is not None
                    else None
                )
                if (
                    not isinstance(lease_id, str)
                    or LEASE_ID_PATTERN.fullmatch(lease_id) is None
                    or row["phase"] != "receiving"
                    or not isinstance(row["kind"], str)
                    or not row["kind"]
                    or expected_path is None
                    or staging_path != str(expected_path)
                    or not self._is_controlled_path(
                        lease_id, "upload", phase, expected_path
                    )
                    or row["reserved_bytes"] < 0
                    or row["actual_bytes"] < 0
                    or row["actual_bytes"] > row["reserved_bytes"]
                ):
                    raise StorageSchemaError("invalid v1 upload lease")
                migrated_rows.append(
                    (
                        lease_id,
                        "upload",
                        row["kind"],
                        "legacy",
                        lease_id,
                        phase,
                        staging_path,
                        row["reserved_bytes"],
                        row["actual_bytes"],
                        row["content_sha256"],
                        row["created_at"],
                    )
                )
            self._create_v2_ledger()
            self._connection.executemany(
                """
                INSERT INTO storage_leases(
                    id, lease_type, resource_kind, owner_kind, owner_id,
                    phase, controlled_path, reserved_bytes, actual_bytes,
                    content_sha256, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                migrated_rows,
            )
            self._connection.execute("DROP TABLE upload_leases")
            changed = self._connection.execute(
                """
                UPDATE schema_meta SET version = 2
                WHERE singleton = 1 AND version = 1
                """,
            ).rowcount
            if changed != 1:
                raise StorageSchemaError(
                    "storage schema version changed during migration"
                )

    def _migrate_v2_to_v3(self) -> None:
        with self._transaction():
            self._verify_v2_ledger()
            self._create_v3_speaker_profiles()
            changed = self._connection.execute(
                """
                UPDATE schema_meta SET version = 3
                WHERE singleton = 1 AND version = 2
                """
            ).rowcount
            if changed != 1:
                raise StorageSchemaError(
                    "storage schema version changed during migration"
                )
            self._verify_v3_schema()

    def _create_v4_job_foundation(self) -> None:
        self._connection.execute(_V4_TRANSCRIPTION_JOBS_DDL)
        self._connection.execute(
            """
            CREATE INDEX transcription_jobs_fifo_idx
            ON transcription_jobs(phase, status, created_at, id)
            """
        )
        self._connection.execute(
            """
            CREATE INDEX transcription_jobs_retention_idx
            ON transcription_jobs(phase, status, finished_at, id)
            """
        )
        self._connection.execute(_V4_SHUTDOWN_MARKER_DDL)

    def _create_v5_job_foundation(self) -> None:
        self._connection.execute(_V5_TRANSCRIPTION_JOBS_DDL)
        self._connection.execute(
            """
            CREATE INDEX transcription_jobs_fifo_idx
            ON transcription_jobs(phase, status, created_at, id)
            """
        )
        self._connection.execute(
            """
            CREATE INDEX transcription_jobs_retention_idx
            ON transcription_jobs(phase, status, finished_at, id)
            """
        )
        self._connection.execute(_V4_SHUTDOWN_MARKER_DDL)

    def _verify_v4_schema(self) -> None:
        self._verify_job_foundation_schema(
            version=4,
            expected_job_ddl=_V4_TRANSCRIPTION_JOBS_DDL,
            expected_job_columns=_V4_TRANSCRIPTION_JOB_COLUMNS,
        )

    def _verify_v5_schema(self) -> None:
        self._verify_job_foundation_schema(
            version=5,
            expected_job_ddl=_V5_TRANSCRIPTION_JOBS_DDL,
            expected_job_columns=_V5_TRANSCRIPTION_JOB_COLUMNS,
        )

    def _verify_job_foundation_schema(
        self,
        *,
        version: int,
        expected_job_ddl: str,
        expected_job_columns: tuple[tuple[object, ...], ...],
    ) -> None:
        self._verify_v3_schema()
        with self._lock:
            job_exists = self._connection.execute(
                """
                SELECT sql FROM sqlite_master
                WHERE type = 'table' AND name = 'transcription_jobs'
                """
            ).fetchone()
            marker_exists = self._connection.execute(
                """
                SELECT sql FROM sqlite_master
                WHERE type = 'table' AND name = 'shutdown_marker'
                """
            ).fetchone()
            job_columns = self._table_columns("transcription_jobs")
            marker_columns = self._table_columns("shutdown_marker")
            job_indexes = {
                row["name"]: row
                for row in self._connection.execute(
                    "PRAGMA index_list(transcription_jobs)"
                )
            }
        if job_exists is None:
            raise StorageSchemaError(
                f"storage schema v{version} transcription job table is missing"
            )
        if job_exists["sql"] != expected_job_ddl:
            raise StorageSchemaError(
                f"storage schema v{version} transcription job table "
                "has unexpected definition"
            )
        if job_columns != expected_job_columns:
            raise StorageSchemaError(
                f"storage schema v{version} transcription job table "
                "has unexpected columns"
            )
        if marker_exists is None:
            raise StorageSchemaError(
                f"storage schema v{version} shutdown marker table is missing"
            )
        if marker_exists["sql"] != _V4_SHUTDOWN_MARKER_DDL:
            raise StorageSchemaError(
                f"storage schema v{version} shutdown marker table "
                "has unexpected definition"
            )
        if marker_columns != _SHUTDOWN_MARKER_COLUMNS:
            raise StorageSchemaError(
                f"storage schema v{version} shutdown marker table "
                "has unexpected columns"
            )
        expected_indexes = {
            "transcription_jobs_fifo_idx": (
                "phase",
                "status",
                "created_at",
                "id",
            ),
            "transcription_jobs_retention_idx": (
                "phase",
                "status",
                "finished_at",
                "id",
            ),
        }
        for name, expected_columns in expected_indexes.items():
            index = job_indexes.get(name)
            if index is None or index["unique"] != 0 or index["partial"] != 0:
                raise StorageSchemaError(
                    f"storage schema v{version} job index {name} is invalid"
                )
            with self._lock:
                indexed_columns = tuple(
                    row["name"]
                    for row in self._connection.execute(f"PRAGMA index_info({name})")
                )
            if indexed_columns != expected_columns:
                raise StorageSchemaError(
                    f"storage schema v{version} job index {name} is invalid"
                )

    def _table_columns(
        self,
        table: str,
    ) -> tuple[tuple[object, ...], ...]:
        return tuple(
            (
                row["name"],
                row["type"],
                row["notnull"],
                row["dflt_value"],
                row["pk"],
            )
            for row in self._connection.execute(f"PRAGMA table_info({table})")
        )

    def _migrate_v3_to_v4(self) -> None:
        with self._transaction():
            self._verify_v3_schema()
            self._create_v4_job_foundation()
            changed = self._connection.execute(
                """
                UPDATE schema_meta SET version = 4
                WHERE singleton = 1 AND version = 3
                """
            ).rowcount
            if changed != 1:
                raise StorageSchemaError(
                    "storage schema version changed during migration"
                )
            self._verify_v4_schema()

    def _migrate_v4_to_v5(self) -> None:
        with self._transaction():
            self._verify_v4_schema()
            if (
                self._connection.execute(
                    "SELECT 1 FROM transcription_jobs LIMIT 1"
                ).fetchone()
                is not None
            ):
                raise StorageSchemaError(
                    "storage schema v4 contains transcription jobs"
                )
            self._connection.execute("DROP INDEX transcription_jobs_fifo_idx")
            self._connection.execute(
                "DROP INDEX transcription_jobs_retention_idx"
            )
            self._connection.execute(
                "ALTER TABLE transcription_jobs RENAME TO transcription_jobs_v4"
            )
            self._connection.execute(_V5_TRANSCRIPTION_JOBS_DDL)
            self._connection.execute(
                """
                CREATE INDEX transcription_jobs_fifo_idx
                ON transcription_jobs(phase, status, created_at, id)
                """
            )
            self._connection.execute(
                """
                CREATE INDEX transcription_jobs_retention_idx
                ON transcription_jobs(phase, status, finished_at, id)
                """
            )
            self._connection.execute("DROP TABLE transcription_jobs_v4")
            changed = self._connection.execute(
                """
                UPDATE schema_meta SET version = 5
                WHERE singleton = 1 AND version = 4
                """
            ).rowcount
            if changed != 1:
                raise StorageSchemaError(
                    "storage schema version changed during migration"
                )
            self._verify_v5_schema()

    def create_speaker_profile(
        self,
        profile: SpeakerProfile,
    ) -> SpeakerProfile:
        if not isinstance(profile, SpeakerProfile):
            raise TypeError("create_speaker_profile requires a SpeakerProfile")
        values = _encode_speaker_profile(profile)
        with self._transaction():
            if (
                self._connection.execute(
                    "SELECT 1 FROM speaker_profiles WHERE id = ?",
                    (profile.id,),
                ).fetchone()
                is not None
            ):
                raise SpeakerProfileIdCollisionError
            if (
                self._connection.execute(
                    "SELECT 1 FROM speaker_profiles WHERE name_key = ?",
                    (profile.name_key,),
                ).fetchone()
                is not None
            ):
                raise SpeakerProfileNameConflictError
            count = self._connection.execute(
                "SELECT COUNT(*) FROM speaker_profiles"
            ).fetchone()[0]
            if count >= MAX_SPEAKER_PROFILES:
                raise SpeakerProfileLimitReachedError
            try:
                self._connection.execute(
                    f"""
                    INSERT INTO speaker_profiles({_SPEAKER_PROFILE_COLUMNS})
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
            except sqlite3.IntegrityError as error:
                raise StorageSchemaError(
                    "speaker profile insert violated storage schema"
                ) from error
        return profile

    def get_speaker_profile(
        self,
        profile_id: str,
    ) -> SpeakerProfile | None:
        if not isinstance(profile_id, str):
            raise TypeError("speaker profile ID must be a string")
        with self._lock:
            row = self._connection.execute(
                f"""
                SELECT {_SPEAKER_PROFILE_COLUMNS}
                FROM speaker_profiles
                WHERE id = ?
                """,
                (profile_id,),
            ).fetchone()
        return None if row is None else _decode_speaker_profile(row)

    def list_speaker_profiles(self) -> tuple[SpeakerProfile, ...]:
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT {_SPEAKER_PROFILE_COLUMNS}
                FROM speaker_profiles
                ORDER BY created_at, id
                """
            ).fetchall()
        return tuple(_decode_speaker_profile(row) for row in rows)

    def update_speaker_profile(
        self,
        profile_id: str,
        update: SpeakerProfileUpdate,
    ) -> SpeakerProfile | None:
        if not isinstance(profile_id, str):
            raise TypeError("speaker profile ID must be a string")
        if type(update) is not SpeakerProfileUpdate:
            raise TypeError(
                "update_speaker_profile requires a SpeakerProfileUpdate"
            )
        with self._transaction():
            row = self._connection.execute(
                f"""
                SELECT {_SPEAKER_PROFILE_COLUMNS}
                FROM speaker_profiles
                WHERE id = ?
                """,
                (profile_id,),
            ).fetchone()
            if row is None:
                return None
            current = _decode_speaker_profile(row)

            description = (
                current.description
                if update.description is KEEP_EXISTING
                else update.description
            )
            if update.embedding is KEEP_EXISTING:
                embedding = current.embedding
                embedding_model_id = current.embedding_model_id
                embedding_model_revision = current.embedding_model_revision
                embedding_dimension = current.embedding_dimension
                embedding_policy_fingerprint = (
                    current.embedding_policy_fingerprint
                )
                sample_count = current.sample_count
            elif type(update.embedding) is SpeakerEmbeddingReplacement:
                replacement = update.embedding
                embedding = replacement.embedding
                embedding_model_id = replacement.embedding_model_id
                embedding_model_revision = replacement.embedding_model_revision
                embedding_dimension = replacement.embedding_dimension
                embedding_policy_fingerprint = (
                    replacement.embedding_policy_fingerprint
                )
                sample_count = replacement.sample_count
            else:
                raise TypeError(
                    "speaker profile embedding update is invalid"
                )

            changed = SpeakerProfile(
                id=current.id,
                name=update.name,
                description=description,
                embedding=embedding,
                embedding_model_id=embedding_model_id,
                embedding_model_revision=embedding_model_revision,
                embedding_dimension=embedding_dimension,
                embedding_policy_fingerprint=embedding_policy_fingerprint,
                sample_count=sample_count,
                created_at=current.created_at,
                updated_at=update.updated_at,
            )
            if changed.updated_at < current.updated_at:
                raise ValueError(
                    "speaker profile updated_at must not move backwards"
                )
            if (
                self._connection.execute(
                    """
                    SELECT 1 FROM speaker_profiles
                    WHERE name_key = ? AND id <> ?
                    """,
                    (changed.name_key, changed.id),
                ).fetchone()
                is not None
            ):
                raise SpeakerProfileNameConflictError
            values = _encode_speaker_profile(changed)
            try:
                rowcount = self._connection.execute(
                    """
                    UPDATE speaker_profiles SET
                        name = ?,
                        name_key = ?,
                        description = ?,
                        embedding = ?,
                        embedding_model_id = ?,
                        embedding_model_revision = ?,
                        embedding_dimension = ?,
                        embedding_policy_fingerprint = ?,
                        sample_count = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        values[1],
                        values[2],
                        values[3],
                        values[4],
                        values[5],
                        values[6],
                        values[7],
                        values[8],
                        values[9],
                        values[11],
                        changed.id,
                    ),
                ).rowcount
            except sqlite3.IntegrityError as error:
                raise StorageSchemaError(
                    "speaker profile update violated storage schema"
                ) from error
            if rowcount != 1:
                raise StorageSchemaError(
                    "speaker profile changed during update"
                )
        return changed

    def delete_speaker_profile(self, profile_id: str) -> bool:
        if not isinstance(profile_id, str):
            raise TypeError("speaker profile ID must be a string")
        with self._transaction():
            try:
                rowcount = self._connection.execute(
                    "DELETE FROM speaker_profiles WHERE id = ?",
                    (profile_id,),
                ).rowcount
            except sqlite3.IntegrityError as error:
                raise StorageSchemaError(
                    "speaker profile delete violated storage schema"
                ) from error
            if rowcount not in {0, 1}:
                raise StorageSchemaError(
                    "speaker profile delete affected unexpected rows"
                )
        return rowcount == 1

    def get_speaker_profiles_by_ids(
        self,
        profile_ids: tuple[str, ...],
    ) -> tuple[SpeakerProfile, ...]:
        if not isinstance(profile_ids, tuple) or any(
            not isinstance(profile_id, str) for profile_id in profile_ids
        ):
            raise TypeError("speaker profile IDs must be a tuple of strings")
        if not profile_ids:
            return ()
        placeholders = ", ".join("?" for _ in profile_ids)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT {_SPEAKER_PROFILE_COLUMNS}
                FROM speaker_profiles
                WHERE id IN ({placeholders})
                ORDER BY id
                """,
                profile_ids,
            ).fetchall()
        return tuple(_decode_speaker_profile(row) for row in rows)

    def begin_job_upload(self, created_at: datetime) -> JobUploadLease:
        encoded_created_at = _encode_job_timestamp(created_at)
        reservation = RESERVATION_QUANTUM
        while True:
            job_id = validate_job_id(generate_job_id())
            partial_path = self.staging_dir / f"{job_id}.partial"
            ready_path = self.staging_dir / f"{job_id}.ready"
            with self._transaction():
                collision = (
                    self._connection.execute(
                        "SELECT 1 FROM transcription_jobs WHERE id = ?",
                        (job_id,),
                    ).fetchone()
                    is not None
                    or self._connection.execute(
                        "SELECT 1 FROM storage_leases WHERE id = ?",
                        (job_id,),
                    ).fetchone()
                    is not None
                )
                if collision:
                    continue

                active = self._connection.execute(
                    """
                    SELECT COUNT(*) FROM storage_leases
                    WHERE lease_type = 'upload' AND phase = 'writing'
                    """
                ).fetchone()[0]
                if active >= self.limits.max_active_uploads:
                    raise StorageAdmissionError(
                        "too_many_active_uploads",
                        "too many active uploads",
                    )
                self._admit_delta(reservation)
                self._connection.execute(
                    """
                    INSERT INTO transcription_jobs(
                        id, phase, status, input_lease_id,
                        processed_samples, attempt_no, crash_recoveries,
                        cancel_requested, input_cleanup_pending, created_at
                    ) VALUES (
                        ?, 'receiving', NULL, ?, 0, 0, 0, 0, 0, ?
                    )
                    """,
                    (job_id, job_id, encoded_created_at),
                )
                self._connection.execute(
                    """
                    INSERT INTO storage_leases(
                        id, lease_type, resource_kind, owner_kind, owner_id,
                        phase, controlled_path, reserved_bytes, actual_bytes
                    ) VALUES (
                        ?, 'upload', 'transcription', 'job', ?,
                        'writing', ?, ?, 0
                    )
                    """,
                    (
                        job_id,
                        job_id,
                        str(partial_path),
                        reservation,
                    ),
                )

            try:
                handle = partial_path.open("xb")
            except OSError as error:
                self._compensate_failed_job_open(job_id, partial_path)
                if isinstance(error, FileExistsError):
                    continue
                raise
            if ready_path.exists() or ready_path.is_symlink():
                handle.close()
                try:
                    partial_path.unlink()
                except FileNotFoundError:
                    pass
                _fsync_directory(self.staging_dir)
                self._compensate_failed_job_open(job_id, partial_path)
                continue

            self._files[job_id] = handle
            return JobUploadLease(
                job_id,
                "transcription",
                "job",
                job_id,
                partial_path,
                reservation,
            )

    def append_job_upload(
        self,
        lease: JobUploadLease,
        data: bytes,
    ) -> None:
        if type(lease) is not JobUploadLease:
            raise TypeError(
                "append_job_upload requires a JobUploadLease"
            )
        with self._lock:
            self._require_job_upload_lease(lease, phase="writing")
            self._append_writing(
                lease=lease,
                lease_type="upload",
                data=data,
            )

    def seal_job_upload(self, lease: JobUploadLease) -> JobInputRef:
        if type(lease) is not JobUploadLease:
            raise TypeError(
                "seal_job_upload requires a JobUploadLease"
            )
        with self._lock:
            self._require_job_upload_lease(lease, phase="writing")
            open_handle = self._files.get(lease.id)
            if open_handle is None:
                raise RuntimeError("job upload lease is closed")
            partial_path = lease.path
            sealed_path = self.staging_dir / f"{lease.id}.ready"
            try:
                open_handle.flush()
                os.fsync(open_handle.fileno())
                open_handle.close()
                self._files.pop(lease.id, None)
                os.replace(partial_path, sealed_path)
                _fsync_directory(self.staging_dir)
                digest = lease._hasher.hexdigest()
                with self._transaction():
                    job_row = self._connection.execute(
                        """
                        SELECT phase, status, input_lease_id
                        FROM transcription_jobs WHERE id = ?
                        """,
                        (lease.id,),
                    ).fetchone()
                    if (
                        job_row is None
                        or tuple(job_row)
                        != ("receiving", None, lease.id)
                    ):
                        raise RuntimeError(
                            "job is no longer receiving"
                        )
                    changed = self._connection.execute(
                        """
                        UPDATE storage_leases SET
                            phase = 'sealed',
                            controlled_path = ?,
                            reserved_bytes = ?,
                            actual_bytes = ?,
                            content_sha256 = ?
                        WHERE id = ? AND lease_type = 'upload'
                          AND resource_kind = 'transcription'
                          AND owner_kind = 'job' AND owner_id = ?
                          AND phase = 'writing' AND controlled_path = ?
                          AND reserved_bytes = ?
                          AND actual_bytes BETWEEN 0 AND ?
                          AND content_sha256 IS NULL
                        """,
                        (
                            str(sealed_path),
                            lease.actual_bytes,
                            lease.actual_bytes,
                            digest,
                            lease.id,
                            lease.id,
                            str(partial_path),
                            lease.reserved_bytes,
                            lease.actual_bytes,
                        ),
                    ).rowcount
                    if changed != 1:
                        raise RuntimeError(
                            "job upload lease is no longer writing"
                        )
            except BaseException:
                self._compensate_failed_job_seal(
                    lease,
                    partial_path=partial_path,
                    sealed_path=sealed_path,
                )
                raise
            lease.path = sealed_path
            lease.reserved_bytes = lease.actual_bytes
            lease.content_sha256 = digest
            lease._state = "sealed"
        return JobInputRef(
            lease.id,
            lease.resource_kind,
            lease.owner_kind,
            lease.owner_id,
            sealed_path,
            lease.actual_bytes,
            digest,
        )

    def abort_job_upload(
        self,
        handle: JobUploadLease | JobInputRef,
    ) -> None:
        if type(handle) not in {JobUploadLease, JobInputRef}:
            raise TypeError(
                "abort_job_upload requires a job upload handle"
            )
        writing = type(handle) is JobUploadLease
        if writing and handle._state == "sealed":
            raise RuntimeError("job upload lease is stale")
        if writing and handle._state not in {"writing", "aborted"}:
            raise RuntimeError("job upload lease has invalid state")
        self._validate_job_handle_identity(handle, writing=writing)

        partial_path = self.staging_dir / f"{handle.id}.partial"
        ready_path = self.staging_dir / f"{handle.id}.ready"
        with self._transaction():
            job_row = self._connection.execute(
                """
                SELECT phase, status, input_lease_id
                FROM transcription_jobs WHERE id = ?
                """,
                (handle.id,),
            ).fetchone()
            lease_row = self._connection.execute(
                """
                SELECT lease_type, resource_kind, owner_kind, owner_id,
                       phase, controlled_path, reserved_bytes, actual_bytes,
                       content_sha256
                FROM storage_leases WHERE id = ?
                """,
                (handle.id,),
            ).fetchone()
            if job_row is None and lease_row is None:
                if partial_path.exists() or ready_path.exists():
                    raise RuntimeError(
                        "job upload cleanup state is inconsistent"
                    )
                if writing:
                    handle._state = "aborted"
                return
            expected_phase = "writing" if writing else "sealed"
            expected_path = partial_path if writing else ready_path
            if (
                job_row is None
                or job_row["phase"] not in {"receiving", "deleting"}
                or job_row["status"] is not None
                or job_row["input_lease_id"] != handle.id
                or not self._job_lease_row_matches(
                    lease_row,
                    handle,
                    phase=expected_phase,
                    path=expected_path,
                )
            ):
                raise RuntimeError("job upload handle is stale")
            if job_row["phase"] == "receiving":
                changed = self._connection.execute(
                    """
                    UPDATE transcription_jobs SET phase = 'deleting'
                    WHERE id = ? AND phase = 'receiving'
                      AND status IS NULL AND input_lease_id = ?
                    """,
                    (handle.id, handle.id),
                ).rowcount
                if changed != 1:
                    raise RuntimeError(
                        "job changed during upload abort"
                    )

        if writing:
            open_handle = self._files.pop(handle.id, None)
            if open_handle is not None:
                open_handle.close()
        for path in (partial_path, ready_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        _fsync_directory(self.staging_dir)

        with self._transaction():
            lease_deleted = self._connection.execute(
                """
                DELETE FROM storage_leases
                WHERE id = ? AND lease_type = 'upload'
                  AND resource_kind = 'transcription'
                  AND owner_kind = 'job' AND owner_id = ?
                  AND phase = ? AND controlled_path = ?
                  AND reserved_bytes = ? AND actual_bytes = ?
                  AND content_sha256 IS ?
                """,
                (
                    handle.id,
                    handle.id,
                    "writing" if writing else "sealed",
                    str(partial_path if writing else ready_path),
                    lease_row["reserved_bytes"],
                    lease_row["actual_bytes"],
                    lease_row["content_sha256"],
                ),
            ).rowcount
            job_deleted = self._connection.execute(
                """
                DELETE FROM transcription_jobs
                WHERE id = ? AND phase = 'deleting'
                  AND status IS NULL AND input_lease_id = ?
                """,
                (handle.id, handle.id),
            ).rowcount
            if lease_deleted != 1 or job_deleted != 1:
                raise RuntimeError(
                    "job changed during upload cleanup"
                )
        if writing:
            handle._state = "aborted"

    def publish_job(
        self,
        input_ref: JobInputRef,
        spec: QueuedJobSpec,
        *,
        speaker_embedding_policy: SpeakerEmbeddingPolicy,
    ) -> DurableJob:
        if type(input_ref) is not JobInputRef:
            raise TypeError("publish_job requires a JobInputRef")
        if type(spec) is not QueuedJobSpec:
            raise TypeError("publish_job requires a QueuedJobSpec")
        if type(speaker_embedding_policy) is not SpeakerEmbeddingPolicy:
            raise TypeError("publish_job requires a speaker embedding policy")
        canonical_options = parse_canonical_options_json(
            spec.canonical_options_json
        )
        self._validate_job_handle_identity(input_ref, writing=False)
        if not self._sealed_job_file_matches(input_ref):
            raise RuntimeError("sealed job input file does not match reference")

        with self._transaction():
            lease_row = self._connection.execute(
                """
                SELECT lease_type, resource_kind, owner_kind, owner_id,
                       phase, controlled_path, reserved_bytes, actual_bytes,
                       content_sha256
                FROM storage_leases WHERE id = ?
                """,
                (input_ref.id,),
            ).fetchone()
            if lease_row is None:
                raise RuntimeError("sealed job input reference is stale")
            input_sha256 = lease_row["content_sha256"]
            if (
                type(input_sha256) is not str
                or re.fullmatch(r"[0-9a-f]{64}", input_sha256) is None
            ):
                raise StorageSchemaError("sealed job input digest is invalid")
            if not self._job_lease_row_matches(
                lease_row,
                input_ref,
                phase="sealed",
                path=input_ref.path,
            ):
                raise RuntimeError("sealed job input reference is stale")
            job_row = self._connection.execute(
                """
                SELECT phase, status, input_lease_id
                FROM transcription_jobs WHERE id = ?
                """,
                (input_ref.id,),
            ).fetchone()
            if (
                job_row is None
                or tuple(job_row)
                != ("receiving", None, input_ref.id)
            ):
                raise RuntimeError("job is no longer receiving")
            queued = self._connection.execute(
                """
                SELECT COUNT(*) FROM transcription_jobs
                WHERE phase = 'visible' AND status = 'queued'
                """
            ).fetchone()[0]
            if queued >= self.limits.max_queued_jobs:
                raise StorageAdmissionError(
                    "too_many_queued_jobs",
                    "too many queued jobs",
                )
            selected_speaker_snapshot = resolve_selected_speaker_snapshot(
                self,
                canonical_options.known_speaker_ids,
                speaker_embedding_policy,
            )
            selected_speaker_snapshot_wire = (
                serialize_selected_speaker_snapshot(
                    selected_speaker_snapshot,
                    speaker_embedding_policy,
                )
            )
            fingerprints = build_request_fingerprints(
                spec.canonical_options_json,
                input_sha256,
                selected_speaker_snapshot_wire,
            )
            changed = self._connection.execute(
                """
                UPDATE transcription_jobs SET
                    phase = 'visible',
                    status = 'queued',
                    canonical_options_json = ?,
                    selected_speaker_snapshot = ?,
                    snapshot_sha256 = ?,
                    input_size_bytes = ?,
                    effective_max_audio_samples = ?,
                    effective_direct_max_audio_samples = ?,
                    total_samples = NULL,
                    request_fingerprint = ?,
                    processor_fingerprint = ?
                WHERE id = ? AND phase = 'receiving'
                  AND status IS NULL AND input_lease_id = ?
                """,
                (
                    spec.canonical_options_json,
                    sqlite3.Binary(selected_speaker_snapshot_wire),
                    fingerprints.snapshot_sha256,
                    input_ref.actual_bytes,
                    spec.effective_max_audio_samples,
                    spec.effective_direct_max_audio_samples,
                    fingerprints.request_fingerprint,
                    spec.processor_fingerprint,
                    input_ref.id,
                    input_ref.id,
                ),
            ).rowcount
            if changed != 1:
                raise RuntimeError("job is no longer receiving")
            row = self._connection.execute(
                f"""
                SELECT {_TRANSCRIPTION_JOB_COLUMNS}
                FROM transcription_jobs WHERE id = ?
                """,
                (input_ref.id,),
            ).fetchone()
            if row is None:
                raise StorageSchemaError("published job row is missing")
            published = _decode_transcription_job(row)
        return published

    def get_visible_job(self, job_id: str) -> DurableJob | None:
        validate_job_id(job_id)
        with self._lock:
            row = self._connection.execute(
                f"""
                SELECT {_TRANSCRIPTION_JOB_COLUMNS}
                FROM transcription_jobs
                WHERE id = ? AND phase = 'visible'
                """,
                (job_id,),
            ).fetchone()
        return None if row is None else _decode_transcription_job(row)

    def claim_next_job(
        self,
        generation: str,
        claimed_at: datetime,
    ) -> DurableJob | None:
        _validate_nonempty_text(generation, name="generation")
        encoded_claimed_at = _encode_job_timestamp(claimed_at)
        with self._transaction():
            marker = self._connection.execute(
                "SELECT 1 FROM shutdown_marker"
            ).fetchone()
            if marker is not None:
                raise RuntimeError(
                    "cannot claim jobs after shutdown has started"
                )
            row = self._connection.execute(
                f"""
                SELECT {_TRANSCRIPTION_JOB_COLUMNS}
                FROM transcription_jobs
                WHERE phase = 'visible' AND status = 'queued'
                ORDER BY created_at, id
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None
            queued = _decode_transcription_job(row)
            if (
                queued.processor_fingerprint
                != self._current_processor_fingerprint
            ):
                raise StorageSchemaError(
                    "processor_fingerprint_mismatch"
                )
            if claimed_at < queued.created_at:
                raise ValueError(
                    "job claim time must not precede creation"
                )
            attempt_token = generate_attempt_token()
            _validate_nonempty_text(
                attempt_token,
                name="attempt token",
            )
            changed = self._connection.execute(
                """
                UPDATE transcription_jobs SET
                    status = 'running',
                    attempt_no = attempt_no + 1,
                    attempt_token = ?,
                    owner_generation = ?,
                    started_at = ?
                WHERE id = ? AND phase = 'visible'
                  AND status = 'queued' AND cancel_requested = 0
                """,
                (
                    attempt_token,
                    generation,
                    encoded_claimed_at,
                    queued.id,
                ),
            ).rowcount
            if changed != 1:
                raise RuntimeError("queued job changed during claim")
            claimed_row = self._connection.execute(
                f"""
                SELECT {_TRANSCRIPTION_JOB_COLUMNS}
                FROM transcription_jobs WHERE id = ?
                """,
                (queued.id,),
            ).fetchone()
        if claimed_row is None:
            raise StorageSchemaError("claimed job row is missing")
        return _decode_transcription_job(claimed_row)

    def update_job_progress(
        self,
        job_id: str,
        attempt_token: str,
        processed_samples: int,
        *,
        total_samples: int | None = None,
    ) -> JobProgressOutcome:
        if type(processed_samples) is not int:
            raise TypeError("processed samples must be an integer")
        if processed_samples < 0:
            raise ValueError("processed samples must be nonnegative")
        if total_samples is not None:
            if type(total_samples) is not int:
                raise TypeError("total samples must be an integer or None")
            if total_samples < 0:
                raise ValueError("total samples must be nonnegative")
            if processed_samples != total_samples:
                raise ValueError(
                    "EOF progress must report equal processed and total samples"
                )
        validate_job_id(job_id)
        _validate_nonempty_text(
            attempt_token,
            name="attempt token",
        )
        with self._transaction():
            row = self._connection.execute(
                """
                SELECT status, attempt_token, processed_samples,
                       total_samples, effective_max_audio_samples,
                       cancel_requested
                FROM transcription_jobs
                WHERE id = ? AND phase = 'visible'
                """,
                (job_id,),
            ).fetchone()
            if (
                row is None
                or row["status"] != "running"
                or row["attempt_token"] != attempt_token
            ):
                return JobProgressOutcome.STALE
            if row["cancel_requested"] == 1:
                return JobProgressOutcome.CANCEL_REQUESTED
            if row["cancel_requested"] != 0:
                raise StorageSchemaError(
                    "job cancellation flag is corrupt"
                )
            effective_max = row["effective_max_audio_samples"]
            if type(effective_max) is not int or effective_max <= 0:
                raise StorageSchemaError("job effective maximum is corrupt")
            if (
                processed_samples < row["processed_samples"]
                or processed_samples > effective_max
            ):
                raise ValueError(
                    "processed samples are outside the current job bounds"
                )
            current_total = row["total_samples"]
            if total_samples is None:
                if (
                    current_total is not None
                    and processed_samples > current_total
                ):
                    raise ValueError(
                        "processed samples exceed the fixed total"
                    )
                changed = self._connection.execute(
                    """
                    UPDATE transcription_jobs SET processed_samples = ?
                    WHERE id = ? AND phase = 'visible'
                      AND status = 'running' AND attempt_token = ?
                      AND cancel_requested = 0
                      AND processed_samples <= ?
                      AND (
                          (
                              total_samples IS NULL
                              AND ? <= effective_max_audio_samples
                          )
                          OR (
                              total_samples IS NOT NULL
                              AND ? <= total_samples
                          )
                      )
                    """,
                    (
                        processed_samples,
                        job_id,
                        attempt_token,
                        processed_samples,
                        processed_samples,
                        processed_samples,
                    ),
                ).rowcount
            else:
                if current_total is not None and current_total != total_samples:
                    raise ValueError(
                        "EOF total samples do not match the fixed total"
                    )
                changed = self._connection.execute(
                    """
                    UPDATE transcription_jobs SET
                        processed_samples = ?,
                        total_samples = ?
                    WHERE id = ? AND phase = 'visible'
                      AND status = 'running' AND attempt_token = ?
                      AND cancel_requested = 0
                      AND processed_samples <= ?
                      AND ? <= effective_max_audio_samples
                      AND (
                          total_samples IS NULL
                          OR total_samples = ?
                      )
                    """,
                    (
                        processed_samples,
                        total_samples,
                        job_id,
                        attempt_token,
                        processed_samples,
                        total_samples,
                        total_samples,
                    ),
                ).rowcount
            if changed != 1:
                return JobProgressOutcome.STALE
        return JobProgressOutcome.UPDATED

    def write_shutdown_marker(
        self,
        generation: str,
        created_at: datetime,
    ) -> None:
        _validate_nonempty_text(generation, name="generation")
        encoded_created_at = _encode_job_timestamp(created_at)
        with self._transaction():
            row = self._connection.execute(
                """
                SELECT generation, created_at
                FROM shutdown_marker WHERE singleton = 1
                """
            ).fetchone()
            if row is None:
                self._connection.execute(
                    """
                    INSERT INTO shutdown_marker(
                        singleton, generation, created_at
                    ) VALUES (1, ?, ?)
                    """,
                    (generation, encoded_created_at),
                )
                return
            if row["generation"] != generation:
                raise RuntimeError(
                    "shutdown marker belongs to another generation"
                )

    def requeue_job_at_shutdown(
        self,
        job_id: str,
        attempt_token: str,
        generation: str,
    ) -> bool:
        validate_job_id(job_id)
        _validate_nonempty_text(
            attempt_token,
            name="attempt token",
        )
        _validate_nonempty_text(generation, name="generation")
        with self._transaction():
            marker = self._connection.execute(
                """
                SELECT generation FROM shutdown_marker
                WHERE singleton = 1
                """
            ).fetchone()
            if marker is None or marker["generation"] != generation:
                return False
            changed = self._connection.execute(
                """
                UPDATE transcription_jobs SET
                    status = 'queued',
                    processed_samples = 0,
                    attempt_token = NULL,
                    owner_generation = NULL,
                    cancel_requested = 0,
                    started_at = NULL,
                    finished_at = NULL
                WHERE id = ? AND phase = 'visible'
                  AND status = 'running'
                  AND attempt_token = ?
                  AND owner_generation = ?
                  AND cancel_requested = 0
                """,
                (job_id, attempt_token, generation),
            ).rowcount
        return changed == 1

    def _compensate_failed_job_open(
        self,
        job_id: str,
        partial_path: Path,
    ) -> None:
        with self._transaction():
            lease_deleted = self._connection.execute(
                """
                DELETE FROM storage_leases
                WHERE id = ? AND lease_type = 'upload'
                  AND resource_kind = 'transcription'
                  AND owner_kind = 'job' AND owner_id = ?
                  AND phase = 'writing' AND controlled_path = ?
                """,
                (job_id, job_id, str(partial_path)),
            ).rowcount
            job_deleted = self._connection.execute(
                """
                DELETE FROM transcription_jobs
                WHERE id = ? AND phase = 'receiving'
                  AND status IS NULL AND input_lease_id = ?
                """,
                (job_id, job_id),
            ).rowcount
            if lease_deleted != 1 or job_deleted != 1:
                raise RuntimeError(
                    "job changed during failed open cleanup"
                )

    def _compensate_failed_job_seal(
        self,
        lease: JobUploadLease,
        *,
        partial_path: Path,
        sealed_path: Path,
    ) -> None:
        open_handle = self._files.pop(lease.id, None)
        if open_handle is not None:
            try:
                open_handle.close()
            except OSError:
                pass
        with self._transaction():
            job_row = self._connection.execute(
                f"""
                SELECT {_TRANSCRIPTION_JOB_COLUMNS}
                FROM transcription_jobs WHERE id = ?
                """,
                (lease.id,),
            ).fetchone()
            lease_row = self._connection.execute(
                "SELECT * FROM storage_leases WHERE id = ?",
                (lease.id,),
            ).fetchone()
            if (
                job_row is None
                or job_row["phase"] not in {"receiving", "deleting"}
                or job_row["status"] is not None
                or job_row["input_lease_id"] != lease.id
                or lease_row is None
                or lease_row["lease_type"] != "upload"
                or lease_row["resource_kind"] != "transcription"
                or lease_row["owner_kind"] != "job"
                or lease_row["owner_id"] != lease.id
                or lease_row["phase"] not in {"writing", "sealed"}
                or lease_row["controlled_path"]
                not in {str(partial_path), str(sealed_path)}
            ):
                raise RuntimeError(
                    "job changed during failed seal cleanup"
                )
            if job_row["phase"] == "receiving":
                changed = self._connection.execute(
                    """
                    UPDATE transcription_jobs SET phase = 'deleting'
                    WHERE id = ? AND phase = 'receiving'
                      AND status IS NULL AND input_lease_id = ?
                    """,
                    (lease.id, lease.id),
                ).rowcount
                if changed != 1:
                    raise RuntimeError(
                        "job changed during failed seal cleanup"
                    )
        expected_job_row = job_row
        expected_lease_row = lease_row

        for path in (partial_path, sealed_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        _fsync_directory(self.staging_dir)

        with self._transaction():
            current_job = self._connection.execute(
                f"""
                SELECT {_TRANSCRIPTION_JOB_COLUMNS}
                FROM transcription_jobs WHERE id = ?
                """,
                (lease.id,),
            ).fetchone()
            current_lease = self._connection.execute(
                "SELECT * FROM storage_leases WHERE id = ?",
                (lease.id,),
            ).fetchone()
            job_keys = expected_job_row.keys()
            expected_job = tuple(
                "deleting" if key == "phase" else expected_job_row[key]
                for key in job_keys
            )
            if (
                current_job is None
                or tuple(current_job) != expected_job
                or current_lease is None
                or tuple(current_lease) != tuple(expected_lease_row)
            ):
                raise RuntimeError(
                    "job changed during failed seal cleanup"
                )
            lease_deleted = self._connection.execute(
                """
                DELETE FROM storage_leases
                WHERE id = ? AND lease_type = 'upload'
                  AND resource_kind = 'transcription'
                  AND owner_kind = 'job' AND owner_id = ?
                  AND phase = ? AND controlled_path = ?
                """,
                (
                    lease.id,
                    lease.id,
                    expected_lease_row["phase"],
                    expected_lease_row["controlled_path"],
                ),
            ).rowcount
            job_deleted = self._connection.execute(
                """
                DELETE FROM transcription_jobs
                WHERE id = ? AND phase = 'deleting' AND status IS NULL
                  AND input_lease_id = ?
                """,
                (lease.id, lease.id),
            ).rowcount
            if lease_deleted != 1 or job_deleted != 1:
                raise RuntimeError(
                    "job changed during failed seal cleanup"
                )
        lease._state = "aborted"

    def _validate_job_handle_identity(
        self,
        handle: JobUploadLease | JobInputRef,
        *,
        writing: bool,
    ) -> None:
        validate_job_id(handle.id)
        expected_path = self.staging_dir / (
            f"{handle.id}.partial" if writing else f"{handle.id}.ready"
        )
        if (
            handle.resource_kind != "transcription"
            or handle.owner_kind != "job"
            or handle.owner_id != handle.id
            or handle.path != expected_path
            or not self._is_controlled_path(
                handle.id,
                "upload",
                "writing" if writing else "sealed",
                handle.path,
            )
        ):
            raise RuntimeError("job upload handle is invalid")

    def _require_job_upload_lease(
        self,
        lease: JobUploadLease,
        *,
        phase: str,
    ) -> None:
        self._validate_job_handle_identity(lease, writing=True)
        self._require_writing(lease, "upload")
        row = self._connection.execute(
            """
            SELECT lease_type, resource_kind, owner_kind, owner_id,
                   phase, controlled_path, reserved_bytes, actual_bytes,
                   content_sha256
            FROM storage_leases WHERE id = ?
            """,
            (lease.id,),
        ).fetchone()
        job_row = self._connection.execute(
            """
            SELECT phase, status, input_lease_id
            FROM transcription_jobs WHERE id = ?
            """,
            (lease.id,),
        ).fetchone()
        if (
            not self._job_lease_row_matches(
                row,
                lease,
                phase=phase,
                path=lease.path,
            )
            or job_row is None
            or tuple(job_row)
            != ("receiving", None, lease.id)
        ):
            raise RuntimeError("job upload lease does not match ledger")

    def _job_lease_row_matches(
        self,
        row: sqlite3.Row | None,
        handle: JobUploadLease | JobInputRef,
        *,
        phase: str,
        path: Path,
    ) -> bool:
        if row is None:
            return False
        expected_reserved = (
            handle.reserved_bytes
            if type(handle) is JobUploadLease
            else handle.actual_bytes
        )
        return bool(
            row["lease_type"] == "upload"
            and row["resource_kind"] == handle.resource_kind
            and row["owner_kind"] == handle.owner_kind
            and row["owner_id"] == handle.owner_id
            and row["phase"] == phase
            and row["controlled_path"] == str(path)
            and row["reserved_bytes"] == expected_reserved
            and (
                type(handle) is JobUploadLease
                or row["actual_bytes"] == handle.actual_bytes
            )
            and (
                type(handle) is JobUploadLease
                or row["content_sha256"] == handle.content_sha256
            )
        )

    def _sealed_job_file_matches(self, ref: JobInputRef) -> bool:
        if (
            type(ref.actual_bytes) is not int
            or ref.actual_bytes < 0
            or type(ref.content_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", ref.content_sha256) is None
        ):
            raise RuntimeError("job upload handle is invalid")
        try:
            stat_result = ref.path.stat(follow_symlinks=False)
            return (
                not ref.path.is_symlink()
                and ref.path.is_file()
                and stat_result.st_size == ref.actual_bytes
            )
        except OSError:
            return False

    def begin_upload(
        self,
        resource_kind: str,
        *,
        owner_kind: str = "sync",
        owner_id: str | None = None,
    ) -> UploadLease:
        if not resource_kind:
            raise ValueError("storage resource kind must not be empty")
        self._validate_owner(owner_kind, owner_id)
        lease_id = uuid.uuid4().hex
        resolved_owner_id = owner_id or lease_id
        path = self.staging_dir / f"{lease_id}.partial"
        reservation = RESERVATION_QUANTUM
        with self._transaction():
            active = self._connection.execute(
                """
                SELECT COUNT(*) FROM storage_leases
                WHERE lease_type = 'upload' AND phase = 'writing'
                """
            ).fetchone()[0]
            if active >= self.limits.max_active_uploads:
                raise StorageAdmissionError(
                    "too_many_active_uploads", "too many active uploads"
                )
            self._admit_delta(reservation)
            self._connection.execute(
                """
                INSERT INTO storage_leases(
                    id, lease_type, resource_kind, owner_kind, owner_id,
                    phase, controlled_path, reserved_bytes, actual_bytes
                ) VALUES (?, 'upload', ?, ?, ?, 'writing', ?, ?, 0)
                """,
                (
                    lease_id,
                    resource_kind,
                    owner_kind,
                    resolved_owner_id,
                    str(path),
                    reservation,
                ),
            )
        try:
            handle = path.open("xb")
        except OSError:
            with self._transaction():
                self._connection.execute(
                    "DELETE FROM storage_leases WHERE id = ?",
                    (lease_id,),
                )
            raise
        self._files[lease_id] = handle
        return UploadLease(
            lease_id,
            resource_kind,
            owner_kind,
            resolved_owner_id,
            path,
            reservation,
        )

    def append(self, lease: UploadLease, data: bytes) -> None:
        if not isinstance(lease, UploadLease):
            raise TypeError("append requires an UploadLease")
        self._append_writing(
            lease=lease,
            lease_type="upload",
            data=data,
        )

    def seal_upload(self, lease: UploadLease) -> InputRef:
        if not isinstance(lease, UploadLease):
            raise TypeError("seal_upload requires an UploadLease")
        sealed_path, digest = self._seal_writing(
            lease=lease,
            lease_type="upload",
            sealed_suffix="ready",
            directory=self.staging_dir,
        )
        return InputRef(
            lease.id,
            lease.resource_kind,
            lease.owner_kind,
            lease.owner_id,
            sealed_path,
            lease.actual_bytes,
            digest,
        )

    def abort_upload(self, lease: UploadLease) -> None:
        if not isinstance(lease, UploadLease):
            raise TypeError("abort_upload requires an UploadLease")
        self._abort_writing(
            lease,
            lease_type="upload",
            directory=self.staging_dir,
            suffixes=("partial", "ready"),
        )

    def resolve_input(self, ref: InputRef) -> Path:
        if not isinstance(ref, InputRef):
            raise TypeError("resolve_input requires an InputRef")
        return self._resolve_ref(ref, lease_type="upload")

    def resolve_job_attempt_input(
        self,
        job_id: str,
        attempt_token: str,
    ) -> tuple[DurableJob, Path]:
        validate_job_id(job_id)
        _validate_nonempty_text(attempt_token, name="attempt token")
        input_path = self.staging_dir / f"{job_id}.ready"
        with self._transaction():
            job_row = self._connection.execute(
                f"""
                SELECT {_TRANSCRIPTION_JOB_COLUMNS}
                FROM transcription_jobs WHERE id = ?
                """,
                (job_id,),
            ).fetchone()
            if job_row is None:
                raise StaleJobAttemptError(
                    "job attempt is no longer running"
                )
            job = _decode_transcription_job(job_row)
            if (
                job.phase is not JobPhase.VISIBLE
                or job.status is not JobStatus.RUNNING
                or job.attempt_token != attempt_token
            ):
                raise StaleJobAttemptError(
                    "job attempt is no longer running"
                )
            if job.cancel_requested:
                raise JobCancellationRequestedError(
                    "job cancellation was requested"
                )
            if (
                job.processor_fingerprint
                != self._current_processor_fingerprint
            ):
                raise StorageSchemaError(
                    "processor_fingerprint_mismatch"
                )

            lease_row = self._connection.execute(
                "SELECT * FROM storage_leases WHERE id = ?",
                (job.input_lease_id,),
            ).fetchone()
            content_sha256 = (
                None
                if lease_row is None
                else lease_row["content_sha256"]
            )
            if (
                job.input_lease_id != job_id
                or lease_row is None
                or lease_row["id"] != job_id
                or lease_row["lease_type"] != "upload"
                or lease_row["resource_kind"] != "transcription"
                or lease_row["owner_kind"] != "job"
                or lease_row["owner_id"] != job_id
                or lease_row["phase"] != "sealed"
                or lease_row["controlled_path"] != str(input_path)
                or lease_row["reserved_bytes"]
                != lease_row["actual_bytes"]
                or lease_row["actual_bytes"] != job.input_size_bytes
                or type(content_sha256) is not str
                or re.fullmatch(r"[0-9a-f]{64}", content_sha256)
                is None
                or not self._is_controlled_path(
                    job_id,
                    "upload",
                    "sealed",
                    input_path,
                )
                or not _is_regular_file_with_size(
                    input_path,
                    lease_row["actual_bytes"],
                )
            ):
                raise StorageSchemaError(
                    "job attempt input is corrupt"
                )
            assert job.canonical_options_json is not None
            assert job.selected_speaker_snapshot is not None
            try:
                fingerprints = build_request_fingerprints(
                    job.canonical_options_json,
                    content_sha256,
                    job.selected_speaker_snapshot,
                )
            except (TypeError, ValueError) as error:
                raise StorageSchemaError(
                    "job attempt fingerprints are corrupt"
                ) from error
            if (
                fingerprints.snapshot_sha256 != job.snapshot_sha256
                or fingerprints.request_fingerprint
                != job.request_fingerprint
            ):
                raise StorageSchemaError(
                    "job attempt fingerprints are corrupt"
                )
        return job, input_path

    def release_input(self, ref: InputRef) -> None:
        if not isinstance(ref, InputRef):
            raise TypeError("release_input requires an InputRef")
        self._release_ref(
            ref,
            lease_type="upload",
            directory=self.staging_dir,
        )

    def begin_job_attempt_artifact(
        self,
        job_id: str,
        attempt_token: str,
    ) -> ReservedByteWriter:
        return self._begin_job_artifact(
            job_id,
            attempt_token,
            resource_kind="segment_jsonl",
        )

    def begin_job_result_artifact(
        self,
        job_id: str,
        attempt_token: str,
    ) -> ReservedByteWriter:
        return self._begin_job_artifact(
            job_id,
            attempt_token,
            resource_kind="result_complete",
        )

    def _begin_job_artifact(
        self,
        job_id: str,
        attempt_token: str,
        *,
        resource_kind: str,
    ) -> ReservedByteWriter:
        validate_job_id(job_id)
        _validate_nonempty_text(attempt_token, name="attempt token")
        lease_id = uuid.uuid4().hex
        path = self.artifact_dir / f"{lease_id}.partial"
        reservation = RESERVATION_QUANTUM
        with self._transaction():
            job_row = self._connection.execute(
                f"""
                SELECT {_TRANSCRIPTION_JOB_COLUMNS}
                FROM transcription_jobs WHERE id = ?
                """,
                (job_id,),
            ).fetchone()
            if job_row is None:
                raise StaleJobAttemptError(
                    "job attempt is no longer running"
                )
            job = _decode_transcription_job(job_row)
            if (
                job.phase is not JobPhase.VISIBLE
                or job.status is not JobStatus.RUNNING
                or job.attempt_token != attempt_token
            ):
                raise StaleJobAttemptError(
                    "job attempt is no longer running"
                )
            if job.cancel_requested:
                raise JobCancellationRequestedError(
                    "job cancellation was requested"
                )
            duplicate = self._connection.execute(
                """
                SELECT 1 FROM storage_leases
                WHERE lease_type = 'artifact'
                  AND resource_kind = ?
                  AND owner_kind = 'job' AND owner_id = ?
                """,
                (resource_kind, job_id),
            ).fetchone()
            if duplicate is not None:
                raise RuntimeError(
                    f"job {resource_kind} artifact already exists"
                )
            self._admit_delta(reservation)
            self._connection.execute(
                """
                INSERT INTO storage_leases(
                    id, lease_type, resource_kind, owner_kind, owner_id,
                    phase, controlled_path, reserved_bytes, actual_bytes
                ) VALUES (
                    ?, 'artifact', ?, 'job', ?,
                    'writing', ?, ?, 0
                )
                """,
                (
                    lease_id,
                    resource_kind,
                    job_id,
                    str(path),
                    reservation,
                ),
            )
        try:
            handle = path.open("xb")
        except OSError:
            with self._transaction():
                self._connection.execute(
                    "DELETE FROM storage_leases WHERE id = ?",
                    (lease_id,),
                )
            raise
        self._files[lease_id] = handle
        return ReservedByteWriter(
            lease_id,
            resource_kind,
            "job",
            job_id,
            path,
            reservation,
        )

    def commit_job_success(
        self,
        job_id: str,
        attempt_token: str,
        result_ref: ArtifactRef,
    ) -> JobSuccessOutcome:
        validate_job_id(job_id)
        _validate_nonempty_text(attempt_token, name="attempt token")
        if type(result_ref) is not ArtifactRef:
            raise TypeError("commit_job_success requires an ArtifactRef")
        if (
            result_ref.resource_kind != "result_complete"
            or result_ref.owner_kind != "job"
            or result_ref.owner_id != job_id
        ):
            raise ValueError("job result artifact identity is invalid")

        with self._lock:
            row = self._connection.execute(
                f"""
                SELECT {_TRANSCRIPTION_JOB_COLUMNS}
                FROM transcription_jobs WHERE id = ?
                """,
                (job_id,),
            ).fetchone()
            already_committed = self._job_result_is_committed(
                row,
                job_id,
                result_ref,
            )
        if already_committed:
            self._cleanup_terminal_job_input(
                job_id,
                JobStatus.SUCCEEDED,
            )
            return JobSuccessOutcome.COMMITTED
        if (
            row is None
            or row["phase"] != "visible"
            or row["status"] != "running"
            or row["attempt_token"] != attempt_token
        ):
            if not (
                row is not None
                and row["status"] == "succeeded"
                and row["result_lease_id"] == result_ref.id
            ):
                self.release_artifact(result_ref)
            return JobSuccessOutcome.STALE
        if row["cancel_requested"] == 1:
            self.release_artifact(result_ref)
            return JobSuccessOutcome.CANCEL_REQUESTED
        if (
            row["cancel_requested"] != 0
            or row["total_samples"] is None
            or row["processed_samples"] != row["total_samples"]
        ):
            self.release_artifact(result_ref)
            return JobSuccessOutcome.STALE

        job = _decode_transcription_job(row)
        manifest = self._validate_job_result(
            job,
            path=self.resolve_artifact(result_ref),
            actual_bytes=result_ref.actual_bytes,
            content_sha256=result_ref.content_sha256,
        )

        committed = False
        cancelled = False
        bound = False
        with self._transaction():
            changed = self._connection.execute(
                f"""
                UPDATE transcription_jobs SET
                    status = 'succeeded',
                    attempt_token = NULL,
                    owner_generation = NULL,
                    result_lease_id = ?,
                    input_cleanup_pending = 1,
                    finished_at = ?
                WHERE id = ? AND phase = 'visible'
                  AND status = 'running' AND attempt_token = ?
                  AND cancel_requested = 0
                  AND total_samples IS NOT NULL
                  AND processed_samples = total_samples
                  AND result_lease_id IS NULL
                  AND EXISTS (
                      SELECT 1 FROM storage_leases
                      WHERE {_EXACT_SEALED_JOB_RESULT_WHERE}
                  )
                """,
                (
                    result_ref.id,
                    _encode_job_timestamp(manifest.finished_at),
                    job_id,
                    attempt_token,
                    result_ref.id,
                    job_id,
                    str(result_ref.path),
                    result_ref.actual_bytes,
                    result_ref.actual_bytes,
                    result_ref.content_sha256,
                ),
            ).rowcount
            if changed == 1:
                committed = True
            else:
                current = self._connection.execute(
                    """
                    SELECT status, attempt_token, cancel_requested,
                           result_lease_id
                    FROM transcription_jobs
                    WHERE id = ? AND phase = 'visible'
                    """,
                    (job_id,),
                ).fetchone()
                committed = self._job_result_is_committed(
                    current,
                    job_id,
                    result_ref,
                )
                if not committed:
                    cancelled = (
                        current is not None
                        and current["status"] == "running"
                        and current["attempt_token"] == attempt_token
                        and current["cancel_requested"] == 1
                    )
                    bound = (
                        current is not None
                        and current["status"] == "succeeded"
                        and current["result_lease_id"] == result_ref.id
                    )
        if committed:
            self._cleanup_terminal_job_input(
                job_id,
                JobStatus.SUCCEEDED,
            )
            return JobSuccessOutcome.COMMITTED
        if not bound:
            self.release_artifact(result_ref)
        if cancelled:
            return JobSuccessOutcome.CANCEL_REQUESTED
        return JobSuccessOutcome.STALE

    def commit_job_failure(
        self,
        job_id: str,
        attempt_token: str,
        error_code: str,
        finished_at: datetime,
    ) -> JobTerminalOutcome:
        validate_job_id(job_id)
        _validate_nonempty_text(attempt_token, name="attempt token")
        if type(error_code) is not str:
            raise TypeError("job failure code must be a string")
        if error_code not in _RUNTIME_JOB_FAILURE_CODES:
            raise ValueError("job failure code is invalid")
        return self._commit_job_terminal(
            job_id,
            attempt_token,
            status=JobStatus.FAILED,
            error_code=error_code,
            finished_at=finished_at,
            expected_cancel_requested=False,
        )

    def commit_job_cancellation(
        self,
        job_id: str,
        attempt_token: str,
        finished_at: datetime,
    ) -> JobTerminalOutcome:
        validate_job_id(job_id)
        _validate_nonempty_text(attempt_token, name="attempt token")
        return self._commit_job_terminal(
            job_id,
            attempt_token,
            status=JobStatus.CANCELLED,
            error_code=None,
            finished_at=finished_at,
            expected_cancel_requested=True,
        )

    def _commit_job_terminal(
        self,
        job_id: str,
        attempt_token: str,
        *,
        status: JobStatus,
        error_code: str | None,
        finished_at: datetime,
        expected_cancel_requested: bool,
    ) -> JobTerminalOutcome:
        encoded_finished_at = _encode_job_timestamp(finished_at)
        outcome = JobTerminalOutcome.STALE
        cleanup_pending = False
        with self._transaction():
            row = self._connection.execute(
                f"""
                SELECT {_TRANSCRIPTION_JOB_COLUMNS}
                FROM transcription_jobs
                WHERE id = ?
                """,
                (job_id,),
            ).fetchone()
            job = (
                None
                if row is None
                else _decode_transcription_job(row)
            )
            same_terminal_identity = (
                job is not None
                and job.status is status
                and (
                    status is JobStatus.CANCELLED
                    or job.error_code == error_code
                )
            )
            if job is None or same_terminal_identity:
                cleanup_pending = True
            elif (
                job.status is not JobStatus.RUNNING
                or job.attempt_token != attempt_token
            ):
                return JobTerminalOutcome.STALE

            if not cleanup_pending:
                if job.started_at is None:
                    raise StorageSchemaError(
                        "running job start timestamp is missing"
                    )
                if finished_at < job.started_at:
                    raise ValueError("job timestamps are out of order")
                if job.cancel_requested != expected_cancel_requested:
                    if (
                        not expected_cancel_requested
                        and job.cancel_requested
                    ):
                        return JobTerminalOutcome.CANCEL_REQUESTED
                    return JobTerminalOutcome.STALE
                changed = self._connection.execute(
                    """
                    UPDATE transcription_jobs SET
                        status = ?,
                        attempt_token = NULL,
                        owner_generation = NULL,
                        error_code = ?,
                        input_cleanup_pending = 1,
                        finished_at = ?
                    WHERE id = ? AND phase = 'visible'
                      AND status = 'running' AND attempt_token = ?
                      AND cancel_requested = ?
                      AND result_lease_id IS NULL
                    """,
                    (
                        status.value,
                        error_code,
                        encoded_finished_at,
                        job_id,
                        attempt_token,
                        int(expected_cancel_requested),
                    ),
                ).rowcount
                if changed != 1:
                    raise RuntimeError(
                        "running job changed during terminal commit"
                    )
                terminal_row = self._connection.execute(
                    f"""
                    SELECT {_TRANSCRIPTION_JOB_COLUMNS}
                    FROM transcription_jobs WHERE id = ?
                    """,
                    (job_id,),
                ).fetchone()
                if terminal_row is None:
                    raise StorageSchemaError(
                        "committed terminal job disappeared"
                    )
                _decode_transcription_job(terminal_row)
                outcome = JobTerminalOutcome.COMMITTED
                cleanup_pending = True

        if cleanup_pending:
            self._cleanup_terminal_job_input(job_id, status)
        return outcome

    def open_succeeded_job_result(
        self,
        job_id: str,
    ) -> StoredJobResult:
        validate_job_id(job_id)
        handle: BinaryIO | None = None
        try:
            with self._transaction():
                job_row = self._connection.execute(
                    f"""
                    SELECT {_TRANSCRIPTION_JOB_COLUMNS}
                    FROM transcription_jobs
                    WHERE id = ? AND phase = 'visible'
                      AND status = 'succeeded'
                    """,
                    (job_id,),
                ).fetchone()
                if job_row is None:
                    raise CanonicalArtifactError(
                        "stored job result is unavailable"
                    )
                job = _decode_transcription_job(job_row)
                lease_row = self._connection.execute(
                    "SELECT * FROM storage_leases WHERE id = ?",
                    (job.result_lease_id,),
                ).fetchone()
                result_path = self.artifact_dir / (
                    f"{job.result_lease_id}.complete"
                )
                if (
                    lease_row is None
                    or lease_row["id"] != job.result_lease_id
                    or re.fullmatch(
                        r"[0-9a-f]{32}",
                        lease_row["id"],
                    )
                    is None
                    or lease_row["lease_type"] != "artifact"
                    or lease_row["resource_kind"] != "result_complete"
                    or lease_row["owner_kind"] != "job"
                    or lease_row["owner_id"] != job_id
                    or lease_row["phase"] != "sealed"
                    or lease_row["controlled_path"] != str(result_path)
                    or lease_row["reserved_bytes"]
                    != lease_row["actual_bytes"]
                    or type(lease_row["content_sha256"]) is not str
                    or re.fullmatch(
                        r"[0-9a-f]{64}",
                        lease_row["content_sha256"],
                    )
                    is None
                    or not self._is_controlled_path(
                        lease_row["id"],
                        "artifact",
                        "sealed",
                        result_path,
                    )
                ):
                    raise CanonicalArtifactError(
                        "stored job result identity is invalid"
                    )
                descriptor = os.open(
                    result_path,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                )
                try:
                    handle = os.fdopen(descriptor, "rb")
                except BaseException:
                    os.close(descriptor)
                    raise
                file_stat = os.fstat(handle.fileno())
                if (
                    not stat.S_ISREG(file_stat.st_mode)
                    or file_stat.st_size != lease_row["actual_bytes"]
                ):
                    raise CanonicalArtifactError(
                        "stored job result file is invalid"
                    )
                reader = ResultEnvelopeReader(
                    result_path,
                    expected_size_bytes=lease_row["actual_bytes"],
                    expected_sha256=lease_row["content_sha256"],
                    expected_job_id=job.id,
                    expected_attempt_no=job.attempt_no,
                    expected_request_fingerprint=(
                        job.request_fingerprint
                    ),
                    expected_processor_fingerprint=(
                        job.processor_fingerprint
                    ),
                    canonical_options=parse_canonical_options_json(
                        job.canonical_options_json
                    ),
                    expected_total_samples=job.total_samples,
                    opener=lambda: handle,
                )
            return StoredJobResult(reader, handle)
        except CanonicalArtifactError:
            if handle is not None:
                handle.close()
            raise
        except OSError as error:
            if handle is not None:
                handle.close()
            raise CanonicalArtifactError(
                "stored job result could not be opened"
            ) from error
        except BaseException:
            if handle is not None:
                handle.close()
            raise

    def _cleanup_terminal_job_input(
        self,
        job_id: str,
        expected_status: JobStatus,
    ) -> None:
        status = expected_status.value
        input_path = self.staging_dir / f"{job_id}.ready"
        with self._transaction():
            job_row = self._connection.execute(
                f"""
                SELECT {_TRANSCRIPTION_JOB_COLUMNS}
                FROM transcription_jobs WHERE id = ?
                """,
                (job_id,),
            ).fetchone()
            if job_row is None:
                orphan_lease = self._connection.execute(
                    "SELECT 1 FROM storage_leases WHERE id = ?",
                    (job_id,),
                ).fetchone()
                if orphan_lease is None:
                    return
                raise StorageSchemaError(
                    "terminal job input lease is orphaned"
                )
            if (
                job_row["phase"] == "deleting"
                and job_row["status"] == status
            ):
                return
            if (
                job_row["phase"] == "visible"
                and job_row["status"] == status
                and job_row["input_lease_id"] is None
                and job_row["input_cleanup_pending"] == 0
            ):
                return
            lease_row = self._connection.execute(
                "SELECT * FROM storage_leases WHERE id = ?",
                (job_id,),
            ).fetchone()
            if (
                job_row["phase"] != "visible"
                or job_row["status"] != status
                or job_row["input_lease_id"] != job_id
                or job_row["input_cleanup_pending"] != 1
                or lease_row is None
                or lease_row["id"] != job_id
                or lease_row["lease_type"] != "upload"
                or lease_row["resource_kind"] != "transcription"
                or lease_row["owner_kind"] != "job"
                or lease_row["owner_id"] != job_id
                or lease_row["phase"] != "sealed"
                or lease_row["controlled_path"] != str(input_path)
                or lease_row["reserved_bytes"]
                != lease_row["actual_bytes"]
                or lease_row["actual_bytes"]
                != job_row["input_size_bytes"]
            ):
                raise StorageSchemaError(
                    "terminal job input cleanup state is corrupt"
                )
            if not _is_absent_or_regular_file_with_size(
                input_path,
                lease_row["actual_bytes"],
            ):
                raise StorageSchemaError(
                    "terminal job input file is corrupt"
                )

        self._unlink_if_present(input_path)
        _fsync_directory(self.staging_dir)

        with self._transaction():
            current_job = self._connection.execute(
                f"""
                SELECT {_TRANSCRIPTION_JOB_COLUMNS}
                FROM transcription_jobs WHERE id = ?
                """,
                (job_id,),
            ).fetchone()
            if current_job is None:
                orphan_lease = self._connection.execute(
                    "SELECT 1 FROM storage_leases WHERE id = ?",
                    (job_id,),
                ).fetchone()
                if orphan_lease is None:
                    return
                raise StorageSchemaError(
                    "terminal job input lease is orphaned"
                )
            if (
                current_job["phase"] == "deleting"
                and current_job["status"] == status
            ):
                return
            if (
                current_job["phase"] == "visible"
                and current_job["status"] == status
                and current_job["input_lease_id"] is None
                and current_job["input_cleanup_pending"] == 0
            ):
                return
            current_lease = self._connection.execute(
                "SELECT * FROM storage_leases WHERE id = ?",
                (job_id,),
            ).fetchone()
            if (
                tuple(current_job) != tuple(job_row)
                or current_lease is None
                or tuple(current_lease) != tuple(lease_row)
            ):
                raise RuntimeError(
                    "terminal job input cleanup state changed"
                )
            lease_deleted = self._connection.execute(
                """
                DELETE FROM storage_leases
                WHERE id = ? AND lease_type = 'upload'
                  AND resource_kind = 'transcription'
                  AND owner_kind = 'job' AND owner_id = ?
                  AND phase = 'sealed' AND controlled_path = ?
                  AND reserved_bytes = ? AND actual_bytes = ?
                  AND content_sha256 IS ?
                  AND created_at = ?
                """,
                (
                    job_id,
                    job_id,
                    str(input_path),
                    lease_row["reserved_bytes"],
                    lease_row["actual_bytes"],
                    lease_row["content_sha256"],
                    lease_row["created_at"],
                ),
            ).rowcount
            changed = self._connection.execute(
                """
                UPDATE transcription_jobs SET
                    input_lease_id = NULL,
                    input_cleanup_pending = 0
                WHERE id = ? AND phase = 'visible'
                  AND status = ?
                  AND input_lease_id = ?
                  AND input_cleanup_pending = 1
                """,
                (job_id, status, job_id),
            ).rowcount
            if lease_deleted != 1 or changed != 1:
                raise RuntimeError(
                    "terminal job changed during input cleanup"
                )
            cleaned_row = self._connection.execute(
                f"""
                SELECT {_TRANSCRIPTION_JOB_COLUMNS}
                FROM transcription_jobs WHERE id = ?
                """,
                (job_id,),
            ).fetchone()
            if cleaned_row is None:
                raise StorageSchemaError(
                    "terminal job disappeared during input cleanup"
                )
            _decode_transcription_job(cleaned_row)

    def _job_result_is_committed(
        self,
        row: sqlite3.Row | None,
        job_id: str,
        result_ref: ArtifactRef,
    ) -> bool:
        if (
            row is None
            or row["status"] != "succeeded"
            or row["result_lease_id"] != result_ref.id
        ):
            return False
        return (
            self._connection.execute(
                f"""
                SELECT 1 FROM storage_leases
                WHERE {_EXACT_SEALED_JOB_RESULT_WHERE}
                """,
                (
                    result_ref.id,
                    job_id,
                    str(result_ref.path),
                    result_ref.actual_bytes,
                    result_ref.actual_bytes,
                    result_ref.content_sha256,
                ),
            ).fetchone()
            is not None
        )

    def _validate_job_result(
        self,
        job: DurableJob,
        *,
        path: Path,
        actual_bytes: int,
        content_sha256: str,
    ) -> ResultEnvelopeManifest:
        if not _is_regular_file_with_size(path, actual_bytes):
            raise CanonicalArtifactError(
                "result envelope size does not match storage"
            )
        manifest = ResultEnvelopeReader(
            path,
            expected_size_bytes=actual_bytes,
            expected_sha256=content_sha256,
            expected_job_id=job.id,
            expected_attempt_no=job.attempt_no,
            expected_request_fingerprint=job.request_fingerprint,
            expected_processor_fingerprint=job.processor_fingerprint,
            canonical_options=parse_canonical_options_json(
                job.canonical_options_json
            ),
            expected_total_samples=job.total_samples,
        ).validate()
        assert job.started_at is not None
        if manifest.finished_at < job.started_at:
            raise ValueError(
                "job result finished before its attempt started"
            )
        return manifest

    def begin_artifact(
        self,
        resource_kind: str,
        *,
        owner_kind: str,
        owner_id: str,
    ) -> ReservedByteWriter:
        if resource_kind not in ARTIFACT_KINDS:
            raise ValueError("unsupported artifact resource kind")
        self._validate_owner(owner_kind, owner_id)
        lease_id = uuid.uuid4().hex
        path = self.artifact_dir / f"{lease_id}.partial"
        reservation = RESERVATION_QUANTUM
        with self._transaction():
            self._admit_delta(reservation)
            self._connection.execute(
                """
                INSERT INTO storage_leases(
                    id, lease_type, resource_kind, owner_kind, owner_id,
                    phase, controlled_path, reserved_bytes, actual_bytes
                ) VALUES (?, 'artifact', ?, ?, ?, 'writing', ?, ?, 0)
                """,
                (
                    lease_id,
                    resource_kind,
                    owner_kind,
                    owner_id,
                    str(path),
                    reservation,
                ),
            )
        try:
            handle = path.open("xb")
        except OSError:
            with self._transaction():
                self._connection.execute(
                    "DELETE FROM storage_leases WHERE id = ?",
                    (lease_id,),
                )
            raise
        self._files[lease_id] = handle
        return ReservedByteWriter(
            lease_id,
            resource_kind,
            owner_kind,
            owner_id,
            path,
            reservation,
        )

    def append_artifact(
        self, writer: ReservedByteWriter, data: bytes
    ) -> None:
        if not isinstance(writer, ReservedByteWriter):
            raise TypeError(
                "append_artifact requires a ReservedByteWriter"
            )
        self._append_writing(
            lease=writer,
            lease_type="artifact",
            data=data,
        )

    def seal_artifact(
        self, writer: ReservedByteWriter
    ) -> ArtifactRef:
        if not isinstance(writer, ReservedByteWriter):
            raise TypeError(
                "seal_artifact requires a ReservedByteWriter"
            )
        sealed_path, digest = self._seal_writing(
            lease=writer,
            lease_type="artifact",
            sealed_suffix="complete",
            directory=self.artifact_dir,
        )
        return ArtifactRef(
            writer.id,
            writer.resource_kind,
            writer.owner_kind,
            writer.owner_id,
            sealed_path,
            writer.actual_bytes,
            digest,
        )

    def abort_artifact(self, writer: ReservedByteWriter) -> None:
        if not isinstance(writer, ReservedByteWriter):
            raise TypeError(
                "abort_artifact requires a ReservedByteWriter"
            )
        self._abort_writing(
            writer,
            lease_type="artifact",
            directory=self.artifact_dir,
            suffixes=("partial", "complete"),
        )

    def resolve_artifact(self, ref: ArtifactRef) -> Path:
        if not isinstance(ref, ArtifactRef):
            raise TypeError("resolve_artifact requires an ArtifactRef")
        return self._resolve_ref(ref, lease_type="artifact")

    def release_artifact(self, ref: ArtifactRef) -> None:
        if not isinstance(ref, ArtifactRef):
            raise TypeError("release_artifact requires an ArtifactRef")
        self._release_ref(
            ref,
            lease_type="artifact",
            directory=self.artifact_dir,
        )

    def _append_writing(
        self,
        *,
        lease: UploadLease | JobUploadLease | ReservedByteWriter,
        lease_type: str,
        data: bytes,
    ) -> None:
        self._require_writing(lease, lease_type)
        if not data:
            return
        next_actual = lease.actual_bytes + len(data)
        required = _round_reservation(next_actual)
        delta = required - lease.reserved_bytes
        if delta > 0:
            with self._transaction():
                self._admit_delta(delta)
                changed = self._connection.execute(
                    """
                    UPDATE storage_leases
                    SET reserved_bytes = ?
                    WHERE id = ? AND lease_type = ? AND phase = 'writing'
                    """,
                    (required, lease.id, lease_type),
                ).rowcount
                if changed != 1:
                    raise RuntimeError(
                        f"{lease_type} lease is no longer writing"
                    )
            lease.reserved_bytes = required

        handle = self._files.get(lease.id)
        if handle is None:
            raise RuntimeError(f"{lease_type} lease is closed")
        written = handle.write(data)
        if written != len(data):
            raise OSError("short write to reserved storage")
        lease._hasher.update(data)
        lease.actual_bytes = next_actual
        checkpoint = (
            next_actual // RESERVATION_QUANTUM
        ) * RESERVATION_QUANTUM
        if checkpoint > lease._checkpointed_bytes:
            with self._transaction():
                changed = self._connection.execute(
                    """
                    UPDATE storage_leases
                    SET actual_bytes = ?
                    WHERE id = ? AND lease_type = ? AND phase = 'writing'
                    """,
                    (checkpoint, lease.id, lease_type),
                ).rowcount
                if changed != 1:
                    raise RuntimeError(
                        f"{lease_type} lease is no longer writing"
                    )
            lease._checkpointed_bytes = checkpoint

    def _seal_writing(
        self,
        *,
        lease: UploadLease | JobUploadLease | ReservedByteWriter,
        lease_type: str,
        sealed_suffix: str,
        directory: Path,
    ) -> tuple[Path, str]:
        self._require_writing(lease, lease_type)
        handle = self._files.get(lease.id)
        if handle is None:
            raise RuntimeError(f"{lease_type} lease is closed")
        sealed_path = directory / f"{lease.id}.{sealed_suffix}"
        try:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            self._files.pop(lease.id, None)
            os.replace(lease.path, sealed_path)
            _fsync_directory(directory)
            digest = lease._hasher.hexdigest()
            with self._transaction():
                changed = self._connection.execute(
                    """
                    UPDATE storage_leases
                    SET phase = 'sealed', controlled_path = ?,
                        reserved_bytes = ?, actual_bytes = ?,
                        content_sha256 = ?
                    WHERE id = ? AND lease_type = ? AND phase = 'writing'
                    """,
                    (
                        str(sealed_path),
                        lease.actual_bytes,
                        lease.actual_bytes,
                        digest,
                        lease.id,
                        lease_type,
                    ),
                ).rowcount
                if changed != 1:
                    raise RuntimeError(
                        f"{lease_type} lease is no longer writing"
                    )
        except BaseException:
            self._compensate_failed_seal(
                lease,
                lease_type=lease_type,
                directory=directory,
                sealed_path=sealed_path,
            )
            raise
        lease.path = sealed_path
        lease.reserved_bytes = lease.actual_bytes
        lease.content_sha256 = digest
        lease._state = "sealed"
        return sealed_path, digest

    def _compensate_failed_seal(
        self,
        lease: UploadLease | JobUploadLease | ReservedByteWriter,
        *,
        lease_type: str,
        directory: Path,
        sealed_path: Path,
    ) -> None:
        partial_path = lease.path
        with self._lock:
            row = self._connection.execute(
                """
                SELECT resource_kind, owner_kind, owner_id, phase,
                       controlled_path
                FROM storage_leases
                WHERE id = ? AND lease_type = ?
                """,
                (lease.id, lease_type),
            ).fetchone()
        if row is None or (
            row["resource_kind"] != lease.resource_kind
            or row["owner_kind"] != lease.owner_kind
            or row["owner_id"] != lease.owner_id
            or row["phase"] not in {"writing", "sealed"}
        ):
            raise RuntimeError(f"{lease_type} lease does not match ledger")
        expected_path = (
            partial_path if row["phase"] == "writing" else sealed_path
        )
        if (
            row["controlled_path"] != str(expected_path)
            or not self._is_controlled_path(
                lease.id,
                lease_type,
                row["phase"],
                expected_path,
            )
            or not self._is_controlled_path(
                lease.id,
                lease_type,
                "writing",
                partial_path,
            )
            or not self._is_controlled_path(
                lease.id,
                lease_type,
                "sealed",
                sealed_path,
            )
        ):
            raise RuntimeError(
                f"{lease_type} lease does not match ledger"
            )
        handle = self._files.get(lease.id)
        if handle is not None:
            handle.close()
            self._files.pop(lease.id, None)
        for path in (partial_path, sealed_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        _fsync_directory(directory)
        with self._transaction():
            changed = self._connection.execute(
                """
                DELETE FROM storage_leases
                WHERE id = ? AND lease_type = ?
                    AND resource_kind = ? AND owner_kind = ?
                    AND owner_id = ? AND phase IN ('writing', 'sealed')
                    AND controlled_path IN (?, ?)
                """,
                (
                    lease.id,
                    lease_type,
                    lease.resource_kind,
                    lease.owner_kind,
                    lease.owner_id,
                    str(partial_path),
                    str(sealed_path),
                ),
            ).rowcount
            if changed != 1:
                raise RuntimeError(
                    f"{lease_type} lease changed during seal cleanup"
                )
        lease._state = "aborted"

    def _abort_writing(
        self,
        lease: UploadLease | ReservedByteWriter,
        *,
        lease_type: str,
        directory: Path,
        suffixes: tuple[str, str],
    ) -> None:
        if lease._state == "aborted":
            return
        self._require_writing(lease, lease_type)
        sealed_path = directory / f"{lease.id}.{suffixes[1]}"
        with self._lock:
            phase_row = self._connection.execute(
                """
                SELECT lease_type, phase
                FROM storage_leases
                WHERE id = ?
                """,
                (lease.id,),
            ).fetchone()
            handle_is_closed = lease.id not in self._files
            cleanup_already_complete = (
                phase_row is None
                and handle_is_closed
                and self._is_controlled_path(
                    lease.id,
                    lease_type,
                    "writing",
                    lease.path,
                )
                and self._is_controlled_path(
                    lease.id,
                    lease_type,
                    "sealed",
                    sealed_path,
                )
                and not lease.path.exists()
                and not sealed_path.exists()
            )
            retry_failed_seal = (
                phase_row is not None
                and phase_row["lease_type"] == lease_type
                and phase_row["phase"] == "sealed"
                and handle_is_closed
            )
            if cleanup_already_complete:
                lease._state = "aborted"
                return
        if retry_failed_seal:
            self._compensate_failed_seal(
                lease,
                lease_type=lease_type,
                directory=directory,
                sealed_path=sealed_path,
            )
            return
        with self._transaction():
            row = self._connection.execute(
                """
                SELECT resource_kind, owner_kind, owner_id, controlled_path
                FROM storage_leases
                WHERE id = ? AND lease_type = ? AND phase = 'writing'
                """,
                (lease.id, lease_type),
            ).fetchone()
            if row is None or (
                row["resource_kind"] != lease.resource_kind
                or row["owner_kind"] != lease.owner_kind
                or row["owner_id"] != lease.owner_id
                or row["controlled_path"] != str(lease.path)
                or not self._is_controlled_path(
                    lease.id, lease_type, "writing", lease.path
                )
            ):
                raise RuntimeError(
                    f"{lease_type} lease does not match ledger"
                )
            handle = self._files.pop(lease.id, None)
            if handle is not None:
                handle.close()
            for suffix in suffixes:
                path = directory / f"{lease.id}.{suffix}"
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            _fsync_directory(directory)
            changed = self._connection.execute(
                """
                DELETE FROM storage_leases
                WHERE id = ? AND lease_type = ? AND phase = 'writing'
                    AND resource_kind = ? AND owner_kind = ?
                    AND owner_id = ? AND controlled_path = ?
                """,
                (
                    lease.id,
                    lease_type,
                    lease.resource_kind,
                    lease.owner_kind,
                    lease.owner_id,
                    str(lease.path),
                ),
            ).rowcount
            if changed != 1:
                raise RuntimeError(
                    f"{lease_type} lease changed during abort"
                )
        lease._state = "aborted"

    def _require_writing(
        self,
        lease: UploadLease | JobUploadLease | ReservedByteWriter,
        lease_type: str,
    ) -> None:
        if lease._state == "sealed":
            raise RuntimeError(f"{lease_type} lease is already sealed")
        if lease._state == "aborted":
            raise RuntimeError(f"{lease_type} lease is already aborted")
        if lease._state != "writing":
            raise RuntimeError(f"{lease_type} lease has invalid state")

    def _resolve_ref(
        self,
        ref: InputRef | ArtifactRef,
        *,
        lease_type: str,
    ) -> Path:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT resource_kind, owner_kind, owner_id, controlled_path
                FROM storage_leases
                WHERE id = ? AND lease_type = ? AND phase = 'sealed'
                """,
                (ref.id, lease_type),
            ).fetchone()
        if row is None or (
            row["resource_kind"] != ref.resource_kind
            or row["owner_kind"] != ref.owner_kind
            or row["owner_id"] != ref.owner_id
            or row["controlled_path"] != str(ref.path)
        ):
            raise RuntimeError(f"sealed {lease_type} reference is stale")
        if not self._is_controlled_path(
            ref.id, lease_type, "sealed", ref.path
        ):
            raise RuntimeError(f"sealed {lease_type} path is not controlled")
        if not ref.path.is_file():
            raise RuntimeError(f"sealed {lease_type} file is missing")
        return ref.path

    def _release_ref(
        self,
        ref: InputRef | ArtifactRef,
        *,
        lease_type: str,
        directory: Path,
    ) -> None:
        if not self._is_controlled_path(
            ref.id, lease_type, "sealed", ref.path
        ):
            raise RuntimeError(f"sealed {lease_type} path is not controlled")
        with self._lock:
            row = self._connection.execute(
                """
                SELECT resource_kind, owner_kind, owner_id, controlled_path
                FROM storage_leases
                WHERE id = ? AND lease_type = ? AND phase = 'sealed'
                """,
                (ref.id, lease_type),
            ).fetchone()
        if row is not None and (
            row["resource_kind"] != ref.resource_kind
            or row["owner_kind"] != ref.owner_kind
            or row["owner_id"] != ref.owner_id
            or row["controlled_path"] != str(ref.path)
        ):
            raise RuntimeError(f"sealed {lease_type} reference is stale")
        try:
            ref.path.unlink()
        except FileNotFoundError:
            pass
        _fsync_directory(directory)
        with self._transaction():
            self._connection.execute(
                """
                DELETE FROM storage_leases
                WHERE id = ? AND lease_type = ? AND phase = 'sealed'
                    AND controlled_path = ?
                """,
                (ref.id, lease_type, str(ref.path)),
            )

    def _reconcile_startup(self) -> None:
        with self._lock:
            marker_rows = self._connection.execute(
                "SELECT * FROM shutdown_marker"
            ).fetchall()
            job_rows = self._connection.execute(
                f"SELECT {_TRANSCRIPTION_JOB_COLUMNS} FROM transcription_jobs"
            ).fetchall()
            lease_rows = self._connection.execute(
                "SELECT * FROM storage_leases"
            ).fetchall()
        if len(marker_rows) > 1:
            raise StorageSchemaError("multiple shutdown markers")
        marker_generation: str | None = None
        if marker_rows:
            marker = marker_rows[0]
            try:
                marker_generation = _validate_nonempty_text(
                    marker["generation"],
                    name="shutdown marker generation",
                )
                _decode_job_timestamp(marker["created_at"])
            except (IndexError, TypeError, ValueError) as error:
                raise StorageSchemaError(
                    "shutdown marker is corrupt"
                ) from error

        jobs_by_id: dict[str, tuple[sqlite3.Row, DurableJob]] = {}
        for row in job_rows:
            job = _decode_transcription_job(row)
            if job.id in jobs_by_id:
                raise StorageSchemaError("duplicate transcription job")
            if not (
                (
                    job.phase in {JobPhase.RECEIVING, JobPhase.DELETING}
                    and job.status is None
                )
                or (
                    job.phase is JobPhase.VISIBLE
                    and job.status
                    in {
                        JobStatus.QUEUED,
                        JobStatus.RUNNING,
                        JobStatus.SUCCEEDED,
                        JobStatus.FAILED,
                        JobStatus.CANCELLED,
                    }
                )
            ):
                raise StorageSchemaError(
                    "unsupported transcription job during startup"
                )
            jobs_by_id[job.id] = (row, job)

        leases_by_id: dict[str, sqlite3.Row] = {}
        generic_cleanup: list[sqlite3.Row] = []
        for row in lease_rows:
            if not self._valid_reconciliation_row(row):
                raise StorageSchemaError("corrupt storage lease")
            lease_id = row["id"]
            if lease_id in leases_by_id:
                raise StorageSchemaError("duplicate storage lease")
            leases_by_id[lease_id] = row
            if row["owner_kind"] in {"sync", "legacy"}:
                generic_cleanup.append(row)

        segment_rows_by_job: dict[str, list[sqlite3.Row]] = {}
        result_rows_by_job: dict[str, list[sqlite3.Row]] = {}
        for row in lease_rows:
            if (
                row["owner_kind"] == "job"
                and row["lease_type"] == "artifact"
            ):
                if row["owner_id"] not in jobs_by_id:
                    raise StorageSchemaError(
                        "job artifact owner is corrupt"
                    )
                rows_by_job = (
                    segment_rows_by_job
                    if row["resource_kind"] == "segment_jsonl"
                    else result_rows_by_job
                )
                rows_by_job.setdefault(row["owner_id"], []).append(row)
        if any(len(rows) != 1 for rows in segment_rows_by_job.values()):
            raise StorageSchemaError("duplicate job segment artifact")
        if any(len(rows) != 1 for rows in result_rows_by_job.values()):
            raise StorageSchemaError("duplicate job result artifact")

        receiving_cleanup: list[tuple[sqlite3.Row, sqlite3.Row]] = []
        terminal_cleanup: list[tuple[sqlite3.Row, sqlite3.Row]] = []
        segment_cleanup: list[sqlite3.Row] = []
        result_cleanup: list[sqlite3.Row] = []
        recoverable_results: dict[
            str, tuple[sqlite3.Row, ResultEnvelopeManifest]
        ] = {}
        protected_paths: set[Path] = set()
        referenced_job_leases: set[str] = set()
        for job_id, (job_row, job) in jobs_by_id.items():
            requires_input = (
                job.phase in {JobPhase.RECEIVING, JobPhase.DELETING}
                or job.status in {JobStatus.QUEUED, JobStatus.RUNNING}
                or job.input_cleanup_pending
            )
            if not requires_input:
                if job.input_lease_id is not None:
                    raise StorageSchemaError(
                        "clean terminal job retains an input reference"
                    )
                continue
            lease_row = leases_by_id.get(job.input_lease_id or "")
            if (
                lease_row is None
                or lease_row["id"] != job_id
                or lease_row["lease_type"] != "upload"
                or lease_row["resource_kind"] != "transcription"
                or lease_row["owner_kind"] != "job"
                or lease_row["owner_id"] != job_id
            ):
                raise StorageSchemaError(
                    "transcription job input reference is corrupt"
                )
            referenced_job_leases.add(lease_row["id"])
            controlled_path = Path(lease_row["controlled_path"])
            if job.phase in {JobPhase.RECEIVING, JobPhase.DELETING}:
                receiving_cleanup.append((job_row, lease_row))
                continue
            if (
                lease_row["phase"] != "sealed"
                or controlled_path
                != self.staging_dir / f"{job_id}.ready"
                or lease_row["reserved_bytes"]
                != lease_row["actual_bytes"]
                or job.input_size_bytes != lease_row["actual_bytes"]
            ):
                raise StorageSchemaError(
                    "visible transcription job input is corrupt"
                )
            if job.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                if not _is_regular_file_with_size(
                    controlled_path,
                    lease_row["actual_bytes"],
                ):
                    raise StorageSchemaError(
                        "active transcription job input is corrupt"
                    )
                protected_paths.add(controlled_path)
            else:
                if not _is_absent_or_regular_file_with_size(
                    controlled_path,
                    lease_row["actual_bytes"],
                ):
                    raise StorageSchemaError(
                        "terminal transcription job input is corrupt"
                    )
                terminal_cleanup.append((job_row, lease_row))

        for job_id, (_, job) in jobs_by_id.items():
            segment_rows = segment_rows_by_job.get(job_id, [])
            if segment_rows:
                segment_row = segment_rows[0]
                referenced_job_leases.add(segment_row["id"])
                segment_cleanup.append(segment_row)
            result_rows = result_rows_by_job.get(job_id, [])
            result_row = result_rows[0] if result_rows else None
            if job.status is JobStatus.SUCCEEDED:
                if (
                    result_row is None
                    or result_row["id"] != job.result_lease_id
                    or result_row["phase"] != "sealed"
                ):
                    raise StorageSchemaError(
                        "succeeded job result reference is corrupt"
                    )
            if result_row is None:
                continue
            referenced_job_leases.add(result_row["id"])
            result_path = Path(result_row["controlled_path"])
            should_recover = (
                job.status is JobStatus.RUNNING
                and not job.cancel_requested
                and job.total_samples is not None
                and job.processed_samples == job.total_samples
            )
            requires_valid_result = (
                job.status is JobStatus.SUCCEEDED or should_recover
            )
            if result_row["phase"] == "writing":
                if job.status is JobStatus.SUCCEEDED:
                    raise StorageSchemaError(
                        "succeeded job result is incomplete"
                    )
                result_cleanup.append(result_row)
                continue
            try:
                result_mode = result_path.lstat().st_mode
            except FileNotFoundError:
                if job.status is JobStatus.SUCCEEDED:
                    raise StorageSchemaError(
                        "succeeded job result is missing"
                    ) from None
                result_cleanup.append(result_row)
                continue
            except OSError as error:
                raise StorageSchemaError(
                    "job result cannot be inspected"
                ) from error
            if not stat.S_ISREG(result_mode):
                raise StorageSchemaError(
                    "job result file is not safely unlinkable"
                )
            if not requires_valid_result:
                result_cleanup.append(result_row)
                continue
            try:
                manifest = self._validate_job_result(
                    job,
                    path=result_path,
                    actual_bytes=result_row["actual_bytes"],
                    content_sha256=result_row["content_sha256"],
                )
            except (CanonicalArtifactError, ValueError) as error:
                storage_mismatch = (
                    isinstance(error, CanonicalArtifactError)
                    and error.reason
                    in {
                        "result envelope size does not match storage",
                        "result envelope SHA-256 does not match storage",
                    }
                )
                if (
                    job.status is JobStatus.SUCCEEDED
                    or storage_mismatch
                ):
                    raise StorageSchemaError(
                        "job result storage is corrupt"
                    ) from error
                result_cleanup.append(result_row)
                continue
            if job.status is JobStatus.SUCCEEDED:
                if manifest.finished_at != job.finished_at:
                    raise StorageSchemaError(
                        "succeeded job result timestamp is corrupt"
                    )
                protected_paths.add(result_path)
            elif should_recover:
                recoverable_results[job_id] = (result_row, manifest)
                protected_paths.add(result_path)
            else:
                result_cleanup.append(result_row)

        for lease_id, row in leases_by_id.items():
            if row["owner_kind"] == "job":
                if lease_id not in referenced_job_leases:
                    raise StorageSchemaError(
                        "job storage lease is unreferenced"
                    )

        preflight_paths: set[Path] = set()
        for _, lease_row in receiving_cleanup + terminal_cleanup:
            preflight_paths.update(
                {
                    self.staging_dir / f"{lease_row['id']}.partial",
                    self.staging_dir / f"{lease_row['id']}.ready",
                }
            )
        for _, job in jobs_by_id.values():
            if job.status is JobStatus.RUNNING:
                preflight_paths.update(
                    {
                        self.staging_dir / f"{job.id}.partial",
                        self.staging_dir / f"{job.id}.ready",
                    }
                )
        for row in result_cleanup:
            preflight_paths.update(
                {
                    self.artifact_dir / f"{row['id']}.partial",
                    self.artifact_dir / f"{row['id']}.complete",
                }
            )
        for row in segment_cleanup:
            preflight_paths.update(
                {
                    self.artifact_dir / f"{row['id']}.partial",
                    self.artifact_dir / f"{row['id']}.complete",
                }
            )
        for row in generic_cleanup:
            lease_type = row["lease_type"]
            directory = (
                self.staging_dir
                if lease_type == "upload"
                else self.artifact_dir
            )
            suffixes = (
                ("partial", "ready")
                if lease_type == "upload"
                else ("partial", "complete")
            )
            preflight_paths.update(
                directory / f"{row['id']}.{suffix}"
                for suffix in suffixes
            )
        for directory, pattern in (
            (self.staging_dir, STAGING_NAME_PATTERN),
            (self.artifact_dir, ARTIFACT_NAME_PATTERN),
        ):
            preflight_paths.update(
                entry
                for entry in directory.iterdir()
                if pattern.fullmatch(entry.name)
                and entry not in protected_paths
            )
        for path in preflight_paths:
            self._preflight_cleanup_path(path)

        running_jobs = tuple(
            job
            for _, job in jobs_by_id.values()
            if job.status is JobStatus.RUNNING
        )
        now = _utc_now()
        _encode_job_timestamp(now)
        with self._transaction():
            for job in running_jobs:
                assert job.attempt_token is not None
                assert job.owner_generation is not None
                assert job.started_at is not None
                if job.cancel_requested:
                    changed = self._connection.execute(
                        """
                        UPDATE transcription_jobs SET
                            status = 'cancelled',
                            attempt_token = NULL,
                            owner_generation = NULL,
                            input_cleanup_pending = 1,
                            finished_at = ?
                        WHERE id = ? AND phase = 'visible'
                          AND status = 'running'
                          AND attempt_token = ?
                          AND owner_generation = ?
                          AND cancel_requested = 1
                        """,
                        (
                            _encode_job_timestamp(
                                max(now, job.started_at)
                            ),
                            job.id,
                            job.attempt_token,
                            job.owner_generation,
                        ),
                    ).rowcount
                elif job.id in recoverable_results:
                    result_row, manifest = recoverable_results[job.id]
                    changed = self._connection.execute(
                        f"""
                        UPDATE transcription_jobs SET
                            status = 'succeeded',
                            attempt_token = NULL,
                            owner_generation = NULL,
                            result_lease_id = ?,
                            input_cleanup_pending = 1,
                            finished_at = ?
                        WHERE id = ? AND phase = 'visible'
                          AND status = 'running'
                          AND attempt_token = ?
                          AND owner_generation = ?
                          AND cancel_requested = 0
                          AND total_samples IS NOT NULL
                          AND processed_samples = total_samples
                          AND result_lease_id IS NULL
                          AND EXISTS (
                              SELECT 1 FROM storage_leases
                              WHERE {_EXACT_SEALED_JOB_RESULT_WHERE}
                          )
                        """,
                        (
                            result_row["id"],
                            _encode_job_timestamp(
                                manifest.finished_at
                            ),
                            job.id,
                            job.attempt_token,
                            job.owner_generation,
                            result_row["id"],
                            job.id,
                            result_row["controlled_path"],
                            result_row["actual_bytes"],
                            result_row["actual_bytes"],
                            result_row["content_sha256"],
                        ),
                    ).rowcount
                elif (
                    marker_generation is not None
                    and job.owner_generation == marker_generation
                ):
                    changed = self._connection.execute(
                        """
                        UPDATE transcription_jobs SET
                            status = 'queued',
                            processed_samples = 0,
                            attempt_token = NULL,
                            owner_generation = NULL,
                            started_at = NULL,
                            finished_at = NULL
                        WHERE id = ? AND phase = 'visible'
                          AND status = 'running'
                          AND attempt_token = ?
                          AND owner_generation = ?
                          AND cancel_requested = 0
                        """,
                        (
                            job.id,
                            job.attempt_token,
                            job.owner_generation,
                        ),
                    ).rowcount
                elif job.crash_recoveries == 0:
                    changed = self._connection.execute(
                        """
                        UPDATE transcription_jobs SET
                            status = 'queued',
                            processed_samples = 0,
                            attempt_token = NULL,
                            owner_generation = NULL,
                            crash_recoveries = 1,
                            started_at = NULL,
                            finished_at = NULL
                        WHERE id = ? AND phase = 'visible'
                          AND status = 'running'
                          AND attempt_token = ?
                          AND owner_generation = ?
                          AND crash_recoveries = 0
                          AND cancel_requested = 0
                        """,
                        (
                            job.id,
                            job.attempt_token,
                            job.owner_generation,
                        ),
                    ).rowcount
                else:
                    changed = self._connection.execute(
                        """
                        UPDATE transcription_jobs SET
                            status = 'failed',
                            attempt_token = NULL,
                            owner_generation = NULL,
                            error_code = 'worker_crashed',
                            input_cleanup_pending = 1,
                            finished_at = ?
                        WHERE id = ? AND phase = 'visible'
                          AND status = 'running'
                          AND attempt_token = ?
                          AND owner_generation = ?
                          AND crash_recoveries = 1
                          AND cancel_requested = 0
                        """,
                        (
                            _encode_job_timestamp(
                                max(now, job.started_at)
                            ),
                            job.id,
                            job.attempt_token,
                            job.owner_generation,
                        ),
                    ).rowcount
                if changed != 1:
                    raise StorageSchemaError(
                        "running job changed during startup"
                    )
            for job_row, _ in receiving_cleanup:
                if job_row["phase"] == "receiving":
                    changed = self._connection.execute(
                        """
                        UPDATE transcription_jobs SET phase = 'deleting'
                        WHERE id = ? AND phase = 'receiving'
                          AND status IS NULL AND input_lease_id = ?
                        """,
                        (job_row["id"], job_row["input_lease_id"]),
                    ).rowcount
                    if changed != 1:
                        raise StorageSchemaError(
                            "transcription job changed during startup"
                        )
            self._connection.execute("DELETE FROM shutdown_marker")
            classified_rows = self._connection.execute(
                f"SELECT {_TRANSCRIPTION_JOB_COLUMNS} FROM transcription_jobs"
            ).fetchall()
            for row in classified_rows:
                _decode_transcription_job(row)

        classified_jobs = {
            row["id"]: (row, _decode_transcription_job(row))
            for row in classified_rows
        }
        receiving_cleanup = []
        terminal_cleanup = []
        protected_paths = set()
        for job_id, (job_row, job) in classified_jobs.items():
            lease_row = leases_by_id.get(job.input_lease_id or "")
            if job.phase is JobPhase.DELETING and job.status is None:
                if lease_row is None:
                    raise StorageSchemaError(
                        "deleting job input lease disappeared"
                    )
                receiving_cleanup.append((job_row, lease_row))
            elif (
                job.status
                in {
                    JobStatus.SUCCEEDED,
                    JobStatus.FAILED,
                    JobStatus.CANCELLED,
                }
                and job.input_cleanup_pending
            ):
                if lease_row is None:
                    raise StorageSchemaError(
                        "terminal job input lease disappeared"
                    )
                terminal_cleanup.append((job_row, lease_row))
            elif job.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                if lease_row is None:
                    raise StorageSchemaError(
                        "active job input lease disappeared"
                    )
                protected_paths.add(
                    self.staging_dir / f"{job_id}.ready"
                )
            if job.status is JobStatus.SUCCEEDED:
                result_row = leases_by_id.get(
                    job.result_lease_id or ""
                )
                if result_row is None:
                    raise StorageSchemaError(
                        "succeeded job result lease disappeared"
                    )
                protected_paths.add(
                    Path(result_row["controlled_path"])
                )

        cleanup_paths: set[Path] = set()
        for _, lease_row in receiving_cleanup + terminal_cleanup:
            cleanup_paths.update(
                {
                    self.staging_dir / f"{lease_row['id']}.partial",
                    self.staging_dir / f"{lease_row['id']}.ready",
                }
            )
        for row in result_cleanup:
            cleanup_paths.update(
                {
                    self.artifact_dir / f"{row['id']}.partial",
                    self.artifact_dir / f"{row['id']}.complete",
                }
            )
        for row in segment_cleanup:
            cleanup_paths.update(
                {
                    self.artifact_dir / f"{row['id']}.partial",
                    self.artifact_dir / f"{row['id']}.complete",
                }
            )
        for row in generic_cleanup:
            directory = (
                self.staging_dir
                if row["lease_type"] == "upload"
                else self.artifact_dir
            )
            suffixes = (
                ("partial", "ready")
                if row["lease_type"] == "upload"
                else ("partial", "complete")
            )
            cleanup_paths.update(
                directory / f"{row['id']}.{suffix}"
                for suffix in suffixes
            )
        for directory, pattern in (
            (self.staging_dir, STAGING_NAME_PATTERN),
            (self.artifact_dir, ARTIFACT_NAME_PATTERN),
        ):
            cleanup_paths.update(
                entry
                for entry in directory.iterdir()
                if pattern.fullmatch(entry.name)
                and entry not in protected_paths
            )
        if not cleanup_paths.issubset(preflight_paths):
            raise StorageSchemaError(
                "startup cleanup plan changed after classification"
            )

        touched_dirs = {path.parent for path in cleanup_paths}
        for path in cleanup_paths:
            self._unlink_if_present(path)
        for directory in touched_dirs:
            _fsync_directory(directory)

        if (
            receiving_cleanup
            or terminal_cleanup
            or segment_cleanup
            or result_cleanup
            or generic_cleanup
        ):
            with self._transaction():
                for job_row, lease_row in receiving_cleanup:
                    current_job = self._connection.execute(
                        f"""
                        SELECT {_TRANSCRIPTION_JOB_COLUMNS}
                        FROM transcription_jobs WHERE id = ?
                        """,
                        (job_row["id"],),
                    ).fetchone()
                    current_lease = self._connection.execute(
                        "SELECT * FROM storage_leases WHERE id = ?",
                        (lease_row["id"],),
                    ).fetchone()
                    job_keys = job_row.keys()
                    expected_job = tuple(
                        "deleting" if key == "phase" else job_row[key]
                        for key in job_keys
                    )
                    if (
                        current_job is None
                        or tuple(current_job) != expected_job
                        or current_lease is None
                        or tuple(current_lease) != tuple(lease_row)
                    ):
                        raise StorageSchemaError(
                            "job cleanup state changed during startup"
                        )
                    self._connection.execute(
                        "DELETE FROM storage_leases WHERE id = ?",
                        (lease_row["id"],),
                    )
                    self._connection.execute(
                        """
                        DELETE FROM transcription_jobs
                        WHERE id = ? AND phase = 'deleting'
                        """,
                        (job_row["id"],),
                    )
                for job_row, lease_row in terminal_cleanup:
                    current_job = self._connection.execute(
                        f"""
                        SELECT {_TRANSCRIPTION_JOB_COLUMNS}
                        FROM transcription_jobs WHERE id = ?
                        """,
                        (job_row["id"],),
                    ).fetchone()
                    current_lease = self._connection.execute(
                        "SELECT * FROM storage_leases WHERE id = ?",
                        (lease_row["id"],),
                    ).fetchone()
                    if (
                        current_job is None
                        or tuple(current_job) != tuple(job_row)
                        or current_lease is None
                        or tuple(current_lease) != tuple(lease_row)
                    ):
                        raise StorageSchemaError(
                            "terminal cleanup state changed during startup"
                        )
                    self._connection.execute(
                        "DELETE FROM storage_leases WHERE id = ?",
                        (lease_row["id"],),
                    )
                    changed = self._connection.execute(
                        """
                        UPDATE transcription_jobs SET
                            input_lease_id = NULL,
                            input_cleanup_pending = 0
                        WHERE id = ? AND phase = 'visible'
                          AND status IN (
                              'succeeded', 'failed', 'cancelled'
                          )
                          AND input_lease_id = ?
                          AND input_cleanup_pending = 1
                        """,
                        (job_row["id"], lease_row["id"]),
                    ).rowcount
                    if changed != 1:
                        raise StorageSchemaError(
                            "terminal job changed during input cleanup"
                        )
                    cleaned_row = self._connection.execute(
                        f"""
                        SELECT {_TRANSCRIPTION_JOB_COLUMNS}
                        FROM transcription_jobs WHERE id = ?
                        """,
                        (job_row["id"],),
                    ).fetchone()
                    if cleaned_row is None:
                        raise StorageSchemaError(
                            "terminal job disappeared during cleanup"
                        )
                    _decode_transcription_job(cleaned_row)
                for row in segment_cleanup:
                    current = self._connection.execute(
                        "SELECT * FROM storage_leases WHERE id = ?",
                        (row["id"],),
                    ).fetchone()
                    if current is None or tuple(current) != tuple(row):
                        raise StorageSchemaError(
                            "job segment lease changed during startup"
                        )
                    self._connection.execute(
                        "DELETE FROM storage_leases WHERE id = ?",
                        (row["id"],),
                    )
                for row in result_cleanup:
                    current = self._connection.execute(
                        "SELECT * FROM storage_leases WHERE id = ?",
                        (row["id"],),
                    ).fetchone()
                    if current is None or tuple(current) != tuple(row):
                        raise StorageSchemaError(
                            "job result lease changed during startup"
                        )
                    self._connection.execute(
                        "DELETE FROM storage_leases WHERE id = ?",
                        (row["id"],),
                    )
                for row in generic_cleanup:
                    current = self._connection.execute(
                        "SELECT * FROM storage_leases WHERE id = ?",
                        (row["id"],),
                    ).fetchone()
                    if current is None or tuple(current) != tuple(row):
                        raise StorageSchemaError(
                            "storage lease changed during startup"
                        )
                    self._connection.execute(
                        "DELETE FROM storage_leases WHERE id = ?",
                        (row["id"],),
                    )

    def _valid_reconciliation_row(self, row: sqlite3.Row) -> bool:
        lease_id = row["id"]
        lease_type = row["lease_type"]
        phase = row["phase"]
        resource_kind = row["resource_kind"]
        owner_kind = row["owner_kind"]
        owner_id = row["owner_id"]
        reserved_bytes = row["reserved_bytes"]
        actual_bytes = row["actual_bytes"]
        content_sha256 = row["content_sha256"]
        return bool(
            type(lease_id) is str
            and LEASE_ID_PATTERN.fullmatch(lease_id)
            and lease_type in LEASE_TYPES
            and phase in {"writing", "sealed"}
            and owner_kind in {"sync", "job", "legacy"}
            and type(owner_id) is str
            and owner_id
            and type(resource_kind) is str
            and resource_kind
            and (
                lease_type == "upload"
                or resource_kind in ARTIFACT_KINDS
            )
            and type(reserved_bytes) is int
            and type(actual_bytes) is int
            and 0 <= actual_bytes <= reserved_bytes
            and (
                content_sha256 is None
                or (
                    type(content_sha256) is str
                    and re.fullmatch(r"[0-9a-f]{64}", content_sha256)
                    is not None
                )
            )
            and type(row["created_at"]) is str
            and bool(row["created_at"])
            and self._is_controlled_path(
                lease_id,
                lease_type,
                phase,
                Path(row["controlled_path"]),
            )
            and (
                owner_kind != "job"
                or (
                    (
                        (
                            owner_id == lease_id
                            and lease_type == "upload"
                            and resource_kind == "transcription"
                        )
                        or (
                            lease_type == "artifact"
                            and resource_kind
                            in {"segment_jsonl", "result_complete"}
                            and re.fullmatch(
                                r"[0-9a-f]{32}", lease_id
                            )
                            is not None
                            and re.fullmatch(
                                r"[0-9A-HJKMNP-TV-Z]{8}", owner_id
                            )
                            is not None
                        )
                    )
                    and (
                        (
                            phase == "writing"
                            and content_sha256 is None
                        )
                        or (
                            phase == "sealed"
                            and reserved_bytes == actual_bytes
                            and content_sha256 is not None
                        )
                    )
                )
            )
        )

    def _unlink_if_present(self, path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def _preflight_cleanup_path(self, path: Path) -> None:
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            return
        except OSError as error:
            raise StorageSchemaError(
                "storage cleanup path cannot be inspected"
            ) from error
        if not (stat.S_ISREG(mode) or stat.S_ISLNK(mode)):
            raise StorageSchemaError(
                "storage cleanup path is not safely unlinkable"
            )

    def total_reserved_bytes(self) -> int:
        with self._lock:
            return int(
                self._connection.execute(
                    """
                    SELECT COALESCE(SUM(reserved_bytes), 0)
                    FROM storage_leases
                    """
                ).fetchone()[0]
            )

    def active_upload_count(self) -> int:
        with self._lock:
            return int(
                self._connection.execute(
                    """
                    SELECT COUNT(*) FROM storage_leases
                    WHERE lease_type = 'upload' AND phase = 'writing'
                    """
                ).fetchone()[0]
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            for handle in self._files.values():
                handle.close()
            self._files.clear()
            self._connection.close()
            self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    def _admit_delta(self, delta: int) -> None:
        reserved = int(
            self._connection.execute(
                """
                SELECT COALESCE(SUM(reserved_bytes), 0)
                FROM storage_leases
                """
            ).fetchone()[0]
        )
        if reserved + delta > self.limits.max_job_storage_bytes:
            raise StorageAdmissionError(
                "storage_capacity_exceeded",
                "storage reservation capacity exceeded",
            )
        outstanding = int(
            self._connection.execute(
                """
                SELECT COALESCE(SUM(reserved_bytes - actual_bytes), 0)
                FROM storage_leases
                """
            ).fetchone()[0]
        )
        if (
            self._free_bytes(self.data_dir) - outstanding - delta
            < self.limits.min_filesystem_free_bytes
        ):
            raise StorageAdmissionError(
                "storage_capacity_exceeded",
                "storage capacity is unavailable",
            )

    def _validate_owner(
        self, owner_kind: str, owner_id: str | None
    ) -> None:
        if owner_kind != "sync":
            raise ValueError("unsupported storage owner kind")
        if owner_id is not None and not owner_id:
            raise ValueError("storage owner id must not be empty")

    def _is_controlled_path(
        self,
        lease_id: str,
        lease_type: str,
        phase: str,
        path: Path,
    ) -> bool:
        if LEASE_ID_PATTERN.fullmatch(lease_id) is None:
            return False
        if path.is_symlink():
            return False
        if lease_type == "upload":
            directory = self.staging_dir
            suffix = "partial" if phase == "writing" else "ready"
        elif lease_type == "artifact":
            directory = self.artifact_dir
            suffix = "partial" if phase == "writing" else "complete"
        else:
            return False
        return (
            path.parent == directory
            and path.resolve().parent == directory
            and path.name == f"{lease_id}.{suffix}"
        )

    def _transaction(self) -> _Transaction:
        return _Transaction(self)


def _encode_speaker_profile(profile: SpeakerProfile) -> tuple[object, ...]:
    return (
        profile.id,
        profile.name,
        profile.name_key,
        profile.description,
        sqlite3.Binary(profile.embedding.to_bytes()),
        profile.embedding_model_id,
        profile.embedding_model_revision,
        profile.embedding_dimension,
        profile.embedding_policy_fingerprint,
        profile.sample_count,
        _encode_profile_timestamp(profile.created_at),
        _encode_profile_timestamp(profile.updated_at),
    )


def _decode_speaker_profile(row: sqlite3.Row) -> SpeakerProfile:
    try:
        dimension = row["embedding_dimension"]
        profile = SpeakerProfile(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            embedding=SpeakerEmbedding.from_bytes(
                row["embedding"],
                dimension=dimension,
            ),
            embedding_model_id=row["embedding_model_id"],
            embedding_model_revision=row["embedding_model_revision"],
            embedding_dimension=dimension,
            embedding_policy_fingerprint=row[
                "embedding_policy_fingerprint"
            ],
            sample_count=row["sample_count"],
            created_at=_decode_profile_timestamp(row["created_at"]),
            updated_at=_decode_profile_timestamp(row["updated_at"]),
        )
        if (
            row["name"] != profile.name
            or row["name_key"] != profile.name_key
            or row["description"] != profile.description
        ):
            raise ValueError("speaker profile text fields are inconsistent")
    except (IndexError, TypeError, ValueError) as error:
        raise StorageSchemaError("speaker profile row is corrupt") from error
    return profile


def _decode_transcription_job(row: sqlite3.Row) -> DurableJob:
    try:
        job = DurableJob(
            id=row["id"],
            phase=JobPhase(row["phase"]),
            status=(
                None
                if row["status"] is None
                else JobStatus(row["status"])
            ),
            input_lease_id=row["input_lease_id"],
            canonical_options_json=row["canonical_options_json"],
            selected_speaker_snapshot=row["selected_speaker_snapshot"],
            snapshot_sha256=row["snapshot_sha256"],
            input_size_bytes=row["input_size_bytes"],
            effective_max_audio_samples=row["effective_max_audio_samples"],
            effective_direct_max_audio_samples=row[
                "effective_direct_max_audio_samples"
            ],
            total_samples=row["total_samples"],
            processed_samples=row["processed_samples"],
            request_fingerprint=row["request_fingerprint"],
            processor_fingerprint=row["processor_fingerprint"],
            attempt_no=row["attempt_no"],
            attempt_token=row["attempt_token"],
            owner_generation=row["owner_generation"],
            crash_recoveries=row["crash_recoveries"],
            cancel_requested=_decode_sqlite_boolean(
                row["cancel_requested"],
                name="cancel_requested",
            ),
            result_lease_id=row["result_lease_id"],
            error_code=row["error_code"],
            input_cleanup_pending=_decode_sqlite_boolean(
                row["input_cleanup_pending"],
                name="input_cleanup_pending",
            ),
            created_at=_decode_job_timestamp(row["created_at"]),
            started_at=(
                None
                if row["started_at"] is None
                else _decode_job_timestamp(row["started_at"])
            ),
            finished_at=(
                None
                if row["finished_at"] is None
                else _decode_job_timestamp(row["finished_at"])
            ),
        )
    except (IndexError, TypeError, ValueError) as error:
        raise StorageSchemaError("transcription job row is corrupt") from error
    return job


def _decode_sqlite_boolean(value: object, *, name: str) -> bool:
    if type(value) is not int or value not in {0, 1}:
        raise ValueError(f"job {name} is not a canonical boolean")
    return bool(value)


def _validate_nonempty_text(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _encode_job_timestamp(value: datetime) -> str:
    if type(value) is not datetime:
        raise TypeError("job timestamp must be a datetime")
    if value.tzinfo is not timezone.utc:
        raise ValueError("job timestamp is not canonical UTC")
    fraction = f".{value.microsecond:06d}" if value.microsecond else ""
    return (
        f"{value.year:04d}-{value.month:02d}-{value.day:02d}"
        f"T{value.hour:02d}:{value.minute:02d}:{value.second:02d}"
        f"{fraction}Z"
    )


def _decode_job_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise TypeError("job timestamp must be text")
    parsed = datetime.strptime(
        value,
        (
            "%Y-%m-%dT%H:%M:%S.%fZ"
            if "." in value
            else "%Y-%m-%dT%H:%M:%SZ"
        ),
    ).replace(tzinfo=timezone.utc)
    if _encode_job_timestamp(parsed) != value:
        raise ValueError("job timestamp is not canonical")
    return parsed


def _encode_profile_timestamp(value: datetime) -> str:
    if value.tzinfo is not timezone.utc:
        raise ValueError("speaker profile timestamp is not canonical UTC")
    return (
        f"{value.year:04d}-{value.month:02d}-{value.day:02d}"
        f"T{value.hour:02d}:{value.minute:02d}:{value.second:02d}"
        f".{value.microsecond:06d}Z"
    )


def _decode_profile_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise TypeError("speaker profile timestamp must be text")
    parsed = datetime.strptime(
        value,
        "%Y-%m-%dT%H:%M:%S.%fZ",
    ).replace(tzinfo=timezone.utc)
    if _encode_profile_timestamp(parsed) != value:
        raise ValueError("speaker profile timestamp is not canonical")
    return parsed


class _Transaction:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    def __enter__(self) -> None:
        self.storage._lock.acquire()
        try:
            self.storage._connection.execute("BEGIN IMMEDIATE")
        except BaseException:
            self.storage._lock.release()
            raise

    def __exit__(self, exc_type, exc, traceback) -> bool:
        try:
            if exc_type is not None:
                self.storage._connection.execute("ROLLBACK")
            else:
                try:
                    self.storage._connection.execute("COMMIT")
                except BaseException:
                    try:
                        self.storage._connection.execute("ROLLBACK")
                    except BaseException:
                        pass
                    raise
        finally:
            self.storage._lock.release()
        return False


def _round_reservation(byte_count: int) -> int:
    return max(
        RESERVATION_QUANTUM,
        ((byte_count + RESERVATION_QUANTUM - 1)
        // RESERVATION_QUANTUM)
        * RESERVATION_QUANTUM,
    )


def _is_regular_file_with_size(path: Path, size: int) -> bool:
    try:
        return (
            not path.is_symlink()
            and path.is_file()
            and path.stat().st_size == size
        )
    except OSError:
        return False


def _is_absent_or_regular_file_with_size(path: Path, size: int) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return _is_regular_file_with_size(path, size)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
