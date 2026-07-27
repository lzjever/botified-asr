from __future__ import annotations

import hashlib
import os
import re
import shutil
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Callable

from botified_asr.config import LimitsConfig, RESERVATION_QUANTUM
from botified_asr.speaker_profiles import (
    KEEP_EXISTING,
    SpeakerEmbedding,
    SpeakerEmbeddingReplacement,
    SpeakerProfile,
    SpeakerProfileUpdate,
)


SCHEMA_VERSION = 3
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
_SPEAKER_PROFILE_COLUMNS = """
    id, name, name_key, description, embedding,
    embedding_model_id, embedding_model_revision,
    embedding_dimension, embedding_policy_fingerprint,
    sample_count, created_at, updated_at
"""


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


class Storage:
    """SQLite-owned generic storage leases and controlled files."""

    def __init__(
        self,
        data_dir: str | Path,
        limits: LimitsConfig,
        *,
        free_bytes: Callable[[Path], int] | None = None,
    ) -> None:
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
            self._reconcile_startup()
        except BaseException:
            self._connection.close()
            raise

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
            self._verify_v3_schema()
            return
        if version == 1:
            self._migrate_v1_to_v2()
            version = 2
        if version == 2:
            self._migrate_v2_to_v3()
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

    def release_input(self, ref: InputRef) -> None:
        if not isinstance(ref, InputRef):
            raise TypeError("release_input requires an InputRef")
        self._release_ref(
            ref,
            lease_type="upload",
            directory=self.staging_dir,
        )

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
        lease: UploadLease | ReservedByteWriter,
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
        lease: UploadLease | ReservedByteWriter,
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
        lease: UploadLease | ReservedByteWriter,
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
        lease: UploadLease | ReservedByteWriter,
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
            rows = self._connection.execute(
                "SELECT * FROM storage_leases"
            ).fetchall()
        for row in rows:
            if not self._valid_reconciliation_row(row):
                raise StorageSchemaError("corrupt storage lease")

        touched_dirs: set[Path] = set()
        for row in rows:
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
            for suffix in suffixes:
                candidate = directory / f"{row['id']}.{suffix}"
                try:
                    candidate.unlink()
                except FileNotFoundError:
                    pass
            touched_dirs.add(directory)
        for directory in touched_dirs:
            _fsync_directory(directory)
        if rows:
            with self._transaction():
                self._connection.execute("DELETE FROM storage_leases")

        self._remove_strict_orphans(
            self.staging_dir, STAGING_NAME_PATTERN
        )
        self._remove_strict_orphans(
            self.artifact_dir, ARTIFACT_NAME_PATTERN
        )

    def _valid_reconciliation_row(self, row: sqlite3.Row) -> bool:
        lease_id = row["id"]
        lease_type = row["lease_type"]
        phase = row["phase"]
        resource_kind = row["resource_kind"]
        return bool(
            isinstance(lease_id, str)
            and LEASE_ID_PATTERN.fullmatch(lease_id)
            and lease_type in LEASE_TYPES
            and phase in {"writing", "sealed"}
            and row["owner_kind"] in {"sync", "legacy"}
            and isinstance(row["owner_id"], str)
            and row["owner_id"]
            and isinstance(resource_kind, str)
            and resource_kind
            and (
                lease_type == "upload"
                or resource_kind in ARTIFACT_KINDS
            )
            and self._is_controlled_path(
                lease_id,
                lease_type,
                phase,
                Path(row["controlled_path"]),
            )
        )

    def _remove_strict_orphans(
        self,
        directory: Path,
        pattern: re.Pattern[str],
    ) -> None:
        removed = False
        for entry in directory.iterdir():
            if (
                pattern.fullmatch(entry.name)
                and (entry.is_file() or entry.is_symlink())
            ):
                try:
                    entry.unlink()
                    removed = True
                except FileNotFoundError:
                    pass
        if removed:
            _fsync_directory(directory)

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


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
