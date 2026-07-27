from __future__ import annotations

import hashlib
import os
import re
import shutil
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Callable

from botified_asr.config import LimitsConfig, RESERVATION_QUANTUM


LEASE_ID_PATTERN = re.compile(
    r"^(?:[0-9a-f]{32}|[0-9A-HJKMNP-TV-Z]{8})$"
)
STAGING_NAME_PATTERN = re.compile(
    r"^(?:[0-9a-f]{32}|[0-9A-HJKMNP-TV-Z]{8})\.(?:partial|ready)$"
)


class StorageAdmissionError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class UploadLease:
    id: str
    kind: str
    path: Path
    reserved_bytes: int
    actual_bytes: int = 0
    content_sha256: str | None = None
    _hasher: object = field(default_factory=hashlib.sha256, repr=False)
    _checkpointed_bytes: int = field(default=0, repr=False)


class Storage:
    """SQLite-owned upload leases and filesystem staging."""

    def __init__(
        self,
        data_dir: str | Path,
        limits: LimitsConfig,
        *,
        free_bytes: Callable[[Path], int] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.staging_dir = self.data_dir / "staging"
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.staging_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.data_dir.chmod(0o700)
        self.staging_dir.chmod(0o700)
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
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        Path(self._connection.execute("PRAGMA database_list").fetchone()[2]).chmod(
            0o600
        )
        self._create_schema()
        self.cleanup_receiving()

    def _create_schema(self) -> None:
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    version INTEGER NOT NULL
                );
                INSERT OR IGNORE INTO schema_meta(singleton, version) VALUES (1, 1);

                CREATE TABLE IF NOT EXISTS upload_leases (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    phase TEXT NOT NULL CHECK (phase = 'receiving'),
                    staging_path TEXT NOT NULL UNIQUE,
                    reserved_bytes INTEGER NOT NULL CHECK (reserved_bytes >= 0),
                    actual_bytes INTEGER NOT NULL CHECK (actual_bytes >= 0),
                    content_sha256 TEXT,
                    created_at TEXT NOT NULL DEFAULT (
                        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    )
                );
                """
            )

    def begin_upload(self, kind: str) -> UploadLease:
        lease_id = uuid.uuid4().hex
        path = self.staging_dir / f"{lease_id}.partial"
        reservation = RESERVATION_QUANTUM
        with self._transaction():
            active = self._connection.execute(
                "SELECT COUNT(*) FROM upload_leases"
            ).fetchone()[0]
            if active >= self.limits.max_active_uploads:
                raise StorageAdmissionError(
                    "too_many_active_uploads", "too many active uploads"
                )
            self._admit_delta(reservation)
            self._connection.execute(
                """
                INSERT INTO upload_leases(
                    id, kind, phase, staging_path, reserved_bytes, actual_bytes
                ) VALUES (?, ?, 'receiving', ?, ?, 0)
                """,
                (lease_id, kind, str(path), reservation),
            )

        try:
            handle = path.open("xb")
        except OSError:
            with self._transaction():
                self._connection.execute(
                    "DELETE FROM upload_leases WHERE id = ?", (lease_id,)
                )
            raise
        self._files[lease_id] = handle
        return UploadLease(lease_id, kind, path, reservation)

    def append(self, lease: UploadLease, data: bytes) -> None:
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
                    UPDATE upload_leases
                    SET reserved_bytes = ?
                    WHERE id = ? AND phase = 'receiving'
                    """,
                    (required, lease.id),
                ).rowcount
                if changed != 1:
                    raise RuntimeError("upload lease is no longer receiving")
            lease.reserved_bytes = required

        handle = self._files.get(lease.id)
        if handle is None:
            raise RuntimeError("upload lease is closed")
        handle.write(data)
        lease._hasher.update(data)
        lease.actual_bytes = next_actual
        checkpoint = (
            next_actual // RESERVATION_QUANTUM
        ) * RESERVATION_QUANTUM
        if checkpoint > lease._checkpointed_bytes:
            with self._transaction():
                changed = self._connection.execute(
                    """
                    UPDATE upload_leases
                    SET actual_bytes = ?
                    WHERE id = ? AND phase = 'receiving'
                    """,
                    (checkpoint, lease.id),
                ).rowcount
                if changed != 1:
                    raise RuntimeError("upload lease is no longer receiving")
            lease._checkpointed_bytes = checkpoint

    def complete_upload(self, lease: UploadLease) -> Path:
        handle = self._files.pop(lease.id, None)
        if handle is None:
            raise RuntimeError("upload lease is closed")
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        ready_path = self.staging_dir / f"{lease.id}.ready"
        os.replace(lease.path, ready_path)
        _fsync_directory(self.staging_dir)
        content_sha256 = lease._hasher.hexdigest()
        with self._transaction():
            changed = self._connection.execute(
                """
                UPDATE upload_leases
                SET staging_path = ?, reserved_bytes = ?, actual_bytes = ?,
                    content_sha256 = ?
                WHERE id = ? AND phase = 'receiving'
                """,
                (
                    str(ready_path),
                    lease.actual_bytes,
                    lease.actual_bytes,
                    content_sha256,
                    lease.id,
                ),
            ).rowcount
            if changed != 1:
                raise RuntimeError("upload lease is no longer receiving")
        lease.path = ready_path
        lease.reserved_bytes = lease.actual_bytes
        lease.content_sha256 = content_sha256
        return ready_path

    def abort_upload(self, lease: UploadLease) -> None:
        handle = self._files.pop(lease.id, None)
        if handle is not None:
            handle.close()
        paths = {
            lease.path,
            self.staging_dir / f"{lease.id}.partial",
            self.staging_dir / f"{lease.id}.ready",
        }
        for path in paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        _fsync_directory(self.staging_dir)
        with self._transaction():
            self._connection.execute(
                "DELETE FROM upload_leases WHERE id = ?", (lease.id,)
            )

    def cleanup_receiving(self) -> None:
        with self._lock:
            rows = self._connection.execute(
                "SELECT id, staging_path FROM upload_leases WHERE phase = 'receiving'"
            ).fetchall()
        for row in rows:
            path = Path(row["staging_path"])
            if self._is_controlled_staging_path(row["id"], path):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            if LEASE_ID_PATTERN.fullmatch(row["id"]):
                for suffix in ("partial", "ready"):
                    sibling = self.staging_dir / f"{row['id']}.{suffix}"
                    try:
                        sibling.unlink()
                    except FileNotFoundError:
                        pass
        for entry in self.staging_dir.iterdir():
            if (
                STAGING_NAME_PATTERN.fullmatch(entry.name)
                and (entry.is_file() or entry.is_symlink())
            ):
                try:
                    entry.unlink()
                except FileNotFoundError:
                    pass
        if rows:
            _fsync_directory(self.staging_dir)
        with self._transaction():
            self._connection.execute(
                "DELETE FROM upload_leases WHERE phase = 'receiving'"
            )

    def total_reserved_bytes(self) -> int:
        with self._lock:
            return int(
                self._connection.execute(
                    "SELECT COALESCE(SUM(reserved_bytes), 0) FROM upload_leases"
                ).fetchone()[0]
            )

    def receiving_count(self) -> int:
        with self._lock:
            return int(
                self._connection.execute(
                    "SELECT COUNT(*) FROM upload_leases WHERE phase = 'receiving'"
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
                "SELECT COALESCE(SUM(reserved_bytes), 0) FROM upload_leases"
            ).fetchone()[0]
        )
        if reserved + delta > self.limits.max_job_storage_bytes:
            raise StorageAdmissionError(
                "storage_capacity_exceeded", "storage reservation capacity exceeded"
            )
        outstanding = int(
            self._connection.execute(
                """
                SELECT COALESCE(SUM(reserved_bytes - actual_bytes), 0)
                FROM upload_leases
                """
            ).fetchone()[0]
        )
        if (
            self._free_bytes(self.data_dir) - outstanding - delta
            < self.limits.min_filesystem_free_bytes
        ):
            raise StorageAdmissionError(
                "filesystem_free_space_exceeded",
                "filesystem free-space floor would be crossed",
            )

    def _is_controlled_staging_path(self, lease_id: str, path: Path) -> bool:
        if LEASE_ID_PATTERN.fullmatch(lease_id) is None:
            return False
        return path.parent == self.staging_dir and path.resolve().parent == self.staging_dir and path.name in {
            f"{lease_id}.partial",
            f"{lease_id}.ready",
        }

    def _transaction(self):
        return _Transaction(self)


class _Transaction:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    def __enter__(self) -> None:
        self.storage._lock.acquire()
        self.storage._connection.execute("BEGIN IMMEDIATE")

    def __exit__(self, exc_type, exc, traceback) -> bool:
        try:
            self.storage._connection.execute(
                "ROLLBACK" if exc_type is not None else "COMMIT"
            )
        finally:
            self.storage._lock.release()
        return False


def _round_reservation(byte_count: int) -> int:
    return max(
        RESERVATION_QUANTUM,
        ((byte_count + RESERVATION_QUANTUM - 1) // RESERVATION_QUANTUM)
        * RESERVATION_QUANTUM,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
