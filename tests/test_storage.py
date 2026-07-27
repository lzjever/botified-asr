from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path

import pytest

from botified_asr.config import LimitsConfig
from botified_asr.storage import (
    RESERVATION_QUANTUM,
    Storage,
    StorageAdmissionError,
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


def test_upload_lease_and_initial_reservation_are_atomic(tmp_path: Path) -> None:
    storage = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    try:
        lease = storage.begin_upload("transcription")

        assert lease.path.name.endswith(".partial")
        assert storage.total_reserved_bytes() == RESERVATION_QUANTUM
        assert storage.receiving_count() == 1
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
        assert recovered.receiving_count() == 0
        assert recovered.total_reserved_bytes() == 0
        assert not lease.path.exists()
    finally:
        recovered.close()


def test_abort_is_idempotent_and_removes_file(tmp_path: Path) -> None:
    storage = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    try:
        lease = storage.begin_upload("transcription")
        storage.append(lease, b"content")

        storage.abort_upload(lease)
        storage.abort_upload(lease)

        assert not lease.path.exists()
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


def test_startup_also_cleans_ready_file_still_in_receiving_phase(
    tmp_path: Path,
) -> None:
    first = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    lease = first.begin_upload("transcription")
    first.append(lease, b"complete-but-not-promoted")
    ready_path = first.complete_upload(lease)
    assert ready_path.exists()
    first.close()

    recovered = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    try:
        assert recovered.receiving_count() == 0
        assert recovered.total_reserved_bytes() == 0
        assert not ready_path.exists()
    finally:
        recovered.close()


def test_completion_records_incremental_content_hash(tmp_path: Path) -> None:
    storage = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    try:
        lease = storage.begin_upload("transcription")
        storage.append(lease, b"first-")
        storage.append(lease, b"second")

        storage.complete_upload(lease)

        assert lease.content_sha256 == hashlib.sha256(
            b"first-second"
        ).hexdigest()
    finally:
        storage.abort_upload(lease)
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

        assert caught.value.code == "filesystem_free_space_exceeded"
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
            if "UPDATE upload_leases" in sql
            else None
        )
        lease = storage.begin_upload("transcription")
        remaining = RESERVATION_QUANTUM + 1
        while remaining:
            size = min(chunk_size, remaining)
            storage.append(lease, b"x" * size)
            remaining -= size
        storage.complete_upload(lease)
        storage.abort_upload(lease)
        storage.close()
        return len(updates)

    small = count_updates(64 * 1024, tmp_path / "small")
    large = count_updates(1024 * 1024, tmp_path / "large")

    assert small == large
    assert small <= 3


def test_cleanup_removes_strict_orphans_but_never_escapes_data_dir(
    tmp_path: Path,
) -> None:
    storage = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    storage.close()
    outside = tmp_path.parent / "outside-must-stay"
    outside.write_bytes(b"safe")
    orphan = tmp_path / "staging" / f"{'a' * 32}.partial"
    orphan.write_bytes(b"orphan")
    unrelated = tmp_path / "staging" / "notes.txt"
    unrelated.write_bytes(b"keep")
    connection = sqlite3.connect(tmp_path / "botified-asr.sqlite3")
    connection.execute(
        """
        INSERT INTO upload_leases(
            id, kind, phase, staging_path, reserved_bytes, actual_bytes
        ) VALUES (?, 'transcription', 'receiving', ?, ?, 0)
        """,
        ("../outside-must-stay", str(outside), RESERVATION_QUANTUM),
    )
    connection.commit()
    connection.close()

    recovered = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    recovered.close()
    recovered_again = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    recovered_again.close()

    assert outside.read_bytes() == b"safe"
    assert not orphan.exists()
    assert unrelated.read_bytes() == b"keep"
