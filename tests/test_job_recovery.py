from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import threading

import pytest

import botified_asr.jobs as jobs
import botified_asr.storage as storage_module
from botified_asr.config import LimitsConfig, RESERVATION_QUANTUM
from botified_asr.storage import Storage, StorageSchemaError


CREATED_AT = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
CLAIMED_AT = CREATED_AT + timedelta(minutes=1)
CANONICAL_OPTIONS_JSON = (
    '{"chunking_strategy":null,"include":[],"known_speaker_ids":[],'
    '"language":"auto","model":"sensevoice","response_format":"json"}'
)


def limits() -> LimitsConfig:
    return LimitsConfig(
        max_upload_bytes=RESERVATION_QUANTUM,
        sync_max_upload_bytes=RESERVATION_QUANTUM,
        max_active_uploads=8,
        max_queued_jobs=8,
        max_job_storage_bytes=8 * RESERVATION_QUANTUM,
        min_filesystem_free_bytes=1,
    )


def patch_job_ids(
    monkeypatch: pytest.MonkeyPatch,
    *job_ids: str,
) -> None:
    source = iter(job_ids)

    def generate_job_id() -> str:
        return next(source)

    monkeypatch.setattr(jobs, "generate_job_id", generate_job_id)
    monkeypatch.setattr(storage_module, "generate_job_id", generate_job_id)


def patch_attempt_tokens(
    monkeypatch: pytest.MonkeyPatch,
    *tokens: str,
) -> None:
    source = iter(tokens)

    def generate_attempt_token() -> str:
        return next(source)

    monkeypatch.setattr(
        jobs,
        "generate_attempt_token",
        generate_attempt_token,
        raising=False,
    )
    monkeypatch.setattr(
        storage_module,
        "generate_attempt_token",
        generate_attempt_token,
        raising=False,
    )


def queue_job(storage: Storage, created_at: datetime) -> jobs.DurableJob:
    upload = storage.begin_job_upload(created_at)
    storage.append_job_upload(upload, b"audio")
    input_ref = storage.seal_job_upload(upload)
    return storage.publish_job(
        input_ref,
        jobs.QueuedJobSpec(
            canonical_options_json=CANONICAL_OPTIONS_JSON,
            selected_speaker_snapshot=b'{"speakers":[]}',
            snapshot_sha256="1" * 64,
            total_samples=32_000,
            request_fingerprint="2" * 64,
            processor_fingerprint="3" * 64,
        ),
    )


def recovery_snapshot(data_dir: Path) -> tuple[object, ...]:
    connection = sqlite3.connect(data_dir / "botified-asr.sqlite3")
    try:
        database = tuple(
            tuple(connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall())
            for table in (
                "transcription_jobs",
                "storage_leases",
                "shutdown_marker",
            )
        )
    finally:
        connection.close()
    files = []
    for directory_name in ("staging", "artifacts"):
        directory = data_dir / directory_name
        for path in sorted(directory.iterdir()):
            kind = "file" if path.is_file() else "directory"
            files.append(
                (
                    f"{directory_name}/{path.name}",
                    kind,
                    path.read_bytes() if kind == "file" else None,
                )
            )
    return (*database, tuple(files))


def test_claim_is_fifo_and_starts_a_token_owned_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert tuple((member.name, member.value) for member in jobs.JobProgressOutcome) == (
        ("UPDATED", "updated"),
        ("STALE", "stale"),
        ("CANCEL_REQUESTED", "cancel_requested"),
    )
    patch_job_ids(monkeypatch, "ABCDEFGH", "01234567", "7K3M9Q2W")
    patch_attempt_tokens(monkeypatch, "token-1", "token-2", "token-3")
    storage = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    try:
        queue_job(storage, CREATED_AT)
        queue_job(storage, CREATED_AT)
        queue_job(storage, CREATED_AT + timedelta(seconds=1))
        storage._connection.execute(
            """
            UPDATE transcription_jobs SET attempt_no = 1
            WHERE id = '01234567'
            """
        )

        claimed = tuple(
            storage.claim_next_job("generation-1", CLAIMED_AT) for _ in range(3)
        )

        assert tuple(job.id for job in claimed) == (
            "01234567",
            "ABCDEFGH",
            "7K3M9Q2W",
        )
        assert tuple(job.attempt_token for job in claimed) == (
            "token-1",
            "token-2",
            "token-3",
        )
        assert all(job.status is jobs.JobStatus.RUNNING for job in claimed)
        assert all(job.processed_samples == 0 for job in claimed)
        assert tuple(job.attempt_no for job in claimed) == (2, 1, 1)
        assert all(job.owner_generation == "generation-1" for job in claimed)
        assert all(job.started_at == CLAIMED_AT for job in claimed)
        assert storage.claim_next_job("generation-1", CLAIMED_AT) is None
    finally:
        storage.close()


def test_two_connections_cannot_claim_the_same_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_job_ids(monkeypatch, "01234567")
    patch_attempt_tokens(monkeypatch, "only-token")
    first = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    second = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    queue_job(first, CREATED_AT)
    barrier = threading.Barrier(2)
    outcomes: list[object] = []

    def claim(storage: Storage) -> None:
        barrier.wait()
        try:
            outcomes.append(storage.claim_next_job("generation-1", CLAIMED_AT))
        except BaseException as error:
            outcomes.append(error)

    threads = (
        threading.Thread(target=claim, args=(first,)),
        threading.Thread(target=claim, args=(second,)),
    )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    try:
        assert not [item for item in outcomes if isinstance(item, BaseException)]
        claimed = [item for item in outcomes if item is not None]
        assert len(claimed) == 1
        assert claimed[0].id == "01234567"
        assert claimed[0].attempt_token == "only-token"
        assert outcomes.count(None) == 1
    finally:
        first.close()
        second.close()


def test_progress_is_monotonic_token_fenced_and_cancel_aware(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_job_ids(monkeypatch, "01234567", "ABCDEFGH")
    patch_attempt_tokens(monkeypatch, "token-1")
    storage = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    try:
        running = queue_job(storage, CREATED_AT)
        queued = queue_job(storage, CREATED_AT + timedelta(seconds=1))
        running = storage.claim_next_job("generation-1", CLAIMED_AT)
        assert storage.get_visible_job(running.id) == running

        for invalid in (True, 1.0, "1"):
            with pytest.raises(TypeError):
                storage.update_job_progress("7K3M9Q2W", "stale", invalid)
        with pytest.raises(ValueError):
            storage.update_job_progress("7K3M9Q2W", "stale", -1)

        assert (
            storage.update_job_progress(queued.id, "stale", 0)
            is jobs.JobProgressOutcome.STALE
        )
        assert (
            storage.update_job_progress("7K3M9Q2W", "stale", 0)
            is jobs.JobProgressOutcome.STALE
        )
        assert (
            storage.update_job_progress(running.id, "wrong-token", 0)
            is jobs.JobProgressOutcome.STALE
        )
        assert (
            storage.update_job_progress(running.id, "token-1", 100)
            is jobs.JobProgressOutcome.UPDATED
        )
        assert (
            storage.update_job_progress(running.id, "token-1", 100)
            is jobs.JobProgressOutcome.UPDATED
        )
        with pytest.raises(ValueError):
            storage.update_job_progress(running.id, "token-1", 99)
        with pytest.raises(ValueError):
            storage.update_job_progress(running.id, "token-1", 32_001)

        storage._connection.execute(
            """
            UPDATE transcription_jobs SET cancel_requested = 1 WHERE id = ?
            """,
            (running.id,),
        )
        assert (
            storage.update_job_progress(running.id, "token-1", 101)
            is jobs.JobProgressOutcome.CANCEL_REQUESTED
        )
        assert storage.get_visible_job(running.id).processed_samples == 100
        assert (
            storage._connection.execute(
                """
                SELECT processed_samples FROM transcription_jobs WHERE id = ?
                """,
                (running.id,),
            ).fetchone()[0]
            == 100
        )
    finally:
        storage.close()


def test_shutdown_marker_is_idempotent_and_fences_requeue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_job_ids(
        monkeypatch,
        "01234567",
        "ABCDEFGH",
        "7K3M9Q2W",
        "JKMNPQRT",
    )
    patch_attempt_tokens(monkeypatch, "token-1", "token-2", "token-3")
    storage = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    try:
        first = queue_job(storage, CREATED_AT)
        second = queue_job(storage, CREATED_AT + timedelta(seconds=1))
        queue_job(storage, CREATED_AT + timedelta(seconds=2))
        queue_job(storage, CREATED_AT + timedelta(seconds=3))
        first = storage.claim_next_job("generation-1", CLAIMED_AT)
        second = storage.claim_next_job("generation-2", CLAIMED_AT)
        cancelled = storage.claim_next_job("generation-1", CLAIMED_AT)

        assert not storage.requeue_job_at_shutdown(
            first.id,
            first.attempt_token,
            "generation-1",
        )
        assert (
            storage.update_job_progress(first.id, first.attempt_token, 100)
            is jobs.JobProgressOutcome.UPDATED
        )

        storage.write_shutdown_marker("generation-1", CLAIMED_AT)
        storage.write_shutdown_marker(
            "generation-1",
            CLAIMED_AT + timedelta(seconds=1),
        )
        marker = storage._connection.execute(
            "SELECT generation, created_at FROM shutdown_marker"
        ).fetchone()
        assert tuple(marker) == (
            "generation-1",
            "2026-07-27T12:01:00Z",
        )
        with pytest.raises(RuntimeError):
            storage.write_shutdown_marker(
                "generation-2",
                CLAIMED_AT + timedelta(seconds=2),
            )
        with pytest.raises(RuntimeError):
            storage.claim_next_job("generation-1", CLAIMED_AT)

        assert not storage.requeue_job_at_shutdown(
            second.id,
            second.attempt_token,
            "generation-1",
        )
        assert not storage.requeue_job_at_shutdown(
            first.id,
            "wrong-token",
            "generation-1",
        )
        assert not storage.requeue_job_at_shutdown(
            first.id,
            first.attempt_token,
            "generation-2",
        )
        storage._connection.execute(
            """
            UPDATE transcription_jobs SET cancel_requested = 1 WHERE id = ?
            """,
            (cancelled.id,),
        )
        assert not storage.requeue_job_at_shutdown(
            cancelled.id,
            cancelled.attempt_token,
            "generation-1",
        )
        assert storage.requeue_job_at_shutdown(
            first.id,
            first.attempt_token,
            "generation-1",
        )
        assert not storage.requeue_job_at_shutdown(
            first.id,
            first.attempt_token,
            "generation-1",
        )

        requeued = storage.get_visible_job(first.id)
        assert requeued.status is jobs.JobStatus.QUEUED
        assert requeued.processed_samples == 0
        assert requeued.attempt_no == 1
        assert requeued.crash_recoveries == 0
        assert requeued.attempt_token is None
        assert requeued.owner_generation is None
        assert requeued.started_at is None
    finally:
        storage.close()


def test_startup_classifies_all_running_jobs_and_clears_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_job_ids(
        monkeypatch,
        "01234567",
        "ABCDEFGH",
        "7K3M9Q2W",
        "JKMNPQRT",
    )
    patch_attempt_tokens(monkeypatch, "token-1", "token-2", "token-3", "token-4")
    storage = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    for offset in range(4):
        queue_job(storage, CREATED_AT + timedelta(seconds=offset))
    claimed = (
        storage.claim_next_job("generation-1", CLAIMED_AT),
        storage.claim_next_job("generation-1", CLAIMED_AT),
        storage.claim_next_job("generation-2", CLAIMED_AT),
        storage.claim_next_job("generation-2", CLAIMED_AT),
    )
    for job, progress in zip(claimed, (10, 20, 30, 40), strict=True):
        assert (
            storage.update_job_progress(job.id, job.attempt_token, progress)
            is jobs.JobProgressOutcome.UPDATED
        )
    storage._connection.execute(
        "UPDATE transcription_jobs SET cancel_requested = 1 WHERE id = ?",
        (claimed[0].id,),
    )
    storage._connection.execute(
        """
        UPDATE transcription_jobs SET crash_recoveries = 1
        WHERE id IN (?, ?)
        """,
        (claimed[0].id, claimed[3].id),
    )
    storage.write_shutdown_marker("generation-1", CLAIMED_AT)
    storage.close()

    monkeypatch.setattr(
        storage_module,
        "_utc_now",
        lambda: CLAIMED_AT - timedelta(seconds=1),
        raising=False,
    )
    recovered = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    try:
        cancelled, graceful, retried, failed = (
            recovered.get_visible_job(job.id) for job in claimed
        )
        assert cancelled.status is jobs.JobStatus.CANCELLED
        assert cancelled.processed_samples == 10
        assert cancelled.crash_recoveries == 1
        assert cancelled.finished_at == CLAIMED_AT
        assert failed.status is jobs.JobStatus.FAILED
        assert failed.error_code == "worker_crashed"
        assert failed.processed_samples == 40
        assert failed.finished_at == CLAIMED_AT
        for terminal in (cancelled, failed):
            assert terminal.input_lease_id is None
            assert not terminal.input_cleanup_pending
            assert not (recovered.staging_dir / f"{terminal.id}.ready").exists()

        assert graceful.status is jobs.JobStatus.QUEUED
        assert graceful.processed_samples == 0
        assert graceful.attempt_no == 1
        assert graceful.crash_recoveries == 0
        assert retried.status is jobs.JobStatus.QUEUED
        assert retried.processed_samples == 0
        assert retried.attempt_no == 1
        assert retried.crash_recoveries == 1
        for queued in (graceful, retried):
            assert queued.attempt_token is None
            assert queued.owner_generation is None
            assert queued.started_at is None
            assert (recovered.staging_dir / f"{queued.id}.ready").is_file()

        assert (
            recovered._connection.execute(
                "SELECT COUNT(*) FROM shutdown_marker"
            ).fetchone()[0]
            == 0
        )
        terminal_ids = (cancelled.id, failed.id)
        assert (
            recovered._connection.execute(
                """
                SELECT COUNT(*) FROM storage_leases WHERE id IN (?, ?)
                """,
                terminal_ids,
            ).fetchone()[0]
            == 0
        )
    finally:
        recovered.close()


def test_startup_marker_delete_fault_rolls_back_classification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_job_ids(monkeypatch, "01234567", "ABCDEFGH")
    patch_attempt_tokens(
        monkeypatch,
        "token-1",
        "token-2",
        "token-3",
        "token-4",
    )
    storage = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    queue_job(storage, CREATED_AT)
    queue_job(storage, CREATED_AT + timedelta(seconds=1))
    graceful = storage.claim_next_job("generation-1", CLAIMED_AT)
    crashed = storage.claim_next_job("generation-2", CLAIMED_AT)
    storage.write_shutdown_marker("generation-1", CLAIMED_AT)
    storage.close()

    connection = sqlite3.connect(tmp_path / "botified-asr.sqlite3")
    try:
        connection.execute(
            """
            CREATE TRIGGER reject_marker_clear
            BEFORE DELETE ON shutdown_marker
            BEGIN
                SELECT RAISE(ABORT, 'injected marker delete failure');
            END
            """
        )
        connection.commit()
    finally:
        connection.close()
    before = recovery_snapshot(tmp_path)

    with pytest.raises(
        sqlite3.IntegrityError,
        match="injected marker delete failure",
    ):
        Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    assert recovery_snapshot(tmp_path) == before

    connection = sqlite3.connect(tmp_path / "botified-asr.sqlite3")
    try:
        connection.execute("DROP TRIGGER reject_marker_clear")
        connection.commit()
    finally:
        connection.close()
    recovered = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    assert recovered.get_visible_job(graceful.id).crash_recoveries == 0
    assert recovered.get_visible_job(crashed.id).crash_recoveries == 1
    assert recovered.claim_next_job("generation-3", CLAIMED_AT).id == graceful.id
    crashed_again = recovered.claim_next_job("generation-3", CLAIMED_AT)
    assert crashed_again.id == crashed.id
    assert crashed_again.attempt_no == 2
    recovered.close()

    after_crash = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    try:
        failed = after_crash.get_visible_job(crashed.id)
        assert failed.status is jobs.JobStatus.FAILED
        assert failed.error_code == "worker_crashed"
        assert failed.attempt_no == 2
        assert failed.crash_recoveries == 1
        assert failed.input_lease_id is None
        assert not failed.input_cleanup_pending
        assert not (after_crash.staging_dir / f"{failed.id}.ready").exists()
    finally:
        after_crash.close()
    reopened = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    try:
        assert reopened.get_visible_job(crashed.id) == failed
    finally:
        reopened.close()


@pytest.mark.parametrize(
    ("phase", "owner_kind"),
    (("writing", "sync"), ("sealed", "legacy")),
)
def test_startup_recovers_jobs_alongside_generic_transient_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    owner_kind: str,
) -> None:
    patch_job_ids(monkeypatch, "01234567")
    patch_attempt_tokens(monkeypatch, "token-1")
    storage = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    queued = queue_job(storage, CREATED_AT)
    running = storage.claim_next_job("generation-1", CLAIMED_AT)
    assert (
        storage.update_job_progress(running.id, running.attempt_token, 100)
        is jobs.JobProgressOutcome.UPDATED
    )

    generic_upload = storage.begin_upload("transcription")
    storage.append(generic_upload, b"generic input")
    generic = generic_upload
    if phase == "sealed":
        generic = storage.seal_upload(generic_upload)
        storage._connection.execute(
            """
            UPDATE storage_leases SET owner_kind = 'legacy',
                owner_id = 'legacy-request'
            WHERE id = ?
            """,
            (generic.id,),
        )
    assert (
        storage._connection.execute(
            "SELECT owner_kind FROM storage_leases WHERE id = ?",
            (generic.id,),
        ).fetchone()[0]
        == owner_kind
    )
    reserved_before = storage.total_reserved_bytes()
    assert reserved_before > queued.input_size_bytes
    storage.write_shutdown_marker("generation-1", CLAIMED_AT)
    storage.close()

    recovered = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    try:
        job = recovered.get_visible_job(running.id)
        assert job.status is jobs.JobStatus.QUEUED
        assert job.processed_samples == 0
        assert job.attempt_no == 1
        assert job.crash_recoveries == 0
        assert job.attempt_token is None
        assert job.owner_generation is None
        assert (recovered.staging_dir / f"{job.id}.ready").is_file()
        assert (
            recovered._connection.execute(
                "SELECT COUNT(*) FROM storage_leases WHERE id = ?",
                (job.id,),
            ).fetchone()[0]
            == 1
        )

        assert not generic.path.exists()
        assert not (recovered.staging_dir / f"{generic.id}.partial").exists()
        assert not (recovered.staging_dir / f"{generic.id}.ready").exists()
        assert (
            recovered._connection.execute(
                "SELECT COUNT(*) FROM storage_leases WHERE id = ?",
                (generic.id,),
            ).fetchone()[0]
            == 0
        )
        assert recovered.total_reserved_bytes() == queued.input_size_bytes
        assert (
            recovered._connection.execute(
                "SELECT COUNT(*) FROM shutdown_marker"
            ).fetchone()[0]
            == 0
        )
    finally:
        recovered.close()


@pytest.mark.parametrize("terminal_status", ("failed", "cancelled"))
@pytest.mark.parametrize("fault_stage", ("fsync", "database"))
def test_startup_terminal_cleanup_resumes_after_phase_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_status: str,
    fault_stage: str,
) -> None:
    patch_job_ids(monkeypatch, "01234567")
    patch_attempt_tokens(monkeypatch, "token-1")
    storage = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    queued = queue_job(storage, CREATED_AT)
    owner = "generation-2" if terminal_status == "failed" else "generation-1"
    running = storage.claim_next_job(owner, CLAIMED_AT)
    if terminal_status == "failed":
        storage._connection.execute(
            "UPDATE transcription_jobs SET crash_recoveries = 1 WHERE id = ?",
            (running.id,),
        )
    else:
        storage._connection.execute(
            "UPDATE transcription_jobs SET cancel_requested = 1 WHERE id = ?",
            (running.id,),
        )
    storage.write_shutdown_marker("generation-1", CLAIMED_AT)
    lease_before = tuple(
        storage._connection.execute(
            "SELECT * FROM storage_leases WHERE id = ?",
            (running.id,),
        ).fetchone()
    )
    reservation_before = storage.total_reserved_bytes()
    storage.close()

    input_path = tmp_path / "staging" / f"{queued.id}.ready"
    original_fsync_directory = storage_module._fsync_directory

    def fail_staging_fsync(directory: Path) -> None:
        if directory == tmp_path / "staging":
            raise OSError("injected terminal cleanup fsync failure")
        original_fsync_directory(directory)

    if fault_stage == "fsync":
        with monkeypatch.context() as fault:
            fault.setattr(storage_module, "_fsync_directory", fail_staging_fsync)
            with pytest.raises(
                OSError,
                match="injected terminal cleanup fsync failure",
            ):
                Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    else:
        connection = sqlite3.connect(tmp_path / "botified-asr.sqlite3")
        try:
            connection.execute(
                f"""
                CREATE TRIGGER reject_terminal_lease_cleanup
                BEFORE DELETE ON storage_leases
                WHEN OLD.id = '{running.id}'
                BEGIN
                    SELECT RAISE(
                        ABORT,
                        'injected terminal cleanup database failure'
                    );
                END
                """
            )
            connection.commit()
        finally:
            connection.close()
        with pytest.raises(
            sqlite3.IntegrityError,
            match="injected terminal cleanup database failure",
        ):
            Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)

    connection = sqlite3.connect(tmp_path / "botified-asr.sqlite3")
    try:
        row = connection.execute(
            """
            SELECT status, input_lease_id, input_cleanup_pending
            FROM transcription_jobs WHERE id = ?
            """,
            (running.id,),
        ).fetchone()
        assert row == (terminal_status, running.id, 1)
        assert (
            tuple(
                connection.execute(
                    "SELECT * FROM storage_leases WHERE id = ?",
                    (running.id,),
                ).fetchone()
            )
            == lease_before
        )
        assert (
            connection.execute(
                "SELECT SUM(reserved_bytes) FROM storage_leases",
            ).fetchone()[0]
            == reservation_before
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM shutdown_marker",
            ).fetchone()[0]
            == 0
        )
    finally:
        connection.close()
    assert not input_path.exists()

    if fault_stage == "database":
        connection = sqlite3.connect(tmp_path / "botified-asr.sqlite3")
        try:
            connection.execute("DROP TRIGGER reject_terminal_lease_cleanup")
            connection.commit()
        finally:
            connection.close()

    recovered = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    try:
        terminal = recovered.get_visible_job(running.id)
        assert terminal.status.value == terminal_status
        assert terminal.input_lease_id is None
        assert not terminal.input_cleanup_pending
        assert (
            recovered._connection.execute(
                "SELECT COUNT(*) FROM storage_leases WHERE id = ?",
                (running.id,),
            ).fetchone()[0]
            == 0
        )
    finally:
        recovered.close()


@pytest.mark.parametrize(
    "corruption",
    (
        "running-owner",
        "terminal-file-type",
        "job-artifact",
        "succeeded-artifact",
    ),
)
def test_startup_preflights_full_recovery_graph_before_mutating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    patch_job_ids(monkeypatch, "01234567", "ABCDEFGH")
    patch_attempt_tokens(monkeypatch, "token-1", "token-2")
    storage = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    queue_job(storage, CREATED_AT)
    queue_job(storage, CREATED_AT + timedelta(seconds=1))
    storage.claim_next_job("generation-1", CLAIMED_AT)
    later = storage.claim_next_job("generation-1", CLAIMED_AT)
    storage.write_shutdown_marker("generation-1", CLAIMED_AT)
    storage.close()

    input_path = tmp_path / "staging" / f"{later.id}.ready"
    connection = sqlite3.connect(tmp_path / "botified-asr.sqlite3")
    try:
        if corruption == "running-owner":
            connection.execute(
                "UPDATE storage_leases SET owner_id = '01234567' WHERE id = ?",
                (later.id,),
            )
        elif corruption == "terminal-file-type":
            connection.execute(
                """
                UPDATE transcription_jobs SET
                    status = 'failed',
                    attempt_token = NULL,
                    owner_generation = NULL,
                    crash_recoveries = 1,
                    error_code = 'worker_crashed',
                    input_cleanup_pending = 1,
                    finished_at = '2026-07-27T12:01:00Z'
                WHERE id = ?
                """,
                (later.id,),
            )
        else:
            artifact_id = "a" * 32
            artifact_path = tmp_path / "artifacts" / f"{artifact_id}.complete"
            artifact_path.write_bytes(b"result")
            connection.execute(
                """
                INSERT INTO storage_leases(
                    id, lease_type, resource_kind, owner_kind, owner_id,
                    phase, controlled_path, reserved_bytes, actual_bytes,
                    content_sha256
                ) VALUES (?, 'artifact', 'result_complete', 'job', ?,
                          'sealed', ?, 6, 6, ?)
                """,
                (
                    artifact_id,
                    later.id,
                    str(artifact_path),
                    "4" * 64,
                ),
            )
            if corruption == "succeeded-artifact":
                connection.execute(
                    """
                    UPDATE transcription_jobs SET
                        status = 'succeeded',
                        processed_samples = total_samples,
                        attempt_token = NULL,
                        owner_generation = NULL,
                        result_lease_id = ?,
                        input_cleanup_pending = 1,
                        finished_at = '2026-07-27T12:01:00Z'
                    WHERE id = ?
                    """,
                    (artifact_id, later.id),
                )
        connection.commit()
    finally:
        connection.close()
    if corruption == "terminal-file-type":
        input_path.unlink()
        input_path.mkdir()

    before = recovery_snapshot(tmp_path)
    with pytest.raises(StorageSchemaError):
        Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    assert recovery_snapshot(tmp_path) == before
