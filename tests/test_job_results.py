from __future__ import annotations

import hashlib
import io
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import botified_asr.jobs as jobs
import botified_asr.storage as storage_module
from botified_asr.canonical_options import (
    parse_canonical_options_json,
)
from botified_asr.config import LimitsConfig, RESERVATION_QUANTUM
from botified_asr.result_artifact import (
    CanonicalArtifactError,
    CanonicalJsonlReader,
    ResultEnvelopeManifest,
    finalize_result_envelope,
)
from botified_asr.speaker_matching import SpeakerLabelMapping
from botified_asr.speakers import SpeakerEmbeddingPolicy
from botified_asr.storage import Storage, StorageSchemaError


CREATED_AT = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
STARTED_AT = CREATED_AT + timedelta(minutes=1)
FINISHED_AT = STARTED_AT + timedelta(minutes=1)
NON_UTC_FINISHED_AT = FINISHED_AT.astimezone(timezone(timedelta(hours=1)))
TOTAL_SAMPLES = 32_000
PROCESSOR_FINGERPRINT = "3" * 64
RUNTIME_FAILURE_CODES = (
    "invalid_audio",
    "audio_too_long",
    "long_audio_requires_vad",
    "too_many_speakers",
    "invalid_model_output",
    "pipeline_not_ready",
    "internal_error",
)
CANONICAL_OPTIONS_JSON = (
    '{"chunking_strategy":null,"include":[],"known_speaker_ids":[],'
    '"language":"auto","model":"sensevoice","response_format":"json"}'
)


def limits() -> LimitsConfig:
    return LimitsConfig(
        max_upload_bytes=RESERVATION_QUANTUM,
        sync_max_upload_bytes=RESERVATION_QUANTUM,
        max_active_uploads=4,
        max_queued_jobs=4,
        max_job_storage_bytes=4 * RESERVATION_QUANTUM,
        min_filesystem_free_bytes=1,
    )


def speaker_policy() -> SpeakerEmbeddingPolicy:
    return SpeakerEmbeddingPolicy(
        model_id="funasr/campplus",
        model_revision="1" * 40,
        embedding_dimension=2,
        sample_rate=16_000,
        downmix_policy_version="ffmpeg-first-audio-stream-ac1-v1",
        window_samples=24_000,
        window_shift_samples=12_000,
        padding_policy_version="right-zero-pad-v1",
        normalization_policy_version="int16-div-32768-l2-v1",
        enrollment_aggregation_policy_version=(
            "sample-centroid-equal-average-v1"
        ),
    )


def patch_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jobs, "generate_job_id", lambda: "7K3M9Q2W")
    monkeypatch.setattr(storage_module, "generate_job_id", lambda: "7K3M9Q2W")
    monkeypatch.setattr(jobs, "generate_attempt_token", lambda: "attempt-1")
    monkeypatch.setattr(
        storage_module,
        "generate_attempt_token",
        lambda: "attempt-1",
    )


def queue_and_claim(storage: Storage) -> jobs.DurableJob:
    upload = storage.begin_job_upload(CREATED_AT)
    storage.append_job_upload(upload, b"audio")
    input_ref = storage.seal_job_upload(upload)
    storage.publish_job(
        input_ref,
        jobs.QueuedJobSpec(
            canonical_options_json=CANONICAL_OPTIONS_JSON,
            effective_max_audio_samples=TOTAL_SAMPLES,
            effective_direct_max_audio_samples=16_000,
            processor_fingerprint="3" * 64,
        ),
        speaker_embedding_policy=speaker_policy(),
    )
    claimed = storage.claim_next_job("generation-1", STARTED_AT)
    assert claimed is not None
    return claimed


class StorageResultWriter:
    def __init__(
        self,
        storage: Storage,
        writer: object,
    ) -> None:
        self.storage = storage
        self.writer = writer

    def write(self, data: bytes) -> None:
        self.storage.append_artifact(self.writer, data)

    def seal(self) -> object:
        return self.storage.seal_artifact(self.writer)

    def abort(self) -> None:
        self.storage.abort_artifact(self.writer)


def seal_result(
    storage: Storage,
    running: jobs.DurableJob,
    *,
    finished_at: datetime = FINISHED_AT,
) -> object:
    assert running.attempt_token is not None
    writer = storage.begin_job_result_artifact(
        running.id,
        running.attempt_token,
    )
    return finalize_result_envelope(
        CanonicalJsonlReader(
            Path("empty.jsonl"),
            opener=lambda: io.BytesIO(),
        ),
        parse_canonical_options_json(running.canonical_options_json),
        TOTAL_SAMPLES,
        writer=StorageResultWriter(storage, writer),
        manifest=ResultEnvelopeManifest(
            version=1,
            job_id=running.id,
            attempt_no=running.attempt_no,
            request_fingerprint=running.request_fingerprint,
            processor_fingerprint=running.processor_fingerprint,
            finished_at=finished_at,
        ),
        speaker_mapping=SpeakerLabelMapping(()),
    )


def recovery_snapshot(data_dir: Path) -> tuple[object, ...]:
    connection = sqlite3.connect(data_dir / "botified-asr.sqlite3")
    try:
        database = tuple(
            tuple(
                connection.execute(
                    f"SELECT * FROM {table} ORDER BY 1"
                ).fetchall()
            )
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


def finish_progress(storage: Storage, running: jobs.DurableJob) -> None:
    assert running.attempt_token is not None
    assert (
        storage.update_job_progress(
            running.id,
            running.attempt_token,
            TOTAL_SAMPLES,
            total_samples=TOTAL_SAMPLES,
        )
        is jobs.JobProgressOutcome.UPDATED
    )


def new_running_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Storage, jobs.DurableJob]:
    patch_ids(monkeypatch)
    storage = Storage(
        tmp_path,
        limits(),
        current_processor_fingerprint=PROCESSOR_FINGERPRINT,
        free_bytes=lambda _: 1 << 40,
    )
    return storage, queue_and_claim(storage)


def request_cancel(storage: Storage, job_id: str) -> None:
    storage._connection.execute(
        "UPDATE transcription_jobs SET cancel_requested = 1 WHERE id = ?",
        (job_id,),
    )


def commit_terminal(
    storage: Storage,
    running: jobs.DurableJob,
    terminal: str,
    *,
    attempt_token: str | None = None,
    failure_code: str = "internal_error",
    finished_at: datetime = FINISHED_AT,
) -> jobs.JobTerminalOutcome:
    token = running.attempt_token if attempt_token is None else attempt_token
    assert token is not None
    if terminal == "failure":
        return storage.commit_job_failure(
            running.id,
            token,
            failure_code,
            finished_at,
        )
    assert terminal == "cancellation"
    return storage.commit_job_cancellation(
        running.id,
        token,
        finished_at,
    )


@pytest.mark.parametrize(
    ("failure_code", "expected_status"),
    (
        *((code, jobs.JobStatus.FAILED) for code in RUNTIME_FAILURE_CODES),
        (None, jobs.JobStatus.CANCELLED),
    ),
)
def test_fenced_terminal_commit_cleans_input_and_is_not_repeatable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_code: str | None,
    expected_status: jobs.JobStatus,
) -> None:
    storage, running = new_running_job(tmp_path, monkeypatch)
    with closing(storage):
        finished_at = STARTED_AT if failure_code is not None else FINISHED_AT
        if failure_code is None:
            request_cancel(storage, running.id)
            terminal_kind = "cancellation"
        else:
            terminal_kind = "failure"

        def commit() -> jobs.JobTerminalOutcome:
            return commit_terminal(
                storage,
                running,
                terminal_kind,
                failure_code=failure_code or "internal_error",
                finished_at=finished_at,
            )

        assert commit() is jobs.JobTerminalOutcome.COMMITTED
        terminal = storage.get_visible_job(running.id)
        assert terminal is not None
        assert terminal.status is expected_status
        assert terminal.attempt_token is None
        assert terminal.owner_generation is None
        assert terminal.result_lease_id is None
        assert terminal.error_code == failure_code
        assert terminal.input_lease_id is None
        assert not terminal.input_cleanup_pending
        assert terminal.finished_at == finished_at
        assert terminal.cancel_requested is (failure_code is None)
        assert not (storage.staging_dir / f"{running.id}.ready").exists()
        assert storage.total_reserved_bytes() == 0
        assert commit() is jobs.JobTerminalOutcome.STALE


@pytest.mark.parametrize("terminal", ("failure", "cancellation"))
def test_terminal_commit_with_stale_token_is_a_zero_change_loser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal: str,
) -> None:
    storage, running = new_running_job(tmp_path, monkeypatch)
    with closing(storage):
        if terminal == "cancellation":
            request_cancel(storage, running.id)
        before = recovery_snapshot(tmp_path)

        outcome = commit_terminal(
            storage,
            running,
            terminal,
            attempt_token="stale-token",
        )

        assert outcome is jobs.JobTerminalOutcome.STALE
        assert recovery_snapshot(tmp_path) == before


def test_cancellation_wins_the_exact_attempt_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, running = new_running_job(tmp_path, monkeypatch)
    with closing(storage):
        before_request = recovery_snapshot(tmp_path)
        assert (
            commit_terminal(storage, running, "cancellation")
            is jobs.JobTerminalOutcome.STALE
        )
        assert recovery_snapshot(tmp_path) == before_request

        request_cancel(storage, running.id)
        after_request = recovery_snapshot(tmp_path)
        assert (
            commit_terminal(storage, running, "failure")
            is jobs.JobTerminalOutcome.CANCEL_REQUESTED
        )
        assert recovery_snapshot(tmp_path) == after_request
        assert (
            commit_terminal(storage, running, "cancellation")
            is jobs.JobTerminalOutcome.COMMITTED
        )


@pytest.mark.parametrize(
    ("terminal", "failure_code", "finished_at"),
    (
        ("failure", "unknown_private_error", FINISHED_AT),
        ("failure", "cancelled", FINISHED_AT),
        ("failure", "worker_crashed", FINISHED_AT),
        ("failure", "internal_error", STARTED_AT - timedelta(microseconds=1)),
        ("failure", "internal_error", NON_UTC_FINISHED_AT),
        ("cancellation", None, STARTED_AT - timedelta(microseconds=1)),
        ("cancellation", None, NON_UTC_FINISHED_AT),
    ),
)
def test_terminal_commit_rejects_invalid_values_before_any_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal: str,
    failure_code: str | None,
    finished_at: datetime,
) -> None:
    storage, running = new_running_job(tmp_path, monkeypatch)
    with closing(storage):
        if terminal == "cancellation":
            request_cancel(storage, running.id)
        before = recovery_snapshot(tmp_path)

        with pytest.raises(ValueError):
            commit_terminal(
                storage,
                running,
                terminal,
                failure_code=failure_code or "internal_error",
                finished_at=finished_at,
            )

        assert recovery_snapshot(tmp_path) == before


@pytest.mark.parametrize("terminal", ("failure", "cancellation"))
def test_terminal_commit_rejects_corrupt_running_shape_before_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal: str,
) -> None:
    storage, running = new_running_job(tmp_path, monkeypatch)
    with closing(storage):
        assert running.attempt_token is not None
        storage._connection.execute(
            """
            UPDATE transcription_jobs
            SET owner_generation = NULL, cancel_requested = ?
            WHERE id = ?
            """,
            (int(terminal == "cancellation"), running.id),
        )
        before = recovery_snapshot(tmp_path)

        with pytest.raises(StorageSchemaError):
            commit_terminal(storage, running, terminal)

        assert recovery_snapshot(tmp_path) == before


@pytest.mark.parametrize("terminal", ("failure", "cancellation"))
def test_repeated_terminal_commit_resumes_only_its_pending_input_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal: str,
) -> None:
    storage, running = new_running_job(tmp_path, monkeypatch)
    with closing(storage):
        assert running.attempt_token is not None
        if terminal == "cancellation":
            request_cancel(storage, running.id)

        original_unlink = storage._unlink_if_present
        faulted = False

        def fail_first_unlink(path: Path) -> None:
            nonlocal faulted
            if not faulted:
                faulted = True
                raise OSError("injected terminal input unlink failure")
            original_unlink(path)

        monkeypatch.setattr(storage, "_unlink_if_present", fail_first_unlink)

        def commit(
            error_code: str = "internal_error",
        ) -> jobs.JobTerminalOutcome:
            return commit_terminal(
                storage,
                running,
                terminal,
                failure_code=error_code,
            )

        with pytest.raises(
            OSError,
            match="injected terminal input unlink failure",
        ):
            commit()

        pending = storage.get_visible_job(running.id)
        assert pending is not None
        assert pending.status is (
            jobs.JobStatus.FAILED
            if terminal == "failure"
            else jobs.JobStatus.CANCELLED
        )
        assert pending.input_lease_id == running.id
        assert pending.input_cleanup_pending
        assert pending.error_code == (
            "internal_error" if terminal == "failure" else None
        )
        assert pending.finished_at == FINISHED_AT
        assert storage.total_reserved_bytes() == len(b"audio")

        if terminal == "failure":
            before_wrong_target = recovery_snapshot(tmp_path)
            assert (
                commit("invalid_audio")
                is jobs.JobTerminalOutcome.STALE
            )
            assert recovery_snapshot(tmp_path) == before_wrong_target

        assert commit() is jobs.JobTerminalOutcome.STALE
        cleaned = storage.get_visible_job(running.id)
        assert cleaned is not None
        assert cleaned.status is pending.status
        assert cleaned.error_code == pending.error_code
        assert cleaned.finished_at == pending.finished_at
        assert cleaned.attempt_token == pending.attempt_token
        assert cleaned.owner_generation == pending.owner_generation
        assert cleaned.input_lease_id is None
        assert not cleaned.input_cleanup_pending
        assert storage.total_reserved_bytes() == 0
        assert not (storage.staging_dir / f"{running.id}.ready").exists()


def test_terminal_cleanup_hands_off_a_deleting_job_without_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, running = new_running_job(tmp_path, monkeypatch)
    with closing(storage):
        original_unlink = storage._unlink_if_present

        def unlink_then_take_ownership(path: Path) -> None:
            original_unlink(path)
            storage._connection.execute(
                "UPDATE transcription_jobs SET phase = 'deleting' WHERE id = ?",
                (running.id,),
            )

        monkeypatch.setattr(
            storage,
            "_unlink_if_present",
            unlink_then_take_ownership,
        )

        assert (
            commit_terminal(storage, running, "failure")
            is jobs.JobTerminalOutcome.COMMITTED
        )
        deleting = storage._connection.execute(
            """
            SELECT phase, status, error_code, input_lease_id,
                   input_cleanup_pending, finished_at
            FROM transcription_jobs WHERE id = ?
            """,
            (running.id,),
        ).fetchone()
        assert tuple(deleting) == (
            "deleting",
            "failed",
            "internal_error",
            running.id,
            1,
            "2026-07-27T12:02:00Z",
        )
        assert storage.total_reserved_bytes() == len(b"audio")
        assert not (storage.staging_dir / f"{running.id}.ready").exists()


@pytest.mark.parametrize("check", ("first", "second"))
def test_terminal_cleanup_rejects_a_missing_row_with_an_orphan_upload_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    check: str,
) -> None:
    storage, running = new_running_job(tmp_path, monkeypatch)
    with closing(storage):
        original_unlink = storage._unlink_if_present

        if check == "first":
            def fail_unlink(_path: Path) -> None:
                raise OSError("injected terminal input unlink failure")

            monkeypatch.setattr(storage, "_unlink_if_present", fail_unlink)
            with pytest.raises(
                OSError,
                match="injected terminal input unlink failure",
            ):
                commit_terminal(storage, running, "failure")
            monkeypatch.setattr(
                storage,
                "_unlink_if_present",
                original_unlink,
            )
            storage._connection.execute(
                "DELETE FROM transcription_jobs WHERE id = ?",
                (running.id,),
            )
            before = recovery_snapshot(tmp_path)

            with pytest.raises(StorageSchemaError):
                commit_terminal(storage, running, "failure")

            assert recovery_snapshot(tmp_path) == before
        else:
            def unlink_then_drop_row(path: Path) -> None:
                original_unlink(path)
                storage._connection.execute(
                    "DELETE FROM transcription_jobs WHERE id = ?",
                    (running.id,),
                )

            monkeypatch.setattr(
                storage,
                "_unlink_if_present",
                unlink_then_drop_row,
            )
            with pytest.raises(StorageSchemaError):
                commit_terminal(storage, running, "failure")

        orphan = storage._connection.execute(
            """
            SELECT lease_type, resource_kind, owner_kind, owner_id, phase
            FROM storage_leases WHERE id = ?""",
            (running.id,),
        ).fetchone()
        assert tuple(orphan) == (
            "upload",
            "transcription",
            "job",
            running.id,
            "sealed",
        )
        assert storage.total_reserved_bytes() == len(b"audio")


def test_job_result_writer_commits_exact_success_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_ids(monkeypatch)
    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    try:
        running = queue_and_claim(storage)
        assert running.attempt_token is not None
        assert (
            storage.update_job_progress(
                running.id,
                running.attempt_token,
                TOTAL_SAMPLES,
                total_samples=TOTAL_SAMPLES,
            )
            is jobs.JobProgressOutcome.UPDATED
        )

        result_ref = seal_result(storage, running)
        result_payload = result_ref.path.read_bytes()
        result_lease = tuple(
            storage._connection.execute(
                "SELECT * FROM storage_leases WHERE id = ?",
                (result_ref.id,),
            ).fetchone()
        )

        assert (
            storage.commit_job_success(
                running.id,
                running.attempt_token,
                result_ref,
            )
            is jobs.JobSuccessOutcome.COMMITTED
        )
        succeeded = storage.get_visible_job(running.id)
        assert succeeded is not None
        assert succeeded.status is jobs.JobStatus.SUCCEEDED
        assert succeeded.processed_samples == succeeded.total_samples
        assert succeeded.attempt_token is None
        assert succeeded.owner_generation is None
        assert succeeded.result_lease_id == result_ref.id
        assert succeeded.input_lease_id is None
        assert not succeeded.input_cleanup_pending
        assert succeeded.finished_at == FINISHED_AT
        assert storage.resolve_artifact(result_ref).is_file()
        assert result_ref.path.read_bytes() == result_payload
        assert (
            tuple(
                storage._connection.execute(
                    "SELECT * FROM storage_leases WHERE id = ?",
                    (result_ref.id,),
                ).fetchone()
            )
            == result_lease
        )
        assert not (storage.staging_dir / f"{running.id}.ready").exists()
        assert storage.total_reserved_bytes() == result_ref.actual_bytes
        assert (
            storage.commit_job_success(
                running.id,
                running.attempt_token,
                result_ref,
            )
            is jobs.JobSuccessOutcome.COMMITTED
        )
        assert storage.resolve_artifact(result_ref).is_file()
        assert result_ref.path.read_bytes() == result_payload
        assert (
            tuple(
                storage._connection.execute(
                    "SELECT * FROM storage_leases WHERE id = ?",
                    (result_ref.id,),
                ).fetchone()
            )
            == result_lease
        )
    finally:
        storage.close()


def test_open_succeeded_job_result_streams_body_once_and_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_ids(monkeypatch)
    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    try:
        running = queue_and_claim(storage)
        finish_progress(storage, running)
        result_ref = seal_result(storage, running)
        expected_body = result_ref.path.read_bytes().split(b"\n", 1)[1]
        assert running.attempt_token is not None
        assert (
            storage.commit_job_success(
                running.id,
                running.attempt_token,
                result_ref,
            )
            is jobs.JobSuccessOutcome.COMMITTED
        )

        stored = storage.open_succeeded_job_result(running.id)
        body = stored.iter_body()
        with pytest.raises(RuntimeError, match="already"):
            stored.iter_body()
        assert b"".join(body) == expected_body
        stored.close()
        stored.close()
    finally:
        storage.close()


@pytest.mark.parametrize("corruption", ("missing", "size"))
def test_open_succeeded_job_result_preflights_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    patch_ids(monkeypatch)
    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    try:
        running = queue_and_claim(storage)
        finish_progress(storage, running)
        result_ref = seal_result(storage, running)
        assert running.attempt_token is not None
        assert (
            storage.commit_job_success(
                running.id,
                running.attempt_token,
                result_ref,
            )
            is jobs.JobSuccessOutcome.COMMITTED
        )
        if corruption == "missing":
            result_ref.path.unlink()
        else:
            with result_ref.path.open("ab") as handle:
                handle.write(b"x")

        with pytest.raises(CanonicalArtifactError):
            storage.open_succeeded_job_result(running.id)
    finally:
        storage.close()


@pytest.mark.parametrize("fault_stage", ("unlink", "fsync", "database"))
def test_success_commit_retries_failed_input_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault_stage: str,
) -> None:
    patch_ids(monkeypatch)
    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    try:
        running = queue_and_claim(storage)
        finish_progress(storage, running)
        result_ref = seal_result(storage, running)
        result_payload = result_ref.path.read_bytes()
        input_path = storage.staging_dir / f"{running.id}.ready"
        input_lease = tuple(
            storage._connection.execute(
                "SELECT * FROM storage_leases WHERE id = ?",
                (running.id,),
            ).fetchone()
        )
        result_lease = tuple(
            storage._connection.execute(
                "SELECT * FROM storage_leases WHERE id = ?",
                (result_ref.id,),
            ).fetchone()
        )
        reservation = storage.total_reserved_bytes()
        assert running.attempt_token is not None

        if fault_stage == "database":
            storage._connection.execute(
                f"""
                CREATE TRIGGER reject_succeeded_input_clear
                BEFORE UPDATE OF input_lease_id ON transcription_jobs
                WHEN OLD.id = '{running.id}'
                  AND NEW.input_lease_id IS NULL
                BEGIN
                    SELECT RAISE(
                        ABORT,
                        'injected input cleanup database failure'
                    );
                END
                """
            )
            with pytest.raises(
                sqlite3.IntegrityError,
                match="injected input cleanup database failure",
            ):
                storage.commit_job_success(
                    running.id,
                    running.attempt_token,
                    result_ref,
                )
            storage._connection.execute(
                "DROP TRIGGER reject_succeeded_input_clear"
            )
        else:
            original_unlink = Path.unlink
            original_fsync = storage_module._fsync_directory

            def fail_input_unlink(path: Path) -> None:
                if path == input_path:
                    raise OSError("injected input cleanup unlink failure")
                original_unlink(path)

            def fail_staging_fsync(directory: Path) -> None:
                if directory == storage.staging_dir:
                    raise OSError("injected input cleanup fsync failure")
                original_fsync(directory)

            with monkeypatch.context() as fault:
                if fault_stage == "unlink":
                    fault.setattr(Path, "unlink", fail_input_unlink)
                else:
                    fault.setattr(
                        storage_module,
                        "_fsync_directory",
                        fail_staging_fsync,
                    )
                with pytest.raises(
                    OSError,
                    match=f"injected input cleanup {fault_stage} failure",
                ):
                    storage.commit_job_success(
                        running.id,
                        running.attempt_token,
                        result_ref,
                    )

        succeeded = storage.get_visible_job(running.id)
        assert succeeded is not None
        assert succeeded.status is jobs.JobStatus.SUCCEEDED
        assert succeeded.input_lease_id == running.id
        assert succeeded.input_cleanup_pending
        assert input_path.exists() == (fault_stage == "unlink")
        assert (
            tuple(
                storage._connection.execute(
                    "SELECT * FROM storage_leases WHERE id = ?",
                    (running.id,),
                ).fetchone()
            )
            == input_lease
        )
        assert storage.total_reserved_bytes() == reservation
        assert (
            tuple(
                storage._connection.execute(
                    "SELECT * FROM storage_leases WHERE id = ?",
                    (result_ref.id,),
                ).fetchone()
            )
            == result_lease
        )
        assert result_ref.path.read_bytes() == result_payload

        assert (
            storage.commit_job_success(
                running.id,
                running.attempt_token,
                result_ref,
            )
            is jobs.JobSuccessOutcome.COMMITTED
        )
        cleaned = storage.get_visible_job(running.id)
        assert cleaned is not None
        assert cleaned.input_lease_id is None
        assert not cleaned.input_cleanup_pending
        assert not input_path.exists()
        assert storage.total_reserved_bytes() == result_ref.actual_bytes
        assert result_ref.path.read_bytes() == result_payload
        assert (
            tuple(
                storage._connection.execute(
                    "SELECT * FROM storage_leases WHERE id = ?",
                    (result_ref.id,),
                ).fetchone()
            )
            == result_lease
        )
    finally:
        storage.close()


@pytest.mark.parametrize(
    "restart_state",
    ("running-after-crash", "running-at-shutdown", "committed"),
)
def test_startup_recovers_or_retains_valid_job_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    restart_state: str,
) -> None:
    patch_ids(monkeypatch)
    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    running = queue_and_claim(storage)
    finish_progress(storage, running)
    result_ref = seal_result(storage, running)
    if restart_state == "committed":
        assert running.attempt_token is not None
        assert (
            storage.commit_job_success(
                running.id,
                running.attempt_token,
                result_ref,
            )
            is jobs.JobSuccessOutcome.COMMITTED
        )
    elif restart_state == "running-at-shutdown":
        storage.write_shutdown_marker("generation-1", FINISHED_AT)
    storage.close()

    reopened = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    try:
        succeeded = reopened.get_visible_job(running.id)
        assert succeeded is not None
        assert succeeded.status is jobs.JobStatus.SUCCEEDED
        assert succeeded.processed_samples == succeeded.total_samples
        assert succeeded.attempt_token is None
        assert succeeded.owner_generation is None
        assert succeeded.crash_recoveries == running.crash_recoveries
        assert succeeded.result_lease_id == result_ref.id
        assert succeeded.input_lease_id is None
        assert not succeeded.input_cleanup_pending
        assert succeeded.finished_at == FINISHED_AT
        assert reopened.resolve_artifact(result_ref).is_file()
        assert reopened.total_reserved_bytes() == result_ref.actual_bytes
    finally:
        reopened.close()

    clean_restart = Storage(
        tmp_path,
        limits(),
        current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40,
    )
    try:
        assert (
            clean_restart.get_visible_job(running.id).result_lease_id
            == result_ref.id
        )
        assert clean_restart.resolve_artifact(result_ref).is_file()
    finally:
        clean_restart.close()


def test_startup_cancel_wins_over_valid_job_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_ids(monkeypatch)
    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    running = queue_and_claim(storage)
    finish_progress(storage, running)
    result_ref = seal_result(storage, running)
    storage._connection.execute(
        """
        UPDATE transcription_jobs SET cancel_requested = 1 WHERE id = ?
        """,
        (running.id,),
    )
    result_ref.path.write_bytes(b"truncated")
    storage.close()

    reopened = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    try:
        cancelled = reopened.get_visible_job(running.id)
        assert cancelled is not None
        assert cancelled.status is jobs.JobStatus.CANCELLED
        assert cancelled.result_lease_id is None
        assert cancelled.input_lease_id is None
        assert not result_ref.path.exists()
        assert (
            reopened._connection.execute(
                "SELECT COUNT(*) FROM storage_leases"
            ).fetchone()[0]
            == 0
        )
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "artifact_state",
    ("writing", "renamed-writing", "invalid-sealed", "orphan"),
)
def test_startup_cleans_uncommitted_result_then_recovers_running_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_state: str,
) -> None:
    patch_ids(monkeypatch)
    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    running = queue_and_claim(storage)
    artifact_paths: tuple[Path, ...]
    if artifact_state in {"writing", "renamed-writing"}:
        assert running.attempt_token is not None
        writer = storage.begin_job_result_artifact(
            running.id,
            running.attempt_token,
        )
        storage.append_artifact(writer, b"partial")
        partial_path = writer.path
        complete_path = partial_path.with_suffix(".complete")
        artifact_paths = (partial_path, complete_path)
        storage.close()
        if artifact_state == "renamed-writing":
            partial_path.replace(complete_path)
    elif artifact_state == "invalid-sealed":
        finish_progress(storage, running)
        result_ref = seal_result(storage, running)
        invalid = b"not-an-envelope"
        result_ref.path.write_bytes(invalid)
        storage._connection.execute(
            """
            UPDATE storage_leases SET
                reserved_bytes = ?, actual_bytes = ?, content_sha256 = ?
            WHERE id = ?
            """,
            (
                len(invalid),
                len(invalid),
                hashlib.sha256(invalid).hexdigest(),
                result_ref.id,
            ),
        )
        artifact_paths = (result_ref.path,)
        storage.close()
    else:
        orphan_path = storage.artifact_dir / f"{'e' * 32}.complete"
        orphan_path.write_bytes(b"orphan")
        artifact_paths = (orphan_path,)
        storage.close()

    reopened = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    try:
        recovered = reopened.get_visible_job(running.id)
        assert recovered is not None
        assert recovered.status is jobs.JobStatus.QUEUED
        assert recovered.crash_recoveries == 1
        assert recovered.result_lease_id is None
        assert all(not path.exists() for path in artifact_paths)
        assert (
            reopened._connection.execute(
                """
                SELECT COUNT(*) FROM storage_leases
                WHERE lease_type = 'artifact' AND owner_kind = 'job'
                """
            ).fetchone()[0]
            == 0
        )
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "corruption",
    (
        "owner",
        "path",
        "size",
        "hash",
        "file-type",
        "duplicate",
        "bound-ref",
        "envelope",
    ),
)
def test_startup_job_result_corruption_fails_closed_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    patch_ids(monkeypatch)
    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    running = queue_and_claim(storage)
    finish_progress(storage, running)
    result_ref = seal_result(storage, running)
    if corruption in {"bound-ref", "envelope"}:
        assert running.attempt_token is not None
        assert (
            storage.commit_job_success(
                running.id,
                running.attempt_token,
                result_ref,
            )
            is jobs.JobSuccessOutcome.COMMITTED
        )
        if corruption == "bound-ref":
            storage._connection.execute(
                """
                UPDATE transcription_jobs
                SET result_lease_id = ? WHERE id = ?
                """,
                ("b" * 32, running.id),
            )
        else:
            invalid = b"not-an-envelope"
            result_ref.path.write_bytes(invalid)
            storage._connection.execute(
                """
                UPDATE storage_leases SET
                    reserved_bytes = ?, actual_bytes = ?,
                    content_sha256 = ?
                WHERE id = ?
                """,
                (
                    len(invalid),
                    len(invalid),
                    hashlib.sha256(invalid).hexdigest(),
                    result_ref.id,
                ),
            )
    elif corruption == "owner":
        storage._connection.execute(
            "UPDATE storage_leases SET owner_id = ? WHERE id = ?",
            ("ABCDEFGH", result_ref.id),
        )
    elif corruption == "path":
        storage._connection.execute(
            "UPDATE storage_leases SET controlled_path = ? WHERE id = ?",
            (
                str(
                    storage.artifact_dir
                    / f"{result_ref.id}.partial"
                ),
                result_ref.id,
            ),
        )
    elif corruption == "size":
        storage._connection.execute(
            """
            UPDATE storage_leases SET
                reserved_bytes = reserved_bytes + 1,
                actual_bytes = actual_bytes + 1
            WHERE id = ?
            """,
            (result_ref.id,),
        )
    elif corruption == "hash":
        storage._connection.execute(
            """
            UPDATE storage_leases SET content_sha256 = ? WHERE id = ?
            """,
            ("f" * 64, result_ref.id),
        )
    elif corruption == "file-type":
        result_ref.path.unlink()
        result_ref.path.mkdir()
    else:
        duplicate_id = "b" * 32
        duplicate_path = (
            storage.artifact_dir / f"{duplicate_id}.complete"
        )
        duplicate_path.write_bytes(result_ref.path.read_bytes())
        storage._connection.execute(
            """
            INSERT INTO storage_leases(
                id, lease_type, resource_kind, owner_kind, owner_id,
                phase, controlled_path, reserved_bytes, actual_bytes,
                content_sha256
            ) VALUES (?, 'artifact', 'result_complete', 'job', ?,
                      'sealed', ?, ?, ?, ?)
            """,
            (
                duplicate_id,
                running.id,
                str(duplicate_path),
                result_ref.actual_bytes,
                result_ref.actual_bytes,
                result_ref.content_sha256,
            ),
        )
    storage.close()
    before = recovery_snapshot(tmp_path)

    with pytest.raises(StorageSchemaError):
        Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)

    assert recovery_snapshot(tmp_path) == before


def test_success_commit_rechecks_exact_result_lease_in_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_ids(monkeypatch)
    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    try:
        running = queue_and_claim(storage)
        finish_progress(storage, running)
        result_ref = seal_result(storage, running)
        original_validate = storage_module.ResultEnvelopeReader.validate

        def validate_then_remove(
            reader: storage_module.ResultEnvelopeReader,
        ) -> object:
            manifest = original_validate(reader)
            storage.release_artifact(result_ref)
            return manifest

        monkeypatch.setattr(
            storage_module.ResultEnvelopeReader,
            "validate",
            validate_then_remove,
        )

        assert running.attempt_token is not None
        assert (
            storage.commit_job_success(
                running.id,
                running.attempt_token,
                result_ref,
            )
            is jobs.JobSuccessOutcome.STALE
        )
        current = storage.get_visible_job(running.id)
        assert current is not None
        assert current.status is jobs.JobStatus.RUNNING
        assert current.result_lease_id is None
    finally:
        storage.close()


def test_job_result_begin_is_token_fenced_and_unique_across_connections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_ids(monkeypatch)
    first = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    second = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    try:
        running = queue_and_claim(first)
        assert running.attempt_token is not None

        with pytest.raises(jobs.StaleJobAttemptError):
            first.begin_job_result_artifact(running.id, "stale-token")
        writer = first.begin_job_result_artifact(
            running.id,
            running.attempt_token,
        )
        with pytest.raises(RuntimeError):
            second.begin_job_result_artifact(
                running.id,
                running.attempt_token,
            )
        first.abort_artifact(writer)
        request_cancel(first, running.id)
        with pytest.raises(jobs.JobCancellationRequestedError):
            first.begin_job_result_artifact(
                running.id,
                running.attempt_token,
            )
    finally:
        second.close()
        first.close()


def test_resolve_job_attempt_input_returns_fresh_job_and_ready_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, running = new_running_job(tmp_path, monkeypatch)
    with closing(storage):
        assert running.attempt_token is not None
        assert (
            storage.update_job_progress(
                running.id,
                running.attempt_token,
                10,
            )
            is jobs.JobProgressOutcome.UPDATED
        )

        resolved, input_path = storage.resolve_job_attempt_input(
            running.id,
            running.attempt_token,
        )

        assert resolved == storage.get_visible_job(running.id)
        assert resolved.processed_samples == 10
        assert resolved.status is jobs.JobStatus.RUNNING
        assert resolved.attempt_token == running.attempt_token
        assert input_path == storage.staging_dir / f"{running.id}.ready"
        assert input_path.read_bytes() == b"audio"


@pytest.mark.parametrize(
    ("operation", "attempt_state"),
    (
        ("resolve", "wrong-token"),
        ("resolve", "missing-job"),
        ("resolve", "cancel"),
        ("segment", "wrong-token"),
        ("segment", "cancel"),
    ),
)
def test_job_attempt_storage_operations_are_typed_fenced_and_cancel_aware(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    attempt_state: str,
) -> None:
    storage, running = new_running_job(tmp_path, monkeypatch)
    with closing(storage):
        assert running.attempt_token is not None
        job_id = running.id
        token = running.attempt_token
        expected_error = (
            jobs.JobCancellationRequestedError
            if attempt_state == "cancel"
            else jobs.StaleJobAttemptError
        )
        if attempt_state == "wrong-token":
            token = "stale-token"
        elif attempt_state == "missing-job":
            job_id = "ABCDEFGH"
        else:
            request_cancel(storage, running.id)
        before = recovery_snapshot(tmp_path)

        with pytest.raises(expected_error):
            if operation == "resolve":
                storage.resolve_job_attempt_input(job_id, token)
            else:
                storage.begin_job_attempt_artifact(job_id, token)

        assert recovery_snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "corruption",
    (
        "job-input-ref",
        "missing-lease",
        "controlled-path",
        "input-size",
        "content-digest",
        "request-fingerprint",
        "processor-fingerprint",
    ),
)
def test_resolve_job_attempt_input_fails_closed_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    storage, running = new_running_job(tmp_path, monkeypatch)
    with closing(storage):
        assert running.attempt_token is not None
        if corruption == "job-input-ref":
            storage._connection.execute(
                """
                UPDATE transcription_jobs SET input_lease_id = ?
                WHERE id = ?
                """,
                ("b" * 32, running.id),
            )
        elif corruption == "missing-lease":
            storage._connection.execute(
                "DELETE FROM storage_leases WHERE id = ?",
                (running.id,),
            )
        elif corruption == "controlled-path":
            storage._connection.execute(
                """
                UPDATE storage_leases SET controlled_path = ?
                WHERE id = ?
                """,
                (
                    str(storage.staging_dir / f"{running.id}.partial"),
                    running.id,
                ),
            )
        elif corruption == "input-size":
            storage._connection.execute(
                """
                UPDATE transcription_jobs
                SET input_size_bytes = input_size_bytes + 1
                WHERE id = ?
                """,
                (running.id,),
            )
        elif corruption == "content-digest":
            storage._connection.execute(
                """
                UPDATE storage_leases SET content_sha256 = ?
                WHERE id = ?
                """,
                ("f" * 64, running.id),
            )
        elif corruption == "request-fingerprint":
            storage._connection.execute(
                """
                UPDATE transcription_jobs SET request_fingerprint = ?
                WHERE id = ?
                """,
                ("f" * 64, running.id),
            )
        else:
            storage._connection.execute(
                """
                UPDATE transcription_jobs SET processor_fingerprint = ?
                WHERE id = ?
                """,
                ("4" * 64, running.id),
            )
        before = recovery_snapshot(tmp_path)

        with pytest.raises(StorageSchemaError):
            storage.resolve_job_attempt_input(
                running.id,
                running.attempt_token,
            )

        assert recovery_snapshot(tmp_path) == before


def test_job_attempt_segment_is_unique_and_can_coexist_with_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, running = new_running_job(tmp_path, monkeypatch)
    with closing(storage):
        assert running.attempt_token is not None
        segment = storage.begin_job_attempt_artifact(
            running.id,
            running.attempt_token,
        )
        storage.append_artifact(segment, b'{"index":0}\n')
        segment_ref = storage.seal_artifact(segment)

        with pytest.raises(RuntimeError, match="already exists"):
            storage.begin_job_attempt_artifact(
                running.id,
                running.attempt_token,
            )

        result = storage.begin_job_result_artifact(
            running.id,
            running.attempt_token,
        )
        kinds = tuple(
            row[0]
            for row in storage._connection.execute(
                """
                SELECT resource_kind FROM storage_leases
                WHERE lease_type = 'artifact'
                  AND owner_kind = 'job' AND owner_id = ?
                ORDER BY resource_kind
                """,
                (running.id,),
            )
        )
        assert kinds == ("result_complete", "segment_jsonl")

        storage.abort_artifact(result)
        storage.release_artifact(segment_ref)


@pytest.mark.parametrize(
    "restart_state",
    ("writing", "sealed", "sealed-with-result"),
)
def test_startup_cleans_attempt_segment_without_treating_it_as_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    restart_state: str,
) -> None:
    patch_ids(monkeypatch)
    storage = Storage(
        tmp_path,
        limits(),
        current_processor_fingerprint=PROCESSOR_FINGERPRINT,
        free_bytes=lambda _: 1 << 40,
    )
    running = queue_and_claim(storage)
    assert running.attempt_token is not None
    segment = storage.begin_job_attempt_artifact(
        running.id,
        running.attempt_token,
    )
    storage.append_artifact(segment, b'{"index":0}\n')
    segment_id = segment.id
    segment_paths = (
        segment.path,
        segment.path.with_suffix(".complete"),
    )
    result_ref = None
    if restart_state != "writing":
        storage.seal_artifact(segment)
    if restart_state == "sealed-with-result":
        finish_progress(storage, running)
        result_ref = seal_result(storage, running)
    storage.close()

    reopened = Storage(
        tmp_path,
        limits(),
        current_processor_fingerprint=PROCESSOR_FINGERPRINT,
        free_bytes=lambda _: 1 << 40,
    )
    try:
        recovered = reopened.get_visible_job(running.id)
        assert recovered is not None
        if result_ref is None:
            assert recovered.status is jobs.JobStatus.QUEUED
            assert recovered.crash_recoveries == 1
            assert recovered.result_lease_id is None
        else:
            assert recovered.status is jobs.JobStatus.SUCCEEDED
            assert recovered.result_lease_id == result_ref.id
            assert reopened.resolve_artifact(result_ref).is_file()
        assert all(not path.exists() for path in segment_paths)
        assert (
            reopened._connection.execute(
                "SELECT COUNT(*) FROM storage_leases WHERE id = ?",
                (segment_id,),
            ).fetchone()[0]
            == 0
        )
    finally:
        reopened.close()


def test_startup_rejects_duplicate_attempt_segments_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, running = new_running_job(tmp_path, monkeypatch)
    assert running.attempt_token is not None
    segment = storage.begin_job_attempt_artifact(
        running.id,
        running.attempt_token,
    )
    storage.append_artifact(segment, b'{"index":0}\n')
    first_ref = storage.seal_artifact(segment)
    duplicate_id = "b" * 32
    duplicate_path = storage.artifact_dir / f"{duplicate_id}.complete"
    duplicate_path.write_bytes(first_ref.path.read_bytes())
    storage._connection.execute(
        """
        INSERT INTO storage_leases(
            id, lease_type, resource_kind, owner_kind, owner_id,
            phase, controlled_path, reserved_bytes, actual_bytes,
            content_sha256
        ) VALUES (?, 'artifact', 'segment_jsonl', 'job', ?,
                  'sealed', ?, ?, ?, ?)
        """,
        (
            duplicate_id,
            running.id,
            str(duplicate_path),
            first_ref.actual_bytes,
            first_ref.actual_bytes,
            first_ref.content_sha256,
        ),
    )
    storage.close()
    before = recovery_snapshot(tmp_path)

    with pytest.raises(StorageSchemaError):
        Storage(
            tmp_path,
            limits(),
            current_processor_fingerprint=PROCESSOR_FINGERPRINT,
            free_bytes=lambda _: 1 << 40,
        )

    assert recovery_snapshot(tmp_path) == before


@pytest.mark.parametrize(
    ("losing_state", "expected"),
    (
        ("stale", "STALE"),
        ("wrong-token", "STALE"),
        ("cancel", "CANCEL_REQUESTED"),
        ("incomplete", "STALE"),
    ),
)
def test_losing_success_commit_cleans_result_without_overwriting_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    losing_state: str,
    expected: str,
) -> None:
    patch_ids(monkeypatch)
    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    try:
        running = queue_and_claim(storage)
        assert running.attempt_token is not None
        if losing_state != "incomplete":
            assert (
                storage.update_job_progress(
                    running.id,
                    running.attempt_token,
                    TOTAL_SAMPLES,
                    total_samples=TOTAL_SAMPLES,
                )
                is jobs.JobProgressOutcome.UPDATED
            )
        result_ref = seal_result(storage, running)
        if losing_state == "stale":
            storage._connection.execute(
                """
                UPDATE transcription_jobs SET
                    status = 'queued', processed_samples = 0,
                    attempt_token = NULL, owner_generation = NULL,
                    started_at = NULL
                WHERE id = ?
                """,
                (running.id,),
            )
        elif losing_state == "cancel":
            storage._connection.execute(
                """
                UPDATE transcription_jobs SET cancel_requested = 1
                WHERE id = ?
                """,
                (running.id,),
            )
        before = storage.get_visible_job(running.id)

        commit_token = (
            "stale-token" if losing_state == "wrong-token" else running.attempt_token
        )
        outcome = storage.commit_job_success(
            running.id,
            commit_token,
            result_ref,
        )

        assert outcome is getattr(jobs.JobSuccessOutcome, expected)
        assert storage.get_visible_job(running.id) == before
        assert not result_ref.path.exists()
        assert (
            storage._connection.execute(
                "SELECT COUNT(*) FROM storage_leases WHERE id = ?",
                (result_ref.id,),
            ).fetchone()[0]
            == 0
        )
    finally:
        storage.close()
