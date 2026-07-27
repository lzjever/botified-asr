from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import threading

import pytest

import botified_asr.jobs as jobs
import botified_asr.storage as storage_module
from botified_asr.config import LimitsConfig, RESERVATION_QUANTUM
from botified_asr.contracts import DIRECT_MAX_SAMPLES, MAX_AUDIO_SAMPLES
from botified_asr.storage import Storage, StorageAdmissionError, StorageSchemaError


CREATED_AT = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
CANONICAL_OPTIONS_JSON = (
    '{"chunking_strategy":null,"include":[],"known_speaker_ids":[],'
    '"language":"auto","model":"sensevoice","response_format":"json"}'
)
def limits(**overrides: int) -> LimitsConfig:
    values = {
        "max_upload_bytes": RESERVATION_QUANTUM,
        "sync_max_upload_bytes": RESERVATION_QUANTUM,
        "max_active_uploads": 4,
        "max_queued_jobs": 4,
        "max_job_storage_bytes": 4 * RESERVATION_QUANTUM,
        "min_filesystem_free_bytes": 1,
    }
    values.update(overrides)
    return LimitsConfig(**values)


def queued_spec(**overrides: object) -> object:
    values = {
        "canonical_options_json": CANONICAL_OPTIONS_JSON,
        "selected_speaker_snapshot": b'{"speakers":[]}',
        "snapshot_sha256": "1" * 64,
        "effective_max_audio_samples": 32_000,
        "effective_direct_max_audio_samples": 16_000,
        "request_fingerprint": "2" * 64,
        "processor_fingerprint": "3" * 64,
    }
    values.update(overrides)
    return jobs.QueuedJobSpec(**values)


def patch_job_ids(
    monkeypatch: pytest.MonkeyPatch,
    *job_ids: str,
) -> None:
    source = iter(job_ids)

    def generate_job_id() -> str:
        return next(source)

    monkeypatch.setattr(
        jobs,
        "generate_job_id",
        generate_job_id,
        raising=False,
    )
    monkeypatch.setattr(
        storage_module,
        "generate_job_id",
        generate_job_id,
        raising=False,
    )


def test_queued_job_spec_validation_and_generated_ids() -> None:
    spec = queued_spec()
    with pytest.raises(FrozenInstanceError):
        spec.effective_max_audio_samples = 1

    generated = tuple(jobs.generate_job_id() for _ in range(64))
    assert all(jobs.validate_job_id(job_id) == job_id for job_id in generated)

    invalid_changes = (
        {"canonical_options_json": ""},
        {"canonical_options_json": CANONICAL_OPTIONS_JSON.encode()},
        {"canonical_options_json": '{"model":"sensevoice"}'},
        {
            "canonical_options_json": CANONICAL_OPTIONS_JSON.replace(
                ":null", ": null"
            )
        },
        {"selected_speaker_snapshot": '{"speakers":[]}'},
        {"total_samples": 1},
        {"effective_max_audio_samples": True},
        {"effective_max_audio_samples": 1.0},
        {"effective_max_audio_samples": 0},
        {"effective_max_audio_samples": MAX_AUDIO_SAMPLES + 1},
        {"effective_direct_max_audio_samples": True},
        {"effective_direct_max_audio_samples": 1.0},
        {"effective_direct_max_audio_samples": 0},
        {
            "effective_max_audio_samples": 16_000,
            "effective_direct_max_audio_samples": 16_001,
        },
        {
            "effective_max_audio_samples": DIRECT_MAX_SAMPLES + 1,
            "effective_direct_max_audio_samples": DIRECT_MAX_SAMPLES + 1,
        },
        {"snapshot_sha256": "A" * 64},
        {"request_fingerprint": "2" * 63},
        {"processor_fingerprint": "g" * 64},
    )
    for changes in invalid_changes:
        with pytest.raises((TypeError, ValueError)):
            queued_spec(**changes)


def test_begin_job_upload_atomically_creates_dedicated_rows_and_retries_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_job_ids(monkeypatch, "01234567", "01234567", "ABCDEFGH")
    storage = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    try:
        first = storage.begin_job_upload(CREATED_AT)
        second = storage.begin_job_upload(CREATED_AT)

        assert isinstance(first, storage_module.JobUploadLease)
        assert isinstance(second, storage_module.JobUploadLease)
        assert (first.id, second.id) == ("01234567", "ABCDEFGH")
        assert first.path == storage.staging_dir / "01234567.partial"
        assert second.path == storage.staging_dir / "ABCDEFGH.partial"
        assert storage.active_upload_count() == 2
        assert storage.total_reserved_bytes() == 2 * RESERVATION_QUANTUM

        job_rows = storage._connection.execute(
            """
            SELECT id, phase, status, input_lease_id, created_at
            FROM transcription_jobs ORDER BY id
            """
        ).fetchall()
        assert tuple(tuple(row) for row in job_rows) == (
            ("01234567", "receiving", None, "01234567", "2026-07-27T12:00:00Z"),
            ("ABCDEFGH", "receiving", None, "ABCDEFGH", "2026-07-27T12:00:00Z"),
        )
        lease_rows = storage._connection.execute(
            """
            SELECT id, lease_type, owner_kind, owner_id, phase,
                   reserved_bytes, actual_bytes
            FROM storage_leases ORDER BY id
            """
        ).fetchall()
        assert tuple(tuple(row) for row in lease_rows) == (
            (
                "01234567",
                "upload",
                "job",
                "01234567",
                "writing",
                RESERVATION_QUANTUM,
                0,
            ),
            (
                "ABCDEFGH",
                "upload",
                "job",
                "ABCDEFGH",
                "writing",
                RESERVATION_QUANTUM,
                0,
            ),
        )
    finally:
        storage.close()


def test_seal_then_publish_roundtrips_visible_job_and_rejects_stale_handles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_job_ids(monkeypatch, "7K3M9Q2W")
    storage = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    try:
        lease = storage.begin_job_upload(CREATED_AT)
        spec = queued_spec()
        with pytest.raises(TypeError):
            storage.publish_job(lease, spec)
        with pytest.raises(TypeError):
            storage.append(lease, b"wrong-handle")
        with pytest.raises(TypeError):
            storage.seal_upload(lease)
        with pytest.raises(TypeError):
            storage.abort_upload(lease)

        storage.append_job_upload(lease, b"audio")
        input_ref = storage.seal_job_upload(lease)

        assert isinstance(input_ref, storage_module.JobInputRef)
        assert input_ref.path == storage.staging_dir / "7K3M9Q2W.ready"
        assert input_ref.actual_bytes == 5
        assert storage.active_upload_count() == 0
        assert storage.total_reserved_bytes() == 5
        assert storage.get_visible_job(lease.id) is None
        receiving = storage._connection.execute(
            """
            SELECT phase, status, canonical_options_json
            FROM transcription_jobs WHERE id = ?
            """,
            (lease.id,),
        ).fetchone()
        assert tuple(receiving) == ("receiving", None, None)
        with pytest.raises(TypeError):
            storage.resolve_input(input_ref)
        with pytest.raises(TypeError):
            storage.release_input(input_ref)

        published = storage.publish_job(input_ref, spec)
        assert published == storage.get_visible_job(lease.id)
        assert published.id == lease.id
        assert published.phase is jobs.JobPhase.VISIBLE
        assert published.status is jobs.JobStatus.QUEUED
        assert published.input_size_bytes == 5
        assert published.effective_max_audio_samples == 32_000
        assert published.effective_direct_max_audio_samples == 16_000
        assert published.total_samples is None
        assert published.processed_samples == 0
        assert published.created_at == CREATED_AT
        assert published.canonical_options_json == spec.canonical_options_json
        assert published.selected_speaker_snapshot == spec.selected_speaker_snapshot
        assert storage.get_visible_job("ABCDEFGH") is None

        with pytest.raises(ValueError):
            replace(
                published,
                status=jobs.JobStatus.SUCCEEDED,
                input_lease_id=None,
                attempt_no=1,
                result_lease_id="a" * 32,
                started_at=CREATED_AT,
                finished_at=CREATED_AT,
            )

        with pytest.raises(RuntimeError):
            storage.publish_job(input_ref, spec)
    finally:
        storage.close()


@pytest.mark.parametrize("sealed", (False, True))
def test_abort_job_upload_is_idempotent_without_leaks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sealed: bool,
) -> None:
    patch_job_ids(monkeypatch, "7K3M9Q2W")
    storage = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    try:
        lease = storage.begin_job_upload(CREATED_AT)
        storage.append_job_upload(lease, b"audio")
        abort_handle = lease
        if sealed:
            abort_handle = storage.seal_job_upload(lease)

        storage.abort_job_upload(abort_handle)
        storage.abort_job_upload(abort_handle)

        assert (
            storage._connection.execute(
                "SELECT COUNT(*) FROM transcription_jobs"
            ).fetchone()[0]
            == 0
        )
        assert (
            storage._connection.execute(
                "SELECT COUNT(*) FROM storage_leases"
            ).fetchone()[0]
            == 0
        )
        assert storage.active_upload_count() == 0
        assert storage.total_reserved_bytes() == 0
        assert not (storage.staging_dir / f"{lease.id}.partial").exists()
        assert not (storage.staging_dir / f"{lease.id}.ready").exists()
        assert storage.get_visible_job(lease.id) is None
    finally:
        storage.close()


@pytest.mark.parametrize(
    "scenario",
    ("max-active", "max-queued", "unknown-partial", "database-trigger"),
)
def test_begin_job_upload_admission_and_fault_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
) -> None:
    if scenario == "max-active":
        configured = limits(max_active_uploads=1, max_queued_jobs=2)
        first = Storage(tmp_path, configured, free_bytes=lambda _: 1 << 40)
        second = Storage(tmp_path, configured, free_bytes=lambda _: 1 << 40)
        ids = iter(("01234567", "ABCDEFGH"))
        id_lock = threading.Lock()

        def generate_job_id() -> str:
            with id_lock:
                return next(ids)

        monkeypatch.setattr(storage_module, "generate_job_id", generate_job_id)
        barrier = threading.Barrier(2)
        outcomes: list[tuple[str, Storage, object]] = []

        def begin(storage: Storage) -> None:
            barrier.wait()
            try:
                outcomes.append(("ok", storage, storage.begin_job_upload(CREATED_AT)))
            except StorageAdmissionError as error:
                outcomes.append((error.code, storage, error))

        threads = (
            threading.Thread(target=begin, args=(first,)),
            threading.Thread(target=begin, args=(second,)),
        )
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        try:
            assert [kind for kind, *_ in outcomes].count("ok") == 1
            assert [kind for kind, *_ in outcomes].count("too_many_active_uploads") == 1
            assert (
                first._connection.execute(
                    "SELECT COUNT(*) FROM transcription_jobs"
                ).fetchone()[0]
                == 1
            )
            assert (
                first._connection.execute(
                    "SELECT COUNT(*) FROM storage_leases"
                ).fetchone()[0]
                == 1
            )
            assert first.total_reserved_bytes() == RESERVATION_QUANTUM
        finally:
            for kind, owner, handle in outcomes:
                if kind == "ok":
                    owner.abort_job_upload(handle)
            first.close()
            second.close()
        return

    if scenario == "max-queued":
        configured = limits(max_active_uploads=2, max_queued_jobs=1)
        first = Storage(tmp_path, configured, free_bytes=lambda _: 1 << 40)
        second = Storage(tmp_path, configured, free_bytes=lambda _: 1 << 40)
        patch_job_ids(monkeypatch, "01234567", "ABCDEFGH")
        try:
            refs = []
            for storage, payload in (
                (first, b"first"),
                (second, b"second"),
            ):
                lease = storage.begin_job_upload(CREATED_AT)
                storage.append_job_upload(lease, payload)
                refs.append(storage.seal_job_upload(lease))

            barrier = threading.Barrier(2)
            outcomes: list[tuple[str, Storage, object, object]] = []

            def publish(storage: Storage, ref: object) -> None:
                barrier.wait()
                try:
                    outcomes.append(
                        (
                            "ok",
                            storage,
                            ref,
                            storage.publish_job(ref, queued_spec()),
                        )
                    )
                except StorageAdmissionError as error:
                    outcomes.append((error.code, storage, ref, error))

            threads = (
                threading.Thread(target=publish, args=(first, refs[0])),
                threading.Thread(target=publish, args=(second, refs[1])),
            )
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            assert [kind for kind, *_ in outcomes].count("ok") == 1
            assert [kind for kind, *_ in outcomes].count("too_many_queued_jobs") == 1
            rejected = next(
                outcome for outcome in outcomes if outcome[0] == "too_many_queued_jobs"
            )
            _, rejected_storage, rejected_ref, _ = rejected
            assert rejected_storage.get_visible_job(rejected_ref.id) is None
            assert tuple(
                rejected_storage._connection.execute(
                    """
                    SELECT phase, status FROM transcription_jobs WHERE id = ?
                    """,
                    (rejected_ref.id,),
                ).fetchone()
            ) == ("receiving", None)
            assert (
                rejected_storage._connection.execute(
                    "SELECT phase FROM storage_leases WHERE id = ?",
                    (rejected_ref.id,),
                ).fetchone()[0]
                == "sealed"
            )

            rejected_storage.abort_job_upload(rejected_ref)

            assert (
                first._connection.execute(
                    "SELECT COUNT(*) FROM transcription_jobs"
                ).fetchone()[0]
                == 1
            )
            assert (
                first._connection.execute(
                    "SELECT COUNT(*) FROM storage_leases"
                ).fetchone()[0]
                == 1
            )
            assert not rejected_ref.path.exists()
        finally:
            first.close()
            second.close()
        return

    patch_job_ids(monkeypatch, "01234567", "ABCDEFGH")
    storage = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    try:
        if scenario == "unknown-partial":
            unknown = storage.staging_dir / "01234567.partial"
            unknown.write_bytes(b"not-this-request")

            lease = storage.begin_job_upload(CREATED_AT)

            assert lease.id == "ABCDEFGH"
            assert unknown.read_bytes() == b"not-this-request"
            assert (
                storage._connection.execute(
                    "SELECT COUNT(*) FROM transcription_jobs"
                ).fetchone()[0]
                == 1
            )
            assert storage.total_reserved_bytes() == RESERVATION_QUANTUM
            storage.abort_job_upload(lease)
        else:
            storage._connection.execute(
                """
                CREATE TRIGGER reject_job_lease
                BEFORE INSERT ON storage_leases
                WHEN NEW.owner_kind = 'job'
                BEGIN
                    SELECT RAISE(ABORT, 'injected job lease failure');
                END
                """
            )
            with pytest.raises(
                sqlite3.IntegrityError,
                match="injected job lease failure",
            ):
                storage.begin_job_upload(CREATED_AT)
            assert (
                storage._connection.execute(
                    "SELECT COUNT(*) FROM transcription_jobs"
                ).fetchone()[0]
                == 0
            )
            assert (
                storage._connection.execute(
                    "SELECT COUNT(*) FROM storage_leases"
                ).fetchone()[0]
                == 0
            )
            assert storage.total_reserved_bytes() == 0
            assert not list(storage.staging_dir.iterdir())
    finally:
        storage.close()


def test_seal_compensation_and_publish_do_not_leave_or_reread_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed_dir = tmp_path / "failed-seal"
    patch_job_ids(monkeypatch, "01234567")
    failed = Storage(failed_dir, limits(), free_bytes=lambda _: 1 << 40)
    lease = failed.begin_job_upload(CREATED_AT)
    failed.append_job_upload(lease, b"audio")
    failed._connection.execute(
        """
        CREATE TRIGGER reject_job_seal
        BEFORE UPDATE OF phase ON storage_leases
        WHEN OLD.owner_kind = 'job' AND NEW.phase = 'sealed'
        BEGIN
            SELECT RAISE(ABORT, 'injected job seal failure');
        END
        """
    )
    try:
        with pytest.raises(
            sqlite3.IntegrityError,
            match="injected job seal failure",
        ):
            failed.seal_job_upload(lease)
        assert (
            failed._connection.execute(
                "SELECT COUNT(*) FROM transcription_jobs"
            ).fetchone()[0]
            == 0
        )
        assert (
            failed._connection.execute(
                "SELECT COUNT(*) FROM storage_leases"
            ).fetchone()[0]
            == 0
        )
        assert failed.total_reserved_bytes() == 0
        assert not list(failed.staging_dir.iterdir())
    finally:
        failed.close()

    published_dir = tmp_path / "publish"
    patch_job_ids(monkeypatch, "ABCDEFGH")
    storage = Storage(published_dir, limits(), free_bytes=lambda _: 1 << 40)
    try:
        upload = storage.begin_job_upload(CREATED_AT)
        storage.append_job_upload(upload, b"audio")
        input_ref = storage.seal_job_upload(upload)
        original_open = Path.open

        def reject_large_input_read(
            path: Path,
            mode: str = "r",
            *args,
            **kwargs,
        ):
            if path == input_ref.path and "r" in mode:
                raise AssertionError("publish reread the uploaded input")
            return original_open(path, mode, *args, **kwargs)

        monkeypatch.setattr(Path, "open", reject_large_input_read)

        published = storage.publish_job(input_ref, queued_spec())

        assert published.status is jobs.JobStatus.QUEUED
        assert published.input_size_bytes == input_ref.actual_bytes
        assert published.request_fingerprint == "2" * 64
    finally:
        storage.close()


def test_seal_compensation_fsync_failure_remains_restart_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_job_ids(monkeypatch, "01234567")
    storage = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    lease = storage.begin_job_upload(CREATED_AT)
    storage.append_job_upload(lease, b"audio")
    storage._connection.execute(
        """
        CREATE TRIGGER reject_job_seal_for_compensation
        BEFORE UPDATE OF phase ON storage_leases
        WHEN OLD.owner_kind = 'job' AND NEW.phase = 'sealed'
        BEGIN
            SELECT RAISE(ABORT, 'injected job seal failure');
        END
        """
    )
    original_fsync_directory = storage_module._fsync_directory
    fsync_calls = 0

    def fail_compensation_fsync(directory: Path) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 2:
            raise OSError("injected compensation fsync failure")
        original_fsync_directory(directory)

    with monkeypatch.context() as fault:
        fault.setattr(
            storage_module,
            "_fsync_directory",
            fail_compensation_fsync,
        )
        with pytest.raises(
            OSError,
            match="injected compensation fsync failure",
        ):
            storage.seal_job_upload(lease)

    assert fsync_calls == 2
    job_row = storage._connection.execute(
        "SELECT phase FROM transcription_jobs WHERE id = '01234567'"
    ).fetchone()
    lease_row = storage._connection.execute(
        "SELECT phase FROM storage_leases WHERE id = '01234567'"
    ).fetchone()
    assert job_row is not None
    assert lease_row is not None
    assert job_row[0] in {"receiving", "deleting"}
    assert lease_row[0] == "writing"
    assert storage.total_reserved_bytes() == RESERVATION_QUANTUM
    storage.close()

    recovered = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    try:
        assert (
            recovered._connection.execute(
                "SELECT COUNT(*) FROM transcription_jobs"
            ).fetchone()[0]
            == 0
        )
        assert (
            recovered._connection.execute(
                "SELECT COUNT(*) FROM storage_leases"
            ).fetchone()[0]
            == 0
        )
        assert recovered.total_reserved_bytes() == 0
        assert not (recovered.staging_dir / "01234567.partial").exists()
        assert not (recovered.staging_dir / "01234567.ready").exists()
    finally:
        recovered.close()


@pytest.mark.parametrize("sealed", (False, True))
def test_startup_cleans_receiving_inputs_and_abort_fault_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sealed: bool,
) -> None:
    patch_job_ids(monkeypatch, "01234567")
    first = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    lease = first.begin_job_upload(CREATED_AT)
    first.append_job_upload(lease, b"audio")
    path = lease.path
    if sealed:
        path = first.seal_job_upload(lease).path
    first.close()

    observed_phases: list[str] = []
    original_fsync_directory = storage_module._fsync_directory

    def observe_deleting_before_fsync(directory: Path) -> None:
        connection = sqlite3.connect(tmp_path / "botified-asr.sqlite3")
        try:
            row = connection.execute(
                "SELECT phase FROM transcription_jobs WHERE id = '01234567'"
            ).fetchone()
            if row is not None:
                observed_phases.append(row[0])
        finally:
            connection.close()
        original_fsync_directory(directory)

    with monkeypatch.context() as observing:
        observing.setattr(
            storage_module,
            "_fsync_directory",
            observe_deleting_before_fsync,
        )
        recovered = Storage(
            tmp_path,
            limits(),
            free_bytes=lambda _: 1 << 40,
        )
    try:
        assert observed_phases and set(observed_phases) == {"deleting"}
        assert (
            recovered._connection.execute(
                "SELECT COUNT(*) FROM transcription_jobs"
            ).fetchone()[0]
            == 0
        )
        assert (
            recovered._connection.execute(
                "SELECT COUNT(*) FROM storage_leases"
            ).fetchone()[0]
            == 0
        )
        assert recovered.total_reserved_bytes() == 0
        assert not path.exists()
    finally:
        recovered.close()

    if not sealed:
        fault_dir = tmp_path / "abort-fault"
        patch_job_ids(monkeypatch, "ABCDEFGH")
        storage = Storage(
            fault_dir,
            limits(),
            free_bytes=lambda _: 1 << 40,
        )
        handle = storage.begin_job_upload(CREATED_AT)
        storage.append_job_upload(handle, b"audio")

        with monkeypatch.context() as fault:

            def fail_fsync(_: Path) -> None:
                raise OSError("injected directory fsync failure")

            fault.setattr(storage_module, "_fsync_directory", fail_fsync)
            with pytest.raises(
                OSError,
                match="injected directory fsync failure",
            ):
                storage.abort_job_upload(handle)
        assert (
            storage._connection.execute(
                "SELECT phase FROM transcription_jobs WHERE id = 'ABCDEFGH'"
            ).fetchone()[0]
            == "deleting"
        )
        storage.close()

        resumed = Storage(
            fault_dir,
            limits(),
            free_bytes=lambda _: 1 << 40,
        )
        try:
            assert (
                resumed._connection.execute(
                    "SELECT COUNT(*) FROM transcription_jobs"
                ).fetchone()[0]
                == 0
            )
            assert resumed.total_reserved_bytes() == 0
        finally:
            resumed.close()


def test_startup_preflight_rejects_late_invalid_job_path_before_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_job_ids(monkeypatch, "01234567", "ABCDEFGH")
    first = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    ordinary = first.begin_job_upload(CREATED_AT)
    first.append_job_upload(ordinary, b"ordinary")
    invalid = first.begin_job_upload(CREATED_AT)
    first.append_job_upload(invalid, b"invalid")
    first.close()

    invalid.path.unlink()
    invalid.path.mkdir()
    connection = sqlite3.connect(tmp_path / "botified-asr.sqlite3")
    try:
        jobs_before = tuple(
            connection.execute(
                "SELECT * FROM transcription_jobs ORDER BY id"
            ).fetchall()
        )
        leases_before = tuple(
            connection.execute("SELECT * FROM storage_leases ORDER BY id").fetchall()
        )
    finally:
        connection.close()

    with pytest.raises(StorageSchemaError):
        Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)

    assert ordinary.path.read_bytes() == b"ordinary"
    assert invalid.path.is_dir()
    connection = sqlite3.connect(tmp_path / "botified-asr.sqlite3")
    try:
        assert (
            tuple(
                connection.execute(
                    "SELECT * FROM transcription_jobs ORDER BY id"
                ).fetchall()
            )
            == jobs_before
        )
        assert (
            tuple(
                connection.execute(
                    "SELECT * FROM storage_leases ORDER BY id"
                ).fetchall()
            )
            == leases_before
        )
        assert {
            row[0] for row in connection.execute("SELECT phase FROM transcription_jobs")
        } == {"receiving"}
    finally:
        connection.close()


def test_startup_preserves_queued_input_and_cleans_unprotected_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_job_ids(monkeypatch, "7K3M9Q2W")
    first = Storage(
        tmp_path,
        limits(max_job_storage_bytes=4 * RESERVATION_QUANTUM),
        free_bytes=lambda _: 1 << 40,
    )
    job_upload = first.begin_job_upload(CREATED_AT)
    first.append_job_upload(job_upload, b"job-audio")
    job_ref = first.seal_job_upload(job_upload)
    queued = first.publish_job(job_ref, queued_spec())

    sync_writing = first.begin_upload("transcription")
    first.append(sync_writing, b"sync-writing")
    sync_sealed_upload = first.begin_upload("transcription")
    first.append(sync_sealed_upload, b"sync-sealed")
    sync_sealed = first.seal_upload(sync_sealed_upload)
    same_id_partial = first.staging_dir / f"{queued.id}.partial"
    unrelated_orphan = first.staging_dir / "JKMNPQRT.ready"
    same_id_partial.write_bytes(b"old-partial")
    unrelated_orphan.write_bytes(b"orphan")
    first.close()

    recovered = Storage(
        tmp_path,
        limits(max_job_storage_bytes=4 * RESERVATION_QUANTUM),
        free_bytes=lambda _: 1 << 40,
    )
    try:
        assert recovered.get_visible_job(queued.id) == queued
        assert job_ref.path.read_bytes() == b"job-audio"
        assert recovered.total_reserved_bytes() == len(b"job-audio")
        assert not sync_writing.path.exists()
        assert not sync_sealed.path.exists()
        assert not same_id_partial.exists()
        assert not unrelated_orphan.exists()
        assert (
            recovered._connection.execute(
                "SELECT COUNT(*) FROM storage_leases"
            ).fetchone()[0]
            == 1
        )
    finally:
        recovered.close()


@pytest.mark.parametrize(
    "corruption",
    (
        "owner",
        "reference",
        "path",
        "file",
        "running",
        "marker",
        "canonical-options",
        "job-artifact",
    ),
)
def test_startup_preflight_rejects_corruption_before_any_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    patch_job_ids(monkeypatch, "7K3M9Q2W")
    first = Storage(
        tmp_path,
        limits(max_job_storage_bytes=4 * RESERVATION_QUANTUM),
        free_bytes=lambda _: 1 << 40,
    )
    job_upload = first.begin_job_upload(CREATED_AT)
    first.append_job_upload(job_upload, b"job-audio")
    job_ref = first.seal_job_upload(job_upload)
    first.publish_job(job_ref, queued_spec())
    sync = first.begin_upload("transcription")
    first.append(sync, b"sync-audio")
    strict_orphan = first.staging_dir / "JKMNPQRT.partial"
    strict_orphan.write_bytes(b"orphan")
    first.close()

    connection = sqlite3.connect(tmp_path / "botified-asr.sqlite3")
    try:
        if corruption == "owner":
            connection.execute(
                "UPDATE storage_leases SET owner_id = 'ABCDEFGH' WHERE id = ?",
                (job_ref.id,),
            )
        elif corruption == "reference":
            connection.execute(
                """
                UPDATE transcription_jobs SET input_lease_id = 'ABCDEFGH'
                WHERE id = ?
                """,
                (job_ref.id,),
            )
        elif corruption == "path":
            connection.execute(
                "UPDATE storage_leases SET controlled_path = ? WHERE id = ?",
                (
                    str(first.staging_dir / f"{job_ref.id}.partial"),
                    job_ref.id,
                ),
            )
        elif corruption == "file":
            job_ref.path.unlink()
        elif corruption == "running":
            connection.execute(
                "UPDATE transcription_jobs SET status = 'running' WHERE id = ?",
                (job_ref.id,),
            )
        elif corruption == "marker":
            connection.execute(
                """
                INSERT INTO shutdown_marker(singleton, generation, created_at)
                VALUES (1, 'generation-1', '2026-07-27T12:00:00.000Z')
                """
            )
        elif corruption == "canonical-options":
            connection.execute(
                """
                UPDATE transcription_jobs SET canonical_options_json = ?
                WHERE id = ?
                """,
                ('{"model":"sensevoice"}', job_ref.id),
            )
        else:
            artifact_path = first.artifact_dir / "ABCDEFGH.partial"
            artifact_path.write_bytes(b"artifact")
            connection.execute(
                """
                INSERT INTO storage_leases(
                    id, lease_type, resource_kind, owner_kind, owner_id,
                    phase, controlled_path, reserved_bytes, actual_bytes
                ) VALUES (
                    'ABCDEFGH', 'artifact', 'segment_jsonl', 'job', ?,
                    'writing', ?, 1, 1
                )
                """,
                (job_ref.id, str(artifact_path)),
            )
        connection.commit()
        database_before = tuple(
            tuple(connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall())
            for table in (
                "transcription_jobs",
                "storage_leases",
                "shutdown_marker",
            )
        )
    finally:
        connection.close()

    existing_paths = {
        path
        for path in (
            job_ref.path,
            sync.path,
            strict_orphan,
            first.artifact_dir / "ABCDEFGH.partial",
        )
        if path.exists()
    }
    file_contents_before = {
        path: path.read_bytes() for path in existing_paths if path.is_file()
    }
    with pytest.raises(StorageSchemaError):
        Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    assert all(path.exists() for path in existing_paths)
    assert {
        path: path.read_bytes() for path in existing_paths if path.is_file()
    } == file_contents_before
    connection = sqlite3.connect(tmp_path / "botified-asr.sqlite3")
    try:
        database_after = tuple(
            tuple(connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall())
            for table in (
                "transcription_jobs",
                "storage_leases",
                "shutdown_marker",
            )
        )
    finally:
        connection.close()
    assert database_after == database_before
