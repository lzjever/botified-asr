from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import threading

import pytest

import botified_asr.jobs as jobs
import botified_asr.speaker_snapshot as speaker_snapshot
import botified_asr.storage as storage_module
from botified_asr.config import LimitsConfig, RESERVATION_QUANTUM
from botified_asr.contracts import DIRECT_MAX_SAMPLES, MAX_AUDIO_SAMPLES
from botified_asr.speaker_profiles import (
    SpeakerEmbedding,
    SpeakerProfile,
)
from botified_asr.speakers import SpeakerEmbeddingPolicy
from botified_asr.storage import Storage, StorageAdmissionError, StorageSchemaError


CREATED_AT = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
PROCESSOR_FINGERPRINT = "3" * 64
CANONICAL_OPTIONS_JSON = (
    '{"chunking_strategy":null,"include":[],"known_speaker_ids":[],'
    '"language":"auto","model":"sensevoice","response_format":"json"}'
)
KNOWN_OPTIONS_JSON = (
    '{"chunking_strategy":"auto","include":[],"known_speaker_ids":'
    '["00000001","00000002"],"language":"auto",'
    '"model":"sensevoice-diarize","response_format":"diarized_json"}'
)
ONE_KNOWN_OPTIONS_JSON = (
    '{"chunking_strategy":"auto","include":[],"known_speaker_ids":'
    '["00000001"],"language":"auto","model":"sensevoice-diarize",'
    '"response_format":"diarized_json"}'
)
INPUT_SHA256 = "6ed8919ce20490a5e3ad8630a4fab69475297abd07db73918dd5f36fcfaeb11b"
EMPTY_SNAPSHOT = b'{"speakers":[],"version":1}'
EMPTY_SNAPSHOT_SHA256 = (
    "37e2de7a783aa3aa11e0b56dbf8faa5ac19217e3ae9c2e2ae228592823009e3f"
)
EMPTY_REQUEST_FINGERPRINT = (
    "5eb4b7227e952d72211424f2f546fc6d0d9c9b31e6da42ff9241dcc129e13334"
)
KNOWN_SNAPSHOT = (
    '{"speakers":['
    '{"embedding":"AACAPwAAAAA=","id":"00000001","name":"艾丽丝"},'
    '{"embedding":"AAAAAAAAgD8=","id":"00000002","name":"Bob"}'
    '],"version":1}'
).encode("utf-8")
KNOWN_SNAPSHOT_SHA256 = (
    "0e0dcec30d8878b3ed5ec43f3a391be54124b02b6a3d4b9ff6ff1fb28bcf2057"
)
KNOWN_REQUEST_FINGERPRINT = (
    "20de60a0e282e43c2ae962cfbba9d129268b5b3636e38989b770df94d7e21593"
)
OLD_ONE_SNAPSHOT = (
    '{"speakers":['
    '{"embedding":"AACAPwAAAAA=","id":"00000001","name":"艾丽丝"}'
    '],"version":1}'
).encode("utf-8")


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
        "effective_max_audio_samples": 32_000,
        "effective_direct_max_audio_samples": 16_000,
        "processor_fingerprint": "3" * 64,
    }
    values.update(overrides)
    return jobs.QueuedJobSpec(**values)


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


def speaker_embedding(axis: int) -> SpeakerEmbedding:
    raw = (
        b"\x00\x00\x80\x3f\x00\x00\x00\x00"
        if axis == 0
        else b"\x00\x00\x00\x00\x00\x00\x80\x3f"
    )
    return SpeakerEmbedding.from_bytes(raw, dimension=2)


def speaker_profile(
    profile_id: str,
    name: str,
    *,
    axis: int,
    compatible: bool = True,
) -> SpeakerProfile:
    policy = speaker_policy()
    return SpeakerProfile(
        id=profile_id,
        name=name,
        description=None,
        embedding=speaker_embedding(axis),
        embedding_model_id=policy.model_id,
        embedding_model_revision=(
            policy.model_revision if compatible else "2" * 40
        ),
        embedding_dimension=policy.embedding_dimension,
        embedding_policy_fingerprint=policy.fingerprint,
        sample_count=2,
        created_at=CREATED_AT,
        updated_at=CREATED_AT,
    )


def seal_job_input(storage: Storage) -> storage_module.JobInputRef:
    upload = storage.begin_job_upload(CREATED_AT)
    storage.append_job_upload(upload, b"audio")
    return storage.seal_job_upload(upload)


def assert_job_remains_receiving(
    storage: Storage,
    input_ref: storage_module.JobInputRef,
) -> None:
    row = storage._connection.execute(
        """
        SELECT phase, status, canonical_options_json,
               selected_speaker_snapshot, snapshot_sha256,
               input_size_bytes, effective_max_audio_samples,
               effective_direct_max_audio_samples, total_samples,
               request_fingerprint, processor_fingerprint
        FROM transcription_jobs WHERE id = ?
        """,
        (input_ref.id,),
    ).fetchone()
    assert tuple(row[:2]) == ("receiving", None)
    assert all(value is None for value in row[2:])
    assert storage.get_visible_job(input_ref.id) is None
    assert input_ref.path.is_file()
    assert (
        storage._connection.execute(
            "SELECT phase FROM storage_leases WHERE id = ?",
            (input_ref.id,),
        ).fetchone()[0]
        == "sealed"
    )


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
    assert tuple(field.name for field in fields(type(spec))) == (
        "canonical_options_json",
        "effective_max_audio_samples",
        "effective_direct_max_audio_samples",
        "processor_fingerprint",
    )
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
    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
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
    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    try:
        lease = storage.begin_job_upload(CREATED_AT)
        spec = queued_spec()
        with pytest.raises(TypeError):
            storage.publish_job(
                lease,
                spec,
                speaker_embedding_policy=speaker_policy(),
            )
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
        assert input_ref.content_sha256 == INPUT_SHA256
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

        published = storage.publish_job(
            input_ref,
            spec,
            speaker_embedding_policy=speaker_policy(),
        )
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
        assert published.selected_speaker_snapshot == EMPTY_SNAPSHOT
        assert published.snapshot_sha256 == EMPTY_SNAPSHOT_SHA256
        assert published.request_fingerprint == EMPTY_REQUEST_FINGERPRINT
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
            storage.publish_job(
                input_ref,
                spec,
                speaker_embedding_policy=speaker_policy(),
            )
    finally:
        storage.close()


def test_publish_reads_and_persists_a_sorted_known_speaker_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_job_ids(monkeypatch, "01234567")
    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    try:
        storage.create_speaker_profile(
            speaker_profile("00000002", "Bob", axis=1)
        )
        storage.create_speaker_profile(
            speaker_profile("00000001", "艾丽丝", axis=0)
        )
        input_ref = seal_job_input(storage)

        published = storage.publish_job(
            input_ref,
            queued_spec(canonical_options_json=KNOWN_OPTIONS_JSON),
            speaker_embedding_policy=speaker_policy(),
        )

        assert published.selected_speaker_snapshot == KNOWN_SNAPSHOT
        assert published.snapshot_sha256 == KNOWN_SNAPSHOT_SHA256
        assert published.request_fingerprint == KNOWN_REQUEST_FINGERPRINT
        assert published.processor_fingerprint == "3" * 64
    finally:
        storage.close()


def test_publish_final_row_decode_failure_rolls_back_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_job_ids(monkeypatch, "01234567")
    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    try:
        input_ref = seal_job_input(storage)

        def reject_final_row(_row: sqlite3.Row) -> object:
            raise StorageSchemaError("injected final job row decode failure")

        monkeypatch.setattr(
            storage_module,
            "_decode_transcription_job",
            reject_final_row,
        )

        with pytest.raises(
            StorageSchemaError,
            match="injected final job row decode failure",
        ):
            storage.publish_job(
                input_ref,
                queued_spec(),
                speaker_embedding_policy=speaker_policy(),
            )

        assert_job_remains_receiving(storage, input_ref)
    finally:
        storage.close()


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("content_sha256", None),
        ("content_sha256", 1),
        ("content_sha256", b"not-a-digest"),
        ("actual_bytes", True),
        ("actual_bytes", -1),
    ),
)
def test_publish_rejects_forged_exact_input_refs_before_database_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    invalid_value: object,
) -> None:
    patch_job_ids(monkeypatch, "01234567")
    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    try:
        input_ref = seal_job_input(storage)
        forged = replace(input_ref, **{field: invalid_value})
        statements: list[str] = []
        storage._connection.set_trace_callback(statements.append)

        with pytest.raises(
            RuntimeError,
            match="job upload handle is invalid",
        ):
            storage.publish_job(
                forged,
                queued_spec(),
                speaker_embedding_policy=speaker_policy(),
            )

        assert statements == []
        assert_job_remains_receiving(storage, input_ref)
    finally:
        storage._connection.set_trace_callback(None)
        storage.close()


@pytest.mark.parametrize("corrupt_digest", (None, "not-a-digest"))
def test_publish_reports_corrupt_sealed_lease_digest_as_schema_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corrupt_digest: str | None,
) -> None:
    patch_job_ids(monkeypatch, "01234567")
    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    try:
        input_ref = seal_job_input(storage)
        with storage._transaction():
            storage._connection.execute(
                """
                UPDATE storage_leases SET content_sha256 = ?
                WHERE id = ?
                """,
                (corrupt_digest, input_ref.id),
            )

        with pytest.raises(StorageSchemaError):
            storage.publish_job(
                input_ref,
                queued_spec(),
                speaker_embedding_policy=speaker_policy(),
            )

        assert_job_remains_receiving(storage, input_ref)
    finally:
        storage.close()


@pytest.mark.parametrize("failure", ("missing", "incompatible"))
def test_publish_snapshot_failure_rolls_back_all_job_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    patch_job_ids(monkeypatch, "01234567")
    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    try:
        storage.create_speaker_profile(
            speaker_profile("00000001", "艾丽丝", axis=0)
        )
        if failure == "incompatible":
            storage.create_speaker_profile(
                speaker_profile(
                    "00000002",
                    "Bob",
                    axis=1,
                    compatible=False,
                )
            )
        input_ref = seal_job_input(storage)
        error_type = (
            speaker_snapshot.SelectedSpeakerNotFoundError
            if failure == "missing"
            else speaker_snapshot.SelectedSpeakerIncompatibleError
        )

        with pytest.raises(error_type):
            storage.publish_job(
                input_ref,
                queued_spec(canonical_options_json=KNOWN_OPTIONS_JSON),
                speaker_embedding_policy=speaker_policy(),
            )

        assert_job_remains_receiving(storage, input_ref)
    finally:
        storage.close()


@pytest.mark.parametrize("mutation", ("update", "delete"))
def test_publish_snapshot_and_profile_mutation_are_transactionally_ordered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    patch_job_ids(monkeypatch, "01234567")
    publisher = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    mutator = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
    publisher.create_speaker_profile(
        speaker_profile("00000001", "艾丽丝", axis=0)
    )
    input_ref = seal_job_input(publisher)

    profile_selected = threading.Event()
    release_publish = threading.Event()
    mutation_attempted = threading.Event()
    published: list[object] = []
    mutated: list[object] = []
    original_decode = storage_module._decode_speaker_profile

    def pause_after_profile_select(row: sqlite3.Row) -> SpeakerProfile:
        profile = original_decode(row)
        if not profile_selected.is_set():
            profile_selected.set()
            if not release_publish.wait(timeout=5):
                raise AssertionError("publish transaction was not released")
        return profile

    monkeypatch.setattr(
        storage_module,
        "_decode_speaker_profile",
        pause_after_profile_select,
    )

    def publish() -> None:
        try:
            published.append(
                publisher.publish_job(
                    input_ref,
                    queued_spec(
                        canonical_options_json=ONE_KNOWN_OPTIONS_JSON,
                    ),
                    speaker_embedding_policy=speaker_policy(),
                )
            )
        except BaseException as error:
            published.append(error)
        finally:
            profile_selected.set()

    def trace_mutation(statement: str) -> None:
        if statement.strip().upper().startswith("BEGIN IMMEDIATE"):
            mutation_attempted.set()

    mutator._connection.set_trace_callback(trace_mutation)

    def mutate() -> None:
        try:
            with mutator._transaction():
                if mutation == "update":
                    mutator._connection.execute(
                        """
                        UPDATE speaker_profiles
                        SET name = 'Updated Alice',
                            name_key = 'updated alice'
                        WHERE id = '00000001'
                        """
                    )
                else:
                    mutator._connection.execute(
                        "DELETE FROM speaker_profiles "
                        "WHERE id = '00000001'"
                    )
            mutated.append(mutator.get_visible_job(input_ref.id))
        except BaseException as error:
            mutated.append(error)

    publish_thread = threading.Thread(target=publish)
    mutation_thread = threading.Thread(target=mutate)
    try:
        publish_thread.start()
        assert profile_selected.wait(timeout=2)

        mutation_thread.start()
        assert mutation_attempted.wait(timeout=2)
        assert mutated == []

        release_publish.set()
        publish_thread.join(timeout=5)
        mutation_thread.join(timeout=5)
        assert not publish_thread.is_alive()
        assert not mutation_thread.is_alive()
        assert len(published) == 1
        assert not isinstance(published[0], BaseException)
        assert len(mutated) == 1
        assert not isinstance(mutated[0], BaseException)

        queued = published[0]
        visible_after_mutation = mutated[0]
        assert queued.selected_speaker_snapshot == OLD_ONE_SNAPSHOT
        assert visible_after_mutation.status is jobs.JobStatus.QUEUED
        assert (
            visible_after_mutation.selected_speaker_snapshot
            == OLD_ONE_SNAPSHOT
        )
        if mutation == "update":
            assert (
                mutator.get_speaker_profile("00000001").name
                == "Updated Alice"
            )
        else:
            assert mutator.get_speaker_profile("00000001") is None
    finally:
        release_publish.set()
        if publish_thread.ident is not None:
            publish_thread.join(timeout=5)
        if mutation_thread.ident is not None:
            mutation_thread.join(timeout=5)
        mutator._connection.set_trace_callback(None)
        publisher.close()
        mutator.close()


@pytest.mark.parametrize("sealed", (False, True))
def test_abort_job_upload_is_idempotent_without_leaks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sealed: bool,
) -> None:
    patch_job_ids(monkeypatch, "7K3M9Q2W")
    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
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
        first = Storage(tmp_path, configured, current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
        second = Storage(tmp_path, configured, current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
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
        first = Storage(tmp_path, configured, current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
        second = Storage(tmp_path, configured, current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
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
                            storage.publish_job(
                                ref,
                                queued_spec(),
                                speaker_embedding_policy=speaker_policy(),
                            ),
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
    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
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
    failed = Storage(failed_dir, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
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
    storage = Storage(published_dir, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
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

        published = storage.publish_job(
            input_ref,
            queued_spec(),
            speaker_embedding_policy=speaker_policy(),
        )

        assert published.status is jobs.JobStatus.QUEUED
        assert published.input_size_bytes == input_ref.actual_bytes
        assert published.request_fingerprint == EMPTY_REQUEST_FINGERPRINT
    finally:
        storage.close()


def test_seal_compensation_fsync_failure_remains_restart_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_job_ids(monkeypatch, "01234567")
    storage = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
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

    recovered = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
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
    first = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
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
            current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40,
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
            current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40,
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
            current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40,
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
    first = Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
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
        Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)

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
        current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40,
    )
    job_upload = first.begin_job_upload(CREATED_AT)
    first.append_job_upload(job_upload, b"job-audio")
    job_ref = first.seal_job_upload(job_upload)
    queued = first.publish_job(
        job_ref,
        queued_spec(),
        speaker_embedding_policy=speaker_policy(),
    )

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
        current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40,
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
        current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40,
    )
    job_upload = first.begin_job_upload(CREATED_AT)
    first.append_job_upload(job_upload, b"job-audio")
    job_ref = first.seal_job_upload(job_upload)
    first.publish_job(
        job_ref,
        queued_spec(),
        speaker_embedding_policy=speaker_policy(),
    )
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
        Storage(tmp_path, limits(), current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40)
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
