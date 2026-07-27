from __future__ import annotations

import io
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
    CanonicalJsonlReader,
    ResultEnvelopeManifest,
    finalize_result_envelope,
)
from botified_asr.speaker_matching import SpeakerLabelMapping
from botified_asr.storage import Storage


CREATED_AT = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
STARTED_AT = CREATED_AT + timedelta(minutes=1)
FINISHED_AT = STARTED_AT + timedelta(minutes=1)
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
            selected_speaker_snapshot=b'{"speakers":[]}',
            snapshot_sha256="1" * 64,
            total_samples=32_000,
            request_fingerprint="2" * 64,
            processor_fingerprint="3" * 64,
        ),
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
        running.total_samples,
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


def test_job_result_writer_commits_exact_success_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_ids(monkeypatch)
    storage = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    try:
        running = queue_and_claim(storage)
        assert running.attempt_token is not None
        assert (
            storage.update_job_progress(
                running.id,
                running.attempt_token,
                running.total_samples,
            )
            is jobs.JobProgressOutcome.UPDATED
        )

        result_ref = seal_result(storage, running)

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
        assert succeeded.input_lease_id == running.input_lease_id
        assert succeeded.input_cleanup_pending
        assert succeeded.finished_at == FINISHED_AT
        assert storage.resolve_artifact(result_ref).is_file()
        assert (storage.staging_dir / f"{running.id}.ready").is_file()
        assert storage.total_reserved_bytes() == (
            running.input_size_bytes + result_ref.actual_bytes
        )
    finally:
        storage.close()


def test_job_result_begin_is_token_fenced_and_unique_across_connections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_ids(monkeypatch)
    first = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    second = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    try:
        running = queue_and_claim(first)
        assert running.attempt_token is not None

        with pytest.raises(RuntimeError):
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
    finally:
        second.close()
        first.close()


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
    storage = Storage(tmp_path, limits(), free_bytes=lambda _: 1 << 40)
    try:
        running = queue_and_claim(storage)
        assert running.attempt_token is not None
        if losing_state != "incomplete":
            assert (
                storage.update_job_progress(
                    running.id,
                    running.attempt_token,
                    running.total_samples,
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
