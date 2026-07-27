from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass
from datetime import datetime, timedelta, timezone
from importlib import import_module
from inspect import Parameter, signature
from types import ModuleType

import pytest


CREATED_AT = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
STARTED_AT = CREATED_AT + timedelta(seconds=1)
FINISHED_AT = STARTED_AT + timedelta(seconds=1)
JOB_FIELDS = (
    "id",
    "phase",
    "status",
    "input_lease_id",
    "canonical_options_json",
    "selected_speaker_snapshot",
    "snapshot_sha256",
    "input_size_bytes",
    "total_samples",
    "processed_samples",
    "request_fingerprint",
    "processor_fingerprint",
    "attempt_no",
    "attempt_token",
    "owner_generation",
    "crash_recoveries",
    "cancel_requested",
    "result_lease_id",
    "error_code",
    "input_cleanup_pending",
    "created_at",
    "started_at",
    "finished_at",
)


def _jobs() -> ModuleType:
    return import_module("botified_asr.jobs")


def _queued_values(jobs: ModuleType) -> dict[str, object]:
    return {
        "id": "7K3M9Q2W",
        "phase": jobs.JobPhase.VISIBLE,
        "status": jobs.JobStatus.QUEUED,
        "input_lease_id": "a" * 32,
        "canonical_options_json": '{"model":"sensevoice"}',
        "selected_speaker_snapshot": b'{"speakers":[]}',
        "snapshot_sha256": "1" * 64,
        "input_size_bytes": 4,
        "total_samples": 32_000,
        "processed_samples": 0,
        "request_fingerprint": "2" * 64,
        "processor_fingerprint": "3" * 64,
        "attempt_no": 0,
        "attempt_token": None,
        "owner_generation": None,
        "crash_recoveries": 0,
        "cancel_requested": False,
        "result_lease_id": None,
        "error_code": None,
        "input_cleanup_pending": False,
        "created_at": CREATED_AT,
        "started_at": None,
        "finished_at": None,
    }


def _job(jobs: ModuleType, **changes: object) -> object:
    values = _queued_values(jobs)
    values.update(changes)
    return jobs.DurableJob(**values)


def _receiving_changes(jobs: ModuleType) -> dict[str, object]:
    return {
        "phase": jobs.JobPhase.RECEIVING,
        "status": None,
        "canonical_options_json": None,
        "selected_speaker_snapshot": None,
        "snapshot_sha256": None,
        "input_size_bytes": None,
        "total_samples": None,
        "request_fingerprint": None,
        "processor_fingerprint": None,
    }


def test_job_domain_types_are_exact_frozen_and_slotted() -> None:
    jobs = _jobs()

    assert tuple(item.value for item in jobs.JobStatus) == (
        "queued",
        "running",
        "succeeded",
        "failed",
        "cancelled",
    )
    assert tuple(item.value for item in jobs.JobPhase) == (
        "receiving",
        "visible",
        "deleting",
    )
    assert is_dataclass(jobs.DurableJob)
    assert tuple(item.name for item in fields(jobs.DurableJob)) == JOB_FIELDS
    assert jobs.DurableJob.__slots__ == JOB_FIELDS
    parameters = signature(jobs.DurableJob).parameters
    assert tuple(parameters) == JOB_FIELDS
    assert all(
        parameter.kind is Parameter.POSITIONAL_OR_KEYWORD
        and parameter.default is Parameter.empty
        for parameter in parameters.values()
    )

    record = _job(jobs)
    with pytest.raises(FrozenInstanceError):
        record.status = jobs.JobStatus.RUNNING


def test_job_id_is_exactly_eight_uppercase_crockford_characters() -> None:
    jobs = _jobs()

    for valid in ("01234567", "ABCDEFGH", "JKMNPQRT", "VWXYZ234"):
        assert jobs.validate_job_id(valid) == valid

    for invalid in (
        "",
        "1234567",
        "123456789",
        "abcdefg2",
        "ABCDEFGI",
        "ABCDEFGL",
        "ABCDEFGO",
        "ABCDEFGU",
        "ABC-1234",
    ):
        with pytest.raises(ValueError):
            jobs.validate_job_id(invalid)
    with pytest.raises(TypeError):
        jobs.validate_job_id(b"01234567")


def test_transcription_job_rejects_noncanonical_local_values() -> None:
    jobs = _jobs()
    invalid_changes = (
        {"id": "ABCDEFGI"},
        {"phase": "visible"},
        {"status": "queued"},
        {"input_lease_id": ""},
        {"canonical_options_json": {"model": "sensevoice"}},
        {"selected_speaker_snapshot": '{"speakers":[]}'},
        {"snapshot_sha256": "A" * 64},
        {"snapshot_sha256": "1" * 63},
        {"input_size_bytes": -1},
        {"total_samples": -1},
        {"processed_samples": True},
        {"request_fingerprint": "g" * 64},
        {"processor_fingerprint": b"3" * 64},
        {"attempt_no": -1},
        {"crash_recoveries": -1},
        {"cancel_requested": 0},
        {"input_cleanup_pending": 0},
        {"created_at": "2026-07-27T12:00:00Z"},
        {"created_at": datetime(2026, 7, 27, 12, 0)},
        {
            "created_at": datetime(
                2026,
                7,
                27,
                12,
                0,
                tzinfo=timezone(timedelta(hours=1)),
            )
        },
    )

    for changes in invalid_changes:
        with pytest.raises((TypeError, ValueError)):
            _job(jobs, **changes)


def test_transcription_job_accepts_each_exact_state_shape() -> None:
    jobs = _jobs()
    valid_changes = (
        _receiving_changes(jobs),
        {},
        {"attempt_no": 1},
        {"attempt_no": 1, "crash_recoveries": 1},
        {
            "status": jobs.JobStatus.RUNNING,
            "processed_samples": 16_000,
            "attempt_no": 1,
            "attempt_token": "attempt-1",
            "owner_generation": "generation-1",
            "started_at": STARTED_AT,
        },
        {
            "status": jobs.JobStatus.RUNNING,
            "processed_samples": 16_000,
            "attempt_no": 1,
            "attempt_token": "attempt-1",
            "owner_generation": "generation-1",
            "cancel_requested": True,
            "started_at": STARTED_AT,
        },
        {
            "status": jobs.JobStatus.SUCCEEDED,
            "input_lease_id": None,
            "processed_samples": 32_000,
            "attempt_no": 1,
            "result_lease_id": "b" * 32,
            "started_at": STARTED_AT,
            "finished_at": FINISHED_AT,
        },
        {
            "status": jobs.JobStatus.SUCCEEDED,
            "processed_samples": 32_000,
            "attempt_no": 1,
            "result_lease_id": "b" * 32,
            "input_cleanup_pending": True,
            "started_at": STARTED_AT,
            "finished_at": FINISHED_AT,
        },
        {
            "status": jobs.JobStatus.FAILED,
            "input_lease_id": None,
            "attempt_no": 1,
            "error_code": "invalid_audio",
            "started_at": STARTED_AT,
            "finished_at": FINISHED_AT,
        },
        {
            "status": jobs.JobStatus.FAILED,
            "attempt_no": 1,
            "error_code": "invalid_audio",
            "input_cleanup_pending": True,
            "started_at": STARTED_AT,
            "finished_at": FINISHED_AT,
        },
        {
            "status": jobs.JobStatus.CANCELLED,
            "input_lease_id": None,
            "finished_at": FINISHED_AT,
        },
        {
            "status": jobs.JobStatus.CANCELLED,
            "input_lease_id": None,
            "attempt_no": 1,
            "cancel_requested": True,
            "started_at": STARTED_AT,
            "finished_at": FINISHED_AT,
        },
        {
            "status": jobs.JobStatus.CANCELLED,
            "attempt_no": 1,
            "cancel_requested": True,
            "input_cleanup_pending": True,
            "started_at": STARTED_AT,
            "finished_at": FINISHED_AT,
        },
        {
            "phase": jobs.JobPhase.DELETING,
            "status": None,
            **{
                key: value
                for key, value in _receiving_changes(jobs).items()
                if key not in {"phase", "status"}
            },
        },
        {
            "phase": jobs.JobPhase.DELETING,
            "status": jobs.JobStatus.SUCCEEDED,
            "input_lease_id": None,
            "processed_samples": 32_000,
            "attempt_no": 1,
            "result_lease_id": "b" * 32,
            "started_at": STARTED_AT,
            "finished_at": FINISHED_AT,
        },
    )

    records = tuple(_job(jobs, **changes) for changes in valid_changes)

    assert records[0].phase is jobs.JobPhase.RECEIVING
    assert all(
        record.phase is jobs.JobPhase.VISIBLE for record in records[1:-2]
    )
    assert all(
        record.phase is jobs.JobPhase.DELETING for record in records[-2:]
    )


def test_transcription_job_rejects_cross_field_state_drift() -> None:
    jobs = _jobs()
    terminal_changes = (
        {
            "status": jobs.JobStatus.SUCCEEDED,
            "input_lease_id": None,
            "processed_samples": 32_000,
            "attempt_no": 1,
            "result_lease_id": "b" * 32,
            "started_at": STARTED_AT,
            "finished_at": FINISHED_AT,
        },
        {
            "status": jobs.JobStatus.FAILED,
            "input_lease_id": None,
            "attempt_no": 1,
            "error_code": "invalid_audio",
            "started_at": STARTED_AT,
            "finished_at": FINISHED_AT,
        },
        {
            "status": jobs.JobStatus.CANCELLED,
            "input_lease_id": None,
            "attempt_no": 1,
            "cancel_requested": True,
            "started_at": STARTED_AT,
            "finished_at": FINISHED_AT,
        },
    )
    invalid_changes = (
        {"status": None},
        {**_receiving_changes(jobs), "status": jobs.JobStatus.QUEUED},
        {"canonical_options_json": None},
        {
            "status": jobs.JobStatus.RUNNING,
            "attempt_no": 1,
            "owner_generation": "generation-1",
            "started_at": STARTED_AT,
        },
        {
            "status": jobs.JobStatus.RUNNING,
            "attempt_no": 1,
            "attempt_token": "attempt-1",
            "started_at": STARTED_AT,
        },
        {
            "status": jobs.JobStatus.SUCCEEDED,
            "input_lease_id": None,
            "attempt_no": 1,
            "started_at": STARTED_AT,
            "finished_at": FINISHED_AT,
        },
        {
            "status": jobs.JobStatus.FAILED,
            "input_lease_id": None,
            "attempt_no": 1,
            "started_at": STARTED_AT,
            "finished_at": FINISHED_AT,
        },
        {
            "status": jobs.JobStatus.CANCELLED,
            "input_lease_id": None,
            "result_lease_id": "b" * 32,
            "finished_at": FINISHED_AT,
        },
        {"processed_samples": 32_001},
        {"crash_recoveries": 2},
        {
            "phase": jobs.JobPhase.DELETING,
            "status": jobs.JobStatus.RUNNING,
            "attempt_no": 1,
            "attempt_token": "attempt-1",
            "owner_generation": "generation-1",
            "started_at": STARTED_AT,
        },
        {
            "status": jobs.JobStatus.FAILED,
            "input_lease_id": None,
            "attempt_no": 1,
            "error_code": "invalid_audio",
            "started_at": STARTED_AT,
            "finished_at": CREATED_AT,
        },
        *(
            {**terminal, field: value}
            for terminal in terminal_changes
            for field, value in (
                ("attempt_token", "attempt-1"),
                ("owner_generation", "generation-1"),
            )
        ),
    )

    for changes in invalid_changes:
        with pytest.raises((TypeError, ValueError)):
            _job(jobs, **changes)
