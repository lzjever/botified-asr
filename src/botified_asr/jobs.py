from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from botified_asr.canonical_options import parse_canonical_options_json
from botified_asr.contracts import DIRECT_MAX_SAMPLES, MAX_AUDIO_SAMPLES

_JOB_ID = re.compile(r"\A[0-9A-HJKMNP-TV-Z]{8}\Z")
_LOWERCASE_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_CROCKFORD_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPhase(str, Enum):
    RECEIVING = "receiving"
    VISIBLE = "visible"
    DELETING = "deleting"


class JobProgressOutcome(str, Enum):
    UPDATED = "updated"
    STALE = "stale"
    CANCEL_REQUESTED = "cancel_requested"


class JobSuccessOutcome(str, Enum):
    COMMITTED = "committed"
    STALE = "stale"
    CANCEL_REQUESTED = "cancel_requested"


def validate_job_id(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("job ID must be a string")
    if _JOB_ID.fullmatch(value) is None:
        raise ValueError("job ID is invalid")
    return value


def generate_job_id() -> str:
    return "".join(secrets.choice(_CROCKFORD_ALPHABET) for _ in range(8))


def generate_attempt_token() -> str:
    return secrets.token_urlsafe(32)


@dataclass(frozen=True, slots=True)
class QueuedJobSpec:
    canonical_options_json: str
    effective_max_audio_samples: int
    effective_direct_max_audio_samples: int
    processor_fingerprint: str

    def __post_init__(self) -> None:
        if type(self.canonical_options_json) is not str:
            raise TypeError("canonical job options must be a string")
        parse_canonical_options_json(self.canonical_options_json)
        for name in (
            "effective_max_audio_samples",
            "effective_direct_max_audio_samples",
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"job {name} must be an integer")
            if value <= 0:
                raise ValueError(f"job {name} must be positive")
        if self.effective_max_audio_samples > MAX_AUDIO_SAMPLES:
            raise ValueError("job effective max audio samples exceeds release limit")
        if self.effective_direct_max_audio_samples > min(
            self.effective_max_audio_samples,
            DIRECT_MAX_SAMPLES,
        ):
            raise ValueError("job effective direct max audio samples is invalid")
        if type(self.processor_fingerprint) is not str:
            raise TypeError("job processor_fingerprint must be a string")
        if _LOWERCASE_SHA256.fullmatch(self.processor_fingerprint) is None:
            raise ValueError("job processor_fingerprint is invalid")


@dataclass(frozen=True, slots=True)
class DurableJob:
    id: str
    phase: JobPhase
    status: JobStatus | None
    input_lease_id: str | None
    canonical_options_json: str | None
    selected_speaker_snapshot: bytes | None
    snapshot_sha256: str | None
    input_size_bytes: int | None
    effective_max_audio_samples: int | None
    effective_direct_max_audio_samples: int | None
    total_samples: int | None
    processed_samples: int
    request_fingerprint: str | None
    processor_fingerprint: str | None
    attempt_no: int
    attempt_token: str | None
    owner_generation: str | None
    crash_recoveries: int
    cancel_requested: bool
    result_lease_id: str | None
    error_code: str | None
    input_cleanup_pending: bool
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    def __post_init__(self) -> None:
        validate_job_id(self.id)
        _validate_local_values(self)
        _validate_state_shape(self)


def _validate_local_values(job: DurableJob) -> None:
    if type(job.phase) is not JobPhase:
        raise TypeError("job phase is invalid")
    if job.status is not None and type(job.status) is not JobStatus:
        raise TypeError("job status is invalid")

    for name in (
        "input_lease_id",
        "canonical_options_json",
        "attempt_token",
        "owner_generation",
        "result_lease_id",
        "error_code",
    ):
        _validate_optional_nonempty_string(getattr(job, name), name=name)
    if job.canonical_options_json is not None:
        parse_canonical_options_json(job.canonical_options_json)

    if (
        job.selected_speaker_snapshot is not None
        and type(job.selected_speaker_snapshot) is not bytes
    ):
        raise TypeError("selected speaker snapshot must be bytes or None")

    for name in (
        "snapshot_sha256",
        "request_fingerprint",
        "processor_fingerprint",
    ):
        _validate_optional_sha256(getattr(job, name), name=name)

    for name in (
        "input_size_bytes",
        "total_samples",
        "processed_samples",
        "attempt_no",
        "crash_recoveries",
    ):
        _validate_optional_nonnegative_integer(
            getattr(job, name),
            name=name,
            optional=name in {"input_size_bytes", "total_samples"},
        )
    for name in (
        "effective_max_audio_samples",
        "effective_direct_max_audio_samples",
    ):
        value = getattr(job, name)
        if value is not None and type(value) is not int:
            raise TypeError(f"job {name} must be an integer or None")
        if value is not None and value <= 0:
            raise ValueError(f"job {name} must be positive")
    if (
        job.effective_max_audio_samples is not None
        and job.effective_max_audio_samples > MAX_AUDIO_SAMPLES
    ):
        raise ValueError("job effective max audio samples exceeds release limit")
    if (
        job.effective_direct_max_audio_samples is not None
        and (
            job.effective_max_audio_samples is None
            or job.effective_direct_max_audio_samples
            > min(job.effective_max_audio_samples, DIRECT_MAX_SAMPLES)
        )
    ):
        raise ValueError("job effective direct max audio samples is invalid")
    if job.crash_recoveries > 1:
        raise ValueError("job crash recoveries must not exceed one")
    if job.crash_recoveries > job.attempt_no:
        raise ValueError("job crash recoveries exceed its attempts")

    for name in ("cancel_requested", "input_cleanup_pending"):
        if type(getattr(job, name)) is not bool:
            raise TypeError(f"job {name} must be a boolean")

    for name in ("created_at", "started_at", "finished_at"):
        _validate_utc_datetime(
            getattr(job, name),
            name=name,
            optional=name != "created_at",
        )
    if job.started_at is not None and job.started_at < job.created_at:
        raise ValueError("job timestamps are out of order")
    if job.finished_at is not None:
        earliest = job.started_at or job.created_at
        if job.finished_at < earliest:
            raise ValueError("job timestamps are out of order")

    if job.total_samples is not None and job.processed_samples > job.total_samples:
        raise ValueError("job processed samples exceed total samples")
    if (
        job.total_samples is not None
        and job.effective_max_audio_samples is not None
        and job.total_samples > job.effective_max_audio_samples
    ):
        raise ValueError("job total samples exceed effective maximum")
    if (
        job.effective_max_audio_samples is not None
        and job.processed_samples > job.effective_max_audio_samples
    ):
        raise ValueError("job processed samples exceed effective maximum")


def _validate_state_shape(job: DurableJob) -> None:
    if job.phase is JobPhase.RECEIVING:
        _validate_receiving_shape(job)
        return
    if job.phase is JobPhase.DELETING and job.status is None:
        _validate_receiving_shape(job)
        return
    if job.status is None:
        raise ValueError("visible jobs must have a status")
    if job.phase is JobPhase.DELETING and job.status in {
        JobStatus.QUEUED,
        JobStatus.RUNNING,
    }:
        raise ValueError("only terminal jobs may be deleting")

    _require_visible_metadata(job)
    if job.status is JobStatus.QUEUED:
        _validate_queued_shape(job)
    elif job.status is JobStatus.RUNNING:
        _validate_running_shape(job)
    else:
        _validate_terminal_shape(job)


def _validate_receiving_shape(job: DurableJob) -> None:
    if job.status is not None:
        raise ValueError("receiving jobs must not have a status")
    absent = (
        job.canonical_options_json,
        job.selected_speaker_snapshot,
        job.snapshot_sha256,
        job.input_size_bytes,
        job.effective_max_audio_samples,
        job.effective_direct_max_audio_samples,
        job.total_samples,
        job.request_fingerprint,
        job.processor_fingerprint,
        job.attempt_token,
        job.owner_generation,
        job.result_lease_id,
        job.error_code,
        job.started_at,
        job.finished_at,
    )
    if any(value is not None for value in absent):
        raise ValueError("receiving job metadata is inconsistent")
    if (
        job.input_lease_id is None
        or job.processed_samples != 0
        or job.attempt_no != 0
        or job.crash_recoveries != 0
        or job.cancel_requested
        or job.input_cleanup_pending
    ):
        raise ValueError("receiving job state is inconsistent")


def _require_visible_metadata(job: DurableJob) -> None:
    required = (
        job.canonical_options_json,
        job.selected_speaker_snapshot,
        job.snapshot_sha256,
        job.input_size_bytes,
        job.effective_max_audio_samples,
        job.effective_direct_max_audio_samples,
        job.request_fingerprint,
        job.processor_fingerprint,
    )
    if any(value is None for value in required):
        raise ValueError("visible job metadata is incomplete")


def _validate_queued_shape(job: DurableJob) -> None:
    if (
        job.input_lease_id is None
        or (job.attempt_no == 0 and job.total_samples is not None)
        or job.processed_samples != 0
        or job.attempt_token is not None
        or job.owner_generation is not None
        or job.cancel_requested
        or job.result_lease_id is not None
        or job.error_code is not None
        or job.input_cleanup_pending
        or job.started_at is not None
        or job.finished_at is not None
    ):
        raise ValueError("queued job state is inconsistent")


def _validate_running_shape(job: DurableJob) -> None:
    if (
        job.input_lease_id is None
        or job.attempt_no < 1
        or job.attempt_token is None
        or job.owner_generation is None
        or job.result_lease_id is not None
        or job.error_code is not None
        or job.input_cleanup_pending
        or job.started_at is None
        or job.finished_at is not None
    ):
        raise ValueError("running job state is inconsistent")


def _validate_terminal_shape(job: DurableJob) -> None:
    if (
        job.attempt_token is not None
        or job.owner_generation is not None
        or job.finished_at is None
    ):
        raise ValueError("terminal job state is inconsistent")
    if job.input_cleanup_pending != (job.input_lease_id is not None):
        raise ValueError("terminal job input cleanup state is inconsistent")

    if job.status is JobStatus.SUCCEEDED:
        if (
            job.attempt_no < 1
            or job.started_at is None
            or job.total_samples is None
            or job.processed_samples != job.total_samples
            or job.result_lease_id is None
            or job.error_code is not None
            or job.cancel_requested
        ):
            raise ValueError("succeeded job state is inconsistent")
    elif job.status is JobStatus.FAILED:
        if (
            job.attempt_no < 1
            or job.started_at is None
            or job.result_lease_id is not None
            or job.error_code is None
            or job.cancel_requested
        ):
            raise ValueError("failed job state is inconsistent")
    elif job.status is JobStatus.CANCELLED:
        if job.result_lease_id is not None or job.error_code is not None:
            raise ValueError("cancelled job state is inconsistent")
        if job.attempt_no == 0:
            if job.started_at is not None or job.cancel_requested:
                raise ValueError("queued cancellation state is inconsistent")
        elif job.started_at is None or not job.cancel_requested:
            raise ValueError("running cancellation state is inconsistent")


def _validate_optional_nonempty_string(value: object, *, name: str) -> None:
    if value is not None and not isinstance(value, str):
        raise TypeError(f"job {name} must be a string or None")
    if value == "":
        raise ValueError(f"job {name} must not be empty")


def _validate_optional_sha256(value: object, *, name: str) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise TypeError(f"job {name} must be a string or None")
    if _LOWERCASE_SHA256.fullmatch(value) is None:
        raise ValueError(f"job {name} is invalid")


def _validate_optional_nonnegative_integer(
    value: object,
    *,
    name: str,
    optional: bool,
) -> None:
    if value is None and optional:
        return
    if type(value) is not int:
        raise TypeError(f"job {name} must be an integer")
    if value < 0:
        raise ValueError(f"job {name} must be nonnegative")


def _validate_utc_datetime(
    value: object,
    *,
    name: str,
    optional: bool,
) -> None:
    if value is None and optional:
        return
    if type(value) is not datetime:
        raise TypeError(f"job {name} must be a datetime")
    if value.tzinfo is not timezone.utc:
        raise ValueError(f"job {name} must use canonical UTC")
