from __future__ import annotations

import json
import threading
import time
import wave
from collections.abc import Callable
from datetime import datetime, timezone
from inspect import Parameter, signature
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from botified_asr import composition as composition_module
from botified_asr import inference as inference_module
from botified_asr import jobs
from botified_asr import pipeline as pipeline_module
from botified_asr import speaker_profiles, speaker_snapshot, speakers
from botified_asr.audio import AudioError, Cancellation, FfmpegAudioFrontend
from botified_asr.canonical_options import serialize_canonical_options
from botified_asr.config import LimitsConfig, RESERVATION_QUANTUM
from botified_asr.contracts import MAX_AUDIO_SAMPLES, CanonicalOptions
from botified_asr.pipeline import (
    AsrResult,
    NormalizingAsrAdapter,
    PipelineError,
    Processor,
    RichAnnotations,
    SegmentRecord,
)
from botified_asr.result_artifact import Projection
from botified_asr.speaker_matching import (
    KnownSpeakerMatch,
    SpeakerLabelMapping,
    SpeakerLabelResolution,
)
from botified_asr.speaker_snapshot import SelectedSpeakerSnapshot
from botified_asr.storage import (
    Storage,
    StorageAdmissionError,
    StorageSchemaError,
)


MODEL_ID = "funasr/campplus"
MODEL_REVISION = "1" * 40
PROCESSOR_FINGERPRINT = "3" * 64
JOB_CREATED_AT = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
JOB_STARTED_AT = datetime(2026, 7, 27, 12, 1, tzinfo=timezone.utc)
JOB_FINISHED_AT = datetime(2026, 7, 27, 12, 2, tzinfo=timezone.utc)


def _speaker_embedding_policy() -> speakers.SpeakerEmbeddingPolicy:
    return speakers.SpeakerEmbeddingPolicy(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        embedding_dimension=2,
        sample_rate=16_000,
        downmix_policy_version="ffmpeg-first-audio-stream-ac1-v1",
        window_samples=24_000,
        window_shift_samples=12_000,
        padding_policy_version="right-zero-pad-v1",
        normalization_policy_version="int16-div-32768-l2-v1",
        enrollment_aggregation_policy_version=("sample-centroid-equal-average-v1"),
    )


def _profile(
    profile_id: str,
    name: str,
    *,
    compatible: bool = True,
) -> speaker_profiles.SpeakerProfile:
    policy = _speaker_embedding_policy()
    vector = np.array([1.0, 0.0], dtype=np.float32)
    embedding = speaker_profiles.SpeakerEmbedding.from_numpy(
        vector,
        dimension=2,
    )
    created_at = datetime(2026, 7, 27, tzinfo=timezone.utc)
    return speaker_profiles.SpeakerProfile(
        id=profile_id,
        name=name,
        description=None,
        embedding=embedding,
        embedding_model_id=MODEL_ID,
        embedding_model_revision=MODEL_REVISION,
        embedding_dimension=2,
        embedding_policy_fingerprint=(policy.fingerprint if compatible else "0" * 64),
        sample_count=2,
        created_at=created_at,
        updated_at=created_at,
    )


def _options(
    response_format: str = "json",
    *,
    include: tuple[str, ...] = (),
    known_speaker_ids: tuple[str, ...] = (),
) -> CanonicalOptions:
    diarized = bool(known_speaker_ids)
    return CanonicalOptions(
        model="sensevoice-diarize" if diarized else "sensevoice",
        language="auto",
        response_format="diarized_json" if diarized else response_format,
        chunking_strategy="auto" if diarized else None,
        include=include,
        known_speaker_ids=known_speaker_ids,
    )


def _storage(
    tmp_path: Path,
    *,
    max_audio_duration_secs: int = 43_200,
    direct_max_audio_duration_secs: int = 30,
    sync_max_audio_duration_secs: int = 3_600,
) -> Storage:
    return Storage(
        tmp_path / "storage",
        LimitsConfig(
            max_upload_bytes=RESERVATION_QUANTUM,
            max_audio_duration_secs=max_audio_duration_secs,
            direct_max_audio_duration_secs=direct_max_audio_duration_secs,
            sync_max_upload_bytes=RESERVATION_QUANTUM,
            sync_max_audio_duration_secs=sync_max_audio_duration_secs,
            max_job_storage_bytes=2 * RESERVATION_QUANTUM,
            min_filesystem_free_bytes=1,
        ),
        current_processor_fingerprint=PROCESSOR_FINGERPRINT, free_bytes=lambda _: 1 << 40,
    )


def _assert_no_artifact(storage: Storage) -> None:
    assert storage.total_reserved_bytes() == 0
    assert not list(storage.artifact_dir.iterdir())
    assert (
        storage._connection.execute(
            """
            SELECT COUNT(*) FROM storage_leases
            WHERE lease_type = 'artifact'
            """
        ).fetchone()[0]
        == 0
    )


def _queue_and_claim_job(
    storage: Storage,
    *,
    options: CanonicalOptions | None = None,
    effective_max_audio_samples: int = 32_000,
    effective_direct_max_audio_samples: int = 16_000,
) -> jobs.DurableJob:
    canonical_options = _options() if options is None else options
    upload = storage.begin_job_upload(JOB_CREATED_AT)
    storage.append_job_upload(upload, b"durable audio")
    input_ref = storage.seal_job_upload(upload)
    storage.publish_job(
        input_ref,
        jobs.QueuedJobSpec(
            canonical_options_json=serialize_canonical_options(
                canonical_options
            ),
            effective_max_audio_samples=effective_max_audio_samples,
            effective_direct_max_audio_samples=(
                effective_direct_max_audio_samples
            ),
            processor_fingerprint=PROCESSOR_FINGERPRINT,
        ),
        speaker_embedding_policy=_speaker_embedding_policy(),
    )
    running = storage.claim_next_job("generation-1", JOB_STARTED_AT)
    assert running is not None
    assert running.attempt_token is not None
    return running


def _artifact_kinds(storage: Storage, job_id: str) -> tuple[str, ...]:
    return tuple(
        row[0]
        for row in storage._connection.execute(
            """
            SELECT resource_kind FROM storage_leases
            WHERE lease_type = 'artifact'
              AND owner_kind = 'job' AND owner_id = ?
            ORDER BY resource_kind
            """,
            (job_id,),
        )
    )


def _execute_attempt(
    storage: Storage,
    processor: object,
    running: jobs.DurableJob,
    cancellation: Cancellation | None = None,
) -> Cancellation:
    current_cancellation = cancellation or Cancellation()
    composition_module.execute_claimed_job_attempt(
        storage,
        processor,  # type: ignore[arg-type]
        running,
        current_cancellation,
        speaker_embedding_policy=_speaker_embedding_policy(),
        now=lambda: JOB_FINISHED_AT,
    )
    return current_cancellation


def _request_job_cancel(storage: Storage, job_id: str) -> None:
    storage._connection.execute(
        """
        UPDATE transcription_jobs SET cancel_requested = 1
        WHERE id = ?
        """,
        (job_id,),
    )


def _assert_clean_terminal(
    storage: Storage,
    job_id: str,
    status: jobs.JobStatus,
) -> jobs.DurableJob:
    terminal = storage.get_visible_job(job_id)
    assert terminal is not None
    assert terminal.status is status
    assert terminal.input_lease_id is None
    assert _artifact_kinds(storage, job_id) == ()
    assert storage.total_reserved_bytes() == 0
    return terminal


class EmittingProcessor:
    def __init__(
        self,
        *,
        behavior: str = "success",
        speaker_mapping: SpeakerLabelMapping | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.behavior = behavior
        self.speaker_mapping = speaker_mapping or SpeakerLabelMapping(())
        self.events = events
        self.calls = 0
        self.selected_snapshots: list[SelectedSpeakerSnapshot] = []
        self.effective_caps: list[tuple[int, int]] = []

    def process(
        self,
        _input_path: Path,
        _options: CanonicalOptions,
        _cancellation: Cancellation,
        progress: Any,
        sink: Any,
        *,
        selected_speaker_snapshot: SelectedSpeakerSnapshot,
        effective_max_audio_samples: int,
        effective_direct_max_audio_samples: int,
    ) -> object:
        self.calls += 1
        self.selected_snapshots.append(selected_speaker_snapshot)
        self.effective_caps.append(
            (
                effective_max_audio_samples,
                effective_direct_max_audio_samples,
            )
        )
        if self.events is not None:
            self.events.append("processor")
        if self.behavior == "480001":
            progress.update(processed_samples=480_001, total_samples=None)
            raise PipelineError(
                "long_audio_requires_vad",
                "chunking_strategy=auto is required",
            )
        if self.behavior == "model":
            raise RuntimeError("model failed")
        if self.behavior == "invalid_progress":
            progress.update(processed_samples=True, total_samples=None)
            raise AssertionError("invalid progress was accepted")
        if self.behavior not in {"zero", "missing_progress"}:
            sink.append(
                SegmentRecord(
                    0,
                    0,
                    16_000,
                    "hello",
                    "en",
                    RichAnnotations("happy", "speech"),
                )
            )
            progress.update(
                processed_samples=16_000,
                total_samples=None,
            )
        elif self.behavior == "zero":
            progress.update(processed_samples=0, total_samples=None)
        if self.behavior not in {"missing_progress", "missing_eof"}:
            total_samples = 0 if self.behavior == "zero" else 16_000
            progress.update(
                processed_samples=total_samples,
                total_samples=total_samples,
            )
        ref = sink.finalize()
        if self.behavior == "bare_ref":
            return ref
        if self.behavior == "invalid_result":
            return object()
        if self.behavior == "nonexact_result":
            result_type = type(
                "NonExactProcessorResult",
                (pipeline_module.ProcessorResult,),
                {"__slots__": ()},
            )
            return result_type(ref, self.speaker_mapping)
        if self.behavior == "wrong_ref":
            return pipeline_module.ProcessorResult(
                object(),
                self.speaker_mapping,
            )
        if self.behavior == "invalid_mapping":
            return pipeline_module.ProcessorResult(
                ref,
                object(),
            )
        if self.behavior == "missing_progress":
            return pipeline_module.ProcessorResult(ref, self.speaker_mapping)
        return pipeline_module.ProcessorResult(ref, self.speaker_mapping)


class ClaimedJobProcessor:
    def __init__(
        self,
        *,
        error: BaseException | None = None,
        before_progress: Callable[[], None] | None = None,
        after_progress: Callable[[int, int | None], None] | None = None,
    ) -> None:
        self.error = error
        self.before_progress = before_progress
        self.after_progress = after_progress
        self.calls = 0
        self.input_paths: list[Path] = []
        self.input_payloads: list[bytes] = []
        self.options: list[CanonicalOptions] = []
        self.cancellations: list[Cancellation] = []
        self.selected_snapshots: list[SelectedSpeakerSnapshot] = []
        self.effective_caps: list[tuple[int, int]] = []

    def process(
        self,
        input_path: Path,
        options: CanonicalOptions,
        cancellation: Cancellation,
        progress: Any,
        sink: Any,
        *,
        selected_speaker_snapshot: SelectedSpeakerSnapshot,
        effective_max_audio_samples: int,
        effective_direct_max_audio_samples: int,
    ) -> object:
        self.calls += 1
        self.input_paths.append(input_path)
        self.input_payloads.append(input_path.read_bytes())
        self.options.append(options)
        self.cancellations.append(cancellation)
        self.selected_snapshots.append(selected_speaker_snapshot)
        self.effective_caps.append(
            (
                effective_max_audio_samples,
                effective_direct_max_audio_samples,
            )
        )
        sink.append(
            SegmentRecord(
                0,
                0,
                16_000,
                "hello",
                "en",
                RichAnnotations("happy", "speech"),
                anonymous_speaker=(
                    "A"
                    if selected_speaker_snapshot.speakers
                    else None
                ),
            )
        )
        if self.before_progress is not None:
            self.before_progress()
        if self.error is not None:
            raise self.error
        for processed_samples, total_samples in (
            (8_000, None),
            (16_000, 16_000),
        ):
            progress.update(
                processed_samples=processed_samples,
                total_samples=total_samples,
            )
            if self.after_progress is not None:
                self.after_progress(
                    processed_samples,
                    total_samples,
                )
        ref = sink.finalize()
        speaker_mapping = SpeakerLabelMapping(())
        if selected_speaker_snapshot.speakers:
            selected = selected_speaker_snapshot.speakers[0]
            speaker_mapping = SpeakerLabelMapping(
                (
                    SpeakerLabelResolution(
                        "A",
                        KnownSpeakerMatch(
                            selected.id,
                            selected.name,
                            1.0,
                        ),
                    ),
                )
            )
        return pipeline_module.ProcessorResult(
            ref,
            speaker_mapping,
        )


class PoolRecordingProcessor:
    def __init__(
        self,
        name: str,
        lane: inference_module.SerialInferenceLane,
        selected: list[str],
        outcomes: list[object] | None = None,
    ) -> None:
        self.name = name
        self.lane = lane
        self.selected = selected
        self.outcomes = [] if outcomes is None else outcomes
        self.invocations: list[tuple[str, str]] = []

    def process(
        self,
        _input_path: Path,
        _options: CanonicalOptions,
        _cancellation: Cancellation,
        _progress: object,
        _sink: object,
        *,
        selected_speaker_snapshot: SelectedSpeakerSnapshot,
        effective_max_audio_samples: int,
        effective_direct_max_audio_samples: int,
    ) -> object:
        assert selected_speaker_snapshot == SelectedSpeakerSnapshot(())
        assert effective_max_audio_samples == 32_000
        assert effective_direct_max_audio_samples == 16_000
        self.selected.append(self.name)
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, BaseException):
                return self.lane.invoke(lambda: (_ for _ in ()).throw(outcome))
        for role in ("vad", "asr", "speaker"):
            self.lane.invoke(
                lambda role=role: self.invocations.append((self.name, role))
            )
        return self.name


def _call_pooled_processor(
    processor: object,
    *,
    cancellation: Cancellation | None = None,
    input_path: Path = Path("input.ready"),
) -> object:
    return processor.process(  # type: ignore[attr-defined]
        input_path,
        _options(),
        Cancellation() if cancellation is None else cancellation,
        object(),
        object(),
        selected_speaker_snapshot=SelectedSpeakerSnapshot(()),
        effective_max_audio_samples=32_000,
        effective_direct_max_audio_samples=16_000,
    )


def _wait_for_pool_lane_waiters(
    lane: inference_module.SerialInferenceLane,
    category: str,
    count: int,
) -> None:
    deadline = time.monotonic() + 2
    waiters = getattr(lane, f"_{category}_waiters")
    while len(waiters) != count:
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for inference waiter")
        time.sleep(0.001)


def test_processor_pool_sync_and_async_share_round_robin_and_request_affinity() -> (
    None
):
    selected: list[str] = []
    lanes = (
        inference_module.SerialInferenceLane(),
        inference_module.SerialInferenceLane(),
    )
    processors = tuple(
        PoolRecordingProcessor(f"lane-{index}", lane, selected)
        for index, lane in enumerate(lanes)
    )
    pool = composition_module.TranscriptionProcessorPool(processors)

    assert _call_pooled_processor(pool.sync_processor) == "lane-0"
    assert _call_pooled_processor(pool.async_processor) == "lane-1"
    assert _call_pooled_processor(pool.async_processor) == "lane-0"
    assert _call_pooled_processor(pool.sync_processor) == "lane-1"

    assert selected == ["lane-0", "lane-1", "lane-0", "lane-1"]
    for processor in processors:
        assert processor.invocations == [
            (processor.name, "vad"),
            (processor.name, "asr"),
            (processor.name, "speaker"),
        ] * 2


def test_processor_pool_routes_repeated_sync_to_idle_lane_not_busy_round_robin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        inference_module,
        "SYNC_INFERENCE_WAIT_SECONDS",
        0.05,
    )
    lanes = (
        inference_module.SerialInferenceLane(),
        inference_module.SerialInferenceLane(),
    )
    busy_entered = threading.Event()
    release_busy = threading.Event()
    selected: list[int] = []

    class UnitProcessor:
        def __init__(
            self,
            index: int,
            lane: inference_module.SerialInferenceLane,
        ) -> None:
            self._index = index
            self._lane = lane

        def process(self, *_args: object, **_kwargs: object) -> int:
            selected.append(self._index)
            if self._index == 0:
                return self._lane.invoke(
                    lambda: (
                        busy_entered.set(),
                        release_busy.wait(timeout=2),
                        self._index,
                    )[-1]
                )
            return self._lane.invoke(lambda: self._index)

    pool = composition_module.TranscriptionProcessorPool(
        tuple(
            UnitProcessor(index, lane)
            for index, lane in enumerate(lanes)
        )
    )
    async_errors: list[BaseException] = []

    def run_long_async() -> None:
        try:
            _call_pooled_processor(pool.async_processor)
        except BaseException as error:
            async_errors.append(error)

    long_async = threading.Thread(target=run_long_async, daemon=True)
    long_async.start()
    assert busy_entered.wait(timeout=1)
    try:
        assert _call_pooled_processor(pool.sync_processor) == 1
        assert _call_pooled_processor(pool.sync_processor) == 1
    finally:
        release_busy.set()
        long_async.join(timeout=1)

    assert not long_async.is_alive()
    assert selected == [0, 1, 1]
    assert async_errors == []


@pytest.mark.parametrize(
    ("failure_mode", "error_type"),
    (
        ("delegate", RuntimeError),
        ("cancellation", PipelineError),
        ("session_enter", TypeError),
    ),
)
def test_processor_pool_failed_session_releases_lane_load(
    failure_mode: str,
    error_type: type[BaseException],
) -> None:
    lanes = (
        inference_module.SerialInferenceLane(),
        inference_module.SerialInferenceLane(),
    )
    delegate_failure = RuntimeError("delegate failed")

    class OutcomeProcessor:
        def __init__(
            self,
            index: int,
            lane: inference_module.SerialInferenceLane,
        ) -> None:
            self._index = index
            self._lane = lane
            self._failed = False

        def process(self, *_args: object, **_kwargs: object) -> int:
            if (
                failure_mode == "delegate"
                and self._index == 0
                and not self._failed
            ):
                self._failed = True
                return self._lane.invoke(
                    lambda: (_ for _ in ()).throw(delegate_failure)
                )
            return self._lane.invoke(lambda: self._index)

    pool = composition_module.TranscriptionProcessorPool(
        tuple(
            OutcomeProcessor(index, lane)
            for index, lane in enumerate(lanes)
        )
    )
    cancellation: object = Cancellation()
    if failure_mode == "cancellation":
        cancellation.cancel()  # type: ignore[union-attr]
    elif failure_mode == "session_enter":
        cancellation = object()

    with pytest.raises(error_type):
        _call_pooled_processor(
            pool.sync_processor,
            cancellation=cancellation,  # type: ignore[arg-type]
        )

    assert _call_pooled_processor(pool.async_processor) == 1
    assert _call_pooled_processor(pool.async_processor) == 0
    assert not hasattr(inference_module._session_local, "current")


def test_processor_pool_facades_use_real_sync_async_lane_handoff() -> None:
    lane = inference_module.SerialInferenceLane()
    holder_entered = threading.Event()
    release_holder = threading.Event()
    order: list[str] = []
    errors: list[BaseException] = []

    class LaneProcessor:
        def process(self, input_path: Path, *_args: object, **_kwargs: object) -> str:
            return lane.invoke(lambda: order.append(input_path.stem))

    pool = composition_module.TranscriptionProcessorPool((LaneProcessor(),))

    def hold_lane() -> None:
        with inference_module.inference_session("async", Cancellation()):
            lane.invoke(
                lambda: (
                    holder_entered.set(),
                    release_holder.wait(timeout=2),
                )
            )

    def run(processor: object, name: str) -> None:
        try:
            _call_pooled_processor(
                processor,
                input_path=Path(f"{name}.ready"),
            )
        except BaseException as error:
            errors.append(error)

    holder = threading.Thread(target=hold_lane, daemon=True)
    holder.start()
    assert holder_entered.wait(timeout=1)
    async_waiter = threading.Thread(
        target=lambda: run(pool.async_processor, "async"),
        daemon=True,
    )
    sync_waiter = threading.Thread(
        target=lambda: run(pool.sync_processor, "sync"),
        daemon=True,
    )
    try:
        async_waiter.start()
        _wait_for_pool_lane_waiters(lane, "async", 1)
        sync_waiter.start()
        _wait_for_pool_lane_waiters(lane, "sync", 1)
        release_holder.set()
    finally:
        release_holder.set()
        holder.join(timeout=1)
        for waiter in (sync_waiter, async_waiter):
            if waiter.ident is not None:
                waiter.join(timeout=1)

    assert not holder.is_alive()
    assert not sync_waiter.is_alive()
    assert not async_waiter.is_alive()
    assert order == ["sync", "async"]
    assert errors == []


def test_processor_pool_does_not_hold_lane_while_delegate_decodes() -> None:
    lane = inference_module.SerialInferenceLane()
    first_decode_entered = threading.Event()
    release_first_decode = threading.Event()
    second_lane_entered = threading.Event()
    call_lock = threading.Lock()
    calls = 0
    operation_order: list[str] = []

    class DecodeBlockingProcessor:
        def process(self, *_args: object, **_kwargs: object) -> str:
            nonlocal calls
            with call_lock:
                call_index = calls
                calls += 1
            if call_index == 0:
                first_decode_entered.set()
                assert release_first_decode.wait(timeout=2)
                return lane.invoke(lambda: operation_order.append("first"))
            return lane.invoke(
                lambda: (
                    operation_order.append("second"),
                    second_lane_entered.set(),
                )
            )

    pool = composition_module.TranscriptionProcessorPool(
        (DecodeBlockingProcessor(),)
    )
    errors: list[BaseException] = []

    def run(processor: object) -> None:
        try:
            _call_pooled_processor(processor)
        except BaseException as error:
            errors.append(error)

    first = threading.Thread(target=lambda: run(pool.sync_processor), daemon=True)
    second = threading.Thread(
        target=lambda: run(pool.async_processor),
        daemon=True,
    )
    first.start()
    assert first_decode_entered.wait(timeout=1)
    second.start()
    try:
        assert second_lane_entered.wait(timeout=1)
    finally:
        release_first_decode.set()
        first.join(timeout=1)
        second.join(timeout=1)

    assert not first.is_alive()
    assert not second.is_alive()
    assert operation_order == ["second", "first"]
    assert errors == []


@pytest.mark.parametrize(
    ("processors", "error_type"),
    (
        ((), ValueError),
        ([], TypeError),
        ((object(),), TypeError),
    ),
)
def test_processor_pool_rejects_invalid_lane_processors(
    processors: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        composition_module.TranscriptionProcessorPool(processors)


class RuntimeModel:
    def __init__(self) -> None:
        self.calls = 0

    def infer(self, pcm: np.ndarray) -> AsrResult:
        self.calls += 1
        assert pcm.dtype == np.float32
        return AsrResult(
            '  runtime "text"  ',
            "en",
            RichAnnotations("happy", "speech"),
        )


class RaisingProjector:
    def prepare(
        self,
        *_args: object,
        speaker_mapping: SpeakerLabelMapping,
        **_kwargs: object,
    ) -> object:
        del speaker_mapping
        raise RuntimeError("projection failed")


class SpyProjector:
    def __init__(self) -> None:
        self.calls = 0
        self.speaker_mapping: SpeakerLabelMapping | None = None

    def prepare(
        self,
        *_args: object,
        speaker_mapping: SpeakerLabelMapping,
        **_kwargs: object,
    ) -> Projection:
        self.calls += 1
        self.speaker_mapping = speaker_mapping
        return Projection(
            content_type="application/json",
            body_factory=lambda: iter((b'{"text":"hello"}',)),
        )


class PermissiveProjector:
    def __init__(self) -> None:
        self.calls = 0

    def prepare(
        self,
        *_args: object,
        speaker_mapping: object,
        **_kwargs: object,
    ) -> Projection:
        del speaker_mapping
        self.calls += 1
        return Projection(
            content_type="application/json",
            body_factory=lambda: iter((b'{"text":"hello"}',)),
        )

def test_progress_accumulator_is_exact_monotonic_and_requires_update() -> None:
    from botified_asr.composition import ProgressAccumulator

    progress = ProgressAccumulator()
    with pytest.raises(RuntimeError, match="progress"):
        progress.finish()

    for processed in (True, 1.0, -1, MAX_AUDIO_SAMPLES + 1):
        with pytest.raises((TypeError, ValueError)):
            progress.update(
                processed_samples=processed,  # type: ignore[arg-type]
                total_samples=None,
            )
    for total in (True, 1.0, -1, MAX_AUDIO_SAMPLES + 1):
        with pytest.raises((TypeError, ValueError)):
            progress.update(
                processed_samples=0,
                total_samples=total,  # type: ignore[arg-type]
            )

    progress.update(processed_samples=2, total_samples=None)
    progress.update(processed_samples=3, total_samples=None)
    with pytest.raises(ValueError, match="monotonic"):
        progress.update(processed_samples=2, total_samples=None)
    with pytest.raises(RuntimeError, match="EOF"):
        progress.finish()
    with pytest.raises(ValueError):
        progress.update(processed_samples=3, total_samples=4)

    progress.update(processed_samples=3, total_samples=3)
    assert progress.finish() == 3
    for total_samples in (None, 3):
        with pytest.raises((RuntimeError, ValueError)):
            progress.update(
                processed_samples=3,
                total_samples=total_samples,
            )


def test_sync_composition_protocol_and_current_effective_caps(
    tmp_path: Path,
) -> None:
    from botified_asr.composition import (
        TranscriptionProcessor,
        prepare_sync_transcription,
    )

    parameters = signature(TranscriptionProcessor.process).parameters
    for name in (
        "selected_speaker_snapshot",
        "effective_max_audio_samples",
        "effective_direct_max_audio_samples",
    ):
        assert parameters[name].kind is Parameter.KEYWORD_ONLY

    storage = _storage(
        tmp_path,
        max_audio_duration_secs=90,
        direct_max_audio_duration_secs=7,
        sync_max_audio_duration_secs=60,
    )
    processor = EmittingProcessor()
    try:
        prepared = prepare_sync_transcription(
            storage,
            processor,
            tmp_path / "input.ready",
            owner_id="request-1",
            options=_options(),
            cancellation=Cancellation(),
            speaker_embedding_policy=_speaker_embedding_policy(),
            projector=SpyProjector(),
        )

        assert processor.effective_caps == [(90 * 16_000, 7 * 16_000)]
        assert b"".join(prepared.iter_body()) == b'{"text":"hello"}'
        _assert_no_artifact(storage)
    finally:
        storage.close()


def test_real_wav_processor_storage_and_three_projections(
    tmp_path: Path,
) -> None:
    from botified_asr.composition import prepare_sync_transcription

    path = tmp_path / "representative.wav"
    samples = np.arange(12_345, dtype=np.int32).astype(np.int16)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(samples.astype("<i2", copy=False).tobytes())
    storage = _storage(tmp_path)
    model = RuntimeModel()
    processor = Processor(
        FfmpegAudioFrontend(),
        NormalizingAsrAdapter(model),
        known_speaker_policy=None,
    )
    try:
        for response_format in ("json", "text", "verbose_json"):
            prepared = prepare_sync_transcription(
                storage,
                processor,
                path,
                owner_id=f"request-{response_format}",
                options=_options(response_format),
                cancellation=Cancellation(),
                speaker_embedding_policy=_speaker_embedding_policy(),
            )
            body = b"".join(prepared.iter_body())
            if response_format == "text":
                assert prepared.content_type == "text/plain; charset=utf-8"
                assert body == b'runtime "text"'
            else:
                assert prepared.content_type == "application/json"
                payload = json.loads(body)
                assert payload["text"] == 'runtime "text"'
                if response_format == "verbose_json":
                    assert payload["duration"] == len(samples) / 16_000
                    assert payload["segments"] == [
                        {
                            "id": "0",
                            "start": 0.0,
                            "end": len(samples) / 16_000,
                            "text": 'runtime "text"',
                        }
                    ]
            _assert_no_artifact(storage)
        assert model.calls == 3
    finally:
        storage.close()


@pytest.mark.parametrize(
    "fault",
    [
        "480001",
        "model",
        "write",
        "seal",
        "projection",
        "bare_ref",
        "invalid_result",
        "nonexact_result",
        "wrong_ref",
        "invalid_mapping",
        "invalid_progress",
        "missing_progress",
        "missing_eof",
    ],
)
def test_composition_faults_discard_every_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    from botified_asr.composition import prepare_sync_transcription

    storage = _storage(tmp_path)
    processor = EmittingProcessor(behavior=fault)
    projector: Any = RaisingProjector() if fault == "projection" else None
    if fault == "write":
        monkeypatch.setattr(
            storage,
            "append_artifact",
            lambda *_args: (_ for _ in ()).throw(OSError("write failed")),
        )
        processor.behavior = "success"
    if fault == "seal":
        monkeypatch.setattr(
            storage,
            "seal_artifact",
            lambda *_args: (_ for _ in ()).throw(OSError("seal failed")),
        )
        processor.behavior = "success"
    try:
        with pytest.raises(
            (OSError, RuntimeError, PipelineError, TypeError, ValueError),
        ):
            prepare_sync_transcription(
                storage,
                processor,
                tmp_path / "input.ready",
                owner_id="request-1",
                options=_options(),
                cancellation=Cancellation(),
                speaker_embedding_policy=_speaker_embedding_policy(),
                **({} if projector is None else {"projector": projector}),
            )

        assert processor.calls == 1
        _assert_no_artifact(storage)
    finally:
        storage.close()


def test_composition_passes_the_identical_speaker_mapping_to_projector(
    tmp_path: Path,
) -> None:
    from botified_asr.composition import prepare_sync_transcription

    mapping = SpeakerLabelMapping((SpeakerLabelResolution("A", None),))
    processor = EmittingProcessor(speaker_mapping=mapping)
    projector = SpyProjector()
    storage = _storage(tmp_path)
    try:
        prepared = prepare_sync_transcription(
            storage,
            processor,
            tmp_path / "input.ready",
            owner_id="request-1",
            options=_options(),
            cancellation=Cancellation(),
            speaker_embedding_policy=_speaker_embedding_policy(),
            projector=projector,
        )

        assert projector.speaker_mapping is mapping
        assert b"".join(prepared.iter_body()) == b'{"text":"hello"}'
        _assert_no_artifact(storage)
    finally:
        storage.close()


def test_anonymous_composition_skips_snapshot_read_and_passes_empty_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from botified_asr.composition import prepare_sync_transcription

    events: list[str] = []
    processor = EmittingProcessor(events=events)
    projector = SpyProjector()
    storage = _storage(tmp_path)
    original_begin = storage.begin_artifact
    monkeypatch.setattr(
        storage,
        "get_speaker_profiles_by_ids",
        lambda _ids: (_ for _ in ()).throw(
            AssertionError("anonymous mode must not read speaker profiles")
        ),
    )

    def begin_artifact(*args: object, **kwargs: object) -> object:
        events.append("begin_artifact")
        return original_begin(*args, **kwargs)

    monkeypatch.setattr(storage, "begin_artifact", begin_artifact)
    try:
        prepared = prepare_sync_transcription(
            storage,
            processor,
            tmp_path / "input.ready",
            owner_id="request-1",
            options=_options(),
            cancellation=Cancellation(),
            speaker_embedding_policy=_speaker_embedding_policy(),
            projector=projector,
        )

        assert events == ["begin_artifact", "processor"]
        assert len(processor.selected_snapshots) == 1
        assert type(processor.selected_snapshots[0]) is SelectedSpeakerSnapshot
        assert processor.selected_snapshots[0].speakers == ()
        assert b"".join(prepared.iter_body()) == b'{"text":"hello"}'
        _assert_no_artifact(storage)
    finally:
        storage.close()


def test_known_composition_reads_one_snapshot_before_processor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from botified_asr.composition import prepare_sync_transcription

    known_ids = ("00000001", "00000002")
    storage = _storage(tmp_path)
    for profile in (
        _profile("00000001", "Alice"),
        _profile("00000002", "Bob"),
    ):
        storage.create_speaker_profile(profile)
    events: list[str] = []
    batch_calls: list[tuple[str, ...]] = []
    resolved_snapshots: list[SelectedSpeakerSnapshot] = []
    original_batch = storage.get_speaker_profiles_by_ids
    original_begin = storage.begin_artifact

    def get_batch(profile_ids: tuple[str, ...]) -> object:
        events.append("snapshot_read")
        batch_calls.append(profile_ids)
        return original_batch(profile_ids)

    def resolve_snapshot(
        reader: object,
        profile_ids: tuple[str, ...],
        policy: speakers.SpeakerEmbeddingPolicy,
    ) -> SelectedSpeakerSnapshot:
        snapshot = speaker_snapshot.resolve_selected_speaker_snapshot(
            reader,  # type: ignore[arg-type]
            profile_ids,
            policy,
        )
        resolved_snapshots.append(snapshot)
        return snapshot

    def begin_artifact(*args: object, **kwargs: object) -> object:
        events.append("begin_artifact")
        return original_begin(*args, **kwargs)

    monkeypatch.setattr(storage, "get_speaker_profiles_by_ids", get_batch)
    monkeypatch.setattr(storage, "begin_artifact", begin_artifact)
    monkeypatch.setattr(
        composition_module,
        "resolve_selected_speaker_snapshot",
        resolve_snapshot,
        raising=False,
    )
    processor = EmittingProcessor(events=events)
    projector = SpyProjector()
    try:
        prepared = prepare_sync_transcription(
            storage,
            processor,
            tmp_path / "input.ready",
            owner_id="request-1",
            options=_options(known_speaker_ids=known_ids),
            cancellation=Cancellation(),
            speaker_embedding_policy=_speaker_embedding_policy(),
            projector=projector,
        )

        assert events == ["snapshot_read", "begin_artifact", "processor"]
        assert batch_calls == [known_ids]
        assert len(resolved_snapshots) == 1
        assert processor.selected_snapshots == [resolved_snapshots[0]]
        assert processor.selected_snapshots[0] is resolved_snapshots[0]
        assert tuple(
            (speaker.id, speaker.name) for speaker in resolved_snapshots[0].speakers
        ) == (("00000001", "Alice"), ("00000002", "Bob"))
        assert b"".join(prepared.iter_body()) == b'{"text":"hello"}'
        _assert_no_artifact(storage)
    finally:
        storage.close()


@pytest.mark.parametrize("failure_kind", ("missing", "incompatible"))
def test_known_snapshot_failure_precedes_artifact_and_processor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    from botified_asr.composition import prepare_sync_transcription

    storage = _storage(tmp_path)
    if failure_kind == "incompatible":
        storage.create_speaker_profile(_profile("00000001", "Alice", compatible=False))
    batch_calls: list[tuple[str, ...]] = []
    original_batch = storage.get_speaker_profiles_by_ids

    def get_batch(profile_ids: tuple[str, ...]) -> object:
        batch_calls.append(profile_ids)
        return original_batch(profile_ids)

    monkeypatch.setattr(storage, "get_speaker_profiles_by_ids", get_batch)
    monkeypatch.setattr(
        storage,
        "begin_artifact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("snapshot failure must precede artifact creation")
        ),
    )
    processor = EmittingProcessor()
    expected_error = (
        speaker_snapshot.SelectedSpeakerNotFoundError
        if failure_kind == "missing"
        else speaker_snapshot.SelectedSpeakerIncompatibleError
    )
    try:
        with pytest.raises(expected_error):
            prepare_sync_transcription(
                storage,
                processor,
                tmp_path / "input.ready",
                owner_id="request-1",
                options=_options(
                    known_speaker_ids=("00000001",),
                ),
                cancellation=Cancellation(),
                speaker_embedding_policy=_speaker_embedding_policy(),
                projector=SpyProjector(),
            )

        assert batch_calls == [("00000001",)]
        assert processor.calls == 0
        _assert_no_artifact(storage)
    finally:
        storage.close()


@pytest.mark.parametrize("invalid_kind", ("object", "subclass"))
def test_composition_rejects_invalid_mapping_before_projector_and_cleans(
    tmp_path: Path,
    invalid_kind: str,
) -> None:
    from botified_asr.composition import prepare_sync_transcription

    if invalid_kind == "object":
        mapping: object = object()
        projector: Any = PermissiveProjector()
    else:
        mapping_type = type(
            "NonExactSpeakerLabelMapping",
            (SpeakerLabelMapping,),
            {"__slots__": ()},
        )
        mapping = mapping_type(())
        projector = SpyProjector()
    processor = EmittingProcessor(
        speaker_mapping=mapping,  # type: ignore[arg-type]
    )
    storage = _storage(tmp_path)
    try:
        with pytest.raises(RuntimeError):
            prepare_sync_transcription(
                storage,
                processor,
                tmp_path / "input.ready",
                owner_id="request-1",
                options=_options(),
                cancellation=Cancellation(),
                speaker_embedding_policy=_speaker_embedding_policy(),
                projector=projector,
            )

        assert projector.calls == 0
        _assert_no_artifact(storage)
    finally:
        storage.close()


def test_zero_sample_response_is_owned_until_stream_finishes(
    tmp_path: Path,
) -> None:
    from botified_asr.composition import prepare_sync_transcription

    storage = _storage(tmp_path)
    try:
        prepared = prepare_sync_transcription(
            storage,
            EmittingProcessor(behavior="zero"),
            tmp_path / "input.ready",
            owner_id="request-1",
            options=_options(),
            cancellation=Cancellation(),
            speaker_embedding_policy=_speaker_embedding_policy(),
        )
        assert len(tuple(storage.artifact_dir.iterdir())) == 1
        assert (
            storage._connection.execute(
                """
                SELECT COUNT(*) FROM storage_leases
                WHERE lease_type = 'artifact'
                """
            ).fetchone()[0]
            == 1
        )
        assert b"".join(prepared.iter_body()) == b'{"text":""}'
        _assert_no_artifact(storage)
    finally:
        storage.close()


def test_prepared_response_is_one_shot_and_early_close_releases(
    tmp_path: Path,
) -> None:
    from botified_asr.composition import prepare_sync_transcription

    storage = _storage(tmp_path)
    try:
        prepared = prepare_sync_transcription(
            storage,
            EmittingProcessor(),
            tmp_path / "input.ready",
            owner_id="request-1",
            options=_options(),
            cancellation=Cancellation(),
            speaker_embedding_policy=_speaker_embedding_policy(),
        )
        body = prepared.iter_body()
        with pytest.raises(RuntimeError, match="already"):
            prepared.iter_body()
        assert next(body)
        body.close()
        _assert_no_artifact(storage)
        prepared.close()
        with pytest.raises(RuntimeError, match="closed"):
            prepared.iter_body()
    finally:
        storage.close()


def test_prepared_response_close_retries_release_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from botified_asr.composition import prepare_sync_transcription

    storage = _storage(tmp_path)
    prepared = prepare_sync_transcription(
        storage,
        EmittingProcessor(),
        tmp_path / "input.ready",
        owner_id="request-1",
        options=_options(),
        cancellation=Cancellation(),
        speaker_embedding_policy=_speaker_embedding_policy(),
    )
    original_release = storage.release_artifact
    calls = 0

    def fail_once(ref: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("release failed")
        original_release(ref)  # type: ignore[arg-type]

    monkeypatch.setattr(storage, "release_artifact", fail_once)
    try:
        with pytest.raises(OSError, match="release failed"):
            prepared.close()
        assert storage.total_reserved_bytes() > 0

        prepared.close()
        prepared.close()
        assert calls == 2
        _assert_no_artifact(storage)
    finally:
        storage.close()


def test_claimed_job_attempt_success_uses_fresh_durable_inputs_and_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    known_id = "00000001"
    options = _options(known_speaker_ids=(known_id,))
    storage = _storage(tmp_path)
    storage.create_speaker_profile(_profile(known_id, "Alice"))
    running = _queue_and_claim_job(
        storage,
        options=options,
        effective_max_audio_samples=32_000,
        effective_direct_max_audio_samples=12_000,
    )
    assert storage.delete_speaker_profile(known_id)
    resolve_calls: list[tuple[jobs.DurableJob, Path]] = []
    original_resolve = storage.resolve_job_attempt_input

    def resolve_attempt(
        job_id: str,
        attempt_token: str,
    ) -> tuple[jobs.DurableJob, Path]:
        resolved = original_resolve(job_id, attempt_token)
        resolve_calls.append(resolved)
        return resolved

    observed_progress: list[tuple[int, int | None, int, int | None]] = []

    def observe_progress(
        processed_samples: int,
        total_samples: int | None,
    ) -> None:
        current = storage.get_visible_job(running.id)
        assert current is not None
        observed_progress.append(
            (
                processed_samples,
                total_samples,
                current.processed_samples,
                current.total_samples,
            )
        )

    monkeypatch.setattr(
        storage,
        "resolve_job_attempt_input",
        resolve_attempt,
    )
    cancellation = Cancellation()
    processor = ClaimedJobProcessor(after_progress=observe_progress)
    try:
        _execute_attempt(storage, processor, running, cancellation)

        assert len(resolve_calls) == 1
        resolved, input_path = resolve_calls[0]
        assert resolved is not running
        assert input_path == storage.staging_dir / f"{running.id}.ready"
        assert processor.input_paths == [input_path]
        assert processor.input_payloads == [b"durable audio"]
        assert processor.options == [options]
        assert processor.cancellations == [cancellation]
        assert processor.effective_caps == [(32_000, 12_000)]
        assert [
            (speaker.id, speaker.name)
            for speaker in processor.selected_snapshots[0].speakers
        ] == [(known_id, "Alice")]
        assert observed_progress == [
            (8_000, None, 8_000, None),
            (16_000, 16_000, 16_000, 16_000),
        ]
        succeeded = storage.get_visible_job(running.id)
        assert succeeded is not None
        assert succeeded.status is jobs.JobStatus.SUCCEEDED
        assert succeeded.processed_samples == 16_000
        assert succeeded.total_samples == 16_000
        assert succeeded.finished_at == JOB_FINISHED_AT
        assert not input_path.exists()
        assert _artifact_kinds(storage, running.id) == (
            "result_complete",
        )
        assert tuple(
            tuple(row)
            for row in storage._connection.execute(
                "SELECT lease_type, resource_kind FROM storage_leases"
            )
        ) == (("artifact", "result_complete"),)
        stored = storage.open_succeeded_job_result(running.id)
        try:
            assert json.loads(b"".join(stored.iter_body()))["text"] == "hello"
        finally:
            stored.close()
    finally:
        storage.close()


@pytest.mark.parametrize("losing_progress", ("cancel", "stale"))
def test_claimed_job_attempt_progress_loser_discards_only_owned_artifact(
    tmp_path: Path,
    losing_progress: str,
) -> None:
    storage = _storage(tmp_path)
    running = _queue_and_claim_job(storage)
    assert running.attempt_token is not None

    def lose_progress() -> None:
        if losing_progress == "cancel":
            storage._connection.execute(
                """
                UPDATE transcription_jobs SET cancel_requested = 1
                WHERE id = ?
                """,
                (running.id,),
            )
        else:
            storage._connection.execute(
                """
                UPDATE transcription_jobs SET attempt_token = ?
                WHERE id = ?
                """,
                ("attempt-new", running.id),
            )

    cancellation = Cancellation()
    processor = ClaimedJobProcessor(before_progress=lose_progress)
    try:
        _execute_attempt(storage, processor, running, cancellation)

        current = storage.get_visible_job(running.id)
        assert current is not None
        assert cancellation.cancelled
        assert _artifact_kinds(storage, running.id) == ()
        if losing_progress == "cancel":
            assert current.status is jobs.JobStatus.CANCELLED
            assert current.input_lease_id is None
            assert not (
                storage.staging_dir / f"{running.id}.ready"
            ).exists()
            assert storage.total_reserved_bytes() == 0
        else:
            assert current.status is jobs.JobStatus.RUNNING
            assert current.attempt_token == "attempt-new"
            assert current.processed_samples == 0
            assert current.total_samples is None
            assert current.input_lease_id == running.id
            assert (
                storage.staging_dir / f"{running.id}.ready"
            ).is_file()
    finally:
        storage.close()


@pytest.mark.parametrize(
    ("error", "expected_code"),
    (
        (
            PipelineError("audio_too_long", "private pipeline detail"),
            "audio_too_long",
        ),
        (
            AudioError("invalid_audio", "private audio detail"),
            "invalid_audio",
        ),
        (RuntimeError("private unknown detail"), "internal_error"),
    ),
)
def test_claimed_job_attempt_maps_processor_failure_without_message_leak(
    tmp_path: Path,
    error: BaseException,
    expected_code: str,
) -> None:
    storage = _storage(tmp_path)
    running = _queue_and_claim_job(storage)
    try:
        _execute_attempt(
            storage,
            ClaimedJobProcessor(error=error),
            running,
        )

        failed = _assert_clean_terminal(
            storage,
            running.id,
            jobs.JobStatus.FAILED,
        )
        assert failed.error_code == expected_code
        assert failed.finished_at == JOB_FINISHED_AT
        assert str(error) not in "\n".join(storage._connection.iterdump())
    finally:
        storage.close()


@pytest.mark.parametrize("race", ("failure", "success"))
def test_claimed_job_attempt_terminal_commit_loses_to_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    race: str,
) -> None:
    storage = _storage(tmp_path)
    running = _queue_and_claim_job(storage)
    saw_result_before_cancel = False

    def request_cancel() -> None:
        _request_job_cancel(storage, running.id)

    if race == "failure":
        original_failure = storage.commit_job_failure

        def cancel_before_failure(*args: Any, **kwargs: Any) -> object:
            request_cancel()
            return original_failure(*args, **kwargs)

        monkeypatch.setattr(
            storage,
            "commit_job_failure",
            cancel_before_failure,
        )
        processor = ClaimedJobProcessor(
            error=PipelineError("invalid_audio", "private"),
        )
    else:
        original_success = storage.commit_job_success

        def cancel_before_success(*args: Any, **kwargs: Any) -> object:
            nonlocal saw_result_before_cancel
            saw_result_before_cancel = (
                "result_complete" in _artifact_kinds(storage, running.id)
            )
            request_cancel()
            return original_success(*args, **kwargs)

        monkeypatch.setattr(
            storage,
            "commit_job_success",
            cancel_before_success,
        )
        processor = ClaimedJobProcessor()
    try:
        _execute_attempt(storage, processor, running)

        cancelled = _assert_clean_terminal(
            storage,
            running.id,
            jobs.JobStatus.CANCELLED,
        )
        assert cancelled.error_code is None
        assert cancelled.finished_at == JOB_FINISHED_AT
        assert saw_result_before_cancel is (race == "success")
    finally:
        storage.close()


def test_claimed_job_attempt_result_admission_failure_commits_internal_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage(tmp_path)
    running = _queue_and_claim_job(storage)
    error = StorageAdmissionError(
        "storage_full",
        "private result admission detail",
    )
    monkeypatch.setattr(
        storage,
        "begin_job_result_artifact",
        lambda *_args: (_ for _ in ()).throw(error),
    )
    try:
        _execute_attempt(storage, ClaimedJobProcessor(), running)

        failed = _assert_clean_terminal(
            storage,
            running.id,
            jobs.JobStatus.FAILED,
        )
        assert failed.error_code == "internal_error"
        assert failed.finished_at == JOB_FINISHED_AT
        assert str(error) not in "\n".join(storage._connection.iterdump())
    finally:
        storage.close()


def test_claimed_job_attempt_fixed_total_mismatch_is_integrity_error(
    tmp_path: Path,
) -> None:
    storage = _storage(tmp_path)
    running = _queue_and_claim_job(storage)
    storage._connection.execute(
        """
        UPDATE transcription_jobs SET total_samples = ?
        WHERE id = ?
        """,
        (16_001, running.id),
    )
    try:
        with pytest.raises(StorageSchemaError):
            _execute_attempt(storage, ClaimedJobProcessor(), running)

        current = storage.get_visible_job(running.id)
        assert current is not None
        assert current.status is jobs.JobStatus.RUNNING
        assert current.total_samples == 16_001
        assert current.processed_samples == 8_000
        assert current.input_lease_id == running.id
        assert _artifact_kinds(storage, running.id) == ()
    finally:
        storage.close()


def test_claimed_job_attempt_propagates_local_cancel_and_integrity_errors(
    tmp_path: Path,
) -> None:
    errors: tuple[BaseException, ...] = (
        PipelineError("cancelled", "local pipeline cancellation"),
        AudioError("cancelled", "local audio cancellation"),
        StorageSchemaError("private integrity failure"),
    )
    for index, error in enumerate(errors):
        storage = _storage(tmp_path / f"case-{index}")
        running = _queue_and_claim_job(storage)
        try:
            with pytest.raises(type(error)) as caught:
                _execute_attempt(
                    storage,
                    ClaimedJobProcessor(error=error),
                    running,
                )

            assert caught.value is error
            current = storage.get_visible_job(running.id)
            assert current is not None
            assert current.status is jobs.JobStatus.RUNNING
            assert current.attempt_token == running.attempt_token
            assert current.error_code is None
            assert current.finished_at is None
            assert current.input_lease_id == running.id
            assert (
                storage.staging_dir / f"{running.id}.ready"
            ).is_file()
            assert _artifact_kinds(storage, running.id) == ()
        finally:
            storage.close()


@pytest.mark.parametrize(
    "error",
    (
        PipelineError("cancelled", "requested pipeline cancellation"),
        AudioError("cancelled", "requested audio cancellation"),
    ),
)
def test_claimed_job_attempt_commits_requested_local_cancellation(
    tmp_path: Path,
    error: BaseException,
) -> None:
    storage = _storage(tmp_path)
    running = _queue_and_claim_job(storage)
    try:
        _execute_attempt(
            storage,
            ClaimedJobProcessor(
                error=error,
                before_progress=lambda: _request_job_cancel(
                    storage,
                    running.id,
                ),
            ),
            running,
        )

        cancelled = _assert_clean_terminal(
            storage,
            running.id,
            jobs.JobStatus.CANCELLED,
        )
        assert cancelled.error_code is None
        assert cancelled.finished_at == JOB_FINISHED_AT
    finally:
        storage.close()
