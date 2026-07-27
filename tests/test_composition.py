from __future__ import annotations

import json
import wave
from datetime import datetime, timezone
from inspect import Parameter, signature
from pathlib import Path
from typing import Any, get_type_hints

import numpy as np
import pytest

from botified_asr import composition as composition_module
from botified_asr import pipeline as pipeline_module
from botified_asr import speaker_profiles, speaker_snapshot, speakers
from botified_asr.audio import Cancellation, FfmpegAudioFrontend
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
    SpeakerLabelMapping,
    SpeakerLabelResolution,
)
from botified_asr.speaker_snapshot import SelectedSpeakerSnapshot
from botified_asr.storage import Storage


MODEL_ID = "funasr/campplus"
MODEL_REVISION = "1" * 40


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


def _storage(tmp_path: Path) -> Storage:
    return Storage(
        tmp_path / "storage",
        LimitsConfig(
            max_upload_bytes=RESERVATION_QUANTUM,
            sync_max_upload_bytes=RESERVATION_QUANTUM,
            max_job_storage_bytes=2 * RESERVATION_QUANTUM,
            min_filesystem_free_bytes=1,
        ),
        free_bytes=lambda _: 1 << 40,
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

    def process(
        self,
        _input_path: Path,
        _options: CanonicalOptions,
        _cancellation: Cancellation,
        progress: Any,
        sink: Any,
        *,
        selected_speaker_snapshot: SelectedSpeakerSnapshot,
    ) -> object:
        self.calls += 1
        self.selected_snapshots.append(selected_speaker_snapshot)
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


def test_composition_protocols_use_the_exact_transport_contract() -> None:
    from botified_asr.composition import (
        ProjectionBuilder,
        TranscriptionProcessor,
        prepare_sync_transcription,
    )

    return_type = get_type_hints(TranscriptionProcessor.process)["return"]
    assert return_type is pipeline_module.ProcessorResult

    parameter = signature(ProjectionBuilder.prepare).parameters["speaker_mapping"]

    assert parameter.kind is Parameter.KEYWORD_ONLY
    assert parameter.default is Parameter.empty

    snapshot_parameter = signature(TranscriptionProcessor.process).parameters[
        "selected_speaker_snapshot"
    ]
    assert snapshot_parameter.kind is Parameter.KEYWORD_ONLY
    assert snapshot_parameter.default is Parameter.empty

    policy_parameter = signature(prepare_sync_transcription).parameters[
        "speaker_embedding_policy"
    ]
    assert policy_parameter.kind is Parameter.KEYWORD_ONLY
    assert policy_parameter.default is Parameter.empty


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

    progress.update(processed_samples=2, total_samples=0)
    progress.update(processed_samples=3, total_samples=None)
    with pytest.raises(ValueError, match="monotonic"):
        progress.update(processed_samples=2, total_samples=None)
    assert progress.finish() == 3


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
