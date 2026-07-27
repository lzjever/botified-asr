from __future__ import annotations

from collections.abc import Callable
from inspect import Parameter, signature
from pathlib import Path

import numpy as np
import pytest

from botified_asr import pipeline, speaker_matching, speakers
from botified_asr.audio import BLOCK_SAMPLES, Cancellation, DecodedBlock, MediaProbe
from botified_asr.contracts import CanonicalOptions
from botified_asr.speaker_profiles import SpeakerEmbedding
from botified_asr.speaker_matching import SpeakerLabelMapping
from botified_asr.speaker_snapshot import SelectedSpeaker, SelectedSpeakerSnapshot


EMPTY_SELECTED_SNAPSHOT = SelectedSpeakerSnapshot(())


class FakeDecoder:
    def __init__(self, blocks: tuple[DecodedBlock, ...]) -> None:
        self.blocks = blocks
        self.closed = 0

    def __iter__(self):
        yield from self.blocks

    def close(self) -> None:
        self.closed += 1


class FakeFrontend:
    def __init__(self, decoder: FakeDecoder) -> None:
        self.decoder = decoder
        self.probe_calls = 0
        self.decode_calls = 0

    def probe(self, _input_path: Path, _cancellation: Cancellation) -> MediaProbe:
        self.probe_calls += 1
        return MediaProbe(1.0, "wav")

    def decode(
        self,
        _input_path: Path,
        _probe: MediaProbe,
        _cancellation: Cancellation,
    ) -> FakeDecoder:
        self.decode_calls += 1
        return self.decoder


class FakeAsrAdapter:
    def __init__(
        self,
        result_factory: Callable[[tuple[np.ndarray, ...]], object] | None = None,
    ) -> None:
        self._result_factory = result_factory or (
            lambda pcms: tuple(
                _result(f"text-{index}") for index, _pcm in enumerate(pcms)
            )
        )
        self.calls: list[tuple[tuple[np.ndarray, ...], str]] = []

    def transcribe(self, _pcm: np.ndarray) -> pipeline.AsrResult:
        raise AssertionError("Processor must not fall back to scalar ASR")

    def transcribe_batch(
        self,
        pcms: tuple[np.ndarray, ...],
        *,
        language: str,
    ) -> object:
        self.calls.append((pcms, language))
        return self._result_factory(pcms)


class RecordingProgress:
    def __init__(self) -> None:
        self.updates: list[tuple[int, int | None]] = []

    def update(self, *, processed_samples: int, total_samples: int | None) -> None:
        self.updates.append((processed_samples, total_samples))


class RecordingSink:
    def __init__(self, events: list[str] | None = None) -> None:
        self.records: list[pipeline.SegmentRecord] = []
        self.finalized = 0
        self.aborted = 0
        self.ref = object()
        self.events = events

    def append(self, record: pipeline.SegmentRecord) -> None:
        if self.events is not None:
            self.events.append("append")
        self.records.append(record)

    def finalize(self) -> object:
        if self.events is not None:
            self.events.append("sink_finalize")
        self.finalized += 1
        return self.ref

    def abort(self) -> None:
        self.aborted += 1


class ScriptedSegmenter:
    def __init__(
        self,
        adapter: object,
        outputs: tuple[tuple[pipeline.BufferedSpeechSegment, ...], ...],
    ) -> None:
        self.adapter = adapter
        self.outputs = outputs
        self.calls: list[tuple[DecodedBlock, bool]] = []

    def process(
        self,
        block: DecodedBlock,
        *,
        is_final: bool,
    ) -> tuple[pipeline.BufferedSpeechSegment, ...]:
        output = self.outputs[len(self.calls)]
        self.calls.append((block, is_final))
        return output


class FakeSpeakerAdapter:
    def __init__(
        self,
        windows: tuple[tuple[speakers.SpeakerEmbeddingWindow, ...], ...] = (),
        *,
        fail_on_call: int | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.windows = windows
        self.fail_on_call = fail_on_call
        self.events = events
        self.calls: list[np.ndarray] = []

    def embed_windows(
        self,
        pcm: np.ndarray,
    ) -> tuple[speakers.SpeakerEmbeddingWindow, ...]:
        call_index = len(self.calls)
        self.calls.append(pcm)
        if self.events is not None:
            self.events.append("cam")
        if call_index == self.fail_on_call:
            raise pipeline.PipelineError(
                "invalid_model_output",
                "CAM++ failed",
            )
        return self.windows[call_index]


def _install_segmenter(
    monkeypatch: pytest.MonkeyPatch,
    outputs: tuple[tuple[pipeline.BufferedSpeechSegment, ...], ...],
) -> list[ScriptedSegmenter]:
    instances: list[ScriptedSegmenter] = []

    def factory(adapter: object) -> ScriptedSegmenter:
        instance = ScriptedSegmenter(adapter, outputs)
        instances.append(instance)
        return instance

    monkeypatch.setattr(pipeline, "StreamingSpeechSegmenter", factory)
    return instances


def _result(text: str) -> pipeline.AsrResult:
    return pipeline.AsrResult(text, "en", pipeline.RichAnnotations())


def _segment(
    start_sample: int,
    end_sample: int,
    *,
    pcm_start_sample: int | None = None,
) -> pipeline.BufferedSpeechSegment:
    pcm_start = start_sample if pcm_start_sample is None else pcm_start_sample
    return pipeline.BufferedSpeechSegment(
        span=pipeline.SpeechSpan(start_sample, end_sample),
        pcm_start_sample=pcm_start,
        pcm=np.arange(end_sample - pcm_start, dtype=np.int64).astype(np.int16),
    )


def _blocks(count: int) -> tuple[DecodedBlock, ...]:
    return tuple(
        DecodedBlock(
            index * BLOCK_SAMPLES,
            np.full(BLOCK_SAMPLES, index, dtype=np.int16),
        )
        for index in range(count)
    )


def _unit(*components: float) -> np.ndarray:
    embedding = np.zeros(speakers.SPEAKER_EMBEDDING_DIMENSION, dtype=np.float32)
    embedding[: len(components)] = components
    embedding /= np.linalg.norm(embedding)
    return embedding


def _window(embedding: np.ndarray) -> tuple[speakers.SpeakerEmbeddingWindow, ...]:
    return (
        speakers.SpeakerEmbeddingWindow(
            start_sample=0,
            end_sample=1,
            embedding=embedding,
        ),
    )


def _policy() -> speakers.AnonymousSpeakerPolicy:
    return speakers.AnonymousSpeakerPolicy(
        threshold=0.5,
        max_speakers=32,
    )


def _match_policy() -> speaker_matching.KnownSpeakerMatchPolicy:
    return speaker_matching.KnownSpeakerMatchPolicy(
        match_threshold=0.5,
        top_two_margin=0.1,
    )


def _selected_snapshot(
    profile_ids: tuple[str, ...] = ("00000001",),
) -> SelectedSpeakerSnapshot:
    return SelectedSpeakerSnapshot(
        tuple(
            SelectedSpeaker(
                profile_id,
                f"Speaker {profile_id}",
                SpeakerEmbedding.from_numpy(
                    _unit(1.0, 0.0),
                    dimension=speakers.SPEAKER_EMBEDDING_DIMENSION,
                ),
            )
            for profile_id in profile_ids
        )
    )


def _options(
    *,
    model: str = "sensevoice",
    language: str = "auto",
    chunking_strategy: str | None = "auto",
    known_speaker_ids: tuple[str, ...] = (),
) -> CanonicalOptions:
    return CanonicalOptions(
        model=model,
        language=language,
        response_format=("diarized_json" if model == "sensevoice-diarize" else "json"),
        chunking_strategy=chunking_strategy,
        include=(),
        known_speaker_ids=known_speaker_ids,
    )


def _process(
    processor: pipeline.Processor,
    sink: RecordingSink,
    *,
    selected_speaker_snapshot: SelectedSpeakerSnapshot,
    language: str = "auto",
    progress: RecordingProgress | None = None,
    canonical_options: CanonicalOptions | None = None,
) -> object:
    return processor.process(
        Path("/internal/input.ready"),
        canonical_options or _options(language=language),
        Cancellation(),
        progress or RecordingProgress(),
        sink,
        selected_speaker_snapshot=selected_speaker_snapshot,
    )


def _assert_empty_processor_result(result: object, ref: object) -> None:
    assert type(result) is pipeline.ProcessorResult
    assert result.artifact_ref is ref
    assert result.speaker_mapping == SpeakerLabelMapping(())


def test_auto_without_vad_fails_before_probe() -> None:
    decoder = FakeDecoder(())
    frontend = FakeFrontend(decoder)
    sink = RecordingSink()

    with pytest.raises(pipeline.PipelineNotReady):
        _process(
            pipeline.Processor(
                frontend,
                FakeAsrAdapter(),
                known_speaker_policy=None,
            ),
            sink,
            selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
        )

    assert frontend.probe_calls == 0
    assert frontend.decode_calls == 0
    assert decoder.closed == 0
    assert sink.finalized == 0
    assert sink.aborted == 1


def test_empty_auto_succeeds_without_vad_or_asr_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances = _install_segmenter(monkeypatch, ())
    decoder = FakeDecoder(())
    asr = FakeAsrAdapter()
    progress = RecordingProgress()
    sink = RecordingSink()

    result = _process(
        pipeline.Processor(
            FakeFrontend(decoder),
            asr,
            vad_adapter=object(),
            known_speaker_policy=None,
        ),
        sink,
        selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
        progress=progress,
    )

    _assert_empty_processor_result(result, sink.ref)
    assert sum(len(instance.calls) for instance in instances) == 0
    assert asr.calls == []
    assert progress.updates == [(0, None)]
    assert decoder.closed == 1
    assert sink.finalized == 1
    assert sink.aborted == 0


def test_auto_uses_lookahead_language_and_canonical_span_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment = _segment(5_920, 24_000, pcm_start_sample=2_720)
    instances = _install_segmenter(monkeypatch, ((), (), (segment,)))
    blocks = _blocks(3)
    decoder = FakeDecoder(blocks)
    asr = FakeAsrAdapter(lambda _pcms: (_result("mapped"),))
    progress = RecordingProgress()
    sink = RecordingSink()

    _process(
        pipeline.Processor(
            FakeFrontend(decoder),
            asr,
            vad_adapter=object(),
            known_speaker_policy=None,
        ),
        sink,
        selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
        language="ja",
        progress=progress,
    )

    assert len(instances) == 1
    assert [is_final for _block, is_final in instances[0].calls] == [
        False,
        False,
        True,
    ]
    assert [block for block, _is_final in instances[0].calls] == list(blocks)
    assert len(asr.calls) == 1
    assert asr.calls[0][1] == "ja"
    np.testing.assert_array_equal(asr.calls[0][0][0], segment.pcm)
    assert sink.records == [
        pipeline.SegmentRecord(
            index=0,
            start_sample=5_920,
            end_sample=24_000,
            text="mapped",
            language="en",
            annotations=pipeline.RichAnnotations(),
        )
    ]
    assert progress.updates == [
        (BLOCK_SAMPLES, None),
        (2 * BLOCK_SAMPLES, None),
        (3 * BLOCK_SAMPLES, None),
    ]


@pytest.mark.parametrize(
    ("segments", "expected_batch_sizes"),
    (
        (
            tuple(_segment(index, index + 1) for index in range(33)),
            [32, 1],
        ),
        (
            (
                _segment(0, 480_000),
                _segment(480_000, 960_000),
                _segment(960_000, 960_001),
            ),
            [2, 1],
        ),
        (
            (
                _segment(0, 1),
                _segment(4_799_999, 4_800_000),
                _segment(4_800_000, 4_800_001),
            ),
            [2, 1],
        ),
    ),
    ids=("32_segments", "960000_pcm", "4800000_wall"),
)
def test_auto_preflushes_before_a_candidate_exceeds_each_inclusive_batch_cap(
    monkeypatch: pytest.MonkeyPatch,
    segments: tuple[pipeline.BufferedSpeechSegment, ...],
    expected_batch_sizes: list[int],
) -> None:
    _install_segmenter(monkeypatch, (segments,))
    asr = FakeAsrAdapter()
    sink = RecordingSink()

    _process(
        pipeline.Processor(
            FakeFrontend(FakeDecoder(_blocks(1))),
            asr,
            vad_adapter=object(),
            known_speaker_policy=None,
        ),
        sink,
        selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
    )

    assert [len(pcms) for pcms, _language in asr.calls] == expected_batch_sizes
    assert all(language == "auto" for _pcms, language in asr.calls)
    assert len(sink.records) == len(segments)
    assert sink.finalized == 1
    assert sink.aborted == 0


@pytest.mark.parametrize(
    "invalid_result",
    (
        [_result("first"), _result("second")],
        (_result("wrong cardinality"),),
        (_result("first"), object()),
    ),
    ids=("list", "cardinality", "element_type"),
)
def test_auto_rejects_invalid_batch_atomically_and_aborts(
    monkeypatch: pytest.MonkeyPatch,
    invalid_result: object,
) -> None:
    segments = (_segment(100, 200), _segment(300, 400))
    _install_segmenter(monkeypatch, (segments,))
    decoder = FakeDecoder(_blocks(1))
    asr = FakeAsrAdapter(lambda _pcms: invalid_result)
    sink = RecordingSink()

    with pytest.raises(pipeline.PipelineError) as caught:
        _process(
            pipeline.Processor(
                FakeFrontend(decoder),
                asr,
                vad_adapter=object(),
                known_speaker_policy=None,
            ),
            sink,
            selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
        )

    assert caught.value.code == "invalid_model_output"
    assert sink.records == []
    assert sink.finalized == 0
    assert sink.aborted == 1
    assert decoder.closed == 1


def test_auto_skips_empty_results_positionally_with_dense_record_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segments = (
        _segment(100, 200),
        _segment(300, 400),
        _segment(500, 600),
    )
    _install_segmenter(monkeypatch, (segments,))
    results = (
        pipeline.AsrResult(
            "first",
            "zh",
            pipeline.RichAnnotations("happy", None),
        ),
        _result(""),
        pipeline.AsrResult(
            "third",
            "ko",
            pipeline.RichAnnotations(None, "speech"),
        ),
    )
    decoder = FakeDecoder(_blocks(1))
    asr = FakeAsrAdapter(lambda _pcms: results)
    sink = RecordingSink()

    result = _process(
        pipeline.Processor(
            FakeFrontend(decoder),
            asr,
            vad_adapter=object(),
            known_speaker_policy=None,
        ),
        sink,
        selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
        language="zh",
    )

    _assert_empty_processor_result(result, sink.ref)
    assert asr.calls[0][1] == "zh"
    assert [
        (record.index, record.start_sample, record.end_sample, record.text)
        for record in sink.records
    ] == [
        (0, 100, 200, "first"),
        (1, 500, 600, "third"),
    ]
    assert decoder.closed == 1
    assert sink.finalized == 1
    assert sink.aborted == 0


def test_speaker_constructor_dependencies_must_be_configured_together() -> None:
    frontend = FakeFrontend(FakeDecoder(()))
    asr = FakeAsrAdapter()
    cam = FakeSpeakerAdapter()

    with pytest.raises(ValueError):
        pipeline.Processor(
            frontend,
            asr,
            speaker_adapter=cam,
            known_speaker_policy=None,
        )
    with pytest.raises(ValueError):
        pipeline.Processor(
            frontend,
            asr,
            speaker_policy=_policy(),
            known_speaker_policy=None,
        )
    with pytest.raises(ValueError):
        pipeline.Processor(
            frontend,
            asr,
            known_speaker_policy=_match_policy(),
        )

    policy_parameter = signature(pipeline.Processor).parameters["known_speaker_policy"]
    assert policy_parameter.kind is Parameter.KEYWORD_ONLY
    assert policy_parameter.default is Parameter.empty

    assert frontend.probe_calls == 0
    assert frontend.decode_calls == 0
    assert cam.calls == []


def test_speaker_embedding_adapter_is_a_structural_window_protocol() -> None:
    adapter_protocol = speakers.SpeakerEmbeddingAdapter

    assert getattr(adapter_protocol, "_is_protocol", False)
    assert callable(adapter_protocol.embed_windows)


@pytest.mark.parametrize(
    ("vad_adapter", "speaker_adapter", "speaker_policy", "known_speaker_ids"),
    (
        (None, FakeSpeakerAdapter(), _policy(), ()),
        (object(), None, None, ()),
    ),
    ids=("missing_vad", "missing_speaker_dependencies"),
)
def test_diarize_readiness_fails_before_probe(
    vad_adapter: object | None,
    speaker_adapter: FakeSpeakerAdapter | None,
    speaker_policy: speakers.AnonymousSpeakerPolicy | None,
    known_speaker_ids: tuple[str, ...],
) -> None:
    decoder = FakeDecoder(_blocks(1))
    frontend = FakeFrontend(decoder)
    sink = RecordingSink()

    with pytest.raises(pipeline.PipelineNotReady):
        _process(
            pipeline.Processor(
                frontend,
                FakeAsrAdapter(),
                vad_adapter=vad_adapter,
                speaker_adapter=speaker_adapter,
                speaker_policy=speaker_policy,
                known_speaker_policy=None,
            ),
            sink,
            selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
            canonical_options=_options(
                model="sensevoice-diarize",
                known_speaker_ids=known_speaker_ids,
            ),
        )

    assert frontend.probe_calls == 0
    assert frontend.decode_calls == 0
    assert decoder.closed == 0
    assert sink.records == []
    assert sink.finalized == 0
    assert sink.aborted == 1
    if speaker_adapter is not None:
        assert speaker_adapter.calls == []


def test_known_matching_snapshot_requires_policy_before_probe() -> None:
    decoder = FakeDecoder(_blocks(1))
    frontend = FakeFrontend(decoder)
    sink = RecordingSink()
    speaker_adapter = FakeSpeakerAdapter()

    with pytest.raises(pipeline.PipelineNotReady):
        _process(
            pipeline.Processor(
                frontend,
                FakeAsrAdapter(),
                vad_adapter=object(),
                speaker_adapter=speaker_adapter,
                speaker_policy=_policy(),
                known_speaker_policy=None,
            ),
            sink,
            selected_speaker_snapshot=_selected_snapshot(),
            canonical_options=_options(
                model="sensevoice-diarize",
                known_speaker_ids=("00000001",),
            ),
        )

    assert frontend.probe_calls == 0
    assert frontend.decode_calls == 0
    assert decoder.closed == 0
    assert speaker_adapter.calls == []
    assert sink.finalized == 0
    assert sink.aborted == 1


@pytest.mark.parametrize(
    "snapshot",
    (
        _selected_snapshot(("00000002", "00000001")),
        _selected_snapshot(("00000001", "00000003")),
    ),
    ids=("order_mismatch", "set_mismatch"),
)
def test_known_snapshot_ids_must_exactly_match_options_before_probe(
    snapshot: SelectedSpeakerSnapshot,
) -> None:
    decoder = FakeDecoder(_blocks(1))
    frontend = FakeFrontend(decoder)
    sink = RecordingSink()
    speaker_adapter = FakeSpeakerAdapter()

    with pytest.raises(RuntimeError):
        _process(
            pipeline.Processor(
                frontend,
                FakeAsrAdapter(),
                vad_adapter=object(),
                speaker_adapter=speaker_adapter,
                speaker_policy=_policy(),
                known_speaker_policy=_match_policy(),
            ),
            sink,
            selected_speaker_snapshot=snapshot,
            canonical_options=_options(
                model="sensevoice-diarize",
                known_speaker_ids=("00000001", "00000002"),
            ),
        )

    assert frontend.probe_calls == 0
    assert frontend.decode_calls == 0
    assert decoder.closed == 0
    assert speaker_adapter.calls == []
    assert sink.finalized == 0
    assert sink.aborted == 1


def test_known_options_reject_empty_snapshot_before_probe() -> None:
    decoder = FakeDecoder(_blocks(1))
    frontend = FakeFrontend(decoder)
    sink = RecordingSink()
    speaker_adapter = FakeSpeakerAdapter()

    with pytest.raises(RuntimeError):
        _process(
            pipeline.Processor(
                frontend,
                FakeAsrAdapter(),
                vad_adapter=object(),
                speaker_adapter=speaker_adapter,
                speaker_policy=_policy(),
                known_speaker_policy=_match_policy(),
            ),
            sink,
            selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
            canonical_options=_options(
                model="sensevoice-diarize",
                known_speaker_ids=("00000001",),
            ),
        )

    assert frontend.probe_calls == 0
    assert frontend.decode_calls == 0
    assert decoder.closed == 0
    assert speaker_adapter.calls == []
    assert sink.finalized == 0
    assert sink.aborted == 1


def test_anonymous_options_reject_nonempty_snapshot_before_probe() -> None:
    decoder = FakeDecoder(_blocks(1))
    frontend = FakeFrontend(decoder)
    sink = RecordingSink()
    speaker_adapter = FakeSpeakerAdapter()

    with pytest.raises(RuntimeError):
        _process(
            pipeline.Processor(
                frontend,
                FakeAsrAdapter(),
                vad_adapter=object(),
                speaker_adapter=speaker_adapter,
                speaker_policy=_policy(),
                known_speaker_policy=None,
            ),
            sink,
            selected_speaker_snapshot=_selected_snapshot(),
            canonical_options=_options(model="sensevoice-diarize"),
        )

    assert frontend.probe_calls == 0
    assert frontend.decode_calls == 0
    assert decoder.closed == 0
    assert speaker_adapter.calls == []
    assert sink.finalized == 0
    assert sink.aborted == 1


def test_normal_modes_never_call_injected_speaker_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment = _segment(100, 200, pcm_start_sample=90)
    _install_segmenter(monkeypatch, ((segment,),))
    cam = FakeSpeakerAdapter((_window(_unit(1.0, 0.0)),))

    direct_sink = RecordingSink()
    _process(
        pipeline.Processor(
            FakeFrontend(FakeDecoder(_blocks(1))),
            FakeAsrAdapter(lambda _pcms: (_result("direct"),)),
            speaker_adapter=cam,
            speaker_policy=_policy(),
            known_speaker_policy=None,
        ),
        direct_sink,
        selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
        canonical_options=_options(chunking_strategy=None),
    )

    auto_sink = RecordingSink()
    _process(
        pipeline.Processor(
            FakeFrontend(FakeDecoder(_blocks(1))),
            FakeAsrAdapter(lambda _pcms: (_result("auto"),)),
            vad_adapter=object(),
            speaker_adapter=cam,
            speaker_policy=_policy(),
            known_speaker_policy=None,
        ),
        auto_sink,
        selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
    )

    assert [record.text for record in direct_sink.records] == ["direct"]
    assert [record.text for record in auto_sink.records] == ["auto"]
    assert cam.calls == []


def test_diarize_dispatches_through_the_existing_vad_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def record_vad_call(
        _self: pipeline.Processor,
        *_args: object,
        **kwargs: object,
    ) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(pipeline.Processor, "_process_vad", record_vad_call)
    vad_adapter = object()
    sink = RecordingSink()

    _process(
        pipeline.Processor(
            FakeFrontend(FakeDecoder(())),
            FakeAsrAdapter(),
            vad_adapter=vad_adapter,
            speaker_adapter=FakeSpeakerAdapter(),
            speaker_policy=_policy(),
            known_speaker_policy=None,
        ),
        sink,
        selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
        canonical_options=_options(model="sensevoice-diarize"),
    )

    assert len(calls) == 1
    assert calls[0]["vad_adapter"] is vad_adapter
    assert sink.finalized == 1
    assert sink.aborted == 0


def test_diarize_uses_canonical_crops_updates_empty_text_and_appends_after_cam_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segments = (
        _segment(100, 104, pcm_start_sample=97),
        _segment(200, 205, pcm_start_sample=196),
        _segment(300, 303, pcm_start_sample=298),
    )
    _install_segmenter(monkeypatch, (segments,))
    results = (
        _result("first"),
        _result(""),
        _result("third"),
    )
    events: list[str] = []
    cam = FakeSpeakerAdapter(
        (
            _window(_unit(1.0, 0.0)),
            _window(_unit(0.0, 1.0)),
            _window(_unit(1.0, 0.0)),
        ),
        events=events,
    )
    fixture_state = speakers.AnonymousSpeakerState(_policy())
    assert [fixture_state.assign_segment(windows) for windows in cam.windows] == [
        "A",
        "B",
        "A",
    ]
    sink = RecordingSink(events)
    selected_snapshot = _selected_snapshot()
    match_policy = _match_policy()
    sentinel_mapping = SpeakerLabelMapping(
        (
            speaker_matching.SpeakerLabelResolution("A", None),
            speaker_matching.SpeakerLabelResolution("B", None),
        )
    )
    original_finalize = speakers.AnonymousSpeakerState.finalize_clusters
    finalized_clusters: list[tuple[speakers.AnonymousSpeakerCluster, ...]] = []

    def finalize_clusters(
        state: speakers.AnonymousSpeakerState,
    ) -> tuple[speakers.AnonymousSpeakerCluster, ...]:
        events.append("finalize_clusters")
        clusters = original_finalize(state)
        finalized_clusters.append(clusters)
        return clusters

    matcher_calls: list[tuple[object, object, object]] = []

    def match_selected_speakers(
        clusters: object,
        snapshot: object,
        policy: object,
    ) -> SpeakerLabelMapping:
        events.append("matcher")
        matcher_calls.append((clusters, snapshot, policy))
        return sentinel_mapping

    monkeypatch.setattr(
        speakers.AnonymousSpeakerState,
        "finalize_clusters",
        finalize_clusters,
    )
    monkeypatch.setattr(
        pipeline,
        "match_selected_speakers",
        match_selected_speakers,
        raising=False,
    )

    result = _process(
        pipeline.Processor(
            FakeFrontend(FakeDecoder(_blocks(1))),
            FakeAsrAdapter(lambda _pcms: results),
            vad_adapter=object(),
            speaker_adapter=cam,
            speaker_policy=_policy(),
            known_speaker_policy=match_policy,
        ),
        sink,
        selected_speaker_snapshot=selected_snapshot,
        canonical_options=_options(
            model="sensevoice-diarize",
            known_speaker_ids=("00000001",),
        ),
    )

    assert type(result) is pipeline.ProcessorResult
    assert result.artifact_ref is sink.ref
    assert result.speaker_mapping is sentinel_mapping
    assert len(finalized_clusters) == 1
    assert tuple(cluster.label for cluster in finalized_clusters[0]) == ("A", "B")
    assert matcher_calls == [(finalized_clusters[0], selected_snapshot, match_policy)]
    assert len(cam.calls) == 3
    for call, segment in zip(cam.calls, segments, strict=True):
        canonical_start = segment.span.start_sample - segment.pcm_start_sample
        canonical_end = segment.span.end_sample - segment.pcm_start_sample
        np.testing.assert_array_equal(
            call,
            segment.pcm[canonical_start:canonical_end],
        )
    assert events == [
        "cam",
        "cam",
        "cam",
        "append",
        "append",
        "finalize_clusters",
        "matcher",
        "sink_finalize",
    ]
    assert sink.records == [
        pipeline.SegmentRecord(
            index=0,
            start_sample=100,
            end_sample=104,
            text="first",
            language="en",
            annotations=pipeline.RichAnnotations(),
            anonymous_speaker="A",
        ),
        pipeline.SegmentRecord(
            index=1,
            start_sample=300,
            end_sample=303,
            text="third",
            language="en",
            annotations=pipeline.RichAnnotations(),
            anonymous_speaker="A",
        ),
    ]


def test_diarize_cam_failure_is_batch_atomic_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segments = (
        _segment(100, 200, pcm_start_sample=90),
        _segment(300, 400, pcm_start_sample=280),
    )
    _install_segmenter(monkeypatch, (segments,))
    decoder = FakeDecoder(_blocks(1))
    cam = FakeSpeakerAdapter(
        (
            _window(_unit(1.0, 0.0)),
            _window(_unit(0.0, 1.0)),
        ),
        fail_on_call=1,
    )
    sink = RecordingSink()

    with pytest.raises(pipeline.PipelineError) as caught:
        _process(
            pipeline.Processor(
                FakeFrontend(decoder),
                FakeAsrAdapter(),
                vad_adapter=object(),
                speaker_adapter=cam,
                speaker_policy=_policy(),
                known_speaker_policy=None,
            ),
            sink,
            selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
            canonical_options=_options(model="sensevoice-diarize"),
        )

    assert caught.value.code == "invalid_model_output"
    assert len(cam.calls) == 2
    assert sink.records == []
    assert sink.finalized == 0
    assert sink.aborted == 1
    assert decoder.closed == 1


@pytest.mark.parametrize("block_count", (0, 1), ids=("zero_decode", "zero_segments"))
def test_empty_diarize_never_calls_asr_or_cam(
    block_count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_segmenter(monkeypatch, ((),))
    asr = FakeAsrAdapter()
    cam = FakeSpeakerAdapter()
    decoder = FakeDecoder(_blocks(block_count))
    sink = RecordingSink()

    result = _process(
        pipeline.Processor(
            FakeFrontend(decoder),
            asr,
            vad_adapter=object(),
            speaker_adapter=cam,
            speaker_policy=_policy(),
            known_speaker_policy=None,
        ),
        sink,
        selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
        canonical_options=_options(model="sensevoice-diarize"),
    )

    _assert_empty_processor_result(result, sink.ref)
    assert asr.calls == []
    assert cam.calls == []
    assert sink.records == []
    assert sink.finalized == 1
    assert sink.aborted == 0
    assert decoder.closed == 1


def test_known_empty_clusters_finalize_once_without_calling_matcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_segmenter(monkeypatch, ((),))
    original_finalize = speakers.AnonymousSpeakerState.finalize_clusters
    finalize_calls = 0

    def finalize_clusters(
        state: speakers.AnonymousSpeakerState,
    ) -> tuple[speakers.AnonymousSpeakerCluster, ...]:
        nonlocal finalize_calls
        finalize_calls += 1
        return original_finalize(state)

    monkeypatch.setattr(
        speakers.AnonymousSpeakerState,
        "finalize_clusters",
        finalize_clusters,
    )
    monkeypatch.setattr(
        pipeline,
        "match_selected_speakers",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("empty clusters must not call matcher")
        ),
        raising=False,
    )
    decoder = FakeDecoder(_blocks(1))
    sink = RecordingSink()

    result = _process(
        pipeline.Processor(
            FakeFrontend(decoder),
            FakeAsrAdapter(),
            vad_adapter=object(),
            speaker_adapter=FakeSpeakerAdapter(),
            speaker_policy=_policy(),
            known_speaker_policy=_match_policy(),
        ),
        sink,
        selected_speaker_snapshot=_selected_snapshot(),
        canonical_options=_options(
            model="sensevoice-diarize",
            known_speaker_ids=("00000001",),
        ),
    )

    _assert_empty_processor_result(result, sink.ref)
    assert finalize_calls == 1
    assert sink.finalized == 1
    assert sink.aborted == 0
    assert decoder.closed == 1


def test_known_match_failure_aborts_before_sink_finalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment = _segment(100, 200, pcm_start_sample=90)
    _install_segmenter(monkeypatch, ((segment,),))
    expected = speaker_matching.SpeakerMatchInputError()
    monkeypatch.setattr(
        pipeline,
        "match_selected_speakers",
        lambda *_args: (_ for _ in ()).throw(expected),
        raising=False,
    )
    decoder = FakeDecoder(_blocks(1))
    sink = RecordingSink()

    with pytest.raises(speaker_matching.SpeakerMatchInputError) as caught:
        _process(
            pipeline.Processor(
                FakeFrontend(decoder),
                FakeAsrAdapter(lambda _pcms: (_result("known"),)),
                vad_adapter=object(),
                speaker_adapter=FakeSpeakerAdapter((_window(_unit(1.0, 0.0)),)),
                speaker_policy=_policy(),
                known_speaker_policy=_match_policy(),
            ),
            sink,
            selected_speaker_snapshot=_selected_snapshot(),
            canonical_options=_options(
                model="sensevoice-diarize",
                known_speaker_ids=("00000001",),
            ),
        )

    assert caught.value is expected
    assert sink.finalized == 0
    assert sink.aborted == 1
    assert decoder.closed == 1


def test_diarize_state_is_request_local_on_reused_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment = _segment(100, 200, pcm_start_sample=90)
    _install_segmenter(monkeypatch, ((segment,),))
    cam = FakeSpeakerAdapter(
        (
            _window(_unit(1.0, 0.0)),
            _window(_unit(0.0, 1.0)),
        )
    )
    processor = pipeline.Processor(
        FakeFrontend(FakeDecoder(_blocks(1))),
        FakeAsrAdapter(),
        vad_adapter=object(),
        speaker_adapter=cam,
        speaker_policy=_policy(),
        known_speaker_policy=None,
    )
    sinks = (RecordingSink(), RecordingSink())

    for sink in sinks:
        _process(
            processor,
            sink,
            selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
            canonical_options=_options(model="sensevoice-diarize"),
        )

    assert len(cam.calls) == 2
    assert [sink.records[0].anonymous_speaker for sink in sinks] == ["A", "A"]
