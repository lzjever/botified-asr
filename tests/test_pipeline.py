from __future__ import annotations

import wave
from dataclasses import FrozenInstanceError
from inspect import Parameter, signature
from pathlib import Path

import numpy as np
import pytest

from botified_asr import pipeline as pipeline_module
from botified_asr.audio import (
    Cancellation,
    DecodedBlock,
    FfmpegAudioFrontend,
    MediaProbe,
)
from botified_asr.contracts import CanonicalOptions
from botified_asr.pipeline import (
    AsrResult,
    CanonicalJsonlSegmentSink,
    NormalizingAsrAdapter,
    PipelineError,
    PipelineNotReady,
    Processor,
    RichAnnotations,
    SegmentRecord,
    canonical_join,
)
from botified_asr.speaker_matching import SpeakerLabelMapping
from botified_asr.speaker_snapshot import SelectedSpeakerSnapshot


EMPTY_SELECTED_SNAPSHOT = SelectedSpeakerSnapshot(())


class FakeDecoder:
    def __init__(
        self,
        blocks: list[DecodedBlock],
        *,
        fail_close: bool = False,
        iteration_error: Exception | None = None,
    ) -> None:
        self.blocks = blocks
        self.closed = 0
        self.fail_close = fail_close
        self.iteration_error = iteration_error

    def __iter__(self):
        yield from self.blocks
        if self.iteration_error is not None:
            raise self.iteration_error

    def close(self) -> None:
        self.closed += 1
        if self.fail_close:
            raise OSError("decoder close failed")


class FakeAdapter:
    def __init__(
        self,
        batch_result: object = (
            AsrResult(
                text="  hello  ",
                language="en",
                annotations=RichAnnotations(
                    emotion="happy",
                    audio_event=None,
                ),
            ),
        ),
    ) -> None:
        self.batch_result = batch_result
        self.calls: list[tuple[tuple[np.ndarray, ...], str]] = []

    def transcribe(self, _pcm: np.ndarray) -> AsrResult:
        raise AssertionError("Processor must use the typed batch adapter contract")

    def transcribe_batch(
        self,
        pcms: tuple[np.ndarray, ...],
        *,
        language: str,
    ) -> object:
        assert len(pcms) == 1
        pcm = pcms[0]
        assert pcm.dtype == np.int16
        assert pcm.ndim == 1
        assert pcm.flags.c_contiguous
        assert len(pcm) <= 480_000
        self.calls.append((pcms, language))
        return self.batch_result


class RecordingSink:
    def __init__(self) -> None:
        self.records: list[SegmentRecord] = []
        self.finalized = 0
        self.aborted = 0
        self.ref = object()

    def append(self, record: SegmentRecord) -> None:
        self.records.append(record)

    def finalize(self) -> object:
        self.finalized += 1
        return self.ref

    def abort(self) -> None:
        self.aborted += 1


class FakeWriter:
    def __init__(
        self,
        *,
        fail_write: bool = False,
        fail_seal: bool = False,
    ) -> None:
        self.payloads: list[bytes] = []
        self.sealed = 0
        self.aborted = 0
        self.ref = object()
        self.fail_write = fail_write
        self.fail_seal = fail_seal

    def write(self, payload: bytes) -> None:
        if self.fail_write:
            raise OSError("disk failed")
        self.payloads.append(payload)

    def seal(self) -> object:
        if self.fail_seal:
            raise OSError("seal failed")
        self.sealed += 1
        return self.ref

    def abort(self) -> None:
        self.aborted += 1


class FloatModel:
    def __init__(self) -> None:
        self.inputs: list[np.ndarray] = []

    def infer(self, pcm: np.ndarray) -> AsrResult:
        assert pcm.dtype == np.float32
        assert pcm.ndim == 1
        assert pcm.flags.c_contiguous
        assert np.isfinite(pcm).all()
        self.inputs.append(pcm)
        return AsrResult("ok", "en", RichAnnotations())


class FakeFrontend:
    def __init__(self, decoder: FakeDecoder) -> None:
        self.decoder = decoder
        self.probe_calls: list[Path] = []
        self.decode_calls: list[Path] = []

    def probe(self, input_path: Path, cancellation: Cancellation) -> MediaProbe:
        self.probe_calls.append(input_path)
        return MediaProbe(1.0, "wav")

    def decode(
        self,
        input_path: Path,
        probe: MediaProbe,
        cancellation: Cancellation,
    ) -> FakeDecoder:
        self.decode_calls.append(input_path)
        return self.decoder


class RecordingProgress:
    def __init__(self) -> None:
        self.updates: list[tuple[int, int | None]] = []

    def update(self, *, processed_samples: int, total_samples: int | None) -> None:
        self.updates.append((processed_samples, total_samples))


def options(
    *,
    model: str = "sensevoice",
    language: str = "auto",
    chunking_strategy: str | None = None,
) -> CanonicalOptions:
    return CanonicalOptions(
        model=model,
        language=language,
        response_format="json",
        chunking_strategy=chunking_strategy,
        include=(),
        known_speaker_ids=(),
    )


def run_processor(
    decoder: FakeDecoder,
    adapter: FakeAdapter,
    sink: RecordingSink | CanonicalJsonlSegmentSink,
    *,
    selected_speaker_snapshot: SelectedSpeakerSnapshot,
    canonical_options: CanonicalOptions | None = None,
    progress: RecordingProgress | None = None,
) -> object:
    frontend = FakeFrontend(decoder)
    return Processor(
        frontend,
        adapter,
        known_speaker_policy=None,
    ).process(
        Path("/internal/input.ready"),
        canonical_options or options(),
        Cancellation(),
        progress or RecordingProgress(),
        sink,
        selected_speaker_snapshot=selected_speaker_snapshot,
    )


def _assert_empty_processor_result(result: object, ref: object) -> None:
    assert type(result) is pipeline_module.ProcessorResult
    assert result.artifact_ref is ref
    assert result.speaker_mapping == SpeakerLabelMapping(())


def test_processor_result_is_exact_frozen_slotted_and_requires_both_fields() -> None:
    result_type = pipeline_module.ProcessorResult
    parameters = signature(result_type).parameters

    assert tuple(parameters) == ("artifact_ref", "speaker_mapping")
    assert all(
        parameter.default is Parameter.empty for parameter in parameters.values()
    )
    assert result_type.__slots__ == ("artifact_ref", "speaker_mapping")

    artifact_ref = object()
    mapping = SpeakerLabelMapping(())
    result = result_type(artifact_ref, mapping)

    assert type(result) is result_type
    assert result.artifact_ref is artifact_ref
    assert result.speaker_mapping is mapping
    with pytest.raises(FrozenInstanceError):
        result.artifact_ref = object()
    with pytest.raises(TypeError):
        result_type()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        result_type(artifact_ref)  # type: ignore[call-arg]

    snapshot_parameter = signature(Processor.process).parameters[
        "selected_speaker_snapshot"
    ]
    assert snapshot_parameter.kind is Parameter.KEYWORD_ONLY
    assert snapshot_parameter.default is Parameter.empty


@pytest.mark.parametrize("sample_count", [0, 1, 480_000])
def test_direct_processor_has_exact_empty_and_max_sample_boundaries(
    sample_count: int,
) -> None:
    adapter = FakeAdapter()
    sink = RecordingSink()
    samples = np.arange(sample_count, dtype=np.int32).astype(np.int16)
    blocks = [
        DecodedBlock(start, samples[start : start + 9_600])
        for start in range(0, sample_count, 9_600)
    ]
    decoder = FakeDecoder(blocks)
    progress = RecordingProgress()

    result = run_processor(
        decoder,
        adapter,
        sink,
        selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
        progress=progress,
    )

    _assert_empty_processor_result(result, sink.ref)
    assert sink.finalized == 1
    assert sink.aborted == 0
    assert decoder.closed == 1
    assert progress.updates[-1][0] == sample_count
    assert len(adapter.calls) == (0 if sample_count == 0 else 1)
    assert len(sink.records) == (0 if sample_count == 0 else 1)
    if sample_count:
        pcms, language = adapter.calls[0]
        assert language == "auto"
        assert len(pcms) == 1
        assert sink.records[0] == SegmentRecord(
            index=0,
            start_sample=0,
            end_sample=sample_count,
            text="  hello  ",
            language="en",
            annotations=RichAnnotations("happy", None),
        )
        np.testing.assert_array_equal(
            pcms[0],
            samples,
        )


def test_normalizing_asr_adapter_has_typed_batch_convenience() -> None:
    samples = np.array([-32_768, -1, 0, 1, 32_767], dtype=np.int16)
    model = FloatModel()

    results = NormalizingAsrAdapter(model).transcribe_batch(
        (samples,),
        language="auto",
    )

    assert len(results) == 1
    assert results[0].text == "ok"
    np.testing.assert_array_equal(
        model.inputs[0],
        samples.astype(np.float32) / np.float32(32768.0),
    )


@pytest.mark.parametrize("language", ("zh", "en", "ko"))
def test_direct_processor_passes_explicit_language_to_one_item_batch(
    language: str,
) -> None:
    samples = np.arange(12_345, dtype=np.int32).astype(np.int16)
    adapter = FakeAdapter()
    sink = RecordingSink()

    result = run_processor(
        FakeDecoder(
            [
                DecodedBlock(start, samples[start : start + 9_600])
                for start in range(0, len(samples), 9_600)
            ]
        ),
        adapter,
        sink,
        selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
        canonical_options=options(language=language),
    )

    _assert_empty_processor_result(result, sink.ref)
    assert len(adapter.calls) == 1
    pcms, requested_language = adapter.calls[0]
    assert requested_language == language
    assert len(pcms) == 1
    np.testing.assert_array_equal(pcms[0], samples)


@pytest.mark.parametrize(
    "batch_result",
    (
        (),
        (
            AsrResult("first", "en", RichAnnotations()),
            AsrResult("second", "en", RichAnnotations()),
        ),
        (object(),),
        [AsrResult("list", "en", RichAnnotations())],
    ),
    ids=("empty", "multiple", "non_asr_result", "non_tuple"),
)
def test_direct_processor_rejects_invalid_typed_batch_results_atomically(
    batch_result: object,
) -> None:
    adapter = FakeAdapter(batch_result)
    sink = RecordingSink()
    decoder = FakeDecoder([DecodedBlock(0, np.ones(4, dtype=np.int16))])

    with pytest.raises(PipelineError) as caught:
        run_processor(
            decoder,
            adapter,
            sink,
            selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
        )

    assert caught.value.code == "invalid_model_output"
    assert len(adapter.calls) == 1
    assert decoder.closed == 1
    assert sink.records == []
    assert sink.finalized == 0
    assert sink.aborted == 1


def test_direct_processor_skips_empty_model_text_and_finalizes_successfully() -> None:
    result = AsrResult(
        text="",
        language="zh",
        annotations=RichAnnotations(
            emotion="unknown:sensevoice:emotion:EMO_UNKNOWN",
            audio_event="unknown:sensevoice:audio_event:Event_UNK",
        ),
    )
    adapter = FakeAdapter((result,))
    sink = RecordingSink()
    progress = RecordingProgress()
    samples = np.arange(12_345, dtype=np.int32).astype(np.int16)

    result = run_processor(
        FakeDecoder(
            [
                DecodedBlock(start, samples[start : start + 9_600])
                for start in range(0, len(samples), 9_600)
            ]
        ),
        adapter,
        sink,
        selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
        canonical_options=options(language="zh"),
        progress=progress,
    )

    _assert_empty_processor_result(result, sink.ref)
    assert len(adapter.calls) == 1
    assert adapter.calls[0][1] == "zh"
    assert progress.updates[-1] == (len(samples), None)
    assert sink.records == []
    assert sink.finalized == 1
    assert sink.aborted == 0


def test_direct_480001_closes_decoder_and_aborts_without_success() -> None:
    adapter = FakeAdapter()
    sink = RecordingSink()
    decoder = FakeDecoder(
        [
            DecodedBlock(start, np.zeros(9_600, dtype=np.int16))
            for start in range(0, 480_000, 9_600)
        ]
        + [DecodedBlock(480_000, np.zeros(1, dtype=np.int16))]
    )

    with pytest.raises(PipelineError) as caught:
        run_processor(
            decoder,
            adapter,
            sink,
            selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
        )

    assert getattr(caught.value, "code", None) == "long_audio_requires_vad"
    assert decoder.closed == 1
    assert adapter.calls == []
    assert sink.records == []
    assert sink.finalized == 0
    assert sink.aborted == 1


def test_unimplemented_branches_are_typed_not_ready_and_never_run_direct() -> None:
    for canonical_options in (
        options(chunking_strategy="auto"),
        options(model="sensevoice-diarize", chunking_strategy="auto"),
    ):
        adapter = FakeAdapter()
        sink = RecordingSink()
        decoder = FakeDecoder([DecodedBlock(0, np.ones(1, dtype=np.int16))])
        frontend = FakeFrontend(decoder)

        with pytest.raises(PipelineNotReady) as caught:
            Processor(
                frontend,
                adapter,
                known_speaker_policy=None,
            ).process(
                Path("/internal/input.ready"),
                canonical_options,
                Cancellation(),
                RecordingProgress(),
                sink,
                selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
            )

        assert caught.value.code == "pipeline_not_ready"
        assert adapter.calls == []
        assert sink.records == []
        assert sink.finalized == 0
        assert sink.aborted == 1
        assert frontend.probe_calls == []
        assert frontend.decode_calls == []
        assert decoder.closed == 0


def test_canonical_jsonl_sink_is_canonical_ordered_and_returns_opaque_ref() -> None:
    writer = FakeWriter()
    sink = CanonicalJsonlSegmentSink(writer)
    annotations = RichAnnotations(emotion="喜悦", audio_event=None)
    record = SegmentRecord(
        index=0,
        start_sample=0,
        end_sample=16_000,
        text='line\n"二"',
        language="zh",
        annotations=annotations,
    )

    sink.append(record)
    artifact_ref = sink.finalize()

    assert artifact_ref is writer.ref
    assert writer.payloads == [
        (
            '{"annotations":{"audio_event":null,"emotion":"喜悦"},'
            '"anonymous_speaker":null,"end_sample":16000,"index":0,"language":"zh",'
            '"start_sample":0,"text":"line\\n\\"二\\""}\n'
        ).encode()
    ]
    assert writer.sealed == 1
    with pytest.raises(FrozenInstanceError):
        record.end_sample = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        annotations.emotion = "other"  # type: ignore[misc]
    with pytest.raises(RuntimeError, match="finalized"):
        sink.append(record)
    with pytest.raises(RuntimeError, match="finalized"):
        sink.finalize()
    with pytest.raises(RuntimeError, match="sealed"):
        sink.abort()

    aborted_writer = FakeWriter()
    aborted_sink = CanonicalJsonlSegmentSink(aborted_writer)
    aborted_sink.abort()
    aborted_sink.abort()
    with pytest.raises(RuntimeError, match="finalized"):
        aborted_sink.append(record)
    with pytest.raises(RuntimeError, match="finalized"):
        aborted_sink.finalize()
    assert aborted_writer.aborted == 1


def test_jsonl_validation_and_writer_fault_cleanup_are_fail_closed() -> None:
    writer = FakeWriter()
    sink = CanonicalJsonlSegmentSink(writer)
    sink.append(SegmentRecord(0, 10, 20, "first", "en", RichAnnotations()))
    with pytest.raises(ValueError, match="index"):
        sink.append(SegmentRecord(2, 20, 30, "gap", "en", RichAnnotations()))
    with pytest.raises(ValueError, match="sample bounds"):
        sink.append(SegmentRecord(1, 19, 30, "overlap", "en", RichAnnotations()))
    payload_count = len(writer.payloads)
    for invalid in (
        SegmentRecord(True, 20, 30, "bool", "en", RichAnnotations()),
        SegmentRecord(1, 20.0, 30, "float", "en", RichAnnotations()),  # type: ignore[arg-type]
        SegmentRecord(1, 20, float("nan"), "nan", "en", RichAnnotations()),  # type: ignore[arg-type]
    ):
        with pytest.raises(TypeError, match="integers"):
            sink.append(invalid)
    assert len(writer.payloads) == payload_count
    for invalid_annotations in (
        RichAnnotations(emotion=True, audio_event=None),  # type: ignore[arg-type]
        RichAnnotations(emotion=None, audio_event=1),  # type: ignore[arg-type]
    ):
        with pytest.raises(TypeError, match="annotation"):
            sink.append(
                SegmentRecord(
                    1,
                    20,
                    30,
                    "invalid annotation",
                    "en",
                    invalid_annotations,
                )
            )
    assert len(writer.payloads) == payload_count
    for invalid_speaker in (True, 1, "Unknown A", "AG"):
        with pytest.raises((TypeError, ValueError)):
            sink.append(
                SegmentRecord(
                    1,
                    20,
                    30,
                    "invalid speaker",
                    "en",
                    RichAnnotations(),
                    anonymous_speaker=invalid_speaker,  # type: ignore[arg-type]
                )
            )
    assert len(writer.payloads) == payload_count

    failing_writer = FakeWriter(fail_write=True)
    failing_sink = CanonicalJsonlSegmentSink(failing_writer)
    with pytest.raises(OSError, match="disk failed"):
        run_processor(
            FakeDecoder([DecodedBlock(0, np.ones(1, dtype=np.int16))]),
            FakeAdapter(),
            failing_sink,
            selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
        )
    assert failing_writer.sealed == 0
    assert failing_writer.aborted == 1

    seal_writer = FakeWriter(fail_seal=True)
    seal_sink = CanonicalJsonlSegmentSink(seal_writer)
    with pytest.raises(OSError, match="seal failed"):
        run_processor(
            FakeDecoder([]),
            FakeAdapter(),
            seal_sink,
            selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
        )
    assert seal_writer.sealed == 0
    assert seal_writer.aborted == 1
    with pytest.raises(RuntimeError, match="finalized"):
        seal_sink.append(SegmentRecord(0, 0, 1, "late", "en", RichAnnotations()))
    with pytest.raises(RuntimeError, match="finalized"):
        seal_sink.finalize()

    discontinuous_sink = RecordingSink()
    with pytest.raises(PipelineError) as discontinuous:
        run_processor(
            FakeDecoder([DecodedBlock(9_600, np.ones(1, dtype=np.int16))]),
            FakeAdapter(),
            discontinuous_sink,
            selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
        )
    assert getattr(discontinuous.value, "code", None) == "invalid_audio"
    assert discontinuous_sink.aborted == 1

    short_nonfinal_sink = RecordingSink()
    with pytest.raises(PipelineError) as short_nonfinal:
        run_processor(
            FakeDecoder(
                [
                    DecodedBlock(0, np.ones(1, dtype=np.int16)),
                    DecodedBlock(1, np.ones(1, dtype=np.int16)),
                ]
            ),
            FakeAdapter(),
            short_nonfinal_sink,
            selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
        )
    assert short_nonfinal.value.code == "invalid_audio"
    assert short_nonfinal_sink.aborted == 1

    close_fault_sink = RecordingSink()
    with pytest.raises(OSError, match="decoder close failed"):
        run_processor(
            FakeDecoder([], fail_close=True),
            FakeAdapter(),
            close_fault_sink,
            selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
        )
    assert close_fault_sink.finalized == 0
    assert close_fault_sink.aborted == 1

    iteration_fault_sink = RecordingSink()
    iteration_fault_decoder = FakeDecoder(
        [],
        iteration_error=OSError("decode iteration failed"),
    )
    with pytest.raises(OSError, match="decode iteration failed"):
        run_processor(
            iteration_fault_decoder,
            FakeAdapter(),
            iteration_fault_sink,
            selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
        )
    assert iteration_fault_decoder.closed == 1
    assert iteration_fault_sink.aborted == 1
    assert iteration_fault_sink.finalized == 0


def test_canonical_join_golden_and_partition_invariant() -> None:
    golden = [
        (["你", "好"], "你好"),
        (["hello", "world"], "hello world"),
        (["Hello.", "Next"], "Hello. Next"),
        (["hello", ","], "hello,"),
        (["中", "English"], "中English"),
        (["English", "中"], "English中"),
        (["(", "hello"], "(hello"),
        (["hello", ")"], "hello)"),
        (["안녕", "하세요"], "안녕 하세요"),
        (["🙂", "🙃"], "🙂 🙃"),
        (["\U0001b000", "\U0001b001"], "\U0001b000\U0001b001"),
        (["A", "\u3007"], "A\u3007"),
        (["A", "\u2f00"], "A\u2f00"),
        (["A", "\U0001f200"], "A\U0001f200"),
        (["A", "\u32d0"], "A\u32d0"),
        (["A", "\u3040"], "A \u3040"),
        (["A", "\U0001afff"], "A \U0001afff"),
        (["“", "hello"], "“hello"),
        (["hello", "”"], "hello”"),
        (["  hello  ", "", "\u2003world\t"], "hello world"),
        (["e\u0301", "x"], "e\u0301 x"),
    ]
    for parts, expected in golden:
        assert canonical_join(parts) == expected

    parts = ["中", "English", "hello", ",", "世界"]
    expected = canonical_join(parts)
    for split in range(len(parts) + 1):
        assert expected == canonical_join(
            [
                canonical_join(parts[:split]),
                canonical_join(parts[split:]),
            ]
        )


def test_runtime_generated_wav_runs_through_public_processor(
    tmp_path: Path,
) -> None:
    path = tmp_path / "representative.wav"
    expected = np.array(
        [((index * 97) % 20_000) - 10_000 for index in range(12_345)],
        dtype=np.int16,
    )
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(expected.astype("<i2", copy=False).tobytes())
    model = FloatModel()
    sink = RecordingSink()
    progress = RecordingProgress()

    result = Processor(
        FfmpegAudioFrontend(),
        NormalizingAsrAdapter(model),
        known_speaker_policy=None,
    ).process(
        path,
        options(),
        Cancellation(),
        progress,
        sink,
        selected_speaker_snapshot=EMPTY_SELECTED_SNAPSHOT,
    )

    _assert_empty_processor_result(result, sink.ref)
    assert sink.records[0].end_sample == len(expected)
    assert progress.updates[-1][0] == len(expected)
    np.testing.assert_array_equal(
        model.inputs[0],
        expected.astype(np.float32) / np.float32(32768.0),
    )
