from __future__ import annotations

import numpy as np
import pytest

from botified_asr import funasr_adapter
from botified_asr.pipeline import (
    AsrResult,
    PipelineError,
    RichAnnotations,
)

CAPTURED_SPEECH_A = {
    "key": "duplicate",
    # The prefix is captured; the body is controlled parser input.
    "text": "<|zh|><|NEUTRAL|><|Speech|><|withitn|>Alpha, 世界!",
}
CAPTURED_SPEECH_B = {
    "key": "duplicate",
    # The prefix is captured; the body is controlled parser input.
    "text": "<|en|><|HAPPY|><|Laughter|><|withitn|>Beta?",
}
CAPTURED_NOSPEECH = {
    "key": "duplicate",
    "text": "<|nospeech|><|EMO_UNKNOWN|><|Event_UNK|><|withitn|>.",
}


class FakeAutoModel:
    def __init__(self, raw_result: object) -> None:
        self.raw_result = raw_result
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.raw_result


class RecordingInferenceLane:
    def __init__(self) -> None:
        self.operations: list[object] = []

    def invoke(self, operation, /):
        self.operations.append(operation)
        return operation()


def _decode(
    raw_result: object,
    *,
    expected_count: int,
    language: str = "auto",
) -> tuple[AsrResult, ...]:
    return funasr_adapter._decode_sensevoice_batch(  # type: ignore[attr-defined, no-any-return]
        raw_result,
        expected_count=expected_count,
        language=language,
    )


def _adapter(model: FakeAutoModel) -> object:
    return funasr_adapter.FunAsrSenseVoiceBatchAdapter(  # type: ignore[attr-defined, no-any-return]
        model,
        inference_lane=RecordingInferenceLane(),
    )


def _raw_text(
    *,
    language: str = "zh",
    emotion: str = "NEUTRAL",
    event: str = "Speech",
    itn: str = "withitn",
    body: str = "controlled text.",
) -> list[dict[str, str]]:
    return [
        {
            "key": "ignored",
            "text": (f"<|{language}|><|{emotion}|><|{event}|><|{itn}|>{body}"),
        }
    ]


def test_parser_decodes_captured_prefixes_with_controlled_body() -> None:
    assert _decode(
        [CAPTURED_SPEECH_A, CAPTURED_SPEECH_B],
        expected_count=2,
    ) == (
        AsrResult(
            text="Alpha, 世界!",
            language="zh",
            annotations=RichAnnotations(
                emotion="neutral",
                audio_event="speech",
            ),
        ),
        AsrResult(
            text="Beta?",
            language="en",
            annotations=RichAnnotations(
                emotion="happy",
                audio_event="laughter",
            ),
        ),
    )


@pytest.mark.parametrize("language", ("zh", "en", "yue", "ja", "ko"))
def test_parser_language_allowlist(language: str) -> None:
    assert (
        _decode(
            _raw_text(language=language),
            expected_count=1,
        )[0].language
        == language
    )


def test_parser_requires_explicit_requested_language_to_match() -> None:
    assert (
        _decode(
            _raw_text(language="zh"),
            expected_count=1,
            language="zh",
        )[0].language
        == "zh"
    )

    with pytest.raises(PipelineError) as caught:
        _decode(
            _raw_text(language="zh"),
            expected_count=1,
            language="en",
        )

    assert caught.value.code == "invalid_model_output"


@pytest.mark.parametrize(
    ("raw_emotion", "expected"),
    (
        ("HAPPY", "happy"),
        ("SAD", "sad"),
        ("ANGRY", "angry"),
        ("NEUTRAL", "neutral"),
        ("FEARFUL", "fearful"),
        ("DISGUSTED", "disgusted"),
        ("SURPRISED", "surprised"),
    ),
)
def test_parser_emotion_allowlist(
    raw_emotion: str,
    expected: str,
) -> None:
    assert (
        _decode(
            _raw_text(emotion=raw_emotion),
            expected_count=1,
        )[0].annotations.emotion
        == expected
    )


@pytest.mark.parametrize(
    ("raw_event", "expected"),
    (
        ("Speech", "speech"),
        ("BGM", "bgm"),
        ("Applause", "applause"),
        ("Laughter", "laughter"),
        ("Cry", "cry"),
        ("Sneeze", "sneeze"),
        ("Breath", "breath"),
        ("Cough", "cough"),
    ),
)
def test_parser_audio_event_allowlist(
    raw_event: str,
    expected: str,
) -> None:
    assert (
        _decode(
            _raw_text(event=raw_event),
            expected_count=1,
        )[0].annotations.audio_event
        == expected
    )


@pytest.mark.parametrize(
    ("emotion", "event", "expected_emotion", "expected_event"),
    (
        (
            "OTHER",
            "Speech",
            "unknown:sensevoice:emotion:OTHER",
            "speech",
        ),
        (
            "NEUTRAL",
            "Sing",
            "neutral",
            "unknown:sensevoice:audio_event:Sing",
        ),
        (
            "NEUTRAL",
            "Speech_Noise",
            "neutral",
            "unknown:sensevoice:audio_event:Speech_Noise",
        ),
    ),
)
def test_parser_namespaces_upstream_tokens_outside_public_typed_allowlists(
    emotion: str,
    event: str,
    expected_emotion: str,
    expected_event: str,
) -> None:
    result = _decode(
        _raw_text(emotion=emotion, event=event),
        expected_count=1,
    )[0]

    assert result.annotations == RichAnnotations(
        emotion=expected_emotion,
        audio_event=expected_event,
    )


def test_parser_fixed_itn_does_not_change_controlled_body() -> None:
    assert (
        _decode(
            _raw_text(itn="withitn", body="Keep punctuation!"),
            expected_count=1,
        )[0].text
        == "Keep punctuation!"
    )


@pytest.mark.parametrize(
    ("emotion", "event", "expected_emotion", "expected_event"),
    (
        (
            "EMO_UNKNOWN",
            "Event_UNK",
            "unknown:sensevoice:emotion:EMO_UNKNOWN",
            "unknown:sensevoice:audio_event:Event_UNK",
        ),
        (
            "CALM",
            "Doorbell",
            "unknown:sensevoice:emotion:CALM",
            "unknown:sensevoice:audio_event:Doorbell",
        ),
        (
            "Doorbell",
            "AmbientNoise",
            "unknown:sensevoice:emotion:Doorbell",
            "unknown:sensevoice:audio_event:AmbientNoise",
        ),
    ),
)
def test_parser_preserves_legal_unknown_tags_in_bounded_namespaces(
    emotion: str,
    event: str,
    expected_emotion: str,
    expected_event: str,
) -> None:
    result = _decode(
        _raw_text(emotion=emotion, event=event),
        expected_count=1,
    )[0]

    assert result.annotations == RichAnnotations(
        emotion=expected_emotion,
        audio_event=expected_event,
    )


@pytest.mark.parametrize(
    ("slot", "raw_tag"),
    (
        ("emotion", "Calm-Quiet"),
        ("emotion", "Calm/Quiet"),
        ("emotion", "1CALM"),
        ("emotion", "_CALM"),
        ("event", "Door-Bell"),
        ("event", "Door/Bell"),
        ("event", "1CALM"),
        ("event", "_CALM"),
    ),
)
def test_parser_rejects_unsafe_unknown_raw_tags(
    slot: str,
    raw_tag: str,
) -> None:
    raw = _raw_text(emotion=raw_tag) if slot == "emotion" else _raw_text(event=raw_tag)

    with pytest.raises(PipelineError) as caught:
        _decode(raw, expected_count=1)

    assert caught.value.code == "invalid_model_output"


@pytest.mark.parametrize(
    ("slot", "known_wrong_role"),
    (
        ("emotion", "zh"),
        ("emotion", "withitn"),
        ("event", "zh"),
        ("event", "withitn"),
    ),
)
def test_parser_rejects_language_or_itn_tokens_in_rich_slots(
    slot: str,
    known_wrong_role: str,
) -> None:
    raw = (
        _raw_text(emotion=known_wrong_role)
        if slot == "emotion"
        else _raw_text(event=known_wrong_role)
    )

    with pytest.raises(PipelineError) as caught:
        _decode(raw, expected_count=1)

    assert caught.value.code == "invalid_model_output"


def test_parser_preserves_isolated_control_terminator_in_body() -> None:
    result = _decode(
        _raw_text(body="literal |> text"),
        expected_count=1,
    )[0]

    assert result.text == "literal |> text"


def test_parser_rejects_isolated_control_opener_in_body() -> None:
    with pytest.raises(PipelineError) as caught:
        _decode(
            _raw_text(body="literal <| text"),
            expected_count=1,
        )

    assert caught.value.code == "invalid_model_output"


def test_parser_preserves_outer_whitespace_for_projection_owner() -> None:
    result = _decode(
        _raw_text(body=" \tcontrolled text. \n"),
        expected_count=1,
    )[0]

    assert result.text == " \tcontrolled text. \n"


@pytest.mark.parametrize("language", ("auto", "zh"))
@pytest.mark.parametrize("body", ("", ".", " \t.\n"))
def test_parser_decodes_exact_nospeech_sentinel_without_dropping_unknowns(
    body: str,
    language: str,
) -> None:
    raw = [
        {
            **CAPTURED_NOSPEECH,
            "text": (f"<|nospeech|><|EMO_UNKNOWN|><|Event_UNK|><|withitn|>{body}"),
        }
    ]

    assert _decode(raw, expected_count=1, language=language) == (
        AsrResult(
            text="",
            language=None,
            annotations=RichAnnotations(
                emotion="unknown:sensevoice:emotion:EMO_UNKNOWN",
                audio_event=("unknown:sensevoice:audio_event:Event_UNK"),
            ),
        ),
    )


@pytest.mark.parametrize(
    "raw_result",
    (
        None,
        {},
        (),
        [None],
        [{}],
        [{"text": None}],
        [{"text": 1}],
    ),
    ids=(
        "none_envelope",
        "mapping_envelope",
        "tuple_envelope",
        "non_mapping_item",
        "missing_text",
        "none_text",
        "integer_text",
    ),
)
def test_parser_rejects_malformed_raw_envelope(raw_result: object) -> None:
    with pytest.raises(PipelineError) as caught:
        _decode(raw_result, expected_count=1)

    assert caught.value.code == "invalid_model_output"


@pytest.mark.parametrize(
    "raw_result",
    (
        _raw_text(body="<|HAPPY|>"),
        _raw_text(body="text <|Laughter|> leaked"),
        _raw_text(emotion="BGM"),
        _raw_text(event="HAPPY"),
        _raw_text(language="fr"),
        _raw_text(itn="woitn"),
        _raw_text(itn="maybeitn"),
        [{"text": ("<|zh|><|NEUTRAL|><|Speech|><|withitn|><|HAPPY|>text")}],
        [{"text": "<|zh|><|NEUTRAL|><|Speech|>text"}],
        [{"text": "<|zh|><|NEUTRAL|><|Speech|><|withitn>text"}],
    ),
    ids=(
        "body_is_control_token",
        "body_leaks_control_token",
        "known_event_in_emotion_slot",
        "known_emotion_in_event_slot",
        "unknown_language",
        "disabled_woitn",
        "unknown_itn",
        "fifth_prefix_token",
        "missing_itn_slot",
        "truncated_delimiter",
    ),
)
def test_parser_rejects_wrong_role_or_leaked_control_tokens(
    raw_result: object,
) -> None:
    with pytest.raises(PipelineError) as caught:
        _decode(raw_result, expected_count=1)

    assert caught.value.code == "invalid_model_output"
    assert "controlled text" not in str(caught.value)


@pytest.mark.parametrize(
    ("emotion", "event", "itn", "body"),
    (
        ("NEUTRAL", "Event_UNK", "withitn", "."),
        ("EMO_UNKNOWN", "Speech", "withitn", "."),
        ("EMO_UNKNOWN", "Event_UNK", "woitn", "."),
        ("EMO_UNKNOWN", "Event_UNK", "withitn", "speech"),
        ("EMO_UNKNOWN", "Event_UNK", "withitn", ".."),
    ),
)
def test_parser_rejects_noncanonical_nospeech_combinations(
    emotion: str,
    event: str,
    itn: str,
    body: str,
) -> None:
    with pytest.raises(PipelineError) as caught:
        _decode(
            _raw_text(
                language="nospeech",
                emotion=emotion,
                event=event,
                itn=itn,
                body=body,
            ),
            expected_count=1,
        )

    assert caught.value.code == "invalid_model_output"


@pytest.mark.parametrize("actual_count", (0, 1, 3))
def test_parser_rejects_batch_cardinality_mismatch(actual_count: int) -> None:
    raw_result = [
        CAPTURED_SPEECH_A,
        CAPTURED_SPEECH_B,
        CAPTURED_NOSPEECH,
    ][:actual_count]

    with pytest.raises(PipelineError) as caught:
        _decode(raw_result, expected_count=2)

    assert caught.value.code == "invalid_model_output"


def test_parser_rejects_the_whole_batch_when_one_item_is_malformed() -> None:
    with pytest.raises(PipelineError) as caught:
        _decode(
            [
                CAPTURED_SPEECH_A,
                {"key": "duplicate", "text": "<|en|>malformed Beta?"},
            ],
            expected_count=2,
        )

    assert caught.value.code == "invalid_model_output"


def test_batch_adapter_is_positional_with_duplicate_upstream_keys() -> None:
    model = FakeAutoModel([CAPTURED_SPEECH_A, CAPTURED_SPEECH_B])
    adapter = _adapter(model)
    first = np.array([-32_768, -1, 0, 1, 32_767], dtype=np.int16)
    second = np.arange(17, dtype=np.int16)
    originals = (first.copy(), second.copy())

    results = adapter.transcribe_batch(  # type: ignore[attr-defined]
        (first, second),
        language="auto",
    )

    assert [result.text for result in results] == ["Alpha, 世界!", "Beta?"]
    assert len(model.calls) == 1
    call = model.calls[0]
    assert set(call) == {
        "input",
        "language",
        "use_itn",
        "batch_size",
        "ban_emo_unk",
    }
    assert call["language"] == "auto"
    assert call["use_itn"] is True
    assert call["batch_size"] == 2
    assert call["ban_emo_unk"] is False
    normalized = call["input"]
    assert isinstance(normalized, list)
    assert len(normalized) == 2
    for actual, source, original in zip(
        normalized,
        (first, second),
        originals,
        strict=True,
    ):
        assert isinstance(actual, np.ndarray)
        assert actual.dtype == np.float32
        assert actual.flags.c_contiguous
        np.testing.assert_array_equal(
            actual,
            source.astype(np.float32) / np.float32(32_768.0),
        )
        np.testing.assert_array_equal(source, original)


def test_batch_adapter_rejects_upstream_cardinality_mismatch() -> None:
    model = FakeAutoModel([CAPTURED_SPEECH_A])
    adapter = _adapter(model)

    with pytest.raises(PipelineError) as caught:
        adapter.transcribe_batch(  # type: ignore[attr-defined]
            (
                np.zeros(4, dtype=np.int16),
                np.ones(4, dtype=np.int16),
            ),
            language="auto",
        )

    assert caught.value.code == "invalid_model_output"
    assert len(model.calls) == 1


def test_batch_adapter_passes_explicit_language_to_model_and_parser() -> None:
    model = FakeAutoModel([CAPTURED_SPEECH_A])
    adapter = _adapter(model)

    result = adapter.transcribe_batch(  # type: ignore[attr-defined]
        (np.zeros(4, dtype=np.int16),),
        language="zh",
    )

    assert result[0].language == "zh"
    assert model.calls[0]["language"] == "zh"


def test_batch_adapter_accepts_480000_and_rejects_480001_before_upstream() -> None:
    accepted_model = FakeAutoModel([CAPTURED_SPEECH_A])
    accepted = _adapter(accepted_model)
    accepted.transcribe_batch(  # type: ignore[attr-defined]
        (np.zeros(480_000, dtype=np.int16),),
        language="auto",
    )

    assert len(accepted_model.calls) == 1

    rejected_model = FakeAutoModel([CAPTURED_SPEECH_A])
    rejected = _adapter(rejected_model)
    with pytest.raises(PipelineError) as caught:
        rejected.transcribe_batch(  # type: ignore[attr-defined]
            (np.zeros(480_001, dtype=np.int16),),
            language="auto",
        )

    assert caught.value.code == "invalid_audio"
    assert rejected_model.calls == []


def test_batch_adapter_validates_every_item_before_calling_upstream() -> None:
    model = FakeAutoModel([CAPTURED_SPEECH_A, CAPTURED_SPEECH_B])
    lane = RecordingInferenceLane()
    adapter = funasr_adapter.FunAsrSenseVoiceBatchAdapter(
        model,
        inference_lane=lane,
    )
    invalid = np.zeros(4, dtype=np.int16)[::2]

    with pytest.raises(PipelineError) as caught:
        adapter.transcribe_batch(  # type: ignore[attr-defined]
            (
                np.zeros(4, dtype=np.int16),
                invalid,
            ),
            language="auto",
        )

    assert caught.value.code == "invalid_audio"
    assert model.calls == []
    assert lane.operations == []


def test_empty_batch_returns_empty_without_calling_upstream() -> None:
    model = FakeAutoModel([])
    adapter = _adapter(model)

    assert (
        adapter.transcribe_batch((), language="auto")  # type: ignore[attr-defined]
        == ()
    )
    assert model.calls == []


def test_single_transcribe_delegates_to_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter(FakeAutoModel([]))
    pcm = np.arange(7, dtype=np.int16)
    expected = AsrResult(
        text="delegated",
        language="en",
        annotations=RichAnnotations("neutral", "speech"),
    )
    calls: list[tuple[tuple[np.ndarray, ...], str]] = []

    def fake_batch(
        pcms: tuple[np.ndarray, ...],
        *,
        language: str,
    ) -> tuple[AsrResult, ...]:
        calls.append((pcms, language))
        return (expected,)

    monkeypatch.setattr(adapter, "transcribe_batch", fake_batch)

    assert adapter.transcribe(pcm) is expected  # type: ignore[attr-defined]
    assert len(calls) == 1
    assert calls[0][0][0] is pcm
    assert len(calls[0][0]) == 1
    assert calls[0][1] == "auto"
