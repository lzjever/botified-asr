from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import numpy as np

from botified_asr.audio import BLOCK_SAMPLES, SAMPLE_RATE, Cancellation, DecodedBlock
from botified_asr.contracts import MAX_AUDIO_SAMPLES
from botified_asr.errors import PipelineError
from botified_asr.pipeline import (
    AudioFrontend,
    BufferedSpeechSegment,
    StreamingSpeechSegmenter,
    StreamingVadAdapter,
)
from botified_asr.speaker_profiles import (
    MAX_SPEAKER_SAMPLES,
    MIN_SPEAKER_SAMPLES,
    SpeakerEmbedding,
    SpeakerEmbeddingReplacement,
)
from botified_asr.speakers import (
    SPEAKER_EMBEDDING_NORM_TOLERANCE,
    SpeakerEmbeddingAdapter,
    SpeakerEmbeddingPolicy,
    SpeakerEmbeddingWindow,
)

MIN_SPEECH_SAMPLES = 5 * SAMPLE_RATE
MAX_SPEECH_SAMPLES = 30 * SAMPLE_RATE


@dataclass(frozen=True, slots=True)
class SpeakerEnrollmentPolicy:
    consistency_threshold: float

    def __post_init__(self) -> None:
        value = self.consistency_threshold
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                "speaker enrollment consistency threshold must be a real number"
            )
        try:
            threshold = float(value)
        except (OverflowError, ValueError) as error:
            raise ValueError(
                "speaker enrollment consistency threshold must be finite"
            ) from error
        if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
            raise ValueError(
                "speaker enrollment consistency threshold must be between -1 and 1"
            )
        object.__setattr__(
            self,
            "consistency_threshold",
            0.0 if threshold == 0.0 else threshold,
        )


class SpeakerEnrollmentProcessor:
    def __init__(
        self,
        frontend: AudioFrontend,
        vad_adapter: StreamingVadAdapter,
        speaker_adapter: SpeakerEmbeddingAdapter,
        embedding_policy: SpeakerEmbeddingPolicy,
        enrollment_policy: SpeakerEnrollmentPolicy,
    ) -> None:
        if not callable(getattr(frontend, "probe", None)) or not callable(
            getattr(frontend, "decode", None)
        ):
            raise TypeError("speaker enrollment audio frontend is invalid")
        if not callable(getattr(vad_adapter, "generate", None)):
            raise TypeError("speaker enrollment VAD adapter is invalid")
        if not callable(getattr(speaker_adapter, "embed_windows", None)):
            raise TypeError("speaker enrollment embedding adapter is invalid")
        if type(embedding_policy) is not SpeakerEmbeddingPolicy:
            raise TypeError("speaker embedding policy is invalid")
        if embedding_policy.sample_rate != SAMPLE_RATE:
            raise ValueError("speaker embedding sample rate is unsupported")
        if type(enrollment_policy) is not SpeakerEnrollmentPolicy:
            raise TypeError("speaker enrollment policy is invalid")
        self._frontend = frontend
        self._vad_adapter = vad_adapter
        self._speaker_adapter = speaker_adapter
        self._embedding_policy = embedding_policy
        self._enrollment_policy = enrollment_policy

    def process(
        self,
        sample_paths: tuple[Path, ...],
        cancellation: Cancellation,
        *,
        effective_max_audio_samples: int,
    ) -> SpeakerEmbeddingReplacement:
        if type(effective_max_audio_samples) is not int:
            raise TypeError("effective maximum audio samples must be an integer")
        if not 1 <= effective_max_audio_samples <= MAX_AUDIO_SAMPLES:
            raise ValueError("effective maximum audio samples are out of range")
        if (
            type(sample_paths) is not tuple
            or not MIN_SPEAKER_SAMPLES <= len(sample_paths) <= MAX_SPEAKER_SAMPLES
            or any(not isinstance(path, Path) for path in sample_paths)
        ):
            raise PipelineError(
                "invalid_speaker_samples",
                "Speaker enrollment requires 2 to 5 samples",
            )
        if type(cancellation) is not Cancellation:
            raise TypeError("speaker enrollment cancellation is invalid")

        sample_centroids = tuple(
            self._process_sample(
                path,
                cancellation,
                effective_max_audio_samples=effective_max_audio_samples,
            )
            for path in sample_paths
        )
        self._validate_consistency(sample_centroids)
        embedding = _embedding_from_mean(
            sample_centroids,
            dimension=self._embedding_policy.embedding_dimension,
        )
        return SpeakerEmbeddingReplacement(
            embedding=embedding,
            embedding_model_id=self._embedding_policy.model_id,
            embedding_model_revision=self._embedding_policy.model_revision,
            embedding_dimension=self._embedding_policy.embedding_dimension,
            embedding_policy_fingerprint=self._embedding_policy.fingerprint,
            sample_count=len(sample_centroids),
        )

    def _process_sample(
        self,
        path: Path,
        cancellation: Cancellation,
        *,
        effective_max_audio_samples: int,
    ) -> SpeakerEmbedding:
        _check_cancellation(cancellation)
        probe = self._frontend.probe(path, cancellation)
        _check_cancellation(cancellation)
        if probe.duration_seconds > effective_max_audio_samples / SAMPLE_RATE:
            raise _audio_too_long_error()
        blocks = self._frontend.decode(path, probe, cancellation)
        try:
            speech_pcm = self._extract_speech_pcm(
                blocks,
                cancellation,
                effective_max_audio_samples=effective_max_audio_samples,
            )
        finally:
            blocks.close()

        if not len(speech_pcm):
            raise PipelineError(
                "no_speech",
                "Speaker sample contains no speech",
            )
        if not MIN_SPEECH_SAMPLES <= len(speech_pcm) <= MAX_SPEECH_SAMPLES:
            raise PipelineError(
                "invalid_speaker_sample_duration",
                "Speaker sample must contain 5 to 30 seconds of speech",
            )

        _check_cancellation(cancellation)
        try:
            windows = self._speaker_adapter.embed_windows(speech_pcm)
        except PipelineError as error:
            if error.code != "invalid_model_output":
                raise
            raise _invalid_embedding_error() from error
        _check_cancellation(cancellation)
        return _embedding_from_windows(
            windows,
            policy=self._embedding_policy,
        )

    def _extract_speech_pcm(
        self,
        blocks: object,
        cancellation: Cancellation,
        *,
        effective_max_audio_samples: int,
    ) -> np.ndarray:
        segmenter = StreamingSpeechSegmenter(self._vad_adapter)
        speech_parts: list[np.ndarray] = []
        speech_samples = 0
        decoded_end_sample = 0
        iterator = iter(blocks)  # type: ignore[arg-type]
        try:
            current = next(iterator)
        except StopIteration:
            return np.empty(0, dtype=np.int16)

        while True:
            try:
                following = next(iterator)
            except StopIteration:
                following = None
            is_final = following is None
            if not isinstance(current, DecodedBlock):
                raise PipelineError(
                    "invalid_audio",
                    "Decoded audio returned an invalid block",
                )
            if current.start_sample != decoded_end_sample:
                raise PipelineError(
                    "invalid_audio",
                    "Decoded audio blocks are not contiguous",
                )
            if not is_final and len(current.pcm) < BLOCK_SAMPLES:
                raise PipelineError(
                    "invalid_audio",
                    "Decoded audio has a non-final short block",
                )
            next_decoded_end_sample = decoded_end_sample + len(current.pcm)
            if next_decoded_end_sample > effective_max_audio_samples:
                raise _audio_too_long_error()

            _check_cancellation(cancellation)
            emitted = segmenter.process(current, is_final=is_final)
            _check_cancellation(cancellation)
            if type(emitted) is not tuple:
                raise PipelineError(
                    "invalid_model_output",
                    "Speech segmenter returned an invalid result",
                )
            for segment in emitted:
                canonical_pcm = _canonical_speech_pcm(segment)
                speech_samples += len(canonical_pcm)
                if speech_samples > MAX_SPEECH_SAMPLES:
                    raise PipelineError(
                        "invalid_speaker_sample_duration",
                        "Speaker sample must contain 5 to 30 seconds of speech",
                    )
                speech_parts.append(canonical_pcm)

            decoded_end_sample = next_decoded_end_sample
            if is_final:
                break
            current = following

        if not speech_parts:
            return np.empty(0, dtype=np.int16)
        if len(speech_parts) == 1:
            return speech_parts[0]
        return np.ascontiguousarray(np.concatenate(speech_parts), dtype=np.int16)

    def _validate_consistency(
        self,
        sample_centroids: tuple[SpeakerEmbedding, ...],
    ) -> None:
        threshold = self._enrollment_policy.consistency_threshold
        vectors = tuple(
            embedding.as_numpy().astype(np.float64, copy=False)
            for embedding in sample_centroids
        )
        for left_index, left in enumerate(vectors):
            for right in vectors[left_index + 1 :]:
                similarity = float(np.dot(left, right))
                if not math.isfinite(similarity):
                    raise _invalid_embedding_error()
                if similarity < threshold:
                    raise PipelineError(
                        "speaker_samples_inconsistent",
                        "Speaker samples are inconsistent",
                    )


def _canonical_speech_pcm(segment: object) -> np.ndarray:
    if not isinstance(segment, BufferedSpeechSegment):
        raise PipelineError(
            "invalid_model_output",
            "Speech segmenter returned an invalid segment",
        )
    canonical_start = segment.span.start_sample - segment.pcm_start_sample
    canonical_end = segment.span.end_sample - segment.pcm_start_sample
    canonical_pcm = np.ascontiguousarray(
        segment.pcm[canonical_start:canonical_end],
        dtype=np.int16,
    )
    if (
        canonical_pcm.ndim != 1
        or not canonical_pcm.flags.c_contiguous
        or len(canonical_pcm) != segment.span.end_sample - segment.span.start_sample
        or not len(canonical_pcm)
    ):
        raise PipelineError(
            "invalid_model_output",
            "Speech segmenter returned an invalid segment",
        )
    return canonical_pcm


def _embedding_from_windows(
    windows: object,
    *,
    policy: SpeakerEmbeddingPolicy,
) -> SpeakerEmbedding:
    if type(windows) is not tuple or not windows:
        raise _invalid_embedding_error()

    embeddings: list[SpeakerEmbedding] = []
    previous_start: int | None = None
    previous_end: int | None = None
    try:
        for window in windows:
            if not isinstance(window, SpeakerEmbeddingWindow):
                raise ValueError
            if (
                type(window.start_sample) is not int
                or type(window.end_sample) is not int
                or window.start_sample < 0
                or window.end_sample <= window.start_sample
                or window.end_sample - window.start_sample > policy.window_samples
                or previous_start is not None
                and window.start_sample <= previous_start
                or previous_end is not None
                and window.end_sample <= previous_end
            ):
                raise ValueError
            embeddings.append(
                SpeakerEmbedding.from_numpy(
                    window.embedding,
                    dimension=policy.embedding_dimension,
                )
            )
            previous_start = window.start_sample
            previous_end = window.end_sample
    except (TypeError, ValueError, OverflowError) as error:
        raise _invalid_embedding_error() from error
    return _embedding_from_mean(
        tuple(embeddings),
        dimension=policy.embedding_dimension,
    )


def _embedding_from_mean(
    embeddings: tuple[SpeakerEmbedding, ...],
    *,
    dimension: int,
) -> SpeakerEmbedding:
    try:
        values = np.stack(
            tuple(
                embedding.as_numpy().astype(np.float64, copy=False)
                for embedding in embeddings
            )
        )
        mean = np.mean(values, axis=0, dtype=np.float64)
        norm = float(np.linalg.norm(mean))
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError
        normalized = np.ascontiguousarray(mean / norm, dtype=np.float32)
        normalized_norm = float(
            np.linalg.norm(normalized.astype(np.float64, copy=False))
        )
        if (
            not math.isfinite(normalized_norm)
            or normalized_norm <= 0.0
            or not math.isclose(
                normalized_norm,
                1.0,
                rel_tol=0.0,
                abs_tol=SPEAKER_EMBEDDING_NORM_TOLERANCE,
            )
        ):
            raise ValueError
        return SpeakerEmbedding.from_numpy(normalized, dimension=dimension)
    except (TypeError, ValueError, OverflowError) as error:
        raise _invalid_embedding_error() from error


def _check_cancellation(cancellation: Cancellation) -> None:
    if cancellation.cancelled:
        raise PipelineError(
            "cancelled",
            "Speaker enrollment was cancelled",
        )


def _invalid_embedding_error() -> PipelineError:
    return PipelineError(
        "invalid_speaker_embedding",
        "Speaker embedding is invalid",
    )


def _audio_too_long_error() -> PipelineError:
    return PipelineError(
        "audio_too_long",
        "Decoded audio exceeds the configured duration limit",
    )
