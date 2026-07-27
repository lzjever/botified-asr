from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from botified_asr.audio import Cancellation
from botified_asr.contracts import MAX_AUDIO_SAMPLES, CanonicalOptions
from botified_asr.pipeline import (
    CanonicalJsonlSegmentSink,
    ProgressSink,
    SegmentSink,
)
from botified_asr.result_artifact import (
    CanonicalJsonlReader,
    Projection,
    ResultProjector,
)
from botified_asr.storage import ArtifactRef, ReservedByteWriter, Storage


class TranscriptionProcessor(Protocol):
    def process(
        self,
        input_path: Path,
        canonical_options: CanonicalOptions,
        cancellation: Cancellation,
        progress_sink: ProgressSink,
        segment_sink: SegmentSink,
    ) -> object: ...


class ProjectionBuilder(Protocol):
    def prepare(
        self,
        reader: CanonicalJsonlReader,
        options: CanonicalOptions,
        total_samples: int,
    ) -> Projection: ...


class StorageArtifactByteWriter:
    def __init__(
        self,
        storage: Storage,
        writer: ReservedByteWriter,
    ) -> None:
        self._storage = storage
        self._writer = writer
        self._state = "OPEN"
        self._sealed_ref: ArtifactRef | None = None

    @property
    def state(self) -> str:
        return self._state

    @property
    def sealed_ref(self) -> ArtifactRef | None:
        return self._sealed_ref

    def write(self, payload: bytes) -> None:
        self._require_open("write")
        self._storage.append_artifact(self._writer, payload)

    def seal(self) -> ArtifactRef:
        self._require_open("seal")
        ref = self._storage.seal_artifact(self._writer)
        self._sealed_ref = ref
        self._state = "SEALED"
        return ref

    def abort(self) -> None:
        if self._state == "ABORTED":
            return
        self._require_open("abort")
        self._storage.abort_artifact(self._writer)
        self._state = "ABORTED"

    def _discard(self) -> None:
        if self._state in {"ABORTED", "RELEASED"}:
            return
        if self._state == "OPEN":
            self._storage.abort_artifact(self._writer)
            self._state = "ABORTED"
            return
        if self._sealed_ref is None:
            raise RuntimeError("sealed artifact writer has no reference")
        self._storage.release_artifact(self._sealed_ref)
        self._state = "RELEASED"

    def _require_open(self, operation: str) -> None:
        if self._state != "OPEN":
            raise RuntimeError(
                f"cannot {operation} artifact writer in {self._state} state"
            )


class ProgressAccumulator:
    def __init__(self) -> None:
        self._last_processed: int | None = None

    def update(
        self,
        *,
        processed_samples: int,
        total_samples: int | None,
    ) -> None:
        _validate_sample_count(
            processed_samples,
            name="processed sample count",
        )
        if total_samples is not None:
            _validate_sample_count(
                total_samples,
                name="total sample count",
            )
        if (
            self._last_processed is not None
            and processed_samples < self._last_processed
        ):
            raise ValueError("processed sample count must be monotonic")
        self._last_processed = processed_samples

    def finish(self) -> int:
        if self._last_processed is None:
            raise RuntimeError("progress did not report processed samples")
        return self._last_processed


def _validate_sample_count(value: int, *, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value <= MAX_AUDIO_SAMPLES:
        raise ValueError(f"{name} is outside the allowed range")


class PreparedSyncResponse:
    def __init__(
        self,
        projection: Projection,
        artifact_writer: StorageArtifactByteWriter,
    ) -> None:
        self.content_type = projection.content_type
        self._body_factory = projection.body_factory
        self._artifact_writer = artifact_writer
        self._body_claimed = False
        self._closed = False
        self._close_lock = threading.Lock()

    def iter_body(self) -> Iterator[bytes]:
        with self._close_lock:
            if self._closed:
                raise RuntimeError("prepared response is closed")
            if self._body_claimed:
                raise RuntimeError("prepared response body was already requested")
            self._body_claimed = True
        try:
            iterator = iter(self._body_factory())
        except BaseException:
            self.close()
            raise

        def body() -> Iterator[bytes]:
            try:
                yield from iterator
            finally:
                try:
                    close = getattr(iterator, "close", None)
                    if close is not None:
                        close()
                finally:
                    self.close()

        return body()

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._artifact_writer._discard()
            self._closed = True


def prepare_sync_transcription(
    storage: Storage,
    processor: TranscriptionProcessor,
    input_path: Path,
    owner_id: str,
    options: CanonicalOptions,
    cancellation: Cancellation,
    projector: ProjectionBuilder | None = None,
) -> PreparedSyncResponse:
    reserved_writer = storage.begin_artifact(
        "segment_jsonl",
        owner_kind="sync",
        owner_id=owner_id,
    )
    writer = StorageArtifactByteWriter(storage, reserved_writer)
    transferred = False
    try:
        sink = CanonicalJsonlSegmentSink(writer)
        progress = ProgressAccumulator()
        returned_ref = processor.process(
            input_path,
            options,
            cancellation,
            progress,
            sink,
        )
        if writer.sealed_ref is None or returned_ref is not writer.sealed_ref:
            raise RuntimeError("processor returned an unexpected artifact reference")
        total_samples = progress.finish()
        artifact_path = storage.resolve_artifact(writer.sealed_ref)
        reader = CanonicalJsonlReader(artifact_path)
        projection_builder = ResultProjector() if projector is None else projector
        projection = projection_builder.prepare(
            reader,
            options,
            total_samples,
        )
        response = PreparedSyncResponse(projection, writer)
        transferred = True
        return response
    finally:
        if not transferred:
            writer._discard()
