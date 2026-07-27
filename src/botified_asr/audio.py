from __future__ import annotations

import json
import math
import os
import queue
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

SAMPLE_RATE = 16_000
BLOCK_SAMPLES = 9_600
BLOCK_BYTES = BLOCK_SAMPLES * 2
MAX_PROBE_OUTPUT_BYTES = 64 * 1024
MAX_DIAGNOSTIC_BYTES = 64 * 1024
MIN_DECODE_WALL_SECONDS = 30.0
MAX_DECODE_WALL_SECONDS = 12 * 60 * 60.0
PROCESS_STOP_GRACE_SECONDS = 5.0
SUPPORTED_DEMUXERS = {
    "3g2",
    "3gp",
    "flac",
    "m4a",
    "matroska",
    "mj2",
    "mov",
    "mp3",
    "mp4",
    "mpeg",
    "ogg",
    "wav",
    "webm",
}

ProcessFactory = Callable[..., Any]


class ProcessGroupController(Protocol):
    def is_alive(self, process: Any) -> bool: ...

    def signal(self, process: Any, sent_signal: int) -> None: ...


class AudioError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class AudioCancelled(AudioError):
    def __init__(self) -> None:
        super().__init__("cancelled", "Audio processing was cancelled")


class Cancellation:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


@dataclass(frozen=True)
class MediaProbe:
    duration_seconds: float
    format_name: str


@dataclass(frozen=True)
class DecodedBlock:
    start_sample: int
    pcm: np.ndarray

    def __post_init__(self) -> None:
        if self.start_sample < 0:
            raise ValueError("decoded block start_sample must be non-negative")
        if (
            self.pcm.dtype != np.int16
            or self.pcm.ndim != 1
            or not self.pcm.flags.c_contiguous
            or not 1 <= len(self.pcm) <= BLOCK_SAMPLES
        ):
            raise ValueError(
                "decoded block must be contiguous one-dimensional int16 PCM"
            )


def probe_media(
    path: str | Path,
    *,
    cancellation: Cancellation | None = None,
    process_factory: ProcessFactory = subprocess.Popen,
    timeout_seconds: float = 10.0,
    process_group_controller: ProcessGroupController | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> MediaProbe:
    media_path = Path(path)
    argv = [
        "ffprobe",
        "-v",
        "error",
        "-protocol_whitelist",
        "file,pipe",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=duration:format=duration,format_name",
        "-of",
        "json",
        str(media_path),
    ]
    process = _spawn(process_factory, argv)
    stdout_reader = _BoundedPipeReader(process.stdout, MAX_PROBE_OUTPUT_BYTES)
    stderr_reader = _BoundedPipeReader(process.stderr, MAX_DIAGNOSTIC_BYTES)
    pump = _ProcessPump(
        process,
        (stdout_reader, stderr_reader),
        process_group_controller or _DEFAULT_PROCESS_GROUP_CONTROLLER,
    )
    completed = False
    deadline = monotonic() + timeout_seconds
    try:
        pump.start()
        returncode = pump.wait_process(
            deadline,
            cancellation=cancellation,
            timeout_code="audio_probe_timeout",
            timeout_message="Audio metadata inspection timed out",
            monotonic=monotonic,
        )
        pump.wait_readers(
            deadline,
            cancellation=cancellation,
            timeout_code="audio_probe_timeout",
            timeout_message="Audio metadata inspection timed out",
            monotonic=monotonic,
        )
        pump.close_streams()
        completed = True
    finally:
        if not completed:
            pump.stop()

    if (
        returncode != 0
        or stdout_reader.overflow
        or stdout_reader.error is not None
        or stderr_reader.error is not None
    ):
        raise AudioError("invalid_audio", "Audio metadata could not be read")
    try:
        payload = json.loads(stdout_reader.output)
        streams = payload.get("streams", [])
        if not streams:
            raise ValueError
        format_name = str(payload.get("format", {}).get("format_name", ""))
        demuxers = set(format_name.split(","))
        if not demuxers or not demuxers.issubset(SUPPORTED_DEMUXERS):
            raise ValueError
        raw_duration = streams[0].get("duration")
        if raw_duration in (None, "N/A"):
            raw_duration = payload.get("format", {}).get("duration")
        duration = float(raw_duration)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AudioError("invalid_audio", "Audio metadata could not be read") from exc
    return MediaProbe(duration, format_name)


def decode_audio(
    path: str | Path,
    probe: MediaProbe,
    *,
    cancellation: Cancellation | None = None,
    process_factory: ProcessFactory = subprocess.Popen,
    monotonic: Callable[[], float] = time.monotonic,
    process_group_controller: ProcessGroupController | None = None,
) -> Iterator[DecodedBlock]:
    wall_budget = _decode_wall_budget(probe.duration_seconds)
    media_path = Path(path)
    argv = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-protocol_whitelist",
        "file,pipe",
        "-i",
        str(media_path),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "pipe:1",
    ]
    process = _spawn(process_factory, argv)
    stdout_reader = _QueuedPipeReader(process.stdout)
    stderr_reader = _BoundedPipeReader(process.stderr, MAX_DIAGNOSTIC_BYTES)
    pump = _ProcessPump(
        process,
        (stdout_reader, stderr_reader),
        process_group_controller or _DEFAULT_PROCESS_GROUP_CONTROLLER,
    )
    deadline = monotonic() + wall_budget
    pending = bytearray()
    start_sample = 0
    completed = False
    try:
        pump.start()
        while True:
            if cancellation is not None and cancellation.cancelled:
                raise AudioCancelled()
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise AudioError("audio_decode_timeout", "Audio decoding timed out")
            try:
                event = stdout_reader.events.get(timeout=min(0.05, remaining))
            except queue.Empty:
                continue
            if event.kind == "eof":
                break
            if event.kind == "read_error":
                raise AudioError("invalid_audio", "Audio could not be decoded")
            pending.extend(event.data)
            while len(pending) >= BLOCK_BYTES:
                block_bytes = bytes(pending[:BLOCK_BYTES])
                del pending[:BLOCK_BYTES]
                pcm = np.frombuffer(block_bytes, dtype="<i2").astype(
                    np.int16, copy=True
                )
                yield DecodedBlock(start_sample, pcm)
                start_sample += BLOCK_SAMPLES

        returncode = pump.wait_process(
            deadline,
            cancellation=cancellation,
            timeout_code="audio_decode_timeout",
            timeout_message="Audio decoding timed out",
            monotonic=monotonic,
        )
        pump.wait_readers(
            deadline,
            cancellation=cancellation,
            timeout_code="audio_decode_timeout",
            timeout_message="Audio decoding timed out",
            monotonic=monotonic,
        )
        pump.close_streams()
        completed = True
        if (
            returncode != 0
            or stdout_reader.error is not None
            or stderr_reader.error is not None
        ):
            raise AudioError("invalid_audio", "Audio could not be decoded")
        if len(pending) % 2:
            raise AudioError("invalid_audio", "Audio could not be decoded")
        if pending:
            pcm = np.frombuffer(bytes(pending), dtype="<i2").astype(np.int16, copy=True)
            yield DecodedBlock(start_sample, pcm)
    finally:
        if not completed:
            pump.stop()


class FfmpegAudioFrontend:
    def __init__(
        self,
        *,
        process_factory: ProcessFactory = subprocess.Popen,
        process_group_controller: ProcessGroupController | None = None,
    ) -> None:
        self._process_factory = process_factory
        self._process_group_controller = process_group_controller

    def probe(
        self,
        input_path: Path,
        cancellation: Cancellation,
    ) -> MediaProbe:
        return probe_media(
            input_path,
            cancellation=cancellation,
            process_factory=self._process_factory,
            process_group_controller=self._process_group_controller,
        )

    def decode(
        self,
        input_path: Path,
        probe: MediaProbe,
        cancellation: Cancellation,
    ) -> Iterator[DecodedBlock]:
        return decode_audio(
            input_path,
            probe,
            cancellation=cancellation,
            process_factory=self._process_factory,
            process_group_controller=self._process_group_controller,
        )


def _spawn(process_factory: ProcessFactory, argv: list[str]) -> Any:
    try:
        return process_factory(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=True,
        )
    except OSError as exc:
        raise AudioError(
            "audio_tool_unavailable", "Audio processing is unavailable"
        ) from exc


@dataclass(frozen=True)
class _PipeEvent:
    kind: str
    data: bytes = b""


class _PipeReader:
    def __init__(self, stream: Any, target: Callable[[], None]) -> None:
        self.stream = stream
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=target, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float) -> None:
        self._thread.join(timeout=timeout)

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def stop(self) -> None:
        pass


class _BoundedPipeReader(_PipeReader):
    def __init__(self, stream: Any, limit: int) -> None:
        self.output = bytearray()
        self.overflow = False
        self._limit = limit
        super().__init__(stream, self._run)

    def _run(self) -> None:
        try:
            while True:
                fragment = self.stream.read(4096)
                if not fragment:
                    return
                remaining = self._limit - len(self.output)
                if remaining > 0:
                    self.output.extend(fragment[:remaining])
                if len(fragment) > max(remaining, 0):
                    self.overflow = True
        except (OSError, ValueError) as exc:
            self.error = exc


class _QueuedPipeReader(_PipeReader):
    def __init__(self, stream: Any) -> None:
        self.events: queue.Queue[_PipeEvent] = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        super().__init__(stream, self._run)

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                fragment = self.stream.read(BLOCK_BYTES)
                if not fragment:
                    self._emit(_PipeEvent("eof"))
                    return
                for offset in range(0, len(fragment), BLOCK_BYTES):
                    if not self._emit(
                        _PipeEvent(
                            "data",
                            fragment[offset : offset + BLOCK_BYTES],
                        )
                    ):
                        return
        except (OSError, ValueError) as exc:
            self.error = exc
            self._emit(_PipeEvent("read_error"))

    def _emit(self, event: _PipeEvent) -> bool:
        while not self._stop.is_set():
            try:
                self.events.put(event, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def stop(self) -> None:
        self._stop.set()


class _ProcessPump:
    def __init__(
        self,
        process: Any,
        readers: tuple[_PipeReader, ...],
        process_group_controller: ProcessGroupController,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.process = process
        self.readers = readers
        self.process_group_controller = process_group_controller
        self.monotonic = monotonic
        self.sleep = sleep
        self._started: list[_PipeReader] = []
        self._closed = False

    def start(self) -> None:
        try:
            for reader in self.readers:
                reader.start()
                self._started.append(reader)
        except BaseException:
            self.stop()
            raise

    def wait_process(
        self,
        deadline: float,
        *,
        cancellation: Cancellation | None,
        timeout_code: str,
        timeout_message: str,
        monotonic: Callable[[], float],
    ) -> int:
        while True:
            if cancellation is not None and cancellation.cancelled:
                raise AudioCancelled()
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise AudioError(timeout_code, timeout_message)
            try:
                return self.process.wait(timeout=min(0.05, remaining))
            except subprocess.TimeoutExpired:
                continue

    def wait_readers(
        self,
        deadline: float,
        *,
        cancellation: Cancellation | None,
        timeout_code: str,
        timeout_message: str,
        monotonic: Callable[[], float],
    ) -> None:
        for reader in self._started:
            while reader.alive:
                if cancellation is not None and cancellation.cancelled:
                    raise AudioCancelled()
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise AudioError(timeout_code, timeout_message)
                reader.join(min(0.05, remaining))

    def close_streams(self) -> None:
        if self._closed:
            return
        _close_process_streams(self.process)
        self._closed = True

    def stop(self) -> None:
        for reader in self._started:
            reader.stop()
        try:
            _terminate_and_reap(
                self.process,
                self.process_group_controller,
                monotonic=self.monotonic,
                sleep=self.sleep,
            )
        finally:
            self._closed = True
            deadline = self.monotonic() + PROCESS_STOP_GRACE_SECONDS
            for reader in self._started:
                while reader.alive:
                    remaining = deadline - self.monotonic()
                    if remaining <= 0:
                        break
                    reader.join(min(0.05, remaining))
            if any(reader.alive for reader in self._started):
                raise AudioError(
                    "audio_cleanup_failed",
                    "Audio process cleanup failed",
                )


def _decode_wall_budget(duration_seconds: float) -> float:
    if not math.isfinite(duration_seconds) or duration_seconds < 0:
        raise AudioError("invalid_audio", "Audio metadata is invalid")
    return min(
        MAX_DECODE_WALL_SECONDS,
        max(MIN_DECODE_WALL_SECONDS, duration_seconds * 2.0 + 10.0),
    )


def _terminate_and_reap(
    process: Any,
    process_group_controller: ProcessGroupController,
    *,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    try:
        if process_group_controller.is_alive(process):
            process_group_controller.signal(process, signal.SIGTERM)
            deadline = monotonic() + PROCESS_STOP_GRACE_SECONDS
            while process_group_controller.is_alive(process):
                remaining = deadline - monotonic()
                if remaining <= 0:
                    process_group_controller.signal(process, signal.SIGKILL)
                    break
                process.poll()
                sleep(min(0.05, remaining))
        process.wait()
    finally:
        _close_process_streams(process)


class _DefaultProcessGroupController:
    def is_alive(self, process: Any) -> bool:
        pid = getattr(process, "pid", None)
        if os.name == "posix" and pid is not None:
            try:
                os.killpg(pid, 0)
            except ProcessLookupError:
                return False
            except PermissionError:
                return True
            return True
        return process.poll() is None

    def signal(self, process: Any, sent_signal: int) -> None:
        pid = getattr(process, "pid", None)
        if os.name == "posix" and pid is not None:
            try:
                os.killpg(pid, sent_signal)
            except ProcessLookupError:
                pass
            return
        if sent_signal == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()


_DEFAULT_PROCESS_GROUP_CONTROLLER = _DefaultProcessGroupController()


def _close_process_streams(process: Any) -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(process, name, None)
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass
