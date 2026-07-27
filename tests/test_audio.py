from __future__ import annotations

import json
import signal
import subprocess
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from botified_asr.audio import (
    MAX_DIAGNOSTIC_BYTES,
    AudioCancelled,
    AudioError,
    Cancellation,
    MediaProbe,
    _BoundedPipeReader,
    _decode_wall_budget,
    _ProcessPump,
    decode_audio,
    probe_media,
)


class FakeStream:
    def __init__(self, fragments: list[bytes]) -> None:
        self.fragments = list(fragments)
        self.closed = False

    def read(self, _size: int = -1) -> bytes:
        if self.fragments:
            return self.fragments.pop(0)
        return b""

    def close(self) -> None:
        self.closed = True


class FakeProcess:
    def __init__(
        self,
        *,
        stdout: list[bytes] | None = None,
        stderr: list[bytes] | None = None,
        returncode: int | None = 0,
        timeout: bool = False,
        pid: int | None = None,
    ) -> None:
        self.stdout = FakeStream(stdout or [])
        self.stderr = FakeStream(stderr or [])
        self.returncode = returncode
        self.timeout = timeout
        if pid is not None:
            self.pid = pid
        self.terminated = 0
        self.killed = 0
        self.wait_timeouts: list[float | None] = []
        self.wait_called = threading.Event()

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        self.wait_called.set()
        if self.timeout and self.returncode is None:
            raise subprocess.TimeoutExpired("redacted", timeout)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.terminated += 1
        self.returncode = -15

    def kill(self) -> None:
        self.killed += 1
        self.returncode = -9


class ProcessFactory:
    def __init__(self, *processes: FakeProcess) -> None:
        self.processes = list(processes)
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> FakeProcess:
        self.calls.append((argv, kwargs))
        return self.processes.pop(0)


class GatedStream(FakeStream):
    def __init__(self, payload: bytes) -> None:
        super().__init__([])
        self.payload = payload
        self.read_started = threading.Event()
        self.release = threading.Event()
        self.returned = False

    def read(self, _size: int = -1) -> bytes:
        if self.returned:
            return b""
        self.read_started.set()
        self.release.wait(timeout=2.0)
        if self.closed:
            raise OSError("closed before drain")
        self.returned = True
        return self.payload

    def close(self) -> None:
        super().close()
        self.release.set()


class ReadErrorStream(FakeStream):
    def read(self, _size: int = -1) -> bytes:
        raise OSError("simulated pipe read failure")


class TrackingStream(FakeStream):
    def __init__(self, fragments: list[bytes]) -> None:
        super().__init__(fragments)
        self.eof = threading.Event()
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        fragment = super().read(size)
        if fragment:
            self.bytes_read += len(fragment)
        else:
            self.eof.set()
        return fragment


class FakeGroupController:
    def __init__(
        self,
        *,
        alive: bool = True,
        ignore_term: bool = False,
    ) -> None:
        self.alive = alive
        self.ignore_term = ignore_term
        self.signals: list[int] = []

    def is_alive(self, _process: Any) -> bool:
        return self.alive

    def signal(self, process: FakeProcess, sent_signal: int) -> None:
        self.signals.append(sent_signal)
        if sent_signal == signal.SIGTERM and not self.ignore_term:
            self.alive = False
            process.returncode = -signal.SIGTERM
        if sent_signal == signal.SIGKILL:
            self.alive = False
            process.returncode = -signal.SIGKILL


class AdvancingClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class PersistentReader:
    def __init__(self, clock: AdvancingClock) -> None:
        self.clock = clock
        self.join_timeouts: list[float] = []
        self.started = 0
        self.stopped = 0

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def join(self, timeout: float) -> None:
        self.join_timeouts.append(timeout)
        self.clock.advance(timeout)

    @property
    def alive(self) -> bool:
        return True


def test_probe_and_decode_use_hardened_argv_and_preserve_fragmented_pcm(
    tmp_path: Path,
) -> None:
    path = tmp_path / "input.wav"
    samples = np.arange(10_000, dtype=np.int16)
    pcm = samples.astype("<i2", copy=False).tobytes()
    probe_process = FakeProcess(
        stdout=[
            json.dumps(
                {
                    "streams": [{"duration": "0.625"}],
                    "format": {"duration": "0.625", "format_name": "wav"},
                }
            ).encode()
        ],
        stderr=[b"x" * (128 * 1024)],
    )
    decode_process = FakeProcess(
        stdout=[pcm[:1], pcm[1:19_199], pcm[19_199:19_203], pcm[19_203:]]
    )
    factory = ProcessFactory(probe_process, decode_process)

    probe = probe_media(path, process_factory=factory)
    blocks = list(decode_audio(path, probe, process_factory=factory))

    assert probe == MediaProbe(duration_seconds=0.625, format_name="wav")
    assert [block.start_sample for block in blocks] == [0, 9_600]
    assert [len(block.pcm) for block in blocks] == [9_600, 400]
    assert all(
        block.pcm.dtype == np.int16
        and block.pcm.ndim == 1
        and block.pcm.flags.c_contiguous
        for block in blocks
    )
    np.testing.assert_array_equal(
        np.concatenate([block.pcm for block in blocks]), samples
    )
    probe_argv, probe_kwargs = factory.calls[0]
    decode_argv, decode_kwargs = factory.calls[1]
    assert probe_argv == [
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
        str(path),
    ]
    assert decode_argv == [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-protocol_whitelist",
        "file,pipe",
        "-i",
        str(path),
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
        "16000",
        "pipe:1",
    ]
    assert _decode_wall_budget(probe.duration_seconds) == 30.0
    assert _decode_wall_budget(100.0) == 210.0
    for kwargs in (probe_kwargs, decode_kwargs):
        assert kwargs["shell"] is False
        assert kwargs["start_new_session"] is True
        assert kwargs["stdin"] is subprocess.DEVNULL
        assert kwargs["stdout"] is subprocess.PIPE
        assert kwargs["stderr"] is subprocess.PIPE
    assert probe_process.stderr.fragments == []


def test_bounded_stderr_reader_drains_concurrently_to_eof() -> None:
    payload = b"x" * (MAX_DIAGNOSTIC_BYTES + 65_536)
    stderr = TrackingStream(
        [payload[index : index + 4_096] for index in range(0, len(payload), 4_096)]
    )
    process = FakeProcess()

    def wait_for_stderr(timeout: float | None = None) -> int:
        if not stderr.eof.wait(timeout=timeout):
            raise subprocess.TimeoutExpired("redacted", timeout)
        process.wait_timeouts.append(timeout)
        process.wait_called.set()
        return 0

    process.wait = wait_for_stderr  # type: ignore[method-assign]
    reader = _BoundedPipeReader(stderr, MAX_DIAGNOSTIC_BYTES)
    pump = _ProcessPump(
        process,
        (reader,),
        FakeGroupController(alive=False),
    )
    pump.start()
    returncode = pump.wait_process(
        1.0,
        cancellation=None,
        timeout_code="timeout",
        timeout_message="timeout",
        monotonic=lambda: 0.0,
    )
    pump.wait_readers(
        1.0,
        cancellation=None,
        timeout_code="timeout",
        timeout_message="timeout",
        monotonic=lambda: 0.0,
    )
    pump.close_streams()

    assert returncode == 0
    assert len(reader.output) == MAX_DIAGNOSTIC_BYTES
    assert reader.overflow is True
    assert stderr.bytes_read == len(payload)
    assert stderr.eof.is_set()


def test_probe_waits_for_gated_pipe_drain_before_closing() -> None:
    payload = (
        b'{"streams":[{"duration":"1"}],"format":{"duration":"1","format_name":"wav"}}'
    )
    process = FakeProcess()
    process.stdout = GatedStream(payload)
    outcome: list[object] = []

    def run_probe() -> None:
        try:
            outcome.append(
                probe_media(
                    Path("/internal/input.wav"),
                    process_factory=ProcessFactory(process),
                )
            )
        except BaseException as exc:
            outcome.append(exc)

    thread = threading.Thread(target=run_probe)
    thread.start()
    assert process.stdout.read_started.wait(timeout=1.0)
    assert process.wait_called.wait(timeout=1.0)
    process.stdout.release.set()
    thread.join(timeout=1.0)

    assert outcome == [MediaProbe(1.0, "wav")]
    assert process.stdout.returned
    assert process.stdout.closed


def test_decode_odd_eof_and_probe_timeout_are_typed_and_redacted(
    tmp_path: Path,
) -> None:
    secret_path = tmp_path / "host-secret.wav"
    odd = FakeProcess(stdout=[b"\x01"], stderr=[b"private diagnostic"])
    with pytest.raises(AudioError) as odd_error:
        list(
            decode_audio(
                secret_path,
                MediaProbe(1.0, "wav"),
                process_factory=ProcessFactory(odd),
            )
        )
    assert odd_error.value.code == "invalid_audio"
    assert str(secret_path) not in str(odd_error.value)
    assert "private diagnostic" not in str(odd_error.value)
    assert odd.wait_timeouts

    timed_out = FakeProcess(returncode=None, timeout=True)
    group = FakeGroupController()

    with pytest.raises(AudioError) as timeout_error:
        probe_media(
            secret_path,
            process_factory=ProcessFactory(timed_out),
            timeout_seconds=0.01,
            process_group_controller=group,
        )
    assert timeout_error.value.code == "audio_probe_timeout"
    assert group.signals == [signal.SIGTERM]
    assert timed_out.wait_timeouts[-1:] == [None]
    assert timed_out.stdout.closed
    assert timed_out.stderr.closed


def test_nonzero_probe_and_decode_are_typed_public_errors(
    tmp_path: Path,
) -> None:
    path = tmp_path / "private.wav"
    failed_probe = FakeProcess(
        stdout=[b'{"streams":[]}'],
        stderr=[b"private probe detail"],
        returncode=1,
    )
    with pytest.raises(AudioError) as probe_error:
        probe_media(path, process_factory=ProcessFactory(failed_probe))
    assert probe_error.value.code == "invalid_audio"

    failed_decode = FakeProcess(
        stdout=[b"\x00\x00"],
        stderr=[b"private decode detail"],
        returncode=1,
    )
    with pytest.raises(AudioError) as decode_error:
        list(
            decode_audio(
                path,
                MediaProbe(1.0, "wav"),
                process_factory=ProcessFactory(failed_decode),
            )
        )
    assert decode_error.value.code == "invalid_audio"
    for error in (probe_error.value, decode_error.value):
        assert str(path) not in str(error)
        assert "private" not in str(error)

    mixed_demuxer = FakeProcess(
        stdout=[
            b'{"streams":[{"duration":"1"}],'
            b'"format":{"duration":"1","format_name":"concat,wav"}}'
        ]
    )
    with pytest.raises(AudioError) as demuxer_error:
        probe_media(path, process_factory=ProcessFactory(mixed_demuxer))
    assert demuxer_error.value.code == "invalid_audio"

    unused_factory = ProcessFactory()
    with pytest.raises(AudioError) as duration_error:
        list(
            decode_audio(
                path,
                MediaProbe(float("nan"), "wav"),
                process_factory=unused_factory,
            )
        )
    assert duration_error.value.code == "invalid_audio"
    assert unused_factory.calls == []


def test_decode_cancellation_terminates_and_reaps_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = FakeProcess(returncode=None, timeout=True, pid=43210)
    cancellation = Cancellation()
    cancellation.cancel()
    signals: list[tuple[int, int]] = []

    def killpg(pid: int, sent_signal: int) -> None:
        if sent_signal == 0:
            if process.returncode is None:
                return
            raise ProcessLookupError
        signals.append((pid, sent_signal))
        process.returncode = -sent_signal

    monkeypatch.setattr("botified_asr.audio.os.killpg", killpg)

    with pytest.raises(AudioCancelled) as caught:
        list(
            decode_audio(
                tmp_path / "input.wav",
                MediaProbe(30.0, "wav"),
                cancellation=cancellation,
                process_factory=ProcessFactory(process),
            )
        )

    assert caught.value.code == "cancelled"
    assert signals == [(43210, signal.SIGTERM)]
    assert process.wait_timeouts[-1] is None

    timed_decode = FakeProcess(returncode=None, timeout=True)
    clock_values = iter((0.0, 31.0))
    with pytest.raises(AudioError) as timeout_error:
        list(
            decode_audio(
                tmp_path / "timeout.wav",
                MediaProbe(1.0, "wav"),
                process_factory=ProcessFactory(timed_decode),
                monotonic=lambda: next(clock_values, 31.0),
            )
        )
    assert timeout_error.value.code == "audio_decode_timeout"
    assert timed_decode.terminated == 1
    assert timed_decode.stdout.closed
    assert timed_decode.stderr.closed


def test_probe_and_decode_cancel_after_process_start(
    tmp_path: Path,
) -> None:
    for operation in ("probe", "decode"):
        process = FakeProcess(returncode=None, timeout=True)
        process.stdout = GatedStream(b"unused")
        cancellation = Cancellation()
        outcome: list[BaseException] = []

        def run() -> None:
            try:
                if operation == "probe":
                    probe_media(
                        tmp_path / "input.wav",
                        cancellation=cancellation,
                        process_factory=ProcessFactory(process),
                    )
                else:
                    list(
                        decode_audio(
                            tmp_path / "input.wav",
                            MediaProbe(1.0, "wav"),
                            cancellation=cancellation,
                            process_factory=ProcessFactory(process),
                        )
                    )
            except BaseException as exc:
                outcome.append(exc)

        thread = threading.Thread(target=run)
        thread.start()
        assert process.stdout.read_started.wait(timeout=1.0)
        cancellation.cancel()
        thread.join(timeout=1.0)

        assert not thread.is_alive()
        assert len(outcome) == 1
        assert isinstance(outcome[0], AudioCancelled)
        assert process.terminated == 1
        assert process.wait_timeouts[-1] is None
        assert process.stdout.closed
        assert process.stderr.closed


@pytest.mark.parametrize("operation", ["probe", "decode"])
def test_probe_and_decode_cancel_during_pipe_drain(
    operation: str,
    tmp_path: Path,
) -> None:
    probe_payload = (
        b'{"streams":[{"duration":"1"}],"format":{"duration":"1","format_name":"wav"}}'
    )
    process = FakeProcess(stdout=[probe_payload] if operation == "probe" else [])
    process.stderr = GatedStream(b"late diagnostic")
    cancellation = Cancellation()
    outcome: list[object] = []

    def run() -> None:
        try:
            if operation == "probe":
                outcome.append(
                    probe_media(
                        tmp_path / "input.wav",
                        cancellation=cancellation,
                        process_factory=ProcessFactory(process),
                    )
                )
            else:
                outcome.append(
                    list(
                        decode_audio(
                            tmp_path / "input.wav",
                            MediaProbe(1.0, "wav"),
                            cancellation=cancellation,
                            process_factory=ProcessFactory(process),
                        )
                    )
                )
        except BaseException as exc:
            outcome.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    assert process.stderr.read_started.wait(timeout=1.0)
    assert process.wait_called.wait(timeout=1.0)
    cancellation.cancel()
    thread.join(timeout=0.5)
    if thread.is_alive():
        process.stderr.release.set()
        thread.join(timeout=3.0)

    assert len(outcome) == 1
    assert isinstance(outcome[0], AudioCancelled)
    assert process.stdout.closed
    assert process.stderr.closed


def test_cleanup_kills_descendant_group_after_leader_has_exited() -> None:
    process = FakeProcess(returncode=0, pid=43210)
    process.stdout = GatedStream(b"held by descendant")
    reader = _BoundedPipeReader(process.stdout, MAX_DIAGNOSTIC_BYTES)
    group = FakeGroupController(ignore_term=True)
    clock = iter((0.0, 0.0, 5.0))
    pump = _ProcessPump(
        process,
        (reader,),
        group,
        monotonic=lambda: next(clock, 5.0),
        sleep=lambda _seconds: None,
    )
    pump.start()
    assert process.stdout.read_started.wait(timeout=1.0)

    pump.stop()

    assert group.signals == [signal.SIGTERM, signal.SIGKILL]
    assert process.wait_timeouts == [None]
    assert process.stdout.closed
    assert process.stderr.closed
    assert not reader.alive


def test_cleanup_readers_share_one_total_grace_deadline() -> None:
    process = FakeProcess(returncode=0)
    clock = AdvancingClock()
    readers = (PersistentReader(clock), PersistentReader(clock))
    pump = _ProcessPump(
        process,
        readers,  # type: ignore[arg-type]
        FakeGroupController(alive=False),
        monotonic=clock,
        sleep=lambda _seconds: None,
    )
    pump.start()

    with pytest.raises(AudioError) as caught:
        pump.stop()

    assert caught.value.code == "audio_cleanup_failed"
    assert [reader.started for reader in readers] == [1, 1]
    assert [reader.stopped for reader in readers] == [1, 1]
    assert sum(
        timeout for reader in readers for timeout in reader.join_timeouts
    ) == pytest.approx(5.0)
    assert readers[0].join_timeouts
    assert readers[1].join_timeouts == []
    assert all(
        timeout <= 0.05 for reader in readers for timeout in reader.join_timeouts
    )
    assert clock.value == pytest.approx(5.0)
    assert process.wait_timeouts == [None]


def test_decode_stdout_read_error_fails_closed(tmp_path: Path) -> None:
    process = FakeProcess()
    process.stdout = ReadErrorStream([])

    with pytest.raises(AudioError) as caught:
        list(
            decode_audio(
                tmp_path / "input.wav",
                MediaProbe(1.0, "wav"),
                process_factory=ProcessFactory(process),
            )
        )

    assert caught.value.code == "invalid_audio"
