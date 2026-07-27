from __future__ import annotations

import importlib
import threading

import pytest


def _inference():
    return importlib.import_module("botified_asr.inference")


def test_serial_inference_lane_never_overlaps_operations() -> None:
    inference = _inference()
    lane = inference.SerialInferenceLane()
    first_entered = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    second_entered = threading.Event()
    state_lock = threading.Lock()
    active = 0
    max_active = 0

    def enter(*, first: bool) -> None:
        nonlocal active
        nonlocal max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        if first:
            first_entered.set()
            assert release_first.wait(timeout=2)
        else:
            second_entered.set()
        with state_lock:
            active -= 1

    first = threading.Thread(
        target=lambda: lane.invoke(lambda: enter(first=True)),
        daemon=True,
    )

    def run_second() -> None:
        second_started.set()
        lane.invoke(lambda: enter(first=False))

    second = threading.Thread(target=run_second, daemon=True)
    first.start()
    assert first_entered.wait(timeout=1)
    second.start()
    assert second_started.wait(timeout=1)
    try:
        assert not second_entered.wait(timeout=0.1)
    finally:
        release_first.set()
        first.join(timeout=1)
        second.join(timeout=1)

    assert not first.is_alive()
    assert not second.is_alive()
    assert second_entered.is_set()
    assert max_active == 1


def test_serial_inference_lane_returns_and_propagates_without_leaking_the_lock() -> (
    None
):
    inference = _inference()
    assert inference.InferenceLane is not None
    lane = inference.SerialInferenceLane()
    expected = object()

    assert lane.invoke(lambda: expected) is expected
    with pytest.raises(TypeError):
        lane.invoke(operation=lambda: expected)

    failure = RuntimeError("model call failed")

    def fail() -> None:
        raise failure

    with pytest.raises(RuntimeError) as caught:
        lane.invoke(fail)

    assert caught.value is failure
    after_failure: list[object] = []
    follower = threading.Thread(
        target=lambda: after_failure.append(lane.invoke(lambda: expected)),
        daemon=True,
    )
    follower.start()
    follower.join(timeout=1)

    assert not follower.is_alive()
    assert after_failure == [expected]
