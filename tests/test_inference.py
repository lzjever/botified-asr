from __future__ import annotations

import importlib
import threading
import time

import pytest

from botified_asr.audio import Cancellation
from botified_asr.errors import PipelineError


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


def test_categorized_lanes_overlap_across_lanes_but_never_within_one() -> None:
    inference = _inference()
    lanes = (inference.SerialInferenceLane(), inference.SerialInferenceLane())
    entered = [threading.Event(), threading.Event()]
    release = threading.Event()
    state_lock = threading.Lock()
    active = 0
    per_lane = [0, 0]
    max_active = 0
    max_per_lane = [0, 0]

    def operation(lane_index: int) -> None:
        nonlocal active, max_active
        with state_lock:
            active += 1
            per_lane[lane_index] += 1
            max_active = max(max_active, active)
            max_per_lane[lane_index] = max(
                max_per_lane[lane_index],
                per_lane[lane_index],
            )
        entered[lane_index].set()
        assert release.wait(timeout=2)
        with state_lock:
            active -= 1
            per_lane[lane_index] -= 1

    def run(lane_index: int) -> None:
        with inference.inference_session("async", Cancellation()):
            lanes[lane_index].invoke(lambda: operation(lane_index))

    threads = [
        threading.Thread(
            target=lambda index=index: run(index),
            daemon=True,
        )
        for index in range(2)
    ]
    for thread in threads:
        thread.start()
    try:
        assert all(event.wait(timeout=2) for event in entered)
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=2)

    assert max_active == 2
    assert max_per_lane == [1, 1]


def test_categorized_lane_fifo_handoff_alternates_to_opposite_class() -> None:
    inference = _inference()
    lane = inference.SerialInferenceLane()
    holder_entered = threading.Event()
    release_holder = threading.Event()
    order: list[str] = []

    def run(name: str, category: str, *, hold: bool = False) -> None:
        with inference.inference_session(category, Cancellation()):
            lane.invoke(
                lambda: (
                    order.append(name),
                    holder_entered.set() if hold else None,
                    release_holder.wait(timeout=2) if hold else None,
                )
            )

    holder = threading.Thread(
        target=lambda: run("async-0", "async", hold=True),
        daemon=True,
    )
    holder.start()
    assert holder_entered.wait(timeout=2)

    waiters: list[threading.Thread] = []
    for name, category in (
        ("async-1", "async"),
        ("async-2", "async"),
        ("sync-1", "sync"),
        ("sync-2", "sync"),
    ):
        thread = threading.Thread(
            target=lambda name=name, category=category: run(name, category),
            daemon=True,
        )
        thread.start()
        waiters.append(thread)
        _wait_for_waiter_count(lane, category, 1 if name.endswith("1") else 2)

    release_holder.set()
    holder.join(timeout=2)
    for thread in waiters:
        thread.join(timeout=2)

    assert order == [
        "async-0",
        "sync-1",
        "async-1",
        "sync-2",
        "async-2",
    ]


def test_sync_timeout_removes_waiter_without_late_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference = _inference()
    monkeypatch.setattr(inference, "SYNC_INFERENCE_WAIT_SECONDS", 0.05)
    lane = inference.SerialInferenceLane()
    holder_entered = threading.Event()
    release_holder = threading.Event()
    late_entry = threading.Event()

    def hold() -> None:
        with inference.inference_session("async", Cancellation()):
            lane.invoke(
                lambda: (
                    holder_entered.set(),
                    release_holder.wait(timeout=2),
                )
            )

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert holder_entered.wait(timeout=2)

    started = time.monotonic()
    with inference.inference_session("sync", Cancellation()):
        with pytest.raises(inference.InferenceSaturated):
            lane.invoke(late_entry.set)
    assert time.monotonic() - started < 1
    release_holder.set()
    holder.join(timeout=2)
    assert not late_entry.is_set()
    _wait_for_waiter_count(lane, "sync", 0)


def test_sync_deadline_wins_when_capacity_arrives_late(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference = _inference()
    monotonic = [100.0]
    monkeypatch.setattr(inference, "monotonic", lambda: monotonic[0])
    monkeypatch.setattr(inference, "SYNC_INFERENCE_WAIT_SECONDS", 1.0)
    monkeypatch.setattr(inference, "_CANCELLATION_POLL_SECONDS", 60.0)
    lane = inference.SerialInferenceLane()
    holder_entered = threading.Event()
    release_holder = threading.Event()
    late_entry = threading.Event()
    errors: list[BaseException] = []

    def hold() -> None:
        with inference.inference_session("async", Cancellation()):
            lane.invoke(
                lambda: (
                    holder_entered.set(),
                    release_holder.wait(timeout=2),
                )
            )

    def wait_sync() -> None:
        try:
            with inference.inference_session("sync", Cancellation()):
                lane.invoke(late_entry.set)
        except BaseException as error:
            errors.append(error)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert holder_entered.wait(timeout=2)
    waiter = threading.Thread(target=wait_sync, daemon=True)
    waiter.start()
    _wait_for_waiter_count(lane, "sync", 1)

    monotonic[0] = 102.0
    release_holder.set()
    holder.join(timeout=2)
    waiter.join(timeout=2)

    assert len(errors) == 1
    assert isinstance(errors[0], inference.InferenceSaturated)
    assert not late_entry.is_set()
    _wait_for_waiter_count(lane, "sync", 0)


@pytest.mark.parametrize("category", ("sync", "async"))
def test_cancelled_waiter_is_removed_and_does_not_leak_lane(
    category: str,
) -> None:
    inference = _inference()
    lane = inference.SerialInferenceLane()
    holder_entered = threading.Event()
    release_holder = threading.Event()
    cancelled_entry = threading.Event()
    cancellation = Cancellation()
    errors: list[BaseException] = []

    def hold() -> None:
        with inference.inference_session("sync", Cancellation()):
            lane.invoke(
                lambda: (
                    holder_entered.set(),
                    release_holder.wait(timeout=2),
                )
            )

    def wait_cancelled() -> None:
        try:
            with inference.inference_session(category, cancellation):
                lane.invoke(cancelled_entry.set)
        except BaseException as error:
            errors.append(error)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert holder_entered.wait(timeout=2)
    waiter = threading.Thread(target=wait_cancelled, daemon=True)
    waiter.start()
    _wait_for_waiter_count(lane, category, 1)
    cancellation.cancel()
    waiter.join(timeout=2)

    assert len(errors) == 1
    assert isinstance(errors[0], PipelineError)
    assert errors[0].code == "cancelled"
    assert not cancelled_entry.is_set()
    _wait_for_waiter_count(lane, category, 0)

    follower_entered = threading.Event()
    follower = threading.Thread(
        target=lambda: lane.invoke(follower_entered.set),
        daemon=True,
    )
    follower.start()
    release_holder.set()
    holder.join(timeout=2)
    follower.join(timeout=2)
    assert follower_entered.is_set()


def _wait_for_waiter_count(lane: object, category: str, count: int) -> None:
    deadline = time.monotonic() + 2
    waiters = getattr(lane, f"_{category}_waiters")
    while len(waiters) != count:
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for inference waiter")
        time.sleep(0.001)
