from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from time import monotonic
from typing import Iterator, Literal, Protocol, TypeVar

from botified_asr.audio import Cancellation
from botified_asr.errors import (
    InferenceSaturated,
    PipelineError,
)

_Result = TypeVar("_Result")
MAX_INFERENCE_LANES = 2
SYNC_INFERENCE_WAIT_SECONDS = 5.0
_CANCELLATION_POLL_SECONDS = 0.05
_InferenceCategory = Literal["sync", "async"]


@dataclass(frozen=True, slots=True)
class _InferenceSession:
    category: _InferenceCategory
    cancellation: Cancellation


_session_local = threading.local()


@contextmanager
def inference_session(
    category: _InferenceCategory,
    cancellation: Cancellation,
) -> Iterator[None]:
    if category not in {"sync", "async"}:
        raise ValueError("inference category must be sync or async")
    if type(cancellation) is not Cancellation:
        raise TypeError("inference cancellation is invalid")
    previous = getattr(_session_local, "current", None)
    _session_local.current = _InferenceSession(category, cancellation)
    try:
        yield
    finally:
        if previous is None:
            del _session_local.current
        else:
            _session_local.current = previous


class InferenceLane(Protocol):
    def invoke(self, operation: Callable[[], _Result], /) -> _Result: ...


class SerialInferenceLane:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._sync_waiters: deque[object] = deque()
        self._async_waiters: deque[object] = deque()
        self._active_category: _InferenceCategory | None = None
        self._active_thread_id: int | None = None
        self._next_category: _InferenceCategory | None = None

    def invoke(self, operation: Callable[[], _Result], /) -> _Result:
        session: _InferenceSession | None = getattr(
            _session_local,
            "current",
            None,
        )
        category: _InferenceCategory = (
            "async" if session is None else session.category
        )
        cancellation = None if session is None else session.cancellation
        deadline = (
            monotonic() + SYNC_INFERENCE_WAIT_SECONDS
            if session is not None and category == "sync"
            else None
        )
        ticket = object()
        waiters = self._waiters(category)
        with self._condition:
            if self._active_thread_id == threading.get_ident():
                raise RuntimeError("inference lane does not allow reentrant calls")
            waiters.append(ticket)
            try:
                while not self._may_enter(category, ticket):
                    if cancellation is not None and cancellation.cancelled:
                        raise PipelineError(
                            "cancelled",
                            "Audio processing was cancelled",
                        )
                    wait_seconds = _CANCELLATION_POLL_SECONDS
                    if deadline is not None:
                        remaining = deadline - monotonic()
                        if remaining <= 0:
                            raise InferenceSaturated()
                        wait_seconds = min(wait_seconds, remaining)
                    elif cancellation is None:
                        wait_seconds = None
                    self._condition.wait(wait_seconds)
                if cancellation is not None and cancellation.cancelled:
                    raise PipelineError(
                        "cancelled",
                        "Audio processing was cancelled",
                    )
                if deadline is not None and monotonic() >= deadline:
                    raise InferenceSaturated()
                waiters.popleft()
                self._active_category = category
                self._active_thread_id = threading.get_ident()
                self._next_category = None
            except BaseException:
                if ticket in waiters:
                    waiters.remove(ticket)
                self._repair_next_category(category)
                self._condition.notify_all()
                raise

        try:
            return operation()
        finally:
            with self._condition:
                completed_category = self._active_category
                self._active_category = None
                self._active_thread_id = None
                if completed_category is not None:
                    opposite: _InferenceCategory = (
                        "sync"
                        if completed_category == "async"
                        else "async"
                    )
                    if self._waiters(opposite):
                        self._next_category = opposite
                    elif self._waiters(completed_category):
                        self._next_category = completed_category
                self._condition.notify_all()

    def _waiters(self, category: _InferenceCategory) -> deque[object]:
        return (
            self._sync_waiters
            if category == "sync"
            else self._async_waiters
        )

    def _may_enter(
        self,
        category: _InferenceCategory,
        ticket: object,
    ) -> bool:
        if self._active_category is not None:
            return False
        waiters = self._waiters(category)
        if not waiters or waiters[0] is not ticket:
            return False
        return (
            self._next_category is None
            or self._next_category == category
        )

    def _repair_next_category(
        self,
        removed_category: _InferenceCategory,
    ) -> None:
        if (
            self._next_category == removed_category
            and not self._waiters(removed_category)
        ):
            opposite: _InferenceCategory = (
                "sync" if removed_category == "async" else "async"
            )
            self._next_category = (
                opposite if self._waiters(opposite) else None
            )
