from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Protocol, TypeVar

_Result = TypeVar("_Result")


class InferenceLane(Protocol):
    def invoke(self, operation: Callable[[], _Result], /) -> _Result: ...


class SerialInferenceLane:
    def __init__(self) -> None:
        self._lock = threading.Lock()

    def invoke(self, operation: Callable[[], _Result], /) -> _Result:
        with self._lock:
            return operation()
