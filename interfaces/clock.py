"""
Shared clock helpers for real-time modules and the simulation harness.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol


class Clock(Protocol):
    """Clock contract shared by runtime modules and the simulator."""

    def time(self) -> float: ...

    def monotonic(self) -> float: ...


class SystemClock:
    """Default wall clock used in production and unit tests."""

    def time(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()


@dataclass
class SimClock:
    """Deterministic clock that advances with the simulation timestep."""

    start_time: float = field(default_factory=time.time)
    _elapsed: float = 0.0

    def time(self) -> float:
        return self.start_time + self._elapsed

    def monotonic(self) -> float:
        return self._elapsed

    def advance(self, dt: float) -> None:
        self._elapsed += dt
