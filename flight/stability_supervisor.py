"""
Stability Supervisor
====================
Monitors wind, oscillation, and position-hold drift to determine
platform stability level.  When conditions degrade, it commands the
shot controller to reduce speed, increase distance, or hover.

Runs at 25 Hz.

Levels:
  NOMINAL  — full performance
  MARGINAL — reduce speed to 70 %
  DEGRADED — reduce speed to 40 %, add 2 m standoff
  CRITICAL — hover (or land, configurable)
"""

from __future__ import annotations

import collections
import logging
import math
from typing import Deque, Optional

from config.parameters import StabilityParams
from interfaces.clock import Clock, SystemClock
from interfaces.event_bus import bus
from interfaces.schemas import (
    DroneTelemetry,
    SafetyOverride,
    StabilityLevel,
    SystemEvent,
    Vec3,
)

log = logging.getLogger(__name__)


class StabilitySupervisor:
    """25 Hz supervisor evaluating platform stability."""

    def __init__(
        self,
        params: StabilityParams | None = None,
        *,
        clock: Clock | None = None,
    ) -> None:
        self.p = params or StabilityParams()
        self.clock = clock or SystemClock()

        self.level: StabilityLevel = StabilityLevel.NOMINAL
        self.speed_scale: float = 1.0
        self.extra_distance: float = 0.0
        self.should_hover: bool = False

        # ── hysteresis ───────────────────────────────────────────────────
        self._level_hold_until: float = 0.0  # monotonic time
        self._HYSTERESIS_S: float = 1.0      # hold a level for at least 1 s

        # ── rolling buffers ──────────────────────────────────────────────
        self._accel_history: Deque[float] = collections.deque(
            maxlen=self.p.accel_jitter_window
        )
        self._pos_history: Deque[tuple[float, Vec3]] = collections.deque(maxlen=50)
        self._prev_velocity: Optional[Vec3] = None

    # ════════════════════════════════════════════════════════════════════
    #  PUBLIC
    # ════════════════════════════════════════════════════════════════════

    def update(
        self,
        telem: DroneTelemetry,
        now_s: float | None = None,
    ) -> StabilityLevel:
        """Call once per tick.  Returns current stability level."""
        now = self.clock.monotonic() if now_s is None else now_s
        timestamp_s = self.clock.time() if now_s is None else now_s

        wind_level = self._assess_wind(telem)
        jitter_level = self._assess_jitter(telem, now)
        drift_level = self._assess_drift(telem, now)

        # Worst of all three assessments
        new_level = max(
            wind_level, jitter_level, drift_level,
            key=lambda l: list(StabilityLevel).index(l),
        )

        if new_level != self.level:
            # Only transition if: escalating (worse) OR hold timer expired
            levels = list(StabilityLevel)
            is_escalation = levels.index(new_level) > levels.index(self.level)
            if is_escalation or now >= self._level_hold_until:
                log.info("Stability %s → %s", self.level.value, new_level.value)
                bus.publish(SystemEvent(
                    timestamp=timestamp_s,
                    source="stability",
                    event="stability_change",
                    payload={"old": self.level.value, "new": new_level.value},
                ))
                self.level = new_level
                self._apply_level(new_level)
                self._level_hold_until = now + self._HYSTERESIS_S

        return self.level

    # ════════════════════════════════════════════════════════════════════
    #  ASSESSORS
    # ════════════════════════════════════════════════════════════════════

    def _assess_wind(self, telem: DroneTelemetry) -> StabilityLevel:
        w = telem.wind_estimate
        speed = math.sqrt(w.x**2 + w.y**2 + w.z**2)
        if speed >= self.p.wind_critical:
            return StabilityLevel.CRITICAL
        if speed >= self.p.wind_degraded:
            return StabilityLevel.DEGRADED
        if speed >= self.p.wind_marginal:
            return StabilityLevel.MARGINAL
        return StabilityLevel.NOMINAL

    def _assess_jitter(self, telem: DroneTelemetry, now: float) -> StabilityLevel:
        """Acceleration variance over a rolling window."""
        # Approximate acceleration as Δv between ticks
        vel = telem.velocity
        if self._prev_velocity is not None:
            accel_mag = (vel - self._prev_velocity).norm()
        else:
            accel_mag = 0.0
        self._prev_velocity = Vec3(vel.x, vel.y, vel.z)
        self._accel_history.append(accel_mag)

        if len(self._accel_history) < 5:
            return StabilityLevel.NOMINAL

        mean = sum(self._accel_history) / len(self._accel_history)
        var = sum((v - mean) ** 2 for v in self._accel_history) / len(self._accel_history)
        std = math.sqrt(var)

        if std >= self.p.accel_jitter_critical:
            return StabilityLevel.CRITICAL
        if std >= self.p.accel_jitter_degraded:
            return StabilityLevel.DEGRADED
        if std >= self.p.accel_jitter_marginal:
            return StabilityLevel.MARGINAL
        return StabilityLevel.NOMINAL

    def _assess_drift(self, telem: DroneTelemetry, now: float) -> StabilityLevel:
        """How much position wanders over a short window (position-hold quality)."""
        horiz_speed = math.sqrt(telem.velocity.x**2 + telem.velocity.y**2)
        if horiz_speed > self.p.drift_speed_gate:
            self._pos_history.clear()
            return StabilityLevel.NOMINAL

        self._pos_history.append((now, Vec3(telem.position.x, telem.position.y, 0)))

        # Need at least drift_window_s of data
        cutoff = now - self.p.drift_window_s
        recent = [(t, p) for t, p in self._pos_history if t >= cutoff]
        if len(recent) < 3:
            return StabilityLevel.NOMINAL

        # Max deviation from mean
        mean_x = sum(p.x for _, p in recent) / len(recent)
        mean_y = sum(p.y for _, p in recent) / len(recent)
        max_dev = max(
            math.sqrt((p.x - mean_x)**2 + (p.y - mean_y)**2)
            for _, p in recent
        )

        if max_dev >= self.p.drift_critical:
            return StabilityLevel.CRITICAL
        if max_dev >= self.p.drift_degraded:
            return StabilityLevel.DEGRADED
        if max_dev >= self.p.drift_marginal:
            return StabilityLevel.MARGINAL
        return StabilityLevel.NOMINAL

    # ════════════════════════════════════════════════════════════════════
    #  RESPONSE
    # ════════════════════════════════════════════════════════════════════

    def _apply_level(self, level: StabilityLevel) -> None:
        if level == StabilityLevel.NOMINAL:
            self.speed_scale = 1.0
            self.extra_distance = 0.0
            self.should_hover = False
        elif level == StabilityLevel.MARGINAL:
            self.speed_scale = self.p.marginal_speed_scale
            self.extra_distance = 0.0
            self.should_hover = False
        elif level == StabilityLevel.DEGRADED:
            self.speed_scale = self.p.degraded_speed_scale
            self.extra_distance = self.p.degraded_distance_add
            self.should_hover = False
        elif level == StabilityLevel.CRITICAL:
            self.speed_scale = 0.0
            self.extra_distance = 0.0
            self.should_hover = (self.p.critical_action == "hover")
