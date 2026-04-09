"""
Safety & Geofence Module
========================
50 Hz hard-real-time safety layer.  Runs at DOUBLE the rate of the rest of
the stack because safety checks must never be skipped.

Responsibilities:
  1. Geofence — cylindrical keep-in volume around home
  2. Battery failsafe — staged (warn → return → land)
  3. Link-loss — staged (warn → return → land)
  4. Subject proximity — hard minimum distance
  5. Emergency stop — immediate motor kill path

The module produces a `SafetyStatus` each tick.  The worst active override
propagates up to the mission state machine.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from config.parameters import SafetyParams
from flight.geofence import GeofenceChecker, GeofenceConfig
from interfaces.clock import Clock, SystemClock
from interfaces.event_bus import bus
from interfaces.schemas import (
    DroneTelemetry,
    FlightSetpoint,
    SafetyOverride,
    SafetyStatus,
    SystemEvent,
    Vec3,
)

log = logging.getLogger(__name__)


class SafetyModule:
    """50 Hz safety monitor — highest-priority module in the stack."""

    def __init__(
        self,
        params: SafetyParams | None = None,
        *,
        clock: Clock | None = None,
    ) -> None:
        self.p = params or SafetyParams()
        self.clock = clock or SystemClock()

        self._last_heartbeat: float = self.clock.monotonic()
        self._emergency_requested: bool = False
        self._last_known_target: Optional[Vec3] = None

        # Build polygon geofence checker (falls back to cylinder if no config)
        geo_cfg = self.p.geofence_config or GeofenceConfig(
            ceiling_m=self.p.geofence_ceiling,
            floor_m=self.p.geofence_floor,
            warn_margin_m=self.p.geofence_warn_margin,
        )
        self._geofence = GeofenceChecker(geo_cfg, fallback_radius=self.p.geofence_radius)

    # ════════════════════════════════════════════════════════════════════
    #  PUBLIC
    # ════════════════════════════════════════════════════════════════════

    def heartbeat(self, now_s: float | None = None) -> None:
        """Called when we receive a heartbeat / command from the app."""
        self._last_heartbeat = self.clock.monotonic() if now_s is None else now_s

    def request_emergency_stop(self) -> None:
        self._emergency_requested = True

    def update(
        self,
        telem: DroneTelemetry,
        setpoint: FlightSetpoint,
        target_world: Optional[Vec3] = None,
        now_s: float | None = None,
    ) -> SafetyStatus:
        """Call every tick (50 Hz).  Returns safety status with override."""
        now = self.clock.monotonic() if now_s is None else now_s
        timestamp_s = self.clock.time() if now_s is None else now_s
        reasons: list[str] = []
        overrides: list[SafetyOverride] = []

        # ── 1.  Emergency stop ───────────────────────────────────────────
        if self._emergency_requested:
            reasons.append("emergency stop requested")
            overrides.append(SafetyOverride.EMERGENCY_STOP)

        # ── 2.  Geofence ────────────────────────────────────────────────
        geo_ok, geo_override, geo_reason = self._check_geofence(telem)
        if not geo_ok:
            reasons.append(geo_reason)
            overrides.append(geo_override)

        # ── 3.  Battery ─────────────────────────────────────────────────
        bat_ok, bat_override, bat_reason = self._check_battery(telem)
        if not bat_ok:
            reasons.append(bat_reason)
            overrides.append(bat_override)

        # ── 4.  Link ────────────────────────────────────────────────────
        link_ok, link_override, link_reason = self._check_link(now)
        if not link_ok:
            reasons.append(link_reason)
            overrides.append(link_override)

        # ── 5.  Subject proximity ───────────────────────────────────────
        min_ok = True
        # Update last known target if we have a fresh one
        prox_target = target_world
        if prox_target is not None:
            self._last_known_target = prox_target
        else:
            prox_target = self._last_known_target  # fall back to stale

        if prox_target is not None:
            min_ok, prox_override, prox_reason = self._check_proximity(
                telem, prox_target
            )
            if not min_ok:
                reasons.append(prox_reason)
                overrides.append(prox_override)

        # ── Determine worst override ─────────────────────────────────────
        active = self._worst_override(overrides)

        status = SafetyStatus(
            timestamp=timestamp_s,
            active_override=active,
            geofence_ok=geo_ok,
            battery_ok=bat_ok,
            link_ok=link_ok,
            min_distance_ok=min_ok,
            reasons=reasons,
        )

        # Emit event if override changed
        if active != SafetyOverride.NONE:
            bus.publish(SystemEvent(
                timestamp=timestamp_s,
                source="safety",
                event="safety_override",
                payload={"override": active.value, "reasons": reasons},
            ))

        return status

    def clamp_setpoint(
        self,
        setpoint: FlightSetpoint,
        telem: DroneTelemetry,
        status: SafetyStatus,
    ) -> FlightSetpoint:
        """Post-process a setpoint to enforce hard safety limits."""
        if status.active_override == SafetyOverride.EMERGENCY_STOP:
            # Zero everything — the autopilot should disarm
            return FlightSetpoint(
                timestamp=self.clock.time(),
                velocity=Vec3(0, 0, self.p.emergency_descent_rate),
                yaw=telem.attitude_euler.z,
            )
        if status.active_override == SafetyOverride.LAND_NOW:
            return FlightSetpoint(
                timestamp=self.clock.time(),
                position=Vec3(telem.position.x, telem.position.y, 0.0),
                velocity=Vec3(0, 0, self.p.emergency_descent_rate),
                yaw=telem.attitude_euler.z,
            )

        # Enforce geofence ceiling/floor on position setpoints
        if setpoint.position:
            alt = -setpoint.position.z
            alt = max(self.p.geofence_floor, min(self.p.geofence_ceiling, alt))
            setpoint.position = Vec3(
                setpoint.position.x, setpoint.position.y, -alt
            )

        return setpoint

    # ════════════════════════════════════════════════════════════════════
    #  CHECKS
    # ════════════════════════════════════════════════════════════════════

    def _check_geofence(
        self, telem: DroneTelemetry
    ) -> tuple[bool, SafetyOverride, str]:
        result = self._geofence.check(telem.position)

        # Hard breach: outside keep-in or inside an exclusion zone
        if not result.inside_keep_in:
            return False, SafetyOverride.RETURN_HOME, (
                "geofence breach: outside keep-in boundary"
            )
        if result.violated_exclusion is not None:
            return False, SafetyOverride.RETURN_HOME, (
                f"exclusion zone breach: {result.violated_exclusion}"
            )

        # Altitude violation
        if not result.altitude_ok:
            alt = -telem.position.z
            if alt > self._geofence.cfg.ceiling_m:
                return False, SafetyOverride.RETURN_HOME, (
                    f"geofence breach: alt={alt:.1f} > ceiling"
                )
            if telem.in_air:
                return False, SafetyOverride.HOVER, f"below floor: alt={alt:.1f}"

        # Warning margin: approaching any horizontal boundary
        if result.keep_in_margin < float("inf"):
            return False, SafetyOverride.REDUCE_SPEED, "approaching keep-in boundary"
        if result.exclusion_margin < float("inf"):
            return False, SafetyOverride.REDUCE_SPEED, "approaching exclusion zone"

        return True, SafetyOverride.NONE, ""

    def _check_battery(
        self, telem: DroneTelemetry
    ) -> tuple[bool, SafetyOverride, str]:
        pct = telem.battery_percent
        if pct <= self.p.battery_land_percent:
            return False, SafetyOverride.LAND_NOW, f"battery critical: {pct:.0f}%"
        if pct <= self.p.battery_return_percent:
            return False, SafetyOverride.RETURN_HOME, f"battery low: {pct:.0f}%"
        if pct <= self.p.battery_warn_percent:
            return False, SafetyOverride.REDUCE_SPEED, f"battery warn: {pct:.0f}%"
        return True, SafetyOverride.NONE, ""

    def _check_link(
        self, now: float
    ) -> tuple[bool, SafetyOverride, str]:
        elapsed = max(0.0, now - self._last_heartbeat)
        if elapsed > self.p.critical_link_timeout_s:
            return False, SafetyOverride.LAND_NOW, (
                f"link lost: {elapsed:.1f}s (critical)"
            )
        if elapsed > self.p.heartbeat_timeout_s:
            return False, SafetyOverride.RETURN_HOME, (
                f"link lost: {elapsed:.1f}s"
            )
        return True, SafetyOverride.NONE, ""

    def _check_proximity(
        self, telem: DroneTelemetry, target: Vec3
    ) -> tuple[bool, SafetyOverride, str]:
        dx = telem.position.x - target.x
        dy = telem.position.y - target.y
        dist = math.sqrt(dx**2 + dy**2)
        if dist < self.p.subject_emergency_dist:
            return False, SafetyOverride.HOVER, (
                f"too close to subject: {dist:.2f}m (emergency)"
            )
        if dist < self.p.subject_min_distance:
            return False, SafetyOverride.INCREASE_DIST, (
                f"too close to subject: {dist:.2f}m"
            )
        return True, SafetyOverride.NONE, ""

    # ════════════════════════════════════════════════════════════════════
    #  HELPERS
    # ════════════════════════════════════════════════════════════════════

    _OVERRIDE_PRIORITY = [
        SafetyOverride.NONE,
        SafetyOverride.REDUCE_SPEED,
        SafetyOverride.INCREASE_DIST,
        SafetyOverride.HOVER,
        SafetyOverride.RETURN_HOME,
        SafetyOverride.LAND_NOW,
        SafetyOverride.EMERGENCY_STOP,
    ]

    def _worst_override(self, overrides: list[SafetyOverride]) -> SafetyOverride:
        if not overrides:
            return SafetyOverride.NONE
        return max(overrides, key=lambda o: self._OVERRIDE_PRIORITY.index(o))
