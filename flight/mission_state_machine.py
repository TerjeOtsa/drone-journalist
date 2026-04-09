"""
Mission State Machine
=====================
Owns the top-level lifecycle:

    IDLE → TAKEOFF → ACQUIRE → LOCK → FILM → DEGRADE → RETURN → LAND

Every tick (25 Hz) it reads perception, telemetry, safety, and app commands,
decides which state we're in, and emits the current MissionState plus
`ready_to_record`.

Design notes
------------
*  Each state is a method `_state_<name>` — easy to read, easy to test.
*  Timers are monotonic-clock-based so they survive wall-clock jumps.
*  Transitions are logged as events on the bus.
"""

from __future__ import annotations

import logging
from typing import Optional

from config.parameters import MissionParams
from interfaces.clock import Clock, SystemClock
from interfaces.event_bus import bus
from interfaces.schemas import (
    AppCommand,
    DroneTelemetry,
    LockState,
    MissionState,
    MissionStatus,
    SafetyOverride,
    SafetyStatus,
    ShotMode,
    StabilityLevel,
    SystemEvent,
    TargetTrack,
)

log = logging.getLogger(__name__)


class MissionStateMachine:
    """25 Hz state machine controlling mission lifecycle."""

    def __init__(
        self,
        params: MissionParams | None = None,
        *,
        clock: Clock | None = None,
    ) -> None:
        self.p = params or MissionParams()
        self.clock = clock or SystemClock()

        # ── state ────────────────────────────────────────────────────────
        self.state: MissionState = MissionState.IDLE
        self.shot_mode: ShotMode = ShotMode.STANDUP
        self.desired_distance: Optional[float] = None
        self.ready_to_record: bool = False
        self.stability: StabilityLevel = StabilityLevel.NOMINAL
        self._now: float = self.clock.monotonic()
        self._timestamp: float = self.clock.time()

        # ── timers (monotonic) ───────────────────────────────────────────
        self._state_enter_time: float = self.clock.monotonic()
        self._lock_stable_since: Optional[float] = None
        self._weak_since: Optional[float] = None
        self._lost_since: Optional[float] = None
        self._relock_count: int = 0

        # ── latest inputs (set each tick) ────────────────────────────────
        self._track: TargetTrack = TargetTrack(timestamp=0.0)
        self._telem: DroneTelemetry = DroneTelemetry()
        self._safety: SafetyStatus = SafetyStatus()
        self._cmd: Optional[AppCommand] = None

    # ════════════════════════════════════════════════════════════════════
    #  PUBLIC
    # ════════════════════════════════════════════════════════════════════

    def update(
        self,
        track: TargetTrack,
        telem: DroneTelemetry,
        safety: SafetyStatus,
        stability: StabilityLevel,
        cmd: Optional[AppCommand] = None,
        now_s: float | None = None,
    ) -> MissionStatus:
        """Call once per tick.  Returns latest mission status."""
        self._track = track
        self._telem = telem
        self._safety = safety
        self.stability = stability
        self._cmd = cmd
        if now_s is None:
            self._now = self.clock.monotonic()
            self._timestamp = self.clock.time()
        else:
            self._now = now_s
            self._timestamp = now_s
        self._apply_operator_tuning()

        # ── global overrides first ───────────────────────────────────────
        if self._handle_emergency_command():
            pass  # state already changed
        elif self._safety.active_override == SafetyOverride.LAND_NOW:
            self._transition(MissionState.LAND, "safety: land now")
        elif self._safety.active_override == SafetyOverride.EMERGENCY_STOP:
            self._transition(MissionState.EMERGENCY, "safety: emergency stop")
        elif self._safety.active_override == SafetyOverride.RETURN_HOME:
            if self.state not in (MissionState.RETURN, MissionState.LAND,
                                  MissionState.EMERGENCY):
                self._transition(MissionState.RETURN, "safety: return home")
        else:
            # ── per-state logic ──────────────────────────────────────────
            handler = getattr(self, f"_state_{self.state.value}", None)
            if handler:
                handler()

        return self._build_status()

    # ════════════════════════════════════════════════════════════════════
    #  STATE HANDLERS
    # ════════════════════════════════════════════════════════════════════

    def _state_idle(self) -> None:
        self.ready_to_record = False
        if self._cmd and self._cmd.action == "start":
            if self._telem.gps_fix >= 3 and self._telem.battery_percent > 20:
                self._transition(MissionState.TAKEOFF, "operator start")
            else:
                log.warning("Cannot start: gps_fix=%d batt=%.0f%%",
                            self._telem.gps_fix, self._telem.battery_percent)

    def _state_takeoff(self) -> None:
        self.ready_to_record = False
        alt = -self._telem.position.z  # NED: z is negative up
        if alt >= self.p.takeoff_altitude * 0.90:
            self._transition(MissionState.ACQUIRE, "takeoff complete")
        elif self._elapsed() > self.p.takeoff_timeout_s:
            self._transition(MissionState.LAND, "takeoff timeout")

    def _state_acquire(self) -> None:
        self.ready_to_record = False
        if self._track.lock_state in (LockState.LOCKED, LockState.CANDIDATE):
            self._transition(MissionState.LOCK, "target found")
        elif self._elapsed() > self.p.acquire_timeout_s:
            self._transition(MissionState.RETURN, "acquire timeout")

    def _state_lock(self) -> None:
        self.ready_to_record = False
        if self._track.lock_state == LockState.LOCKED:
            if self._lock_stable_since is None:
                self._lock_stable_since = self._now
            elif self._now - self._lock_stable_since >= self.p.lock_confirm_time_s:
                self._transition(MissionState.FILM, "lock confirmed")
        else:
            self._lock_stable_since = None
            if self._track.lock_state == LockState.LOST:
                self._transition(MissionState.ACQUIRE, "lock lost during confirm")

    def _state_film(self) -> None:
        self.ready_to_record = True
        self._relock_count = 0

        if self._track.lock_state == LockState.WEAK:
            if self._weak_since is None:
                self._weak_since = self._now
            elif self._now - self._weak_since >= self.p.weak_lock_timeout_s:
                self._transition(MissionState.DEGRADE, "weak lock timeout")
        else:
            self._weak_since = None

        if self._track.lock_state == LockState.LOST:
            if self._lost_since is None:
                self._lost_since = self._now
            elif self._now - self._lost_since >= self.p.lost_lock_timeout_s:
                self._transition(MissionState.DEGRADE, "lost lock timeout")
        else:
            self._lost_since = None

        # Operator commands while filming
        if self._cmd:
            if self._cmd.action == "stop":
                self._transition(MissionState.RETURN, "operator stop")
            elif self._cmd.action == "set_shot_mode" and self._cmd.shot_mode:
                self.shot_mode = self._cmd.shot_mode
                log.info("Shot mode → %s", self.shot_mode.value)

    def _state_degrade(self) -> None:
        self.ready_to_record = False

        # If lock recovers → back to FILM
        if self._track.lock_state == LockState.LOCKED:
            self._relock_count += 1
            if self._relock_count <= self.p.relock_attempts:
                self._transition(MissionState.FILM, "re-locked")
                return

        if self._track.lock_state == LockState.LOST:
            if self._elapsed() > self.p.degrade_hold_time_s:
                self._transition(MissionState.RETURN, "degrade timeout")

        if self._cmd and self._cmd.action == "relock":
            self._transition(MissionState.ACQUIRE, "operator relock")

    def _state_return(self) -> None:
        self.ready_to_record = False
        # Flight interface will steer us home; we just wait until close
        dist = (self._telem.position - self._telem.home_position).norm()
        if dist < 2.0:
            self._transition(MissionState.LAND, "home reached")

    def _state_land(self) -> None:
        self.ready_to_record = False
        if not self._telem.in_air:
            self._transition(MissionState.IDLE, "landed")

    def _state_emergency(self) -> None:
        self.ready_to_record = False
        if not self._telem.in_air:
            self._transition(MissionState.IDLE, "emergency landed")

    # ════════════════════════════════════════════════════════════════════
    #  HELPERS
    # ════════════════════════════════════════════════════════════════════

    def _apply_operator_tuning(self) -> None:
        if not self._cmd:
            return
        if self._cmd.action != "set_distance":
            return
        if self._cmd.desired_distance is None:
            return
        if self._cmd.desired_distance <= 0.0:
            log.warning("Ignoring invalid desired distance %.2f m", self._cmd.desired_distance)
            return
        self.desired_distance = self._cmd.desired_distance
        log.info("Desired distance -> %.1f m", self.desired_distance)

    def _handle_emergency_command(self) -> bool:
        if self._cmd and self._cmd.action == "emergency_stop":
            self._transition(MissionState.EMERGENCY, "operator emergency stop")
            return True
        if self._cmd and self._cmd.action == "stop" and self.state not in (
            MissionState.IDLE, MissionState.LAND, MissionState.EMERGENCY
        ):
            self._transition(MissionState.RETURN, "operator stop")
            return True
        return False

    def _transition(self, new: MissionState, reason: str) -> None:
        old = self.state
        if old == new:
            return
        self.state = new
        self._state_enter_time = self._now
        self._lock_stable_since = None
        self._weak_since = None
        self._lost_since = None
        log.info("STATE %s → %s  (%s)", old.value, new.value, reason)
        bus.publish(SystemEvent(
            timestamp=self._timestamp,
            source="mission",
            event="state_change",
            payload={"old": old.value, "new": new.value, "reason": reason},
        ))

    def _elapsed(self) -> float:
        return self._now - self._state_enter_time

    def _build_status(self) -> MissionStatus:
        return MissionStatus(
            timestamp=self._timestamp,
            state=self.state,
            shot_mode=self.shot_mode,
            desired_distance=self.desired_distance,
            ready_to_record=self.ready_to_record,
            stability=self.stability,
            safety=self._safety,
            message=f"{self.state.value}",
        )
