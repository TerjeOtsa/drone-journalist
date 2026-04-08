"""
Flight Interface Module
=======================
Translates FlightSetpoint into MAVLink-compatible commands for PX4/ArduPilot.
In simulation mode it writes to a simple state struct; in production this
would use pymavlink or MAVSDK.

Runs at 50 Hz (highest rate in the companion computer).

Responsibilities:
  • Convert NED setpoints → MAVLink SET_POSITION_TARGET_LOCAL_NED
  • Fallback to hover if no setpoint received within timeout
  • Arm / disarm / mode-change commands
  • Watchdog: if this module stops publishing, autopilot falls back to HOLD
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable, Optional, Protocol

from config.parameters import FlightInterfaceParams
from interfaces.clock import Clock, SystemClock
from interfaces.schemas import DroneTelemetry, FlightSetpoint, Vec3

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AutopilotSetpointCommand:
    """Transport-ready command mirroring MAVLink local-NED fields."""

    time_boot_ms: int
    coordinate_frame: str
    type_mask: int
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    yaw: float
    yaw_rate: float
    source_timestamp: float


class AutopilotTransport(Protocol):
    """Abstract transport seam for simulation, SITL, or hardware adapters."""

    def publish_setpoint(self, command: AutopilotSetpointCommand) -> None: ...

    def arm(self) -> None: ...

    def disarm(self) -> None: ...

    def set_offboard_mode(self) -> None: ...


class MemoryAutopilotTransport:
    """In-memory transport used by tests and the simulator harness."""

    def __init__(self) -> None:
        self.last_command: Optional[AutopilotSetpointCommand] = None
        self.published_commands: int = 0
        self.arm_calls: int = 0
        self.disarm_calls: int = 0
        self.offboard_calls: int = 0

    def publish_setpoint(self, command: AutopilotSetpointCommand) -> None:
        self.last_command = command
        self.published_commands += 1

    def arm(self) -> None:
        self.arm_calls += 1

    def disarm(self) -> None:
        self.disarm_calls += 1

    def set_offboard_mode(self) -> None:
        self.offboard_calls += 1


_IGNORE_PX = 1 << 0
_IGNORE_PY = 1 << 1
_IGNORE_PZ = 1 << 2
_IGNORE_VX = 1 << 3
_IGNORE_VY = 1 << 4
_IGNORE_VZ = 1 << 5
_IGNORE_AX = 1 << 6
_IGNORE_AY = 1 << 7
_IGNORE_AZ = 1 << 8
_IGNORE_YAW_RATE = 1 << 11


class FlightInterface:
    """
    50 Hz bridge between the autonomy stack and the autopilot.

    In production, `send()` would serialise to MAVLink and push to serial/UDP.
    Here it stores the last setpoint for the simulator to consume.
    """

    def __init__(
        self,
        params: FlightInterfaceParams | None = None,
        *,
        transport: AutopilotTransport | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.p = params or FlightInterfaceParams()
        self.clock = clock or SystemClock()
        self.transport = transport or MemoryAutopilotTransport()

        self._last_setpoint: Optional[FlightSetpoint] = None
        self._last_setpoint_time: float = self.clock.monotonic()
        self._last_command: Optional[AutopilotSetpointCommand] = None
        self._armed: bool = False
        self._in_offboard: bool = False
        self.transport_healthy: bool = True
        self.last_transport_error: Optional[str] = None

        # Stats
        self.setpoints_sent: int = 0
        self.watchdog_hovers: int = 0

    # ════════════════════════════════════════════════════════════════════
    #  PUBLIC
    # ════════════════════════════════════════════════════════════════════

    def send(self, setpoint: FlightSetpoint, now_s: float | None = None) -> None:
        """Accept a new setpoint from the shot controller / safety module."""
        self._last_setpoint = setpoint
        self._last_setpoint_time = self.clock.monotonic() if now_s is None else now_s
        self.setpoints_sent += 1

    def tick(
        self,
        telem: DroneTelemetry,
        now_s: float | None = None,
    ) -> Optional[FlightSetpoint]:
        """
        Called at 50 Hz.  Returns the setpoint that was actually pushed
        to the autopilot (may be a watchdog hover).
        """
        now = self.clock.monotonic() if now_s is None else now_s
        elapsed = max(0.0, now - self._last_setpoint_time)
        timestamp_s = self.clock.time() if now_s is None else now_s

        if elapsed > self.p.timeout_no_setpoint_s or self._last_setpoint is None:
            # Watchdog: hold current position
            self.watchdog_hovers += 1
            hover = FlightSetpoint(
                timestamp=timestamp_s,
                position=Vec3(telem.position.x, telem.position.y, telem.position.z),
                velocity=Vec3(),
                yaw=telem.attitude_euler.z,
            )
            self._publish_to_autopilot(hover)
            return hover

        self._publish_to_autopilot(self._last_setpoint)
        return self._last_setpoint

    def arm(self) -> None:
        if self._invoke_transport("arm", self.transport.arm):
            log.info("ARM command sent")
            self._armed = True

    def disarm(self) -> None:
        if self._invoke_transport("disarm", self.transport.disarm):
            log.info("DISARM command sent")
            self._armed = False

    def set_offboard_mode(self) -> None:
        if self._invoke_transport("set offboard mode", self.transport.set_offboard_mode):
            log.info("OFFBOARD mode requested")
            self._in_offboard = True

    @property
    def is_armed(self) -> bool:
        return self._armed

    @property
    def is_offboard(self) -> bool:
        return self._in_offboard

    @property
    def last_command(self) -> Optional[AutopilotSetpointCommand]:
        return self._last_command

    # ════════════════════════════════════════════════════════════════════
    #  INTERNALS
    # ════════════════════════════════════════════════════════════════════

    def _publish_to_autopilot(self, sp: FlightSetpoint) -> None:
        """
        In production this builds the exact command shape a MAVLink/MAVSDK
        transport would need. The default in-memory transport keeps the flight
        interface fully testable without hardware dependencies.
        """
        command = self._build_command(sp)
        self._last_command = command
        self._invoke_transport(
            "publish setpoint",
            lambda: self.transport.publish_setpoint(command),
        )

    def _build_command(self, sp: FlightSetpoint) -> AutopilotSetpointCommand:
        position = sp.position or Vec3()
        velocity = sp.velocity or Vec3()
        coordinate_frame = "BODY_NED" if sp.is_body_frame else "LOCAL_NED"

        return AutopilotSetpointCommand(
            time_boot_ms=int(round(self.clock.monotonic() * 1000.0)),
            coordinate_frame=coordinate_frame,
            type_mask=self._build_type_mask(sp),
            x=position.x,
            y=position.y,
            z=position.z,
            vx=velocity.x,
            vy=velocity.y,
            vz=velocity.z,
            yaw=sp.yaw,
            yaw_rate=0.0 if sp.yaw_rate is None else sp.yaw_rate,
            source_timestamp=sp.timestamp,
        )

    def _build_type_mask(self, sp: FlightSetpoint) -> int:
        mask = _IGNORE_AX | _IGNORE_AY | _IGNORE_AZ
        if sp.position is None:
            mask |= _IGNORE_PX | _IGNORE_PY | _IGNORE_PZ
        if sp.velocity is None:
            mask |= _IGNORE_VX | _IGNORE_VY | _IGNORE_VZ
        if sp.yaw_rate is None:
            mask |= _IGNORE_YAW_RATE
        return mask

    def _invoke_transport(self, action: str, callback: Callable[[], None]) -> bool:
        try:
            callback()
            self.transport_healthy = True
            self.last_transport_error = None
            return True
        except Exception as exc:
            self.transport_healthy = False
            self.last_transport_error = f"{action}: {exc}"
            log.exception("Autopilot transport failed to %s", action)
            return False
