"""
Optional pymavlink-based transport for the flight interface.

This module keeps the hardware dependency out of the core autonomy stack while
providing a concrete path to SITL or real autopilots when `pymavlink` is
installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from flight.flight_interface import AutopilotSetpointCommand, AutopilotTransport


@dataclass(frozen=True)
class PymavlinkConnectionConfig:
    endpoint: str
    baud: int = 57600
    source_system: int = 255
    source_component: int = 191
    wait_heartbeat: bool = True
    heartbeat_timeout_s: float = 30.0
    offboard_mode: str = "GUIDED"
    target_system: Optional[int] = None
    target_component: Optional[int] = None


class PymavlinkTransport(AutopilotTransport):
    """Sync transport implementation backed by `pymavlink`."""

    def __init__(
        self,
        master: Any,
        mavlink: Any,
        *,
        offboard_mode: str = "GUIDED",
        target_system: Optional[int] = None,
        target_component: Optional[int] = None,
    ) -> None:
        self.master = master
        self.mavlink = mavlink
        self.offboard_mode = offboard_mode
        self._target_system_override = target_system
        self._target_component_override = target_component

    @classmethod
    def connect(cls, config: PymavlinkConnectionConfig) -> "PymavlinkTransport":
        try:
            from pymavlink import mavutil
        except ImportError as exc:
            raise RuntimeError(
                "pymavlink is not installed. Install requirements-hardware.txt "
                "or add pymavlink to your environment."
            ) from exc

        master = mavutil.mavlink_connection(
            config.endpoint,
            baud=config.baud,
            source_system=config.source_system,
            source_component=config.source_component,
        )
        if config.wait_heartbeat:
            master.wait_heartbeat(timeout=config.heartbeat_timeout_s)

        return cls(
            master,
            mavutil.mavlink,
            offboard_mode=config.offboard_mode,
            target_system=config.target_system,
            target_component=config.target_component,
        )

    def publish_setpoint(self, command: AutopilotSetpointCommand) -> None:
        target_system, target_component = self._resolve_target_ids()
        self.master.mav.set_position_target_local_ned_send(
            command.time_boot_ms,
            target_system,
            target_component,
            self._frame_id(command.coordinate_frame),
            command.type_mask,
            command.x,
            command.y,
            command.z,
            command.vx,
            command.vy,
            command.vz,
            0.0,
            0.0,
            0.0,
            command.yaw,
            command.yaw_rate,
        )

    def arm(self) -> None:
        self._send_arm_command(1.0)

    def disarm(self) -> None:
        self._send_arm_command(0.0)

    def set_offboard_mode(self) -> None:
        mode_mapping = self.master.mode_mapping()
        if not mode_mapping or self.offboard_mode not in mode_mapping:
            raise RuntimeError(f"Autopilot mode '{self.offboard_mode}' is not available")
        target_system, _ = self._resolve_target_ids()
        self.master.mav.set_mode_send(
            target_system,
            self.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_mapping[self.offboard_mode],
        )

    def _send_arm_command(self, arm_value: float) -> None:
        target_system, target_component = self._resolve_target_ids()
        self.master.mav.command_long_send(
            target_system,
            target_component,
            self.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            arm_value,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    def _resolve_target_ids(self) -> tuple[int, int]:
        target_system = self._target_system_override
        if target_system is None:
            target_system = int(getattr(self.master, "target_system", 0))

        target_component = self._target_component_override
        if target_component is None:
            target_component = int(getattr(self.master, "target_component", 0))

        return target_system, target_component

    def _frame_id(self, coordinate_frame: str) -> int:
        frame_map = {
            "LOCAL_NED": self.mavlink.MAV_FRAME_LOCAL_NED,
            "BODY_NED": self.mavlink.MAV_FRAME_BODY_NED,
        }
        if coordinate_frame not in frame_map:
            raise ValueError(f"Unsupported coordinate frame: {coordinate_frame}")
        return frame_map[coordinate_frame]
