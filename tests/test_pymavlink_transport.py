"""
Unit tests for the optional pymavlink transport adapter.
"""

from types import SimpleNamespace

from flight.flight_interface import AutopilotSetpointCommand
from flight.pymavlink_transport import PymavlinkTransport


class _FakeMav:
    def __init__(self) -> None:
        self.position_calls: list[tuple[object, ...]] = []
        self.command_calls: list[tuple[object, ...]] = []
        self.mode_calls: list[tuple[object, ...]] = []

    def set_position_target_local_ned_send(self, *args):
        self.position_calls.append(args)

    def command_long_send(self, *args):
        self.command_calls.append(args)

    def set_mode_send(self, *args):
        self.mode_calls.append(args)


class _FakeMaster:
    def __init__(self) -> None:
        self.target_system = 1
        self.target_component = 191
        self.mav = _FakeMav()

    def mode_mapping(self):
        return {"GUIDED": 4}


def _command() -> AutopilotSetpointCommand:
    return AutopilotSetpointCommand(
        time_boot_ms=120,
        coordinate_frame="LOCAL_NED",
        type_mask=2496,
        x=1.0,
        y=2.0,
        z=-3.0,
        vx=0.4,
        vy=0.5,
        vz=0.6,
        yaw=0.7,
        yaw_rate=0.8,
        source_timestamp=1.25,
    )


class TestPymavlinkTransport:
    def test_publish_setpoint_uses_mavlink_send_api(self):
        master = _FakeMaster()
        mavlink = SimpleNamespace(
            MAV_FRAME_LOCAL_NED=1,
            MAV_FRAME_BODY_NED=8,
            MAV_CMD_COMPONENT_ARM_DISARM=400,
            MAV_MODE_FLAG_CUSTOM_MODE_ENABLED=1,
        )
        transport = PymavlinkTransport(master, mavlink)

        transport.publish_setpoint(_command())

        call = master.mav.position_calls[0]
        assert call[0] == 120
        assert call[1] == 1
        assert call[2] == 191
        assert call[3] == 1
        assert call[4] == 2496
        assert call[5:11] == (1.0, 2.0, -3.0, 0.4, 0.5, 0.6)
        assert call[-2:] == (0.7, 0.8)

    def test_arm_disarm_and_mode_commands_are_forwarded(self):
        master = _FakeMaster()
        mavlink = SimpleNamespace(
            MAV_FRAME_LOCAL_NED=1,
            MAV_FRAME_BODY_NED=8,
            MAV_CMD_COMPONENT_ARM_DISARM=400,
            MAV_MODE_FLAG_CUSTOM_MODE_ENABLED=1,
        )
        transport = PymavlinkTransport(master, mavlink, offboard_mode="GUIDED")

        transport.arm()
        transport.disarm()
        transport.set_offboard_mode()

        assert master.mav.command_calls[0][:5] == (1, 191, 400, 0, 1.0)
        assert master.mav.command_calls[1][:5] == (1, 191, 400, 0, 0.0)
        assert master.mav.mode_calls[0] == (1, 1, 4)

    def test_unknown_mode_raises_clear_error(self):
        master = _FakeMaster()
        mavlink = SimpleNamespace(
            MAV_FRAME_LOCAL_NED=1,
            MAV_FRAME_BODY_NED=8,
            MAV_CMD_COMPONENT_ARM_DISARM=400,
            MAV_MODE_FLAG_CUSTOM_MODE_ENABLED=1,
        )
        transport = PymavlinkTransport(master, mavlink, offboard_mode="OFFBOARD")

        try:
            transport.set_offboard_mode()
        except RuntimeError as exc:
            assert "OFFBOARD" in str(exc)
        else:
            raise AssertionError("Expected RuntimeError for unavailable mode")
