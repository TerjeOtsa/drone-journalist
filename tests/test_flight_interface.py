"""
Unit tests for the Flight Interface boundary adapter.
"""

from config.parameters import FlightInterfaceParams
from flight.flight_interface import FlightInterface, MemoryAutopilotTransport
from interfaces.clock import SimClock
from interfaces.schemas import DroneTelemetry, FlightSetpoint, Vec3


def _telem(x=0.0, y=0.0, z=-2.5, yaw=0.0) -> DroneTelemetry:
    return DroneTelemetry(
        position=Vec3(x, y, z),
        attitude_euler=Vec3(0.0, 0.0, yaw),
    )


class TestFlightInterface:
    def test_builds_transport_command_from_setpoint(self):
        clock = SimClock(start_time=0.0)
        transport = MemoryAutopilotTransport()
        fi = FlightInterface(
            FlightInterfaceParams(),
            transport=transport,
            clock=clock,
        )

        sp = FlightSetpoint(
            timestamp=1.25,
            position=Vec3(1.0, 2.0, -3.0),
            velocity=Vec3(0.4, 0.5, 0.6),
            yaw=0.7,
        )
        fi.send(sp, now_s=clock.monotonic())
        applied = fi.tick(_telem(), now_s=clock.monotonic())

        cmd = transport.last_command
        assert applied == sp
        assert cmd is not None
        assert cmd.time_boot_ms == 0
        assert cmd.coordinate_frame == "LOCAL_NED"
        assert cmd.type_mask == 2496
        assert cmd.x == 1.0 and cmd.y == 2.0 and cmd.z == -3.0
        assert cmd.vx == 0.4 and cmd.vy == 0.5 and cmd.vz == 0.6
        assert cmd.yaw == 0.7
        assert cmd.yaw_rate == 0.0
        assert cmd.source_timestamp == 1.25

    def test_watchdog_publishes_hover_setpoint(self):
        clock = SimClock(start_time=0.0)
        transport = MemoryAutopilotTransport()
        fi = FlightInterface(
            FlightInterfaceParams(timeout_no_setpoint_s=0.1),
            transport=transport,
            clock=clock,
        )

        telem = _telem(x=3.0, y=-1.0, z=-2.0, yaw=1.2)
        clock.advance(0.2)
        applied = fi.tick(telem, now_s=clock.monotonic())

        assert applied is not None
        assert applied.position == Vec3(3.0, -1.0, -2.0)
        assert applied.velocity == Vec3()
        assert applied.yaw == 1.2
        assert fi.watchdog_hovers == 1
        assert transport.last_command is not None

    def test_control_commands_are_forwarded_to_transport(self):
        transport = MemoryAutopilotTransport()
        fi = FlightInterface(transport=transport)

        fi.arm()
        fi.set_offboard_mode()
        fi.disarm()

        assert transport.arm_calls == 1
        assert transport.offboard_calls == 1
        assert transport.disarm_calls == 1
        assert fi.is_offboard is True
        assert fi.is_armed is False
