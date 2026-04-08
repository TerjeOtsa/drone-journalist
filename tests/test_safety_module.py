"""
Unit tests for the Safety & Geofence Module.
"""

import math
import time
import pytest

from config.parameters import SafetyParams
from flight.safety_module import SafetyModule
from interfaces.schemas import (
    DroneTelemetry,
    FlightSetpoint,
    SafetyOverride,
    Vec3,
)


def _telem(x=0, y=0, alt=2.5, batt=80, in_air=True) -> DroneTelemetry:
    return DroneTelemetry(
        position=Vec3(x, y, -alt),
        in_air=in_air,
        battery_percent=batt,
        home_position=Vec3(0, 0, 0),
    )


class TestGeofence:
    def test_inside_geofence_ok(self):
        sm = SafetyModule()
        sm.heartbeat()
        status = sm.update(_telem(x=10, y=10), FlightSetpoint())
        assert status.geofence_ok is True

    def test_breach_returns_home(self):
        sm = SafetyModule(SafetyParams(geofence_radius=50))
        sm.heartbeat()
        status = sm.update(_telem(x=60, y=0), FlightSetpoint())
        assert status.geofence_ok is False
        assert status.active_override == SafetyOverride.RETURN_HOME

    def test_ceiling_breach(self):
        sm = SafetyModule(SafetyParams(geofence_ceiling=20))
        sm.heartbeat()
        status = sm.update(_telem(alt=25), FlightSetpoint())
        assert status.geofence_ok is False

    def test_warn_margin(self):
        sm = SafetyModule(SafetyParams(geofence_radius=100, geofence_warn_margin=10))
        sm.heartbeat()
        status = sm.update(_telem(x=92, y=0), FlightSetpoint())
        assert status.active_override == SafetyOverride.REDUCE_SPEED


class TestBattery:
    def test_normal_battery_ok(self):
        sm = SafetyModule()
        sm.heartbeat()
        status = sm.update(_telem(batt=80), FlightSetpoint())
        assert status.battery_ok is True

    def test_low_battery_return(self):
        sm = SafetyModule()
        sm.heartbeat()
        status = sm.update(_telem(batt=18), FlightSetpoint())
        assert status.active_override == SafetyOverride.RETURN_HOME

    def test_critical_battery_land(self):
        sm = SafetyModule()
        sm.heartbeat()
        status = sm.update(_telem(batt=8), FlightSetpoint())
        assert status.active_override == SafetyOverride.LAND_NOW


class TestLinkLoss:
    def test_recent_heartbeat_ok(self):
        sm = SafetyModule()
        sm.heartbeat()
        status = sm.update(_telem(), FlightSetpoint())
        assert status.link_ok is True

    def test_heartbeat_timeout_return(self):
        sm = SafetyModule(SafetyParams(heartbeat_timeout_s=0.01))
        sm.heartbeat()
        time.sleep(0.02)
        status = sm.update(_telem(), FlightSetpoint())
        assert status.link_ok is False
        assert status.active_override in (SafetyOverride.RETURN_HOME, SafetyOverride.LAND_NOW)


class TestSubjectProximity:
    def test_safe_distance_ok(self):
        sm = SafetyModule()
        sm.heartbeat()
        status = sm.update(_telem(x=0, y=0), FlightSetpoint(), target_world=Vec3(5, 0, 0))
        assert status.min_distance_ok is True

    def test_too_close_increases_distance(self):
        sm = SafetyModule(SafetyParams(subject_min_distance=3.0))
        sm.heartbeat()
        status = sm.update(
            _telem(x=4, y=0), FlightSetpoint(), target_world=Vec3(5, 0, 0)
        )
        assert status.min_distance_ok is False

    def test_emergency_proximity_hovers(self):
        sm = SafetyModule(SafetyParams(subject_emergency_dist=1.5))
        sm.heartbeat()
        status = sm.update(
            _telem(x=4.5, y=0), FlightSetpoint(), target_world=Vec3(5, 0, 0)
        )
        assert status.active_override == SafetyOverride.HOVER


class TestEmergencyStop:
    def test_emergency_stop_overrides_all(self):
        sm = SafetyModule()
        sm.heartbeat()
        sm.request_emergency_stop()
        status = sm.update(_telem(), FlightSetpoint())
        assert status.active_override == SafetyOverride.EMERGENCY_STOP


class TestLastKnownTargetProximity:
    def test_proximity_uses_stale_target_when_lost(self):
        """Proximity check should use last known position when target is lost."""
        sm = SafetyModule(SafetyParams(subject_min_distance=3.0))
        sm.heartbeat()
        # First update: target visible and too close
        status1 = sm.update(
            _telem(x=4, y=0), FlightSetpoint(), target_world=Vec3(5, 0, 0)
        )
        assert status1.min_distance_ok is False
        # Second update: target lost (None) — should still use stale position
        status2 = sm.update(
            _telem(x=4, y=0), FlightSetpoint(), target_world=None
        )
        assert status2.min_distance_ok is False

    def test_no_proximity_without_any_target(self):
        """If we never had a target, proximity should be ok."""
        sm = SafetyModule()
        sm.heartbeat()
        status = sm.update(_telem(), FlightSetpoint(), target_world=None)
        assert status.min_distance_ok is True


class TestSetpointClamping:
    def test_ceiling_clamp(self):
        sm = SafetyModule(SafetyParams(geofence_ceiling=20))
        sm.heartbeat()
        sp = FlightSetpoint(position=Vec3(0, 0, -50))  # 50 m up
        status = sm.update(_telem(), sp)
        clamped = sm.clamp_setpoint(sp, _telem(), status)
        assert clamped.position is not None
        assert -clamped.position.z <= 20.0

    def test_floor_clamp(self):
        sm = SafetyModule(SafetyParams(geofence_floor=1.0))
        sm.heartbeat()
        sp = FlightSetpoint(position=Vec3(0, 0, -0.3))  # 0.3 m up
        status = sm.update(_telem(), sp)
        clamped = sm.clamp_setpoint(sp, _telem(), status)
        assert clamped.position is not None
        assert -clamped.position.z >= 1.0
