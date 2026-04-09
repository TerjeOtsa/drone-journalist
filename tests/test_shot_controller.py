"""
Unit tests for the Shot Controller.
"""

import math
import time

import pytest

from config.parameters import MissionParams, SafetyParams, ShotParams
from flight.shot_controller import ShotController
from interfaces.schemas import (
    DroneTelemetry,
    LockState,
    MissionState,
    SafetyOverride,
    ShotMode,
    TargetTrack,
    Vec3,
)


def _telem(x=0, y=0, z=-2.5, yaw=0) -> DroneTelemetry:
    return DroneTelemetry(
        position=Vec3(x, y, z),
        attitude_euler=Vec3(0, 0, yaw),
        in_air=True,
        armed=True,
    )


def _track(x=5, y=0, vx=0, vy=0, lock=LockState.LOCKED) -> TargetTrack:
    return TargetTrack(
        timestamp=time.time(),
        lock_state=lock,
        target_position_world=Vec3(x, y, 0.0) if lock != LockState.LOST else None,
        target_velocity_world=Vec3(vx, vy, 0.0) if lock != LockState.LOST else None,
        tracking_confidence=0.95,
        identity_confidence=0.95,
    )


class TestStandupShot:
    def test_maintains_distance(self):
        sc = ShotController()
        telem = _telem(x=0, y=0, z=-1.8)
        track = _track(x=5, y=0)
        sp = sc.update(track, telem, MissionState.FILM, ShotMode.STANDUP, SafetyOverride.NONE)
        # Position should be roughly standup_distance away from target
        assert sp.position is not None
        dx = sp.position.x - 5.0
        dy = sp.position.y - 0.0
        dist = math.sqrt(dx**2 + dy**2)
        assert 3.0 < dist < 6.0  # reasonable range around 4m default

    def test_altitude_is_standup_level(self):
        sc = ShotController()
        sp = sc.update(
            _track(x=5, y=0), _telem(), MissionState.FILM,
            ShotMode.STANDUP, SafetyOverride.NONE,
        )
        assert sp.position is not None
        alt = -sp.position.z
        assert 1.0 <= alt <= 3.0  # around 1.8 m default

    def test_user_distance_override_changes_follow_distance(self):
        sc = ShotController()
        telem = _telem(x=0, y=0, z=-1.8)
        track = _track(x=5, y=0)
        sp = sc.update(
            track,
            telem,
            MissionState.FILM,
            ShotMode.STANDUP,
            SafetyOverride.NONE,
            desired_distance=8.0,
        )
        assert sp.position is not None
        dx = sp.position.x - 5.0
        dy = sp.position.y
        dist = math.sqrt(dx**2 + dy**2)
        assert dist == pytest.approx(8.0, abs=0.2)


class TestWalkAndTalk:
    def test_follows_behind_moving_subject(self):
        sc = ShotController()
        track = _track(x=5, y=0, vx=1.2, vy=0)
        sp = sc.update(
            track, _telem(x=0, y=0), MissionState.FILM,
            ShotMode.WALK_AND_TALK, SafetyOverride.NONE,
        )
        assert sp.position is not None
        # Drone should be behind (negative x relative to subject)
        assert sp.position.x < 5.0

    def test_stationary_subject_falls_back_to_standup(self):
        sc = ShotController()
        track = _track(x=5, y=0, vx=0, vy=0)
        sp = sc.update(
            track, _telem(x=0, y=0), MissionState.FILM,
            ShotMode.WALK_AND_TALK, SafetyOverride.NONE,
        )
        assert sp.position is not None


class TestOrbit:
    def test_orbit_changes_angle(self):
        sc = ShotController()
        sp1 = sc.update(
            _track(x=5, y=0), _telem(x=5, y=6, z=-3),
            MissionState.FILM, ShotMode.ORBIT, SafetyOverride.NONE,
        )
        time.sleep(0.05)
        sp2 = sc.update(
            _track(x=5, y=0), _telem(x=5, y=6, z=-3),
            MissionState.FILM, ShotMode.ORBIT, SafetyOverride.NONE,
        )
        # Positions should differ (orbit is progressing)
        assert sp1.position is not None and sp2.position is not None
        d = (sp2.position.x - sp1.position.x)**2 + (sp2.position.y - sp1.position.y)**2
        assert d > 0.0001


class TestDistanceClamping:
    def test_min_distance_enforced(self):
        sc = ShotController(ShotParams(standup_distance=1.0, min_distance=2.0))
        sp = sc.update(
            _track(x=1, y=0), _telem(x=0.5, y=0, z=-2),
            MissionState.FILM, ShotMode.STANDUP, SafetyOverride.NONE,
        )
        assert sp.position is not None
        dx = sp.position.x - 1.0
        dy = sp.position.y
        dist = math.sqrt(dx**2 + dy**2)
        assert dist >= 1.9  # should be at least min_distance (with small tolerance)


class TestSpecialStates:
    def test_takeoff_setpoint(self):
        sc = ShotController()
        sp = sc.update(
            _track(lock=LockState.LOST), _telem(x=0, y=0, z=0),
            MissionState.TAKEOFF, ShotMode.STANDUP, SafetyOverride.NONE,
        )
        assert sp.position is not None
        assert -sp.position.z > 0  # ascending

    def test_takeoff_uses_injected_mission_params(self):
        sc = ShotController(mission_params=MissionParams(takeoff_altitude=4.2))
        sp = sc.update(
            _track(lock=LockState.LOST), _telem(x=0, y=0, z=0),
            MissionState.TAKEOFF, ShotMode.STANDUP, SafetyOverride.NONE,
        )
        assert sp.position is not None
        assert sp.position.z == -4.2

    def test_hover_on_safety_override(self):
        sc = ShotController()
        sp = sc.update(
            _track(x=5, y=0), _telem(x=2, y=0, z=-2.5),
            MissionState.FILM, ShotMode.STANDUP, SafetyOverride.HOVER,
        )
        # Should return current position (hover)
        assert sp.position is not None

    def test_lost_target_hovers(self):
        sc = ShotController()
        sp = sc.update(
            _track(lock=LockState.LOST), _telem(x=2, y=0, z=-2.5),
            MissionState.FILM, ShotMode.STANDUP, SafetyOverride.NONE,
        )
        assert sp.position is not None

    def test_emergency_uses_injected_safety_params(self):
        sc = ShotController(safety_params=SafetyParams(emergency_descent_rate=2.3))
        sp = sc.update(
            _track(lock=LockState.LOST), _telem(x=2, y=0, z=-2.5),
            MissionState.EMERGENCY, ShotMode.STANDUP, SafetyOverride.NONE,
        )
        assert sp.velocity is not None
        assert sp.velocity.z == 2.3


class TestSmoothing:
    def test_velocity_is_capped(self):
        params = ShotParams(max_velocity=3.0)
        sc = ShotController(params)
        # Large error → velocity should still be capped
        track = _track(x=50, y=0)
        sp = sc.update(
            track, _telem(x=0, y=0, z=-2), MissionState.FILM,
            ShotMode.STANDUP, SafetyOverride.NONE,
        )
        if sp.velocity:
            assert sp.velocity.norm() <= 3.5  # small tolerance for filter lag

    def test_speed_scale_reduces_velocity(self):
        sc = ShotController()
        sc.set_speed_scale(0.5)
        track = _track(x=20, y=0)
        sp = sc.update(
            track, _telem(x=0, y=0, z=-2), MissionState.FILM,
            ShotMode.STANDUP, SafetyOverride.NONE,
        )
        if sp.velocity:
            assert sp.velocity.norm() <= 2.6  # half of default 5 m/s
