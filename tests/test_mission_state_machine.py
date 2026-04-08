"""
Unit tests for the Mission State Machine.
"""

import time
import pytest
from unittest.mock import patch

from config.parameters import MissionParams
from flight.mission_state_machine import MissionStateMachine
from interfaces.schemas import (
    AppCommand,
    DroneTelemetry,
    LockState,
    MissionState,
    SafetyOverride,
    SafetyStatus,
    ShotMode,
    StabilityLevel,
    TargetTrack,
    Vec3,
)


def _telem(armed=True, in_air=False, gps_fix=3, batt=80.0, alt=0.0) -> DroneTelemetry:
    return DroneTelemetry(
        armed=armed, in_air=in_air, gps_fix=gps_fix,
        battery_percent=batt,
        position=Vec3(0, 0, -alt),
    )

def _track(lock=LockState.LOST, world=None) -> TargetTrack:
    return TargetTrack(
        timestamp=time.time(),
        lock_state=lock,
        target_position_world=world,
        tracking_confidence=0.9 if lock == LockState.LOCKED else 0.0,
        identity_confidence=0.9 if lock == LockState.LOCKED else 0.0,
    )

def _safety(override=SafetyOverride.NONE) -> SafetyStatus:
    return SafetyStatus(active_override=override)


class TestMissionIdleToTakeoff:
    def test_start_command_transitions_to_takeoff(self):
        msm = MissionStateMachine()
        cmd = AppCommand(action="start")
        msm.update(_track(), _telem(gps_fix=3, batt=80), _safety(), StabilityLevel.NOMINAL, cmd)
        assert msm.state == MissionState.TAKEOFF

    def test_start_with_low_battery_stays_idle(self):
        msm = MissionStateMachine()
        cmd = AppCommand(action="start")
        msm.update(_track(), _telem(gps_fix=3, batt=15), _safety(), StabilityLevel.NOMINAL, cmd)
        assert msm.state == MissionState.IDLE

    def test_start_with_no_gps_stays_idle(self):
        msm = MissionStateMachine()
        cmd = AppCommand(action="start")
        msm.update(_track(), _telem(gps_fix=0, batt=80), _safety(), StabilityLevel.NOMINAL, cmd)
        assert msm.state == MissionState.IDLE


class TestTakeoffToAcquire:
    def test_altitude_reached_transitions_to_acquire(self):
        msm = MissionStateMachine()
        msm.state = MissionState.TAKEOFF
        msm._state_enter_time = time.monotonic()
        telem = _telem(in_air=True, alt=2.4)  # 90% of 2.5
        msm.update(_track(), telem, _safety(), StabilityLevel.NOMINAL)
        assert msm.state == MissionState.ACQUIRE

    def test_takeoff_timeout_goes_to_land(self):
        params = MissionParams(takeoff_timeout_s=0.01)
        msm = MissionStateMachine(params)
        msm.state = MissionState.TAKEOFF
        msm._state_enter_time = time.monotonic() - 1.0  # well past timeout
        msm.update(_track(), _telem(in_air=True, alt=0.5), _safety(), StabilityLevel.NOMINAL)
        assert msm.state == MissionState.LAND


class TestAcquireToLock:
    def test_locked_target_transitions_to_lock(self):
        msm = MissionStateMachine()
        msm.state = MissionState.ACQUIRE
        msm._state_enter_time = time.monotonic()
        msm.update(
            _track(LockState.LOCKED, Vec3(5, 0, 0)),
            _telem(in_air=True, alt=2.5),
            _safety(), StabilityLevel.NOMINAL,
        )
        assert msm.state == MissionState.LOCK

    def test_acquire_timeout_goes_to_return(self):
        params = MissionParams(acquire_timeout_s=0.01)
        msm = MissionStateMachine(params)
        msm.state = MissionState.ACQUIRE
        msm._state_enter_time = time.monotonic() - 1.0
        msm.update(_track(LockState.LOST), _telem(in_air=True, alt=2.5), _safety(), StabilityLevel.NOMINAL)
        assert msm.state == MissionState.RETURN


class TestLockToFilm:
    def test_stable_lock_transitions_to_film(self):
        params = MissionParams(lock_confirm_time_s=0.01)
        msm = MissionStateMachine(params)
        msm.state = MissionState.LOCK
        msm._state_enter_time = time.monotonic()
        # First tick starts the lock timer
        msm.update(
            _track(LockState.LOCKED, Vec3(5, 0, 0)),
            _telem(in_air=True, alt=2.5), _safety(), StabilityLevel.NOMINAL,
        )
        time.sleep(0.02)
        # Second tick should confirm
        msm.update(
            _track(LockState.LOCKED, Vec3(5, 0, 0)),
            _telem(in_air=True, alt=2.5), _safety(), StabilityLevel.NOMINAL,
        )
        assert msm.state == MissionState.FILM


class TestFilmDegradation:
    def test_lost_lock_goes_to_degrade(self):
        params = MissionParams(lost_lock_timeout_s=0.01)
        msm = MissionStateMachine(params)
        msm.state = MissionState.FILM
        msm._state_enter_time = time.monotonic()
        msm.update(_track(LockState.LOST), _telem(in_air=True, alt=2.5), _safety(), StabilityLevel.NOMINAL)
        time.sleep(0.02)
        msm.update(_track(LockState.LOST), _telem(in_air=True, alt=2.5), _safety(), StabilityLevel.NOMINAL)
        assert msm.state == MissionState.DEGRADE

    def test_ready_to_record_true_when_filming(self):
        msm = MissionStateMachine()
        msm.state = MissionState.FILM
        msm._state_enter_time = time.monotonic()
        status = msm.update(
            _track(LockState.LOCKED, Vec3(5, 0, 0)),
            _telem(in_air=True, alt=2.5), _safety(), StabilityLevel.NOMINAL,
        )
        assert status.ready_to_record is True


class TestDegradeRelock:
    def test_relock_transitions_back_to_film(self):
        """DEGRADE → FILM when perception re-locks within relock_attempts."""
        params = MissionParams(lost_lock_timeout_s=0.01, degrade_hold_time_s=30.0)
        msm = MissionStateMachine(params)
        msm.state = MissionState.FILM
        msm._state_enter_time = time.monotonic()

        # Lose lock long enough to enter DEGRADE
        msm.update(_track(LockState.LOST), _telem(in_air=True, alt=2.5), _safety(), StabilityLevel.NOMINAL)
        time.sleep(0.02)
        msm.update(_track(LockState.LOST), _telem(in_air=True, alt=2.5), _safety(), StabilityLevel.NOMINAL)
        assert msm.state == MissionState.DEGRADE

        # Re-lock should go back to FILM
        msm.update(
            _track(LockState.LOCKED, Vec3(5, 0, 0)),
            _telem(in_air=True, alt=2.5), _safety(), StabilityLevel.NOMINAL,
        )
        assert msm.state == MissionState.FILM

    def test_degrade_timeout_goes_to_return(self):
        """DEGRADE → RETURN when degrade_hold_time_s exceeded without re-lock."""
        params = MissionParams(lost_lock_timeout_s=0.01, degrade_hold_time_s=0.01)
        msm = MissionStateMachine(params)
        msm.state = MissionState.DEGRADE
        msm._state_enter_time = time.monotonic() - 1.0  # well past hold time
        msm.update(_track(LockState.LOST), _telem(in_air=True, alt=2.5), _safety(), StabilityLevel.NOMINAL)
        assert msm.state == MissionState.RETURN

    def test_operator_relock_goes_to_acquire(self):
        """Operator 'relock' command during DEGRADE → ACQUIRE."""
        msm = MissionStateMachine()
        msm.state = MissionState.DEGRADE
        msm._state_enter_time = time.monotonic()
        cmd = AppCommand(action="relock")
        msm.update(_track(LockState.LOST), _telem(in_air=True, alt=2.5), _safety(), StabilityLevel.NOMINAL, cmd)
        assert msm.state == MissionState.ACQUIRE


class TestReturnAndLand:
    def test_return_to_land_when_home_reached(self):
        """RETURN → LAND when distance to home < 2 m."""
        msm = MissionStateMachine()
        msm.state = MissionState.RETURN
        msm._state_enter_time = time.monotonic()
        # Position very close to home (0,0,0) — norm must be < 2.0 (includes Z)
        telem = DroneTelemetry(
            armed=True, in_air=True, gps_fix=3, battery_percent=50,
            position=Vec3(0.5, 0.3, -0.2), home_position=Vec3(0, 0, 0),
        )
        msm.update(_track(LockState.LOST), telem, _safety(), StabilityLevel.NOMINAL)
        assert msm.state == MissionState.LAND

    def test_return_stays_if_far_from_home(self):
        """RETURN stays RETURN when still far from home."""
        msm = MissionStateMachine()
        msm.state = MissionState.RETURN
        msm._state_enter_time = time.monotonic()
        telem = DroneTelemetry(
            armed=True, in_air=True, gps_fix=3, battery_percent=50,
            position=Vec3(30, 20, -2.5), home_position=Vec3(0, 0, 0),
        )
        msm.update(_track(LockState.LOST), telem, _safety(), StabilityLevel.NOMINAL)
        assert msm.state == MissionState.RETURN

    def test_land_to_idle_when_on_ground(self):
        """LAND → IDLE once in_air=False."""
        msm = MissionStateMachine()
        msm.state = MissionState.LAND
        msm._state_enter_time = time.monotonic()
        msm.update(_track(), _telem(in_air=False, alt=0), _safety(), StabilityLevel.NOMINAL)
        assert msm.state == MissionState.IDLE


class TestOperatorStopDuringFlight:
    def test_stop_during_film_goes_to_return(self):
        """Operator 'stop' during FILM → RETURN."""
        msm = MissionStateMachine()
        msm.state = MissionState.FILM
        msm._state_enter_time = time.monotonic()
        cmd = AppCommand(action="stop")
        msm.update(
            _track(LockState.LOCKED, Vec3(5, 0, 0)),
            _telem(in_air=True, alt=2.5), _safety(), StabilityLevel.NOMINAL, cmd,
        )
        assert msm.state == MissionState.RETURN

    def test_stop_during_acquire_goes_to_return(self):
        """Operator 'stop' during ACQUIRE → RETURN."""
        msm = MissionStateMachine()
        msm.state = MissionState.ACQUIRE
        msm._state_enter_time = time.monotonic()
        cmd = AppCommand(action="stop")
        msm.update(_track(LockState.LOST), _telem(in_air=True, alt=2.5), _safety(), StabilityLevel.NOMINAL, cmd)
        assert msm.state == MissionState.RETURN


class TestOperatorDistance:
    def test_set_distance_updates_operator_preference(self):
        msm = MissionStateMachine()
        cmd = AppCommand(action="set_distance", desired_distance=7.5)

        status = msm.update(
            _track(LockState.LOST),
            _telem(in_air=False, alt=0.0),
            _safety(),
            StabilityLevel.NOMINAL,
            cmd,
        )

        assert msm.desired_distance == 7.5
        assert status.desired_distance == 7.5
        assert msm.state == MissionState.IDLE

    def test_invalid_distance_is_ignored(self):
        msm = MissionStateMachine()
        msm.desired_distance = 5.0
        cmd = AppCommand(action="set_distance", desired_distance=-1.0)

        status = msm.update(
            _track(LockState.LOST),
            _telem(in_air=False, alt=0.0),
            _safety(),
            StabilityLevel.NOMINAL,
            cmd,
        )

        assert msm.desired_distance == 5.0
        assert status.desired_distance == 5.0


class TestSafetyOverrides:
    def test_land_now_overrides_film(self):
        msm = MissionStateMachine()
        msm.state = MissionState.FILM
        msm._state_enter_time = time.monotonic()
        msm.update(
            _track(LockState.LOCKED, Vec3(5, 0, 0)),
            _telem(in_air=True, alt=2.5),
            _safety(SafetyOverride.LAND_NOW),
            StabilityLevel.NOMINAL,
        )
        assert msm.state == MissionState.LAND

    def test_emergency_stop_command(self):
        msm = MissionStateMachine()
        msm.state = MissionState.FILM
        msm._state_enter_time = time.monotonic()
        cmd = AppCommand(action="emergency_stop")
        msm.update(
            _track(LockState.LOCKED),
            _telem(in_air=True, alt=2.5),
            _safety(), StabilityLevel.NOMINAL, cmd,
        )
        assert msm.state == MissionState.EMERGENCY

    def test_return_home_override(self):
        msm = MissionStateMachine()
        msm.state = MissionState.FILM
        msm._state_enter_time = time.monotonic()
        msm.update(
            _track(LockState.LOCKED),
            _telem(in_air=True, alt=2.5),
            _safety(SafetyOverride.RETURN_HOME),
            StabilityLevel.NOMINAL,
        )
        assert msm.state == MissionState.RETURN
