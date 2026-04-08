"""
Tests for the product-side adapters.
"""

import time

from interfaces.schemas import (
    AppCommand,
    LockState,
    MissionState,
    MissionStatus,
    SafetyOverride,
    SafetyStatus,
    ShotMode,
    StabilityLevel,
    SystemEvent,
    TargetTrack,
    Vec3,
)
from product.adapters import EventMirror, ProductCommandAdapter, ProductStateAdapter
from product.schemas import LinkStats, RecordingState


def _track(
    lock: LockState = LockState.LOCKED,
    identity: float = 0.91,
    tracking: float = 0.87,
) -> TargetTrack:
    return TargetTrack(
        timestamp=time.time(),
        lock_state=lock,
        identity_confidence=identity,
        tracking_confidence=tracking,
        target_position_world=Vec3(5.0, 0.0, 0.0),
        target_velocity_world=Vec3(1.0, 0.0, 0.0),
    )


def _mission() -> MissionStatus:
    return MissionStatus(
        timestamp=time.time(),
        state=MissionState.FILM,
        shot_mode=ShotMode.WALK_AND_TALK,
        desired_distance=6.5,
        ready_to_record=True,
        stability=StabilityLevel.NOMINAL,
        safety=SafetyStatus(active_override=SafetyOverride.NONE),
        message="film",
    )


class TestProductStateAdapter:
    def test_lock_view_uses_min_confidence_as_effective(self):
        adapter = ProductStateAdapter()
        view = adapter.make_lock_view(_track(identity=0.95, tracking=0.61))

        assert view.lock_state == LockState.LOCKED
        assert view.confidence.identity == 0.95
        assert view.confidence.tracking == 0.61
        assert view.confidence.effective == 0.61
        assert view.target_world_valid is True
        assert view.target_velocity_valid is True

    def test_snapshot_includes_mission_lock_and_link(self):
        adapter = ProductStateAdapter(drone_id="drn_001", profile="walk")
        snapshot = adapter.make_snapshot(
            "sess_001",
            _mission(),
            _track(),
            recording_state=RecordingState.RECORDING_CONFIRMED,
            active_clip_id="clip_001",
            link=LinkStats(connected=True, rssi_dbm=-58, preview_latency_ms=120),
        )
        payload = snapshot.to_dict()

        assert payload["drone_id"] == "drn_001"
        assert payload["session_id"] == "sess_001"
        assert payload["mission"]["mission_state"] == "film"
        assert payload["mission"]["shot_mode"] == "walk_and_talk"
        assert payload["mission"]["desired_distance"] == 6.5
        assert payload["recording_state"] == "recording_confirmed"
        assert payload["active_clip_id"] == "clip_001"
        assert payload["link"]["rssi_dbm"] == -58
        assert payload["lock"]["confidence"]["effective"] == 0.87


class TestProductCommandAdapter:
    def test_launch_with_requested_mode_creates_deferred_mode_plan(self):
        adapter = ProductCommandAdapter()
        plan = adapter.launch_plan(
            timestamp=12.3,
            requested_shot_mode=ShotMode.ORBIT,
        )

        assert len(plan.immediate) == 1
        assert isinstance(plan.immediate[0], AppCommand)
        assert plan.immediate[0].action == "start"
        assert plan.deferred_until_state == "film"
        assert len(plan.deferred) == 1
        assert plan.deferred[0].action == "set_shot_mode"
        assert plan.deferred[0].shot_mode == ShotMode.ORBIT

    def test_mode_and_relock_commands_map_cleanly(self):
        adapter = ProductCommandAdapter()

        mode_cmd = adapter.set_mode(5.0, ShotMode.STANDUP)
        distance_cmd = adapter.set_distance(5.5, 8.0)
        relock_cmd = adapter.relock(6.0)
        stop_cmd = adapter.stop(7.0)
        emergency_cmd = adapter.emergency_stop(8.0)

        assert mode_cmd.action == "set_shot_mode"
        assert mode_cmd.shot_mode == ShotMode.STANDUP
        assert distance_cmd.action == "set_distance"
        assert distance_cmd.desired_distance == 8.0
        assert relock_cmd.action == "relock"
        assert stop_cmd.action == "stop"
        assert emergency_cmd.action == "emergency_stop"


class TestEventMirror:
    def test_mirror_assigns_monotonic_sequence(self):
        mirror = EventMirror()

        first = mirror.mirror_system_event(
            SystemEvent(
                timestamp=1.0,
                source="mission",
                event="state_change",
                payload={"old": "lock", "new": "film"},
            ),
            session_id="sess_001",
            monotonic_time=0.5,
        )
        second = mirror.mirror_system_event(
            SystemEvent(
                timestamp=2.0,
                source="safety",
                event="safety_override",
                payload={"override": "return_home"},
            ),
            session_id="sess_001",
            monotonic_time=0.6,
        )

        assert first.seq == 1
        assert second.seq == 2
        assert second.monotonic_time == 0.6
        assert second.payload["override"] == "return_home"
