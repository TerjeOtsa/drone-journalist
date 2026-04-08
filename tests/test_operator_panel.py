"""
Tests for the operator-panel interaction model.
"""

import time

from interfaces.schemas import LockState, MissionState, ShotMode
from product.operator_panel import OperatorPanelController
from product.schemas import (
    ConfidenceSnapshot,
    LinkStats,
    LockStatusView,
    MissionView,
    ProductSnapshot,
    RecordingState,
)


def _snapshot(
    *,
    mission_state: MissionState = MissionState.IDLE,
    shot_mode: ShotMode = ShotMode.WALK_AND_TALK,
    desired_distance: float | None = None,
    ready_to_record: bool = False,
    recording_state: RecordingState = RecordingState.IDLE,
    battery_percent: float | None = 82.0,
    clip_id: str | None = None,
) -> ProductSnapshot:
    return ProductSnapshot(
        timestamp=time.time(),
        drone_id="drn_001",
        session_id="sess_001",
        profile="walk",
        mission=MissionView(
            timestamp=time.time(),
            mission_state=mission_state,
            shot_mode=shot_mode,
            desired_distance=desired_distance,
            ready_to_record=ready_to_record,
            message=mission_state.value,
        ),
        lock=LockStatusView(
            timestamp=time.time(),
            lock_state=LockState.LOCKED,
            confidence=ConfidenceSnapshot(identity=0.9, tracking=0.8, effective=0.8),
            target_world_valid=True,
            target_velocity_valid=True,
        ),
        recording_state=recording_state,
        active_clip_id=clip_id,
        battery_percent=battery_percent,
        link=LinkStats(),
    )


class TestOperatorPanelController:
    def test_launch_sends_start_and_distance_and_defers_shot_mode(self):
        controller = OperatorPanelController(default_shot_mode=ShotMode.ORBIT, default_distance_m=8.0)

        commands = controller.launch_commands(timestamp=10.0)

        assert [command.action for command in commands] == ["start", "set_distance"]
        assert commands[1].desired_distance == 8.0

        deferred = controller.maybe_emit_deferred_commands(
            _snapshot(mission_state=MissionState.FILM)
        )
        assert len(deferred) == 1
        assert deferred[0].action == "set_shot_mode"
        assert deferred[0].shot_mode == ShotMode.ORBIT

    def test_distance_is_clamped_to_safe_range(self):
        controller = OperatorPanelController(min_distance_m=2.0, max_distance_m=10.0)

        assert controller.set_selected_distance(0.5) == 2.0
        assert controller.set_selected_distance(14.0) == 10.0

    def test_view_state_prefers_snapshot_distance_and_recording_badge(self):
        controller = OperatorPanelController(default_distance_m=5.0)

        state = controller.build_view_state(
            _snapshot(
                mission_state=MissionState.FILM,
                desired_distance=7.5,
                ready_to_record=True,
                recording_state=RecordingState.RECORDING_CONFIRMED,
                clip_id="clip_009",
            )
        )

        assert state.primary_badge == "Recording"
        assert state.distance_label == "7.5 m follow distance"
        assert "Clip: clip_009" in state.detail_label

    def test_default_view_state_is_action_oriented(self):
        controller = OperatorPanelController(default_distance_m=6.0)

        state = controller.build_view_state()

        assert state.primary_badge == "Ready"
        assert state.distance_label == "6.0 m follow distance"
        assert "Choose a shot" in state.detail_label
