"""
Tests for perception result adapters into shared project interfaces.
"""

from interfaces.schemas import LockState, Vec3
from perception.adapter import result_to_lock_event, result_to_target_track
from perception.schemas import ImageTarget, LockResult, PerceptionDiagnostics, WorldTarget


def test_result_to_target_track_maps_bbox_and_world_fields():
    result = LockResult(
        timestamp_ms=1234,
        frame_id=12,
        lock_state=LockState.LOCKED,
        identity_confidence=0.88,
        tracking_confidence=0.91,
        target_id="subject_primary",
        target_position_image=ImageTarget(
            cx_norm=0.60,
            cy_norm=0.50,
            w_norm=0.20,
            h_norm=0.40,
        ),
        target_position_world=WorldTarget(
            position_m=Vec3(3.0, -1.0, 0.0),
            velocity_mps=Vec3(0.5, 0.1, 0.0),
        ),
        diagnostics=PerceptionDiagnostics(),
    )

    track = result_to_target_track(result)

    assert track.timestamp == 1.234
    assert track.target_position_image == (0.60, 0.50)
    assert track.target_position_world == Vec3(3.0, -1.0, 0.0)
    assert track.target_velocity_world == Vec3(0.5, 0.1, 0.0)
    assert track.bounding_box == (0.50, 0.30, 0.20, 0.40)


def test_result_to_lock_event_uses_conservative_confidence():
    result = LockResult(
        timestamp_ms=1000,
        frame_id=7,
        lock_state=LockState.WEAK,
        identity_confidence=0.82,
        tracking_confidence=0.44,
        target_id="subject_primary",
        diagnostics=PerceptionDiagnostics(),
    )

    event = result_to_lock_event(result)

    assert event == {
        "lock_state": "weak",
        "confidence": 0.44,
        "target_id": "subject_primary",
    }


def test_result_to_target_track_skips_non_ned_world_frames():
    result = LockResult(
        timestamp_ms=1000,
        frame_id=4,
        lock_state=LockState.LOCKED,
        identity_confidence=0.90,
        tracking_confidence=0.88,
        target_id="subject_primary",
        target_position_image=ImageTarget(
            cx_norm=0.50,
            cy_norm=0.50,
            w_norm=0.20,
            h_norm=0.40,
        ),
        target_position_world=WorldTarget(
            position_m=Vec3(2.0, 0.2, 0.0),
            frame="camera_local",
        ),
        diagnostics=PerceptionDiagnostics(),
    )

    track = result_to_target_track(result)

    assert track.target_position_world is None
    assert track.target_velocity_world is None
