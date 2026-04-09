"""
Adapters from perception lock results to project-wide shared schemas.
"""

from __future__ import annotations

from interfaces.schemas import TargetTrack
from perception.schemas import LockResult


def result_to_target_track(result: LockResult) -> TargetTrack:
    """Map a rich perception result onto the shared autonomy boundary."""
    image = result.target_position_image
    world = result.target_position_world
    target_position_world = None
    target_velocity_world = None
    if world is not None and world.frame == "local_ned":
        target_position_world = world.position_m
        target_velocity_world = world.velocity_mps

    return TargetTrack(
        timestamp=result.timestamp_ms / 1000.0,
        target_position_image=(
            (image.cx_norm, image.cy_norm) if image is not None else (0.0, 0.0)
        ),
        target_position_world=target_position_world,
        target_velocity_world=target_velocity_world,
        identity_confidence=result.identity_confidence,
        tracking_confidence=result.tracking_confidence,
        lock_state=result.lock_state,
        bounding_box=image.as_bbox() if image is not None else None,
        face_score=result.face_score,
    )


def result_to_lock_event(result: LockResult) -> dict[str, object]:
    """Reduce the rich perception output to the app-facing lock event."""
    return {
        "lock_state": result.lock_state.value,
        "confidence": min(result.identity_confidence, result.tracking_confidence),
        "target_id": result.target_id,
    }
