"""
Perception-side utilities for target identity, lock state, and adapters.
"""

from perception.adapter import result_to_lock_event, result_to_target_track
from perception.geometry import MonocularGroundProjector, MonocularProjectorParams
from perception.identity_lock import IdentityLockManager
from perception.live_camera import LiveCameraTracker, LiveTrackerParams
from perception.parameters import IdentityLockParams
from perception.schemas import (
    CandidateObservation,
    IdentityCues,
    ImageTarget,
    LockResult,
    PerceptionDiagnostics,
    TrackingCues,
    WorldTarget,
)

__all__ = [
    "CandidateObservation",
    "IdentityCues",
    "IdentityLockManager",
    "IdentityLockParams",
    "ImageTarget",
    "LiveCameraTracker",
    "LiveTrackerParams",
    "LockResult",
    "MonocularGroundProjector",
    "MonocularProjectorParams",
    "PerceptionDiagnostics",
    "TrackingCues",
    "WorldTarget",
    "result_to_lock_event",
    "result_to_target_track",
]
