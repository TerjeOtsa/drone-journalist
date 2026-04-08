"""
Internal schemas for the perception / identity subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from interfaces.schemas import LockState, Vec3


@dataclass
class ImageTarget:
    """Normalized image-space target geometry."""

    cx_norm: float
    cy_norm: float
    w_norm: float
    h_norm: float
    footpoint_norm: Optional[tuple[float, float]] = None
    velocity_px_s: tuple[float, float] = (0.0, 0.0)
    bbox_confidence: float = 0.0

    def as_bbox(self) -> tuple[float, float, float, float]:
        """Return (x, y, w, h) using top-left origin."""
        return (
            self.cx_norm - 0.5 * self.w_norm,
            self.cy_norm - 0.5 * self.h_norm,
            self.w_norm,
            self.h_norm,
        )


@dataclass
class WorldTarget:
    """World-frame target estimate."""

    position_m: Vec3
    velocity_mps: Optional[Vec3] = None
    covariance_diag: tuple[float, float, float] = (1.0, 1.0, 1.0)
    frame: str = "local_ned"
    source: str = "ground_plane_intersection"


@dataclass
class IdentityCues:
    """Optional identity cues in the [0, 1] range."""

    body: Optional[float] = None
    face: Optional[float] = None
    phone: Optional[float] = None
    temporal: Optional[float] = None
    pose: Optional[float] = None


@dataclass
class TrackingCues:
    """Optional tracking cues in the [0, 1] range."""

    detection: Optional[float] = None
    association: Optional[float] = None
    motion: Optional[float] = None
    visibility: Optional[float] = None
    quality: Optional[float] = None


@dataclass
class CandidateObservation:
    """
    Best-current target hypothesis emitted by a perception front-end.

    `candidate_id` is the stable identity hypothesis token, not a raw track ID.
    `track_id` is only for diagnostics.
    """

    timestamp: float
    frame_id: int = 0
    candidate_id: Optional[str] = None
    track_id: Optional[str] = None
    image_target: Optional[ImageTarget] = None
    world_target: Optional[WorldTarget] = None
    identity: IdentityCues = field(default_factory=IdentityCues)
    tracking: TrackingCues = field(default_factory=TrackingCues)
    best_candidate_score: Optional[float] = None
    second_best_score: float = 0.0
    has_fresh_detection: bool = True
    occlusion_ratio: float = 0.0
    blur_score: float = 0.0
    face_used: bool = False
    phone_used: bool = False
    failure_reason: Optional[str] = None
    assist_request: Optional[str] = None


@dataclass
class PerceptionDiagnostics:
    """Optional diagnostic payload for logs and evaluation."""

    target_track_id: Optional[str] = None
    best_candidate_score: float = 0.0
    second_best_score: float = 0.0
    occlusion_ratio: float = 0.0
    blur_score: float = 0.0
    face_used: bool = False
    phone_used: bool = False
    failure_reason: Optional[str] = None
    assist_request: Optional[str] = None


@dataclass
class LockResult:
    """Stateful lock-manager output."""

    timestamp_ms: int
    frame_id: int
    lock_state: LockState
    identity_confidence: float
    tracking_confidence: float
    target_id: Optional[str] = None
    target_position_image: Optional[ImageTarget] = None
    target_position_world: Optional[WorldTarget] = None
    diagnostics: PerceptionDiagnostics = field(default_factory=PerceptionDiagnostics)
    face_score: float = 0.0  # [0,1] face visibility / quality signal for framing

    @property
    def lost_target(self) -> bool:
        return self.lock_state == LockState.LOST

    def to_dict(self) -> dict:
        """Return a JSON-like structure aligned with the design spec."""
        image = self.target_position_image
        world = self.target_position_world
        return {
            "timestamp_ms": self.timestamp_ms,
            "frame_id": self.frame_id,
            "lock_state": self.lock_state.value,
            "lost_target": self.lost_target,
            "identity_confidence": self.identity_confidence,
            "tracking_confidence": self.tracking_confidence,
            "target_position_image": {
                "valid": image is not None,
                "cx_norm": image.cx_norm if image else None,
                "cy_norm": image.cy_norm if image else None,
                "w_norm": image.w_norm if image else None,
                "h_norm": image.h_norm if image else None,
                "footpoint_norm": image.footpoint_norm if image else None,
                "velocity_px_s": image.velocity_px_s if image else None,
                "bbox_confidence": image.bbox_confidence if image else None,
            },
            "target_position_world": {
                "valid": world is not None,
                "frame": world.frame if world else None,
                "position_m": (
                    [world.position_m.x, world.position_m.y, world.position_m.z]
                    if world else None
                ),
                "velocity_mps": (
                    [world.velocity_mps.x, world.velocity_mps.y, world.velocity_mps.z]
                    if world and world.velocity_mps else None
                ),
                "covariance_diag": list(world.covariance_diag) if world else None,
                "source": world.source if world else None,
            },
            "diagnostics": {
                "target_track_id": self.diagnostics.target_track_id,
                "best_candidate_score": self.diagnostics.best_candidate_score,
                "second_best_score": self.diagnostics.second_best_score,
                "occlusion_ratio": self.diagnostics.occlusion_ratio,
                "blur_score": self.diagnostics.blur_score,
                "face_used": self.diagnostics.face_used,
                "phone_used": self.diagnostics.phone_used,
                "failure_reason": self.diagnostics.failure_reason,
                "assist_request": self.diagnostics.assist_request,
            },
        }
