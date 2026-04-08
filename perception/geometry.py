"""
Geometry helpers for camera-relative subject estimation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from interfaces.schemas import Vec3
from perception.schemas import ImageTarget, WorldTarget


@dataclass
class MonocularProjectorParams:
    """
    Camera-relative ground-plane projector.

    This produces a camera-local estimate suitable for bench testing.
    It does not replace a telemetry-aware NED projection for real flight.
    """

    image_width: int = 1280
    image_height: int = 720
    hfov_deg: float = 78.0
    vfov_deg: float = 49.0
    camera_height_m: float = 1.6
    camera_down_tilt_deg: float = 18.0
    min_depression_deg: float = 2.5


class MonocularGroundProjector:
    """Project a person's image foot-point onto a flat ground plane."""

    def __init__(self, params: MonocularProjectorParams | None = None) -> None:
        self.p = params or MonocularProjectorParams()

    def project(self, image_target: ImageTarget) -> Optional[WorldTarget]:
        foot = image_target.footpoint_norm
        if foot is None:
            foot = (image_target.cx_norm, image_target.cy_norm + 0.5 * image_target.h_norm)

        u = max(0.0, min(1.0, foot[0]))
        v = max(0.0, min(1.0, foot[1]))

        hfov = math.radians(self.p.hfov_deg)
        vfov = math.radians(self.p.vfov_deg)
        azimuth = (u - 0.5) * hfov
        vertical_offset = (v - 0.5) * vfov
        depression = math.radians(self.p.camera_down_tilt_deg) + vertical_offset

        if depression <= math.radians(self.p.min_depression_deg):
            return None

        ground_range = self.p.camera_height_m / math.tan(depression)
        x = ground_range * math.cos(azimuth)
        y = ground_range * math.sin(azimuth)

        return WorldTarget(
            position_m=Vec3(x, y, 0.0),
            velocity_mps=None,
            covariance_diag=(0.8, 0.8, 2.0),
            frame="camera_local",
            source="monocular_ground_plane",
        )
