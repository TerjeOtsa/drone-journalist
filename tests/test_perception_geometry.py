"""
Unit tests for camera-relative ground projection utilities.
"""

from perception.geometry import MonocularGroundProjector, MonocularProjectorParams
from perception.schemas import ImageTarget


def test_project_center_footpoint_to_forward_ground_point():
    projector = MonocularGroundProjector(
        MonocularProjectorParams(
            image_width=1280,
            image_height=720,
            hfov_deg=80.0,
            vfov_deg=50.0,
            camera_height_m=1.6,
            camera_down_tilt_deg=20.0,
        )
    )

    target = ImageTarget(
        cx_norm=0.50,
        cy_norm=0.65,
        w_norm=0.10,
        h_norm=0.30,
        footpoint_norm=(0.50, 0.80),
    )
    world = projector.project(target)

    assert world is not None
    assert world.frame == "camera_local"
    assert world.position_m.x > 0.0
    assert abs(world.position_m.y) < 1e-6


def test_project_returns_none_near_horizon():
    projector = MonocularGroundProjector(
        MonocularProjectorParams(
            camera_height_m=1.6,
            camera_down_tilt_deg=5.0,
            min_depression_deg=4.0,
        )
    )

    target = ImageTarget(
        cx_norm=0.50,
        cy_norm=0.48,
        w_norm=0.10,
        h_norm=0.20,
        footpoint_norm=(0.50, 0.45),
    )
    assert projector.project(target) is None
