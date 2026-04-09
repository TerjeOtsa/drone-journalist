"""
3D visualization for the simulator.

Usage:
    python -m sim.visualize_sim_3d
    python -m sim.visualize_sim_3d --duration 30 --seed 5
    python -m sim.visualize_sim_3d --snapshot sim_3d_snapshot.png --duration 20
    python -m sim.visualize_sim_3d --gif sim_3d.gif --duration 20
    python -m sim.visualize_sim_3d --gif sim_3d.gif --duration 20 --frame-step 3 --export-fps 15
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def _arg_value(name: str) -> str | None:
    if name not in sys.argv:
        return None
    idx = sys.argv.index(name)
    if idx + 1 >= len(sys.argv):
        return None
    return sys.argv[idx + 1]


forced_backend = _arg_value("--backend")
if forced_backend:
    import matplotlib

    matplotlib.use(forced_backend)
elif "--snapshot" in sys.argv or "--gif" in sys.argv:
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from sim.sim_harness import run_simulation  # noqa: E402

LOCK_COLORS = {
    "candidate": "#f4a261",
    "locked": "#2a9d8f",
    "weak": "#e9c46a",
    "lost": "#e76f51",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="3D visualize the drone simulator")
    parser.add_argument("--duration", type=float, default=70.0, help="Simulation duration in seconds")
    parser.add_argument("--dt", type=float, default=0.02, help="Simulation timestep in seconds")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Save a static PNG snapshot instead of opening an interactive window",
    )
    parser.add_argument(
        "--gif",
        type=Path,
        default=None,
        help="Save an animated GIF instead of opening an interactive window",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Optional Matplotlib backend override, e.g. Agg or WebAgg",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=220,
        help="Number of recent samples to keep in trails",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Render every Nth simulation frame for playback/export",
    )
    parser.add_argument(
        "--export-fps",
        type=float,
        default=None,
        help="Optional GIF export FPS override",
    )
    return parser


def _set_equal_3d_bounds(ax, xs, ys, zs, pad: float = 2.0) -> None:
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    span = max(max_x - min_x, max_y - min_y, max_z - min_z, 8.0)
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    cz = 0.5 * (min_z + max_z)
    radius = 0.5 * span + pad
    ax.set_xlim(cx - radius, cx + radius)
    ax.set_ylim(cy - radius, cy + radius)
    ax.set_zlim(max(0.0, cz - radius * 0.30), cz + radius * 0.70)


def _normalize_2d(dx: float, dy: float) -> tuple[float, float]:
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return 1.0, 0.0
    return dx / norm, dy / norm


def _rotation_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ]
    )
    ry = np.array(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ]
    )
    rz = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return rz @ ry @ rx


def _transform_points(points: np.ndarray, origin: tuple[float, float, float], rotation: np.ndarray) -> np.ndarray:
    origin_vec = np.asarray(origin, dtype=float)
    return np.asarray(points, dtype=float) @ rotation.T + origin_vec


def _cuboid_faces(size: tuple[float, float, float]) -> list[np.ndarray]:
    hx, hy, hz = (0.5 * value for value in size)
    vertices = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=float,
    )
    return [
        vertices[[0, 1, 2, 3]],
        vertices[[4, 5, 6, 7]],
        vertices[[0, 1, 5, 4]],
        vertices[[1, 2, 6, 5]],
        vertices[[2, 3, 7, 6]],
        vertices[[3, 0, 4, 7]],
    ]


def _transformed_faces(
    size: tuple[float, float, float],
    origin: tuple[float, float, float],
    rotation: np.ndarray,
) -> list[np.ndarray]:
    return [_transform_points(face, origin, rotation) for face in _cuboid_faces(size)]


def _circle_points(
    center: tuple[float, float, float],
    radius: float,
    *,
    count: int = 40,
    z: float | None = None,
) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * math.pi, count)
    z_value = center[2] if z is None else z
    return np.column_stack(
        [
            center[0] + radius * np.cos(angles),
            center[1] + radius * np.sin(angles),
            np.full_like(angles, z_value),
        ]
    )


def _disc_points(
    local_center: tuple[float, float, float],
    radius: float,
    rotation: np.ndarray,
    origin: tuple[float, float, float],
    *,
    count: int = 28,
) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * math.pi, count)
    local = np.column_stack(
        [
            local_center[0] + radius * np.cos(angles),
            local_center[1] + radius * np.sin(angles),
            np.full_like(angles, local_center[2]),
        ]
    )
    return _transform_points(local, origin, rotation)


def _subject_pose(
    x: float,
    y: float,
    z_ground: float,
    heading_xy: tuple[float, float],
    speed: float,
    t: float,
) -> dict[str, tuple[float, float, float]]:
    fx, fy = heading_xy
    sx, sy = -fy, fx

    lean = 0.05 * min(speed / 1.4, 1.0)
    hip_h = z_ground + 0.96
    chest_h = z_ground + 1.44
    neck_h = z_ground + 1.60
    head_h = z_ground + 1.74

    hip = (x, y, hip_h)
    chest = (x + lean * fx, y + lean * fy, chest_h)
    neck = (x + 1.15 * lean * fx, y + 1.15 * lean * fy, neck_h)
    head = (x + 1.28 * lean * fx, y + 1.28 * lean * fy, head_h)

    shoulder_offset = 0.20
    hip_offset = 0.12
    upper_arm = 0.30
    lower_arm = 0.28
    upper_leg = 0.46
    _lower_leg = 0.45  # noqa: F841  reserved for future leg segment rendering

    gait_amp = 0.20 * min(max(speed / 1.15, 0.10), 1.25)
    phase = 2.0 * math.pi * 1.8 * t * min(max(speed / 1.0, 0.3), 1.6)
    gait = math.sin(phase)
    anti = math.sin(phase + math.pi)

    l_shoulder = (chest[0] + shoulder_offset * sx, chest[1] + shoulder_offset * sy, chest_h)
    r_shoulder = (chest[0] - shoulder_offset * sx, chest[1] - shoulder_offset * sy, chest_h)
    l_hip = (x + hip_offset * sx, y + hip_offset * sy, hip_h)
    r_hip = (x - hip_offset * sx, y - hip_offset * sy, hip_h)

    l_elbow = (
        l_shoulder[0] - 0.12 * anti * fx + 0.05 * sx,
        l_shoulder[1] - 0.12 * anti * fy + 0.05 * sy,
        chest_h - upper_arm + 0.10 * anti,
    )
    r_elbow = (
        r_shoulder[0] - 0.12 * gait * fx - 0.05 * sx,
        r_shoulder[1] - 0.12 * gait * fy - 0.05 * sy,
        chest_h - upper_arm + 0.10 * gait,
    )
    l_hand = (
        l_elbow[0] + 0.20 * anti * fx + 0.03 * sx,
        l_elbow[1] + 0.20 * anti * fy + 0.03 * sy,
        l_elbow[2] - lower_arm + 0.06 * anti,
    )
    r_hand = (
        r_elbow[0] + 0.20 * gait * fx - 0.03 * sx,
        r_elbow[1] + 0.20 * gait * fy - 0.03 * sy,
        r_elbow[2] - lower_arm + 0.06 * gait,
    )

    l_knee = (
        l_hip[0] + 0.15 * gait_amp * fx + 0.03 * sx,
        l_hip[1] + 0.15 * gait_amp * fy + 0.03 * sy,
        hip_h - upper_leg + 0.10 * max(0.0, anti),
    )
    r_knee = (
        r_hip[0] + 0.15 * gait_amp * anti * fx - 0.03 * sx,
        r_hip[1] + 0.15 * gait_amp * anti * fy - 0.03 * sy,
        hip_h - upper_leg + 0.10 * max(0.0, gait),
    )
    l_foot = (
        l_knee[0] + 0.20 * gait_amp * fx + 0.03 * sx,
        l_knee[1] + 0.20 * gait_amp * fy + 0.03 * sy,
        z_ground,
    )
    r_foot = (
        r_knee[0] + 0.20 * gait_amp * anti * fx - 0.03 * sx,
        r_knee[1] + 0.20 * gait_amp * anti * fy - 0.03 * sy,
        z_ground,
    )

    return {
        "hip": hip,
        "chest": chest,
        "neck": neck,
        "head": head,
        "l_shoulder": l_shoulder,
        "r_shoulder": r_shoulder,
        "l_hip": l_hip,
        "r_hip": r_hip,
        "l_elbow": l_elbow,
        "r_elbow": r_elbow,
        "l_hand": l_hand,
        "r_hand": r_hand,
        "l_knee": l_knee,
        "r_knee": r_knee,
        "l_foot": l_foot,
        "r_foot": r_foot,
    }


def _subject_heading(xs: list[float], ys: list[float], i: int) -> tuple[float, float]:
    if i <= 0:
        return _normalize_2d(xs[min(1, len(xs) - 1)] - xs[0], ys[min(1, len(ys) - 1)] - ys[0])
    return _normalize_2d(xs[i] - xs[i - 1], ys[i] - ys[i - 1])


def _speed(xs: list[float], ys: list[float], times: list[float], i: int) -> float:
    if i <= 0:
        return 0.0
    dt = max(times[i] - times[i - 1], 1e-6)
    return math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]) / dt


def _update_line3d(line, p0: tuple[float, float, float], p1: tuple[float, float, float]) -> None:
    line.set_data_3d([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]])


def _set_poly_verts(poly: Poly3DCollection, faces: list[np.ndarray]) -> None:
    poly.set_verts([face.tolist() for face in faces])


def _drone_geometry(
    drone_pos: tuple[float, float, float],
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> dict[str, object]:
    rotation = _rotation_matrix(roll_deg, pitch_deg, yaw_deg)
    visual_scale = 1.22
    arm = 0.34 * visual_scale
    rotor_centers_local = np.array(
        [
            [arm, arm, 0.02],
            [arm, -arm, 0.02],
            [-arm, arm, 0.02],
            [-arm, -arm, 0.02],
        ],
        dtype=float,
    )
    rotor_centers = _transform_points(rotor_centers_local, drone_pos, rotation)

    arm_a = (tuple(rotor_centers[0]), tuple(rotor_centers[3]))
    arm_b = (tuple(rotor_centers[1]), tuple(rotor_centers[2]))

    landing_left = _transform_points(
        visual_scale * np.array([[-0.08, 0.16, -0.11], [0.12, 0.16, -0.11]], dtype=float),
        drone_pos,
        rotation,
    )
    landing_right = _transform_points(
        visual_scale * np.array([[-0.08, -0.16, -0.11], [0.12, -0.16, -0.11]], dtype=float),
        drone_pos,
        rotation,
    )
    landing_front = _transform_points(
        visual_scale * np.array([[0.12, -0.16, -0.11], [0.12, 0.16, -0.11]], dtype=float),
        drone_pos,
        rotation,
    )
    landing_back = _transform_points(
        visual_scale * np.array([[-0.08, -0.16, -0.11], [-0.08, 0.16, -0.11]], dtype=float),
        drone_pos,
        rotation,
    )
    nose_line = _transform_points(
        visual_scale * np.array([[0.12, 0.0, 0.0], [0.24, 0.0, -0.05]], dtype=float),
        drone_pos,
        rotation,
    )
    camera_boom = _transform_points(
        visual_scale * np.array([[0.22, 0.0, -0.05], [0.32, 0.0, -0.15]], dtype=float),
        drone_pos,
        rotation,
    )

    rotor_rings = [
        _disc_points(tuple(local_center), 0.09 * visual_scale, rotation, drone_pos, count=26)
        for local_center in rotor_centers_local
    ]
    body_faces = _transformed_faces(
        (0.28 * visual_scale, 0.18 * visual_scale, 0.08 * visual_scale),
        drone_pos,
        rotation,
    )

    return {
        "body_faces": body_faces,
        "arm_a": arm_a,
        "arm_b": arm_b,
        "landing_left": landing_left,
        "landing_right": landing_right,
        "landing_front": landing_front,
        "landing_back": landing_back,
        "nose_line": nose_line,
        "camera_boom": camera_boom,
        "rotor_rings": rotor_rings,
        "rotor_centers": rotor_centers,
    }


def visualize_simulation_3d(
    duration: float = 70.0,
    dt: float = 0.02,
    seed: int | None = None,
    snapshot: Path | None = None,
    gif: Path | None = None,
    tail: int = 220,
    frame_step: int = 1,
    export_fps: float | None = None,
) -> None:
    records = run_simulation(duration=duration, dt=dt, seed=seed)
    if not records:
        raise RuntimeError("Simulation produced no records")
    frame_step = max(1, frame_step)

    times = [r["t"] for r in records]
    drone_x = [r["drone_x"] for r in records]
    drone_y = [r["drone_y"] for r in records]
    drone_z = [-r["drone_z"] for r in records]
    subj_x = [r["subj_x"] for r in records]
    subj_y = [r["subj_y"] for r in records]
    subj_z = [0.0 for _ in records]
    roll = [r["roll_deg"] for r in records]
    pitch = [r["pitch_deg"] for r in records]
    yaw = [r.get("yaw_deg", 0.0) for r in records]
    subject_heading_deg = [r.get("subject_heading_deg", 0.0) for r in records]
    subject_speed = [r.get("subject_speed", 0.0) for r in records]
    battery = [r["battery_v"] for r in records]
    wind_est = [r["wind_est"] for r in records]
    airspeed = [r["airspeed"] for r in records]
    tracking = [r["tracking_conf"] for r in records]
    identity = [r["identity_conf"] for r in records]
    face_score = [r.get("face_score", 0.0) for r in records]
    lock = [r["lock"] for r in records]

    fig = plt.figure(figsize=(15.5, 9.2))
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.8, 1.0],
        height_ratios=[1.0, 1.0, 0.62],
        wspace=0.18,
        hspace=0.20,
    )
    ax_scene = fig.add_subplot(grid[:, 0], projection="3d")
    ax_conf = fig.add_subplot(grid[0, 1])
    ax_status = fig.add_subplot(grid[1, 1])
    ax_info = fig.add_subplot(grid[2, 1])

    fig.suptitle("Drone Simulator 3D Playback", y=0.98, fontsize=15)

    ax_scene.set_title("Flight Space", pad=18)
    ax_scene.set_xlabel("X (m)")
    ax_scene.set_ylabel("Y (m)")
    ax_scene.set_zlabel("Height (m)")
    ax_scene.set_facecolor("#eef6fb")
    ax_scene.xaxis.pane.set_facecolor((0.95, 0.98, 1.0, 0.55))
    ax_scene.yaxis.pane.set_facecolor((0.95, 0.98, 1.0, 0.55))
    ax_scene.zaxis.pane.set_facecolor((0.98, 1.0, 1.0, 0.15))
    ax_scene.xaxis._axinfo["grid"]["color"] = (0.55, 0.62, 0.70, 0.25)
    ax_scene.yaxis._axinfo["grid"]["color"] = (0.55, 0.62, 0.70, 0.25)
    ax_scene.zaxis._axinfo["grid"]["color"] = (0.55, 0.62, 0.70, 0.22)
    _set_equal_3d_bounds(ax_scene, drone_x + subj_x + [0.0], drone_y + subj_y + [0.0], drone_z + [0.0, 2.0])
    ax_scene.view_init(elev=34, azim=-60)
    ax_scene.set_box_aspect((1.0, 1.0, 0.55))

    ground_x = np.linspace(ax_scene.get_xlim()[0], ax_scene.get_xlim()[1], 28)
    ground_y = np.linspace(ax_scene.get_ylim()[0], ax_scene.get_ylim()[1], 28)
    xx, yy = np.meshgrid(ground_x, ground_y)
    zz = -0.05 + 0.015 * np.sin(xx * 0.13) * np.cos(yy * 0.11)
    grass = 0.55 + 0.12 * np.sin(xx * 0.07 + yy * 0.05) * np.cos(yy * 0.09)
    facecolors = np.empty(xx.shape + (4,), dtype=float)
    facecolors[..., 0] = 0.13 + 0.09 * grass
    facecolors[..., 1] = 0.40 + 0.24 * grass
    facecolors[..., 2] = 0.18 + 0.08 * grass
    facecolors[..., 3] = 0.82
    ax_scene.plot_surface(xx, yy, zz, facecolors=facecolors, shade=False, linewidth=0, antialiased=False)

    pad_ring = _circle_points((0.0, 0.0, 0.03), 0.95, count=72)
    pad_inner = _circle_points((0.0, 0.0, 0.03), 0.55, count=72)
    ax_scene.plot(pad_ring[:, 0], pad_ring[:, 1], pad_ring[:, 2], color="#1d3557", lw=1.8, alpha=0.9)
    ax_scene.plot(pad_inner[:, 0], pad_inner[:, 1], pad_inner[:, 2], color="#457b9d", lw=1.0, alpha=0.85)
    ax_scene.plot([-0.25, 0.25], [0.0, 0.0], [0.04, 0.04], color="#f1faee", lw=2.0, alpha=0.9)
    ax_scene.plot([0.0, 0.0], [-0.20, 0.20], [0.04, 0.04], color="#f1faee", lw=2.0, alpha=0.9)

    init_heading = _normalize_2d(
        math.cos(math.radians(subject_heading_deg[0])),
        math.sin(math.radians(subject_heading_deg[0])),
    )
    _subject_pose(subj_x[0], subj_y[0], 0.0, init_heading, subject_speed[0], times[0])
    init_drone = _drone_geometry((drone_x[0], drone_y[0], drone_z[0]), roll[0], pitch[0], yaw[0])

    ax_scene.scatter([0.0], [0.0], [0.05], marker="x", s=85, color="#0b1f2a", label="Home pad")
    drone_trail, = ax_scene.plot([], [], [], color="#4f83cc", lw=2.0, alpha=0.85, label="Drone path")
    subj_trail, = ax_scene.plot([], [], [], color="#ff7f5f", lw=1.8, alpha=0.82, label="Subject path")
    drone_shadow, = ax_scene.plot([], [], [], color="#1d3557", lw=2.2, alpha=0.18)
    subj_shadow, = ax_scene.plot([], [], [], color="#8d5524", lw=2.0, alpha=0.18)
    los_line, = ax_scene.plot([], [], [], color=LOCK_COLORS["candidate"], lw=1.7, alpha=0.95, label="Line of sight")

    drone_body = Poly3DCollection([], facecolors="#274c77", edgecolors="#18314a", linewidths=0.8, alpha=0.96)
    ax_scene.add_collection3d(drone_body)
    _set_poly_verts(drone_body, init_drone["body_faces"])

    drone_arm_a, = ax_scene.plot([], [], [], color="#18314a", lw=3.6)
    drone_arm_b, = ax_scene.plot([], [], [], color="#18314a", lw=3.6)
    drone_skid_left, = ax_scene.plot([], [], [], color="#556270", lw=2.0)
    drone_skid_right, = ax_scene.plot([], [], [], color="#556270", lw=2.0)
    drone_skid_front, = ax_scene.plot([], [], [], color="#556270", lw=1.4, alpha=0.85)
    drone_skid_back, = ax_scene.plot([], [], [], color="#556270", lw=1.4, alpha=0.85)
    drone_nose, = ax_scene.plot([], [], [], color="#f1faee", lw=2.0)
    drone_camera, = ax_scene.plot([], [], [], color="#ffb703", lw=1.5, alpha=0.95)
    rotor_lines = [
        ax_scene.plot([], [], [], color="#5dade2", lw=1.1, alpha=0.75)[0]
        for _ in range(4)
    ]
    rotor_hubs = ax_scene.scatter([], [], [], s=24, color="#0b1f2a")

    subject_torso = Poly3DCollection([], facecolors="#c97b63", edgecolors="none", alpha=0.65)
    ax_scene.add_collection3d(subject_torso)
    torso, = ax_scene.plot([], [], [], color="#8d5524", lw=3.2)
    shoulders, = ax_scene.plot([], [], [], color="#8d5524", lw=2.6)
    hips, = ax_scene.plot([], [], [], color="#8d5524", lw=2.4)
    l_upper_arm, = ax_scene.plot([], [], [], color="#264653", lw=2.2)
    l_lower_arm, = ax_scene.plot([], [], [], color="#264653", lw=2.1)
    r_upper_arm, = ax_scene.plot([], [], [], color="#264653", lw=2.2)
    r_lower_arm, = ax_scene.plot([], [], [], color="#264653", lw=2.1)
    l_upper_leg, = ax_scene.plot([], [], [], color="#6d597a", lw=2.6)
    l_lower_leg, = ax_scene.plot([], [], [], color="#6d597a", lw=2.4)
    r_upper_leg, = ax_scene.plot([], [], [], color="#6d597a", lw=2.6)
    r_lower_leg, = ax_scene.plot([], [], [], color="#6d597a", lw=2.4)
    subject_head = ax_scene.scatter([], [], [], s=62, color="#d62828")

    scene_text = ax_scene.text2D(
        0.03,
        0.97,
        "",
        transform=ax_scene.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.88, "edgecolor": "none"},
    )
    ax_scene.legend(loc="lower right", frameon=True, framealpha=0.88)

    ax_conf.set_title("Perception Confidence")
    ax_conf.set_xlabel("Time (s)")
    ax_conf.set_ylabel("Confidence")
    ax_conf.set_ylim(0.0, 1.05)
    ax_conf.grid(True, alpha=0.24)
    ax_conf.plot(times, tracking, color="#457b9d", lw=1.8, label="Tracking")
    ax_conf.plot(times, identity, color="#2a9d8f", lw=1.8, label="Identity")
    ax_conf.plot(times, face_score, color="#6d597a", lw=1.5, label="Face score")
    conf_cursor = ax_conf.axvline(times[0], color="#0b1f2a", lw=1.2, alpha=0.70)
    ax_conf.legend(loc="lower right")

    ax_status.set_title("Vehicle Status")
    ax_status.set_xlabel("Time (s)")
    ax_status.grid(True, alpha=0.24)
    ax_status.plot(times, battery, color="#6a994e", lw=1.7, label="Battery (V)")
    ax_status.plot(times, wind_est, color="#bc6c25", lw=1.7, label="Wind est (m/s)")
    ax_status.plot(times, airspeed, color="#577590", lw=1.6, label="Airspeed (m/s)")
    status_cursor = ax_status.axvline(times[0], color="#0b1f2a", lw=1.2, alpha=0.70)
    ax_status.legend(loc="upper right")

    ax_info.axis("off")
    info_text = ax_info.text(
        0.02,
        0.98,
        "",
        transform=ax_info.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f7fbfc", "alpha": 0.96, "edgecolor": "#d8e2dc"},
    )

    def draw_frame(i: int) -> None:
        lo = max(0, i - tail)
        _set_equal_3d_bounds(
            ax_scene,
            drone_x[lo : i + 1] + subj_x[lo : i + 1],
            drone_y[lo : i + 1] + subj_y[lo : i + 1],
            drone_z[lo : i + 1] + [0.0, 2.0],
            pad=1.8,
        )

        drone_trail.set_data_3d(drone_x[lo : i + 1], drone_y[lo : i + 1], drone_z[lo : i + 1])
        subj_trail.set_data_3d(subj_x[lo : i + 1], subj_y[lo : i + 1], subj_z[lo : i + 1])
        drone_shadow.set_data_3d(
            drone_x[lo : i + 1],
            drone_y[lo : i + 1],
            [0.03 for _ in range(i - lo + 1)],
        )
        subj_shadow.set_data_3d(
            subj_x[lo : i + 1],
            subj_y[lo : i + 1],
            [0.03 for _ in range(i - lo + 1)],
        )

        heading = _normalize_2d(
            math.cos(math.radians(subject_heading_deg[i])),
            math.sin(math.radians(subject_heading_deg[i])),
        )
        if subject_speed[i] <= 0.05:
            heading = _subject_heading(subj_x, subj_y, i)
        pose = _subject_pose(subj_x[i], subj_y[i], 0.0, heading, subject_speed[i], times[i])

        body_quad = [
            np.array(
                [
                    pose["l_shoulder"],
                    pose["r_shoulder"],
                    pose["r_hip"],
                    pose["l_hip"],
                ]
            )
        ]
        _set_poly_verts(subject_torso, body_quad)
        subject_head._offsets3d = ([pose["head"][0]], [pose["head"][1]], [pose["head"][2]])
        _update_line3d(torso, pose["hip"], pose["neck"])
        _update_line3d(shoulders, pose["l_shoulder"], pose["r_shoulder"])
        _update_line3d(hips, pose["l_hip"], pose["r_hip"])
        _update_line3d(l_upper_arm, pose["l_shoulder"], pose["l_elbow"])
        _update_line3d(l_lower_arm, pose["l_elbow"], pose["l_hand"])
        _update_line3d(r_upper_arm, pose["r_shoulder"], pose["r_elbow"])
        _update_line3d(r_lower_arm, pose["r_elbow"], pose["r_hand"])
        _update_line3d(l_upper_leg, pose["l_hip"], pose["l_knee"])
        _update_line3d(l_lower_leg, pose["l_knee"], pose["l_foot"])
        _update_line3d(r_upper_leg, pose["r_hip"], pose["r_knee"])
        _update_line3d(r_lower_leg, pose["r_knee"], pose["r_foot"])

        frame = _drone_geometry((drone_x[i], drone_y[i], drone_z[i]), roll[i], pitch[i], yaw[i])
        _set_poly_verts(drone_body, frame["body_faces"])
        _update_line3d(drone_arm_a, *frame["arm_a"])
        _update_line3d(drone_arm_b, *frame["arm_b"])
        drone_skid_left.set_data_3d(frame["landing_left"][:, 0], frame["landing_left"][:, 1], frame["landing_left"][:, 2])
        drone_skid_right.set_data_3d(frame["landing_right"][:, 0], frame["landing_right"][:, 1], frame["landing_right"][:, 2])
        drone_skid_front.set_data_3d(frame["landing_front"][:, 0], frame["landing_front"][:, 1], frame["landing_front"][:, 2])
        drone_skid_back.set_data_3d(frame["landing_back"][:, 0], frame["landing_back"][:, 1], frame["landing_back"][:, 2])
        drone_nose.set_data_3d(frame["nose_line"][:, 0], frame["nose_line"][:, 1], frame["nose_line"][:, 2])
        drone_camera.set_data_3d(frame["camera_boom"][:, 0], frame["camera_boom"][:, 1], frame["camera_boom"][:, 2])
        for ring_line, ring in zip(rotor_lines, frame["rotor_rings"], strict=False):
            ring_line.set_data_3d(ring[:, 0], ring[:, 1], ring[:, 2])
        rotor_hubs._offsets3d = (
            frame["rotor_centers"][:, 0],
            frame["rotor_centers"][:, 1],
            frame["rotor_centers"][:, 2],
        )

        los_color = LOCK_COLORS.get(lock[i], "#999999")
        los_line.set_color(los_color)
        _update_line3d(los_line, (drone_x[i], drone_y[i], drone_z[i]), pose["head"])

        conf_cursor.set_xdata([times[i], times[i]])
        status_cursor.set_xdata([times[i], times[i]])

        scene_text.set_text(
            "\n".join(
                [
                    f"t = {times[i]:.1f}s   lock = {lock[i]}",
                    f"drone yaw = {yaw[i]:.0f} deg   subject heading = {subject_heading_deg[i]:.0f} deg",
                    f"track = {tracking[i]:.2f}   id = {identity[i]:.2f}   face = {face_score[i]:.2f}",
                ]
            )
        )
        scene_text.set_bbox(
            {
                "boxstyle": "round,pad=0.35",
                "facecolor": los_color,
                "alpha": 0.18,
                "edgecolor": "none",
            }
        )

        r = records[i]
        desired_distance = r.get("desired_distance")
        desired_distance_text = f"{desired_distance:.1f} m" if desired_distance is not None else "shot default"
        info_text.set_text(
            "\n".join(
                [
                    f"Mission: {r['state']}   Shot: {r['shot']}   Safety: {r['safety']}",
                    f"Stability: {r['stability']}   Desired distance: {desired_distance_text}",
                    f"Drone: ({drone_x[i]:.1f}, {drone_y[i]:.1f}, {drone_z[i]:.1f}) m   Subject: ({subj_x[i]:.1f}, {subj_y[i]:.1f}) m",
                    f"Airspeed: {r['airspeed']:.2f} m/s   Subject speed: {r.get('subject_speed', 0.0):.2f} m/s",
                    f"Battery: {r['battery_v']:.2f} V   Wind est: {r['wind_est']:.1f} m/s   Current: {r['battery_current_a']:.1f} A",
                    f"Thrust: {r.get('thrust_n', 0.0):.1f} N / {r.get('max_thrust_n', 0.0):.1f} N   Throttle: {r.get('throttle', 0.0):.2f}",
                    f"Ground fx: {r.get('ground_effect_gain', 1.0):.3f}   ETL: {r.get('trans_lift_gain', 1.0):.3f}   VRS: {r.get('vortex_ring_penalty', 0.0):.3f}",
                ]
            )
        )

    frame_indices = list(range(0, len(records), frame_step))
    if frame_indices[-1] != len(records) - 1:
        frame_indices.append(len(records) - 1)

    if snapshot is not None:
        draw_frame(len(records) - 1)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(snapshot, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved 3D snapshot to {snapshot}")
        return

    if gif is not None and frame_step == 1 and len(frame_indices) > 360:
        auto_step = max(1, math.ceil(len(frame_indices) / 360))
        frame_indices = list(range(0, len(records), auto_step))
        if frame_indices[-1] != len(records) - 1:
            frame_indices.append(len(records) - 1)
        frame_step = auto_step
        print(
            f"GIF export auto-selected --frame-step {frame_step} "
            f"to keep the render manageable ({len(frame_indices)} frames).",
            flush=True,
        )

    interval_ms = max(1, int(1000 * dt * frame_step))
    anim = FuncAnimation(fig, draw_frame, frames=frame_indices, interval=interval_ms, repeat=False)
    fig._anim = anim

    if gif is not None:
        from matplotlib.animation import PillowWriter

        gif.parent.mkdir(parents=True, exist_ok=True)
        fps = export_fps
        if fps is None:
            fps = min(20.0, max(4.0, 1.0 / max(dt * frame_step, 1e-6)))
        total = len(frame_indices)

        def _progress(current: int, total_frames: int) -> None:
            if total_frames <= 0:
                return
            if current == 0 or current == total_frames - 1 or current % max(1, total_frames // 12) == 0:
                print(f"Rendering GIF frame {current + 1}/{total_frames}...", flush=True)

        print(f"Saving GIF with {total} rendered frames at {fps:.1f} FPS...", flush=True)
        try:
            anim.save(gif, writer=PillowWriter(fps=fps), progress_callback=_progress)
        except TypeError:
            anim.save(gif, writer=PillowWriter(fps=fps))
        plt.close(fig)
        print(f"Saved 3D animation to {gif}", flush=True)
        return

    try:
        plt.show()
    except KeyboardInterrupt:
        plt.close(fig)
        raise RuntimeError(
            "Interactive GUI backend did not open cleanly. "
            "Try --gif sim_3d.gif or --snapshot sim_3d_snapshot.png, "
            "or override the backend with --backend WebAgg or --backend Agg."
        ) from None


def main() -> None:
    args = _build_parser().parse_args()
    visualize_simulation_3d(
        duration=args.duration,
        dt=args.dt,
        seed=args.seed,
        snapshot=args.snapshot,
        gif=args.gif,
        tail=args.tail,
        frame_step=args.frame_step,
        export_fps=args.export_fps,
    )


if __name__ == "__main__":
    main()
