"""
Top-down visualization for the simulator.

Usage:
    python -m sim.visualize_sim
    python -m sim.visualize_sim --duration 30 --seed 5
    python -m sim.visualize_sim --snapshot sim_snapshot.png --duration 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if "--snapshot" in sys.argv:
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from sim.sim_harness import run_simulation


LOCK_COLORS = {
    "candidate": "#f4a261",
    "locked": "#2a9d8f",
    "weak": "#e9c46a",
    "lost": "#e76f51",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize the drone simulator")
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
        "--tail",
        type=int,
        default=200,
        help="Number of recent samples to keep in the path trail",
    )
    return parser


def _set_equal_bounds(ax, xs, ys, pad: float = 2.0) -> None:
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span = max(max_x - min_x, max_y - min_y, 6.0)
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    radius = 0.5 * span + pad
    ax.set_xlim(cx - radius, cx + radius)
    ax.set_ylim(cy - radius, cy + radius)
    ax.set_aspect("equal", adjustable="box")


def visualize_simulation(
    duration: float = 70.0,
    dt: float = 0.02,
    seed: int | None = None,
    snapshot: Path | None = None,
    tail: int = 200,
) -> None:
    records = run_simulation(duration=duration, dt=dt, seed=seed)
    if not records:
        raise RuntimeError("Simulation produced no records")

    times = [r["t"] for r in records]
    drone_x = [r["drone_x"] for r in records]
    drone_y = [r["drone_y"] for r in records]
    subj_x = [r["subj_x"] for r in records]
    subj_y = [r["subj_y"] for r in records]
    battery = [r["battery_v"] for r in records]
    wind_est = [r["wind_est"] for r in records]
    tracking = [r["tracking_conf"] for r in records]
    identity = [r["identity_conf"] for r in records]
    lock = [r["lock"] for r in records]

    fig = plt.figure(figsize=(12, 8))
    grid = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0])
    ax_map = fig.add_subplot(grid[0, 0])
    ax_conf = fig.add_subplot(grid[0, 1])
    ax_status = fig.add_subplot(grid[1, :])

    fig.suptitle("Drone Simulator Playback")

    # Top-down map
    ax_map.set_title("Top-Down Position")
    ax_map.set_xlabel("X (m)")
    ax_map.set_ylabel("Y (m)")
    ax_map.grid(True, alpha=0.25)
    _set_equal_bounds(ax_map, drone_x + subj_x + [0.0], drone_y + subj_y + [0.0])

    home = ax_map.scatter([0.0], [0.0], marker="x", s=80, color="black", label="Home")
    drone_dot = ax_map.scatter([], [], s=90, color="#1d3557", label="Drone")
    subject_dot = ax_map.scatter([], [], s=90, color="#d62828", label="Subject")
    drone_path, = ax_map.plot([], [], color="#457b9d", lw=1.5, alpha=0.7)
    subject_path, = ax_map.plot([], [], color="#e76f51", lw=1.5, alpha=0.7)
    lock_text = ax_map.text(
        0.02,
        0.98,
        "",
        transform=ax_map.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )
    ax_map.legend(loc="lower right")

    # Confidence chart
    ax_conf.set_title("Confidence")
    ax_conf.set_xlabel("Time (s)")
    ax_conf.set_ylabel("Confidence")
    ax_conf.set_ylim(0.0, 1.05)
    ax_conf.grid(True, alpha=0.25)
    ax_conf.plot(times, tracking, color="#457b9d", label="Tracking")
    ax_conf.plot(times, identity, color="#2a9d8f", label="Identity")
    conf_cursor = ax_conf.axvline(times[0], color="black", lw=1.5, alpha=0.7)
    ax_conf.legend(loc="lower right")

    # Status chart
    ax_status.set_title("Battery / Wind")
    ax_status.set_xlabel("Time (s)")
    ax_status.grid(True, alpha=0.25)
    ax_status.plot(times, battery, color="#6a994e", label="Battery (V)")
    ax_status.plot(times, wind_est, color="#bc6c25", label="Wind est (m/s)")
    status_cursor = ax_status.axvline(times[0], color="black", lw=1.5, alpha=0.7)
    ax_status.legend(loc="upper right")

    state_text = ax_status.text(
        0.01,
        0.96,
        "",
        transform=ax_status.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )

    def draw_frame(i: int) -> None:
        lo = max(0, i - tail)
        drone_dot.set_offsets([[drone_x[i], drone_y[i]]])
        subject_dot.set_offsets([[subj_x[i], subj_y[i]]])
        drone_path.set_data(drone_x[lo:i + 1], drone_y[lo:i + 1])
        subject_path.set_data(subj_x[lo:i + 1], subj_y[lo:i + 1])
        conf_cursor.set_xdata([times[i], times[i]])
        status_cursor.set_xdata([times[i], times[i]])

        lock_value = lock[i]
        lock_text.set_text(
            "\n".join(
                [
                    f"t = {times[i]:.1f}s",
                    f"lock = {lock_value}",
                    f"track = {tracking[i]:.2f}",
                    f"id = {identity[i]:.2f}",
                ]
            )
        )
        lock_text.set_bbox(
            {
                "boxstyle": "round,pad=0.3",
                "facecolor": LOCK_COLORS.get(lock_value, "white"),
                "alpha": 0.22,
                "edgecolor": "none",
            }
        )

        r = records[i]
        state_text.set_text(
            "\n".join(
                [
                    f"state = {r['state']} | shot = {r['shot']} | safety = {r['safety']}",
                    f"stability = {r['stability']} | battery = {r['battery_v']:.2f} V | wind_est = {r['wind_est']:.1f} m/s",
                    f"airspeed = {r['airspeed']:.2f} m/s | thrust = {r.get('thrust_n', 0.0):.1f} N | throttle = {r.get('throttle', 0.0):.2f}",
                    f"ground fx = {r.get('ground_effect_gain', 1.0):.3f} | ETL = {r.get('trans_lift_gain', 1.0):.3f} | VRS = {r.get('vortex_ring_penalty', 0.0):.3f}",
                ]
            )
        )

    if snapshot is not None:
        draw_frame(len(records) - 1)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(snapshot, dpi=140)
        plt.close(fig)
        print(f"Saved snapshot to {snapshot}")
        return

    interval_ms = max(1, int(1000 * dt))
    anim = FuncAnimation(fig, draw_frame, frames=len(records), interval=interval_ms, repeat=False)
    fig._anim = anim  # keep a live reference for interactive backends
    fig.tight_layout()
    plt.show()


def main() -> None:
    args = _build_parser().parse_args()
    visualize_simulation(
        duration=args.duration,
        dt=args.dt,
        seed=args.seed,
        snapshot=args.snapshot,
        tail=args.tail,
    )


if __name__ == "__main__":
    main()
