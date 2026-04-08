"""
Interactive simulator dashboard.

This runs the autonomy stack in a live loop and exposes the most useful
operator and environment controls:

- start / stop / lock-loss / pause
- shot mode and follow distance
- walking target speed and heading
- wind target in X / Y
- live force and confidence readouts
"""

from __future__ import annotations

import argparse
import math

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, RadioButtons, Slider

from interfaces.schemas import ShotMode
from sim.sim_harness import SimulationSession


LOCK_COLORS = {
    "candidate": "#f4a261",
    "locked": "#2a9d8f",
    "weak": "#e9c46a",
    "lost": "#e76f51",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive drone simulator")
    parser.add_argument("--dt", type=float, default=0.02, help="Simulation timestep in seconds")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument(
        "--steps-per-frame",
        type=int,
        default=2,
        help="Simulation steps to advance per screen refresh",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=300,
        help="Number of recent samples to keep in the map trail",
    )
    return parser


def _set_equal_bounds(ax, xs, ys, pad: float = 3.0) -> None:
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span = max(max_x - min_x, max_y - min_y, 10.0)
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    radius = 0.5 * span + pad
    ax.set_xlim(cx - radius, cx + radius)
    ax.set_ylim(cy - radius, cy + radius)
    ax.set_aspect("equal", adjustable="box")


class InteractiveSimulationApp:
    def __init__(self, *, dt: float, seed: int | None, steps_per_frame: int, tail: int) -> None:
        self.dt = dt
        self.seed = seed
        self.steps_per_frame = max(1, steps_per_frame)
        self.tail = max(60, tail)
        self.running = True
        self.walking_enabled = False

        self.fig = plt.figure(figsize=(15, 9))
        self.fig.suptitle("Interactive Follow-and-Film Simulator")
        self._build_session()
        self._build_layout()
        self._connect_events()

        # Show one settled frame so the dashboard opens with meaningful text.
        self.session.step()
        self._draw_latest()

    def _build_session(self) -> None:
        self.session = SimulationSession(
            scenario=[],
            dt=self.dt,
            seed=self.seed,
            log_interval_s=None,
        )

    def _build_layout(self) -> None:
        gs = self.fig.add_gridspec(
            2,
            3,
            width_ratios=[1.65, 1.05, 0.95],
            height_ratios=[1.0, 1.0],
            wspace=0.28,
            hspace=0.25,
        )

        self.ax_map = self.fig.add_subplot(gs[:, 0])
        self.ax_force = self.fig.add_subplot(gs[0, 1])
        self.ax_status = self.fig.add_subplot(gs[1, 1])
        self.ax_panel = self.fig.add_subplot(gs[:, 2])
        self.ax_panel.axis("off")

        self.ax_map.set_title("Top-Down Scene")
        self.ax_map.set_xlabel("X (m)")
        self.ax_map.set_ylabel("Y (m)")
        self.ax_map.grid(True, alpha=0.22)
        _set_equal_bounds(self.ax_map, [0.0, 8.0], [0.0, 8.0])

        self.drone_dot = self.ax_map.scatter([], [], s=95, color="#1d3557", label="Drone")
        self.subject_dot = self.ax_map.scatter([], [], s=90, color="#d62828", label="Subject")
        self.drone_path, = self.ax_map.plot([], [], color="#457b9d", lw=1.6, alpha=0.8)
        self.subject_path, = self.ax_map.plot([], [], color="#e76f51", lw=1.6, alpha=0.8)
        self.home_marker = self.ax_map.scatter([0.0], [0.0], marker="x", s=70, color="black", label="Home")
        self.line_of_sight, = self.ax_map.plot([], [], color=LOCK_COLORS["lost"], lw=1.6, alpha=0.9)
        self.map_text = self.ax_map.text(
            0.02,
            0.98,
            "",
            transform=self.ax_map.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85},
        )
        self.ax_map.legend(loc="lower right")

        self.ax_force.set_title("Altitude / Force Terms")
        self.ax_force.set_xlabel("Time (s)")
        self.ax_force.grid(True, alpha=0.22)
        self.altitude_line, = self.ax_force.plot([], [], color="#577590", label="Altitude (m)")
        self.thrust_line, = self.ax_force.plot([], [], color="#bc6c25", label="Thrust (N)")
        self.force_cursor = self.ax_force.axvline(0.0, color="black", lw=1.2, alpha=0.7)
        self.ax_force.legend(loc="upper right")

        self.ax_status.set_title("Battery / Wind / Confidence")
        self.ax_status.set_xlabel("Time (s)")
        self.ax_status.grid(True, alpha=0.22)
        self.battery_line, = self.ax_status.plot([], [], color="#6a994e", label="Battery (V)")
        self.wind_line, = self.ax_status.plot([], [], color="#f4a261", label="Wind est (m/s)")
        self.conf_line, = self.ax_status.plot([], [], color="#2a9d8f", label="Tracking conf")
        self.status_cursor = self.ax_status.axvline(0.0, color="black", lw=1.2, alpha=0.7)
        self.status_text = self.ax_status.text(
            0.02,
            0.96,
            "",
            transform=self.ax_status.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.86},
        )
        self.ax_status.legend(loc="upper right")

        self.pause_button = Button(self.fig.add_axes([0.735, 0.88, 0.095, 0.05]), "Pause")
        self.start_button = Button(self.fig.add_axes([0.845, 0.88, 0.095, 0.05]), "Start")
        self.stop_button = Button(self.fig.add_axes([0.735, 0.81, 0.095, 0.05]), "Stop")
        self.reset_button = Button(self.fig.add_axes([0.845, 0.81, 0.095, 0.05]), "Reset")
        self.lock_button = Button(self.fig.add_axes([0.735, 0.74, 0.095, 0.05]), "Force Lock")
        self.lost_button = Button(self.fig.add_axes([0.845, 0.74, 0.095, 0.05]), "Lose 3 s")
        self.walk_button = Button(self.fig.add_axes([0.735, 0.67, 0.205, 0.05]), "Walking: Off")

        self.distance_slider = Slider(
            self.fig.add_axes([0.735, 0.59, 0.205, 0.03]),
            "Distance (m)",
            2.0,
            20.0,
            valinit=5.0,
            valstep=0.5,
        )
        self.speed_slider = Slider(
            self.fig.add_axes([0.735, 0.52, 0.205, 0.03]),
            "Subject Speed",
            0.0,
            3.0,
            valinit=1.2,
            valstep=0.1,
        )
        self.heading_slider = Slider(
            self.fig.add_axes([0.735, 0.45, 0.205, 0.03]),
            "Heading (deg)",
            -180.0,
            180.0,
            valinit=0.0,
            valstep=5.0,
        )
        self.wind_x_slider = Slider(
            self.fig.add_axes([0.735, 0.38, 0.205, 0.03]),
            "Wind X",
            -12.0,
            12.0,
            valinit=0.0,
            valstep=0.5,
        )
        self.wind_y_slider = Slider(
            self.fig.add_axes([0.735, 0.31, 0.205, 0.03]),
            "Wind Y",
            -12.0,
            12.0,
            valinit=0.0,
            valstep=0.5,
        )

        self.shot_radio = RadioButtons(
            self.fig.add_axes([0.735, 0.12, 0.205, 0.14]),
            ("Stand-up", "Walk & Talk", "Wide Safety", "Orbit"),
            active=1,
        )
        self.panel_text = self.fig.text(
            0.735,
            0.05,
            "Keys: space pause  s start  x stop  l lock  k lost",
            fontsize=10,
        )

        self.pause_button.on_clicked(self._toggle_pause)
        self.start_button.on_clicked(lambda _event: self.session.start())
        self.stop_button.on_clicked(lambda _event: self.session.stop())
        self.reset_button.on_clicked(self._reset_session)
        self.lock_button.on_clicked(lambda _event: self.session.force_lock())
        self.lost_button.on_clicked(lambda _event: self.session.force_lost(3.0))
        self.walk_button.on_clicked(self._toggle_walking)
        self.distance_slider.on_changed(self._on_distance_changed)
        self.speed_slider.on_changed(self._on_motion_changed)
        self.heading_slider.on_changed(self._on_motion_changed)
        self.wind_x_slider.on_changed(self._on_wind_changed)
        self.wind_y_slider.on_changed(self._on_wind_changed)
        self.shot_radio.on_clicked(self._on_shot_mode_changed)

    def _connect_events(self) -> None:
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

    def _toggle_pause(self, _event) -> None:
        self.running = not self.running
        self.pause_button.label.set_text("Resume" if not self.running else "Pause")

    def _toggle_walking(self, _event) -> None:
        self.walking_enabled = not self.walking_enabled
        self.walk_button.label.set_text(f"Walking: {'On' if self.walking_enabled else 'Off'}")
        self._apply_motion()

    def _reset_session(self, _event) -> None:
        self._build_session()
        self.walking_enabled = False
        self.walk_button.label.set_text("Walking: Off")
        self.running = True
        self.pause_button.label.set_text("Pause")
        self._apply_controls()
        self.session.step()
        self._draw_latest()

    def _on_distance_changed(self, value: float) -> None:
        self.session.set_distance(value)

    def _on_motion_changed(self, _value: float) -> None:
        self._apply_motion()

    def _apply_motion(self) -> None:
        self.session.set_subject_motion(
            walking=self.walking_enabled,
            speed=self.speed_slider.val,
            heading_deg=self.heading_slider.val,
        )

    def _on_wind_changed(self, _value: float) -> None:
        self.session.set_wind_target(self.wind_x_slider.val, self.wind_y_slider.val)

    def _apply_controls(self) -> None:
        self._on_wind_changed(0.0)
        self._apply_motion()
        self._on_distance_changed(self.distance_slider.val)
        self._on_shot_mode_changed(self.shot_radio.value_selected)

    def _on_shot_mode_changed(self, label: str) -> None:
        mapping = {
            "Stand-up": ShotMode.STANDUP,
            "Walk & Talk": ShotMode.WALK_AND_TALK,
            "Wide Safety": ShotMode.WIDE_SAFETY,
            "Orbit": ShotMode.ORBIT,
        }
        self.session.set_shot_mode(mapping[label])

    def _on_key_press(self, event) -> None:
        if event.key == " ":
            self._toggle_pause(event)
        elif event.key == "s":
            self.session.start()
        elif event.key == "x":
            self.session.stop()
        elif event.key == "l":
            self.session.force_lock()
        elif event.key == "k":
            self.session.force_lost(3.0)

    def _draw_latest(self) -> None:
        if self.session.latest_record is None:
            return
        records = self.session.records
        rec = self.session.latest_record
        i = len(records) - 1
        lo = max(0, i - self.tail)

        times = [r["t"] for r in records]
        drone_x = [r["drone_x"] for r in records]
        drone_y = [r["drone_y"] for r in records]
        drone_alt = [-r["drone_z"] for r in records]
        subj_x = [r["subj_x"] for r in records]
        subj_y = [r["subj_y"] for r in records]
        thrust = [r["thrust_n"] for r in records]
        battery = [r["battery_v"] for r in records]
        wind = [r["wind_est"] for r in records]
        track = [r["tracking_conf"] for r in records]

        self.drone_dot.set_offsets([[drone_x[i], drone_y[i]]])
        self.subject_dot.set_offsets([[subj_x[i], subj_y[i]]])
        self.drone_path.set_data(drone_x[lo:i + 1], drone_y[lo:i + 1])
        self.subject_path.set_data(subj_x[lo:i + 1], subj_y[lo:i + 1])
        self.line_of_sight.set_data([drone_x[i], subj_x[i]], [drone_y[i], subj_y[i]])
        self.line_of_sight.set_color(LOCK_COLORS.get(rec["lock"], "#999999"))
        _set_equal_bounds(self.ax_map, drone_x[lo:i + 1] + subj_x[lo:i + 1] + [0.0], drone_y[lo:i + 1] + subj_y[lo:i + 1] + [0.0])

        self.map_text.set_text(
            "\n".join(
                [
                    f"t = {rec['t']:.1f}s | state = {rec['state']} | shot = {rec['shot']}",
                    f"lock = {rec['lock']} | safety = {rec['safety']} | stability = {rec['stability']}",
                    f"distance = {rec['desired_distance'] if rec['desired_distance'] is not None else 'auto'} m",
                    f"drone = ({rec['drone_x']:.1f}, {rec['drone_y']:.1f}, {-rec['drone_z']:.1f}) m",
                ]
            )
        )

        self.altitude_line.set_data(times, drone_alt)
        self.thrust_line.set_data(times, thrust)
        self.force_cursor.set_xdata([times[i], times[i]])
        self.ax_force.relim()
        self.ax_force.autoscale_view()

        self.battery_line.set_data(times, battery)
        self.wind_line.set_data(times, wind)
        self.conf_line.set_data(times, track)
        self.status_cursor.set_xdata([times[i], times[i]])
        self.ax_status.relim()
        self.ax_status.autoscale_view()

        self.status_text.set_text(
            "\n".join(
                [
                    f"battery = {rec['battery_v']:.2f} V   current = {rec['battery_current_a']:.1f} A   wind = {rec['wind_est']:.1f} m/s",
                    f"throttle = {rec['throttle']:.2f}   thrust = {rec['thrust_n']:.1f}/{rec['max_thrust_n']:.1f} N",
                    f"ground fx = {rec['ground_effect_gain']:.3f}   ETL = {rec['trans_lift_gain']:.3f}   VRS = {rec['vortex_ring_penalty']:.3f}",
                    f"airspeed = {rec['airspeed']:.2f} m/s   accel_xy = {rec['accel_xy']:.2f} m/s²   accel_z = {rec['accel_z']:.2f} m/s²",
                ]
            )
        )

    def _update_frame(self, _frame_index: int) -> None:
        if self.running:
            self._apply_motion()
            self._on_wind_changed(0.0)
            self.session.step(self.steps_per_frame)
        self._draw_latest()

    def run(self) -> None:
        anim = FuncAnimation(self.fig, self._update_frame, interval=max(20, int(self.dt * 1000 * self.steps_per_frame)))
        self.fig._anim = anim
        self.fig.tight_layout(rect=[0.0, 0.0, 0.98, 0.97])
        plt.show()


def launch_interactive_simulation(
    *,
    dt: float = 0.02,
    seed: int | None = None,
    steps_per_frame: int = 2,
    tail: int = 300,
) -> None:
    app = InteractiveSimulationApp(
        dt=dt,
        seed=seed,
        steps_per_frame=steps_per_frame,
        tail=tail,
    )
    app.run()


def main() -> None:
    args = _build_parser().parse_args()
    launch_interactive_simulation(
        dt=args.dt,
        seed=args.seed,
        steps_per_frame=args.steps_per_frame,
        tail=args.tail,
    )


if __name__ == "__main__":
    main()
