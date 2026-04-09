"""
Operator-facing control panel for the follow-and-film product layer.

This module separates the command/state logic from the optional Tkinter UI so
the interaction model remains testable in headless environments.
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, cast

from config.parameters import ShotParams
from interfaces.schemas import AppCommand, MissionState, ShotMode
from product.adapters import ProductCommandAdapter
from product.schemas import ProductSnapshot

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:  # pragma: no cover - depends on local Python installation
    tk = cast(Any, None)
    ttk = cast(Any, None)


DISTANCE_PRESETS: tuple[tuple[str, float], ...] = (
    ("Close", 3.0),
    ("Default", 5.0),
    ("Wide", 8.0),
    ("Far", 12.0),
)


def _title_case(value: str) -> str:
    return value.replace("_", " ").title()


def _shot_mode_label(mode: ShotMode) -> str:
    labels = {
        ShotMode.STANDUP: "Stand-up",
        ShotMode.WALK_AND_TALK: "Walk & Talk",
        ShotMode.WIDE_SAFETY: "Wide Safety",
        ShotMode.ORBIT: "Orbit",
    }
    return labels[mode]


@dataclass(frozen=True)
class OperatorPanelViewState:
    """Immutable snapshot of labels and badge for the operator UI."""

    mission_label: str
    lock_label: str
    recording_label: str
    confidence_label: str
    distance_label: str
    detail_label: str
    primary_badge: str
    primary_badge_color: str


class OperatorPanelController:
    """UI behavior layer for operator commands and status presentation."""

    def __init__(
        self,
        *,
        adapter: ProductCommandAdapter | None = None,
        min_distance_m: float | None = None,
        max_distance_m: float | None = None,
        default_distance_m: float = 5.0,
        default_shot_mode: ShotMode = ShotMode.WALK_AND_TALK,
    ) -> None:
        shot_defaults = ShotParams()
        self.adapter = adapter or ProductCommandAdapter()
        self.min_distance_m = shot_defaults.min_distance if min_distance_m is None else min_distance_m
        self.max_distance_m = shot_defaults.max_distance if max_distance_m is None else max_distance_m
        self.selected_distance_m = self.clamp_distance(default_distance_m)
        self.selected_shot_mode = default_shot_mode
        self._deferred_until_state: Optional[str] = None
        self._deferred_commands: list[AppCommand] = []

    def clamp_distance(self, meters: float) -> float:
        """Clamp *meters* to the configured min/max range."""
        return max(self.min_distance_m, min(self.max_distance_m, float(meters)))

    def set_selected_distance(self, meters: float) -> float:
        """Update the selected follow distance and return the clamped value."""
        self.selected_distance_m = self.clamp_distance(meters)
        return self.selected_distance_m

    def set_selected_shot_mode(self, shot_mode: ShotMode) -> ShotMode:
        """Update the selected shot mode and return it."""
        self.selected_shot_mode = shot_mode
        return self.selected_shot_mode

    def launch_commands(self, timestamp: float | None = None) -> list[AppCommand]:
        """Build the start + distance + deferred shot-mode commands."""
        ts = time.time() if timestamp is None else timestamp
        plan = self.adapter.launch_plan(ts, requested_shot_mode=self.selected_shot_mode)
        self._deferred_until_state = plan.deferred_until_state
        self._deferred_commands = list(plan.deferred)
        return [
            *plan.immediate,
            self.adapter.set_distance(ts, self.selected_distance_m),
        ]

    def maybe_emit_deferred_commands(self, snapshot: ProductSnapshot) -> list[AppCommand]:
        """Emit deferred commands once the mission reaches the expected state."""
        if not self._deferred_until_state or not self._deferred_commands:
            return []
        if snapshot.mission.mission_state.value != self._deferred_until_state:
            return []
        commands = list(self._deferred_commands)
        self._deferred_until_state = None
        self._deferred_commands = []
        return commands

    def distance_command(self, timestamp: float | None = None) -> AppCommand:
        """Create a distance command from the currently selected distance."""
        ts = time.time() if timestamp is None else timestamp
        return self.adapter.set_distance(ts, self.selected_distance_m)

    def mode_command(self, timestamp: float | None = None) -> AppCommand:
        """Create a shot-mode command from the currently selected mode."""
        ts = time.time() if timestamp is None else timestamp
        return self.adapter.set_mode(ts, self.selected_shot_mode)

    def build_view_state(self, snapshot: ProductSnapshot | None = None) -> OperatorPanelViewState:
        """Derive UI labels and badge from the latest product snapshot."""
        if snapshot is None:
            return OperatorPanelViewState(
                mission_label="Idle",
                lock_label="No subject locked",
                recording_label="Recorder idle",
                confidence_label="Confidence n/a",
                distance_label=f"{self.selected_distance_m:.1f} m follow distance",
                detail_label="Choose a shot, set distance, then press Start.",
                primary_badge="Ready",
                primary_badge_color="#245c3f",
            )

        mission = _title_case(snapshot.mission.mission_state.value)
        lock = _title_case(snapshot.lock.lock_state.value)
        recording = _title_case(snapshot.recording_state.value)
        effective_conf = snapshot.lock.confidence.effective * 100.0
        distance_m = (
            self.selected_distance_m
            if snapshot.mission.desired_distance is None
            else snapshot.mission.desired_distance
        )

        if snapshot.recording_state.value == "recording_confirmed":
            badge = ("Recording", "#8a1c1c")
        elif snapshot.mission.ready_to_record:
            badge = ("Ready", "#245c3f")
        elif snapshot.mission.mission_state == MissionState.EMERGENCY:
            badge = ("Emergency", "#7a1111")
        else:
            badge = ("Waiting", "#705c14")

        detail_parts = [
            f"Shot: {_shot_mode_label(snapshot.mission.shot_mode)}",
            f"Safety: {_title_case(snapshot.mission.safety_override.value)}",
        ]
        if snapshot.battery_percent is not None:
            detail_parts.append(f"Battery: {snapshot.battery_percent:.0f}%")
        if snapshot.active_clip_id:
            detail_parts.append(f"Clip: {snapshot.active_clip_id}")

        return OperatorPanelViewState(
            mission_label=f"Mission: {mission}",
            lock_label=f"Subject: {lock}",
            recording_label=f"Recorder: {recording}",
            confidence_label=f"Confidence: {effective_conf:.0f}%",
            distance_label=f"{distance_m:.1f} m follow distance",
            detail_label="   |   ".join(detail_parts),
            primary_badge=badge[0],
            primary_badge_color=badge[1],
        )


class OperatorControlPanel:
    """Small Tkinter control panel for live demos and local operator testing."""

    def __init__(
        self,
        *,
        send_command: Callable[[AppCommand], None],
        controller: OperatorPanelController | None = None,
        title: str = "Follow and Film Control Panel",
    ) -> None:
        if tk is None or ttk is None:  # pragma: no cover - depends on local Python installation
            raise RuntimeError("Tkinter is not available in this Python environment")

        self.send_command = send_command
        self.controller = controller or OperatorPanelController()
        self.last_snapshot: ProductSnapshot | None = None

        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("760x440")
        self.root.minsize(700, 420)

        self._build_widgets()
        self._apply_view_state(self.controller.build_view_state())

    def _build_widgets(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        outer = ttk.Frame(self.root, padding=18)
        outer.grid(sticky="nsew")
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)

        status = ttk.LabelFrame(outer, text="Live Status", padding=14)
        status.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        status.columnconfigure(0, weight=1)

        self.badge_var = tk.StringVar()
        self.mission_var = tk.StringVar()
        self.lock_var = tk.StringVar()
        self.recording_var = tk.StringVar()
        self.confidence_var = tk.StringVar()
        self.distance_var = tk.StringVar()
        self.detail_var = tk.StringVar()

        self.badge_label = tk.Label(
            status,
            textvariable=self.badge_var,
            fg="white",
            bg="#245c3f",
            padx=12,
            pady=8,
            font=("Segoe UI", 12, "bold"),
        )
        self.badge_label.grid(row=0, column=0, sticky="w", pady=(0, 10))

        for row, variable in enumerate(
            [
                self.mission_var,
                self.lock_var,
                self.recording_var,
                self.confidence_var,
                self.distance_var,
                self.detail_var,
            ],
            start=1,
        ):
            ttk.Label(status, textvariable=variable).grid(row=row, column=0, sticky="w", pady=2)

        commands = ttk.LabelFrame(outer, text="Flight Actions", padding=14)
        commands.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        commands.columnconfigure(0, weight=1)
        commands.columnconfigure(1, weight=1)

        ttk.Button(commands, text="Start", command=self._on_start).grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        ttk.Button(commands, text="Stop", command=self._emit_stop).grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(commands, text="Re-lock", command=self._emit_relock).grid(row=1, column=0, sticky="ew", padx=4, pady=4)
        ttk.Button(commands, text="Emergency Stop", command=self._emit_emergency).grid(row=1, column=1, sticky="ew", padx=4, pady=4)

        modes = ttk.LabelFrame(outer, text="Shot Mode", padding=14)
        modes.grid(row=1, column=1, sticky="nsew", padx=(6, 0))
        modes.columnconfigure(0, weight=1)
        modes.columnconfigure(1, weight=1)

        self.shot_mode_var = tk.StringVar(value=self.controller.selected_shot_mode.value)
        for idx, mode in enumerate(ShotMode):
            ttk.Radiobutton(
                modes,
                text=_shot_mode_label(mode),
                value=mode.value,
                variable=self.shot_mode_var,
                command=self._on_mode_changed,
            ).grid(row=idx // 2, column=idx % 2, sticky="w", padx=4, pady=4)

        distance = ttk.LabelFrame(outer, text="Follow Distance", padding=14)
        distance.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        distance.columnconfigure(0, weight=1)

        self.distance_scale = tk.Scale(
            distance,
            from_=self.controller.min_distance_m,
            to=self.controller.max_distance_m,
            orient="horizontal",
            resolution=0.5,
            showvalue=False,
            command=self._on_distance_slider,
        )
        self.distance_scale.set(self.controller.selected_distance_m)
        self.distance_scale.grid(row=0, column=0, columnspan=5, sticky="ew")

        self.distance_readout_var = tk.StringVar()
        ttk.Label(distance, textvariable=self.distance_readout_var).grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Button(distance, text="Apply Distance", command=self._emit_distance).grid(row=1, column=4, sticky="e", pady=(6, 0))

        for idx, (label, meters) in enumerate(DISTANCE_PRESETS):
            ttk.Button(
                distance,
                text=f"{label} {meters:.0f} m",
                command=functools.partial(self._apply_distance_preset, meters),
            ).grid(row=2, column=idx, sticky="ew", padx=4, pady=(10, 0))

        ttk.Label(
            distance,
            text="Tip: presets are one-tap. Use the slider only when you want a custom distance.",
        ).grid(row=3, column=0, columnspan=5, sticky="w", pady=(10, 0))

        self._sync_distance_readout()

    def _apply_view_state(self, state: OperatorPanelViewState) -> None:
        self.badge_var.set(state.primary_badge)
        self.badge_label.configure(bg=state.primary_badge_color)
        self.mission_var.set(state.mission_label)
        self.lock_var.set(state.lock_label)
        self.recording_var.set(state.recording_label)
        self.confidence_var.set(state.confidence_label)
        self.distance_var.set(state.distance_label)
        self.detail_var.set(state.detail_label)

    def _sync_distance_readout(self) -> None:
        self.distance_readout_var.set(
            f"Selected distance: {self.controller.selected_distance_m:.1f} m"
        )

    def _send(self, command: AppCommand) -> None:
        self.send_command(command)

    def _on_start(self) -> None:
        for command in self.controller.launch_commands():
            self._send(command)

    def _emit_stop(self) -> None:
        self._send(self.controller.adapter.stop(time.time()))

    def _emit_relock(self) -> None:
        self._send(self.controller.adapter.relock(time.time()))

    def _emit_emergency(self) -> None:
        self._send(self.controller.adapter.emergency_stop(time.time()))

    def _emit_distance(self) -> None:
        self._send(self.controller.distance_command())

    def _on_mode_changed(self) -> None:
        selected = ShotMode(self.shot_mode_var.get())
        self.controller.set_selected_shot_mode(selected)
        if (
            self.last_snapshot is not None
            and self.last_snapshot.mission.mission_state == MissionState.FILM
        ):
            self._send(self.controller.mode_command())

    def _on_distance_slider(self, value: str) -> None:
        self.controller.set_selected_distance(float(value))
        self._sync_distance_readout()

    def _apply_distance_preset(self, meters: float) -> None:
        self.controller.set_selected_distance(meters)
        self.distance_scale.set(self.controller.selected_distance_m)
        self._sync_distance_readout()
        self._emit_distance()

    def apply_snapshot(self, snapshot: ProductSnapshot) -> None:
        """Update the panel with a new product snapshot."""
        self.last_snapshot = snapshot
        self._apply_view_state(self.controller.build_view_state(snapshot))
        for command in self.controller.maybe_emit_deferred_commands(snapshot):
            self._send(command)

    def run(self) -> None:  # pragma: no cover - requires UI event loop
        self.root.mainloop()


def _demo_send_command(command: AppCommand) -> None:  # pragma: no cover - console helper
    print(f"[operator-panel] {command}")


def main() -> None:  # pragma: no cover - manual UI demo
    panel = OperatorControlPanel(send_command=_demo_send_command)
    panel.run()


if __name__ == "__main__":  # pragma: no cover - manual UI demo
    main()
