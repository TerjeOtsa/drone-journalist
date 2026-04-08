# Autonomous Follow-and-Film Drone — Autonomy & Flight Systems

## Architecture Overview

This system sits **between** the perception/identity pipeline and the low-level autopilot
(PX4/ArduPilot). It owns everything related to *where to fly*, *how to move*, and *when to abort*.

```
┌─────────────────────────────────────────────────────────────────┐
│                     COMPANION COMPUTER (50 Hz)                  │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌───────────────────┐   │
│  │  Perception   │──▶│   Mission    │──▶│  Shot Controller  │   │
│  │  (external)   │   │ State Machine│   │  (relative pos)   │   │
│  └──────────────┘   └──────┬───────┘   └────────┬──────────┘   │
│                            │                     │              │
│                     ┌──────▼───────┐   ┌────────▼──────────┐   │
│                     │  Stability   │◀─▶│  Safety &          │   │
│                     │  Supervisor  │   │  Geofence Module   │   │
│                     └──────┬───────┘   └────────┬──────────┘   │
│                            │                     │              │
│                     ┌──────▼─────────────────────▼──────────┐   │
│                     │       Flight Interface Module          │   │
│                     │  (MAVLink setpoints → autopilot)       │   │
│                     └────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────┬───────────────────────────────────┘
                              │ MAVLink / Serial
                     ┌────────▼────────┐
                     │   PX4/ArduPilot │
                     │  (stabilization) │
                     └─────────────────┘
```

## Module Inventory

| Module                  | File                              | Rate   | Runs On          |
|-------------------------|-----------------------------------|--------|------------------|
| Flight Interface        | `flight/flight_interface.py`      | 50 Hz  | Companion        |
| Shot Controller         | `flight/shot_controller.py`       | 25 Hz  | Companion        |
| Stability Supervisor    | `flight/stability_supervisor.py`  | 25 Hz  | Companion        |
| Mission State Machine   | `flight/mission_state_machine.py` | 25 Hz  | Companion        |
| Safety & Geofence       | `flight/safety_module.py`         | 50 Hz  | Companion        |
| Configuration           | `config/parameters.py`            | —      | Companion        |
| Interfaces / Schemas    | `interfaces/schemas.py`           | —      | Shared           |
| Event Bus               | `interfaces/event_bus.py`         | —      | Companion        |
| Simulation Harness      | `sim/sim_harness.py`              | 50 Hz  | Dev workstation  |

## Quick Start

```bash
pip install -r requirements.txt
# Run the full simulation
python -m sim.sim_harness
# Visualize the simulation
python -m sim.visualize_sim
# 3D visualize the simulation
python -m sim.visualize_sim_3d
# Run the live interactive simulator
python -m sim.interactive_sim
# Save 3D playback without opening a GUI window
python -m sim.visualize_sim_3d --gif sim_3d.gif
# Faster GIF export for longer runs
python -m sim.visualize_sim_3d --gif sim_3d.gif --duration 20 --frame-step 3 --export-fps 15
# Bench-test live tracking with a camera
python -m perception.live_camera --camera 0
# Open the operator control panel
python -m product.operator_panel
# Run unit tests
python -m pytest tests/ -v
```

## Dev And Hardware Installs

```bash
# Dev tooling (pytest + lint/type tooling)
pip install -r requirements-dev.txt

# Optional hardware / SITL transport support
pip install -r requirements-hardware.txt
```

The production-facing MAVLink transport lives in
`flight/pymavlink_transport.py` and plugs into `FlightInterface` through the
transport seam added in `flight/flight_interface.py`.

The operator-facing desktop control panel lives in
`product/operator_panel.py`. It gives you a simple Start / Stop / Re-lock /
Emergency workflow plus clear shot-mode and follow-distance controls.

The simulator now also has a live dashboard in `sim/interactive_sim.py` with
start/stop, forced lock/loss, shot mode, follow distance, target walking, and
wind controls, along with live force readouts such as thrust, ground effect,
translational lift, and vortex-ring penalty.

## Related Spec

The external perception and identity contract is defined in
`VISION_TRACKING_IDENTITY.md`.

## Live Camera Test

You can now bench-test the perception stack with a USB camera or video file:

```bash
python -m perception.live_camera --camera 0
python -m perception.live_camera --video sample.mp4
python -m perception.live_camera --camera 0 --log live_tracks.ndjson
```

Controls:

- press `l` to draw/select the journalist ROI
- press `r` to reset tracking
- press `q` to quit

Notes:

- this is a perception bench test, not flight control
- it uses manual enrollment plus optical flow/template tracking, with HOG person detection to help relock
- the optional world estimate is camera-local only and is not yet suitable for direct flight control

## Key Design Principles

1. **Reliability over performance** — every module has a degraded fallback.
2. **Assume imperfect tracking** — the system smooths, predicts, and gracefully degrades.
3. **Single subject only** — simplifies state machine and safety logic.
4. **Conservative motion** — jerk-limited, velocity-capped, always smooth.
5. **Predictable safety** — geofence, battery, and link-loss behave deterministically.
