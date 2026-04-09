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

### Flight (`flight/`)

| Module                | File                              | Rate   | Runs On   |
|-----------------------|-----------------------------------|--------|-----------|
| Flight Interface      | `flight/flight_interface.py`      | 50 Hz  | Companion |
| Shot Controller       | `flight/shot_controller.py`       | 25 Hz  | Companion |
| Stability Supervisor  | `flight/stability_supervisor.py`  | 25 Hz  | Companion |
| Mission State Machine | `flight/mission_state_machine.py` | 25 Hz  | Companion |
| Safety Module         | `flight/safety_module.py`         | 50 Hz  | Companion |
| Geofence              | `flight/geofence.py`              | 50 Hz  | Companion |
| MAVLink Transport     | `flight/pymavlink_transport.py`   | 50 Hz  | Companion |

### Perception (`perception/`)

| Module              | File                          | Rate   | Runs On   |
|---------------------|-------------------------------|--------|-----------|
| Perception Adapter  | `perception/adapter.py`       | 25 Hz  | Companion |
| Geometry            | `perception/geometry.py`      | —      | Companion |
| Identity Lock       | `perception/identity_lock.py` | 25 Hz  | Companion |
| Live Camera Tracker | `perception/live_camera.py`   | 30 fps | Companion |
| Perception Params   | `perception/parameters.py`    | —      | Companion |
| Perception Schemas  | `perception/schemas.py`       | —      | Shared    |

### Product (`product/`)

| Module          | File                          | Rate | Runs On        |
|-----------------|-------------------------------|------|----------------|
| Product Adapters| `product/adapters.py`         | —    | Companion      |
| Config Profiles | `product/config_profiles.py`  | —    | Companion      |
| Operator Panel  | `product/operator_panel.py`   | —    | Dev workstation |
| Session Log     | `product/session_log.py`      | —    | Companion      |
| Product Schemas | `product/schemas.py`          | —    | Shared         |

### Shared (`interfaces/`, `config/`)

| Module        | File                    | Rate | Runs On   |
|---------------|-------------------------|------|-----------|
| Event Bus     | `interfaces/event_bus.py`| —   | Companion |
| Clock         | `interfaces/clock.py`   | —    | Companion |
| Schemas       | `interfaces/schemas.py` | —    | Shared    |
| Configuration | `config/parameters.py`  | —    | Companion |

### Simulation (`sim/`)

| Module            | File                        | Rate   | Runs On         |
|-------------------|-----------------------------|--------|-----------------|
| Simulation Harness| `sim/sim_harness.py`        | 50 Hz  | Dev workstation |
| Interactive Sim   | `sim/interactive_sim.py`    | 50 Hz  | Dev workstation |
| Regression Runner | `sim/regression_runner.py`  | 50 Hz  | Dev workstation |
| Param Sweep       | `sim/param_sweep.py`        | 50 Hz  | Dev workstation |
| 3D Visualizer     | `sim/visualize_sim_3d.py`   | —      | Dev workstation |
| 2D Visualizer     | `sim/visualize_sim.py`      | —      | Dev workstation |

## Quick Start

```bash
# Clone and set up
git clone https://github.com/TerjeOtsa/drone-journalist.git
cd drone-journalist
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate          # macOS / Linux
pip install -e ".[dev]"

# Run the test suite (157 tests)
python -m pytest tests/ -v

# Run the full simulation
python -m sim.sim_harness

# Interactive dashboard with live controls
python -m sim.interactive_sim

# 2D / 3D visualization and GIF export
python -m sim.visualize_sim
python -m sim.visualize_sim_3d
python -m sim.visualize_sim_3d --gif sim_3d.gif

# Run regression scenarios
python -m sim.regression_runner

# Grid-search parameter tuning
python -m sim.param_sweep

# Bench-test live tracking with a camera
python -m perception.live_camera --camera 0

# Operator control panel
python -m product.operator_panel
```

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for the full developer guide, project layout, and PR checklist.

## Dev And Hardware Installs

```bash
# Dev tooling (included in pip install -e ".[dev]")
pip install -e ".[dev]"

# Optional hardware / SITL transport support
pip install -e ".[hardware]"
```

The MAVLink transport lives in `flight/pymavlink_transport.py` and plugs into
`FlightInterface` through the transport seam in `flight/flight_interface.py`.

## Documentation

| Document | What it covers |
|---|---|
| [CONTRIBUTING.md](CONTRIBUTING.md) | Developer setup, project layout, coding rules, PR checklist |
| [DESIGN.md](DESIGN.md) | Full technical design — state machines, shot modes, safety, parameters |
| [VISION_TRACKING_IDENTITY.md](VISION_TRACKING_IDENTITY.md) | Perception & identity lock contract |
| [VIDEO_APP_INFRASTRUCTURE.md](VIDEO_APP_INFRASTRUCTURE.md) | Video pipeline, phone app, infrastructure target architecture |

## Live Camera Test

Bench-test the perception stack with a USB camera or video file:

```bash
python -m perception.live_camera --camera 0
python -m perception.live_camera --video sample.mp4
python -m perception.live_camera --camera 0 --log live_tracks.ndjson
```

Keys: `l` = select target, `r` = reset, `q` = quit.

This is a perception-only test — it does not run flight control.

## Key Design Principles

1. **Reliability over performance** — every module has a degraded fallback.
2. **Assume imperfect tracking** — the system smooths, predicts, and gracefully degrades.
3. **Single subject only** — simplifies state machine and safety logic.
4. **Conservative motion** — jerk-limited, velocity-capped, always smooth.
5. **Predictable safety** — geofence, battery, and link-loss behave deterministically.
6. **No magic numbers** — every constant in `config/parameters.py` with a unit comment.
