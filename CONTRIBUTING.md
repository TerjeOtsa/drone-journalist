# Contributing

## Prerequisites

- Python 3.12+
- Git
- (Optional) A webcam for live perception testing
- (Optional) PX4 SITL + Gazebo for hardware-in-the-loop testing

## First-time Setup

```bash
# Clone the repo
git clone https://github.com/TerjeOtsa/drone-journalist.git
cd drone-journalist

# Create a virtual environment
python -m venv .venv

# Activate it
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

## Everyday Commands

| What                          | Command                                          |
|-------------------------------|--------------------------------------------------|
| Run all tests                 | `python -m pytest`                               |
| Run tests verbose             | `python -m pytest tests/ -v`                     |
| Run one test file             | `python -m pytest tests/test_shot_controller.py` |
| Lint (auto-fix)               | `ruff check --fix .`                             |
| Lint (check only)             | `ruff check .`                                   |
| Format                        | `ruff format .`                                  |
| Type check                    | `mypy config flight interfaces perception product sim` |
| Run simulation                | `python -m sim.sim_harness`                      |
| Interactive sim dashboard     | `python -m sim.interactive_sim`                  |
| 3D visualization              | `python -m sim.visualize_sim_3d`                 |
| Regression scenarios          | `python -m sim.regression_runner`                |
| Parameter sweep               | `python -m sim.param_sweep`                      |
| Live camera test              | `python -m perception.live_camera --camera 0`    |
| Operator panel                | `python -m product.operator_panel`               |

## Project Layout

```
drone-journalist/
├── config/              # All tunable parameters (dataclasses)
│   └── parameters.py    # SystemConfig → ShotParams, SafetyParams, MissionParams, …
│
├── interfaces/          # Shared data types and infrastructure
│   ├── schemas.py       # TargetTrack, FlightSetpoint, DroneTelemetry, Vec3, enums
│   ├── event_bus.py     # In-process pub/sub for system events
│   └── clock.py         # Mockable wall-clock abstraction
│
├── flight/              # Core autonomy modules (the "brain")
│   ├── mission_state_machine.py   # IDLE → TAKEOFF → ACQUIRE → LOCK → FILM → …
│   ├── shot_controller.py         # 4 shot modes + face-aware framing bias
│   ├── stability_supervisor.py    # Wind/jitter assessment → speed scaling
│   ├── safety_module.py           # Geofence, battery, link-loss, proximity
│   ├── geofence.py                # Polygon keep-in / exclusion zone math
│   ├── flight_interface.py        # Setpoint throttling + watchdog → autopilot
│   └── pymavlink_transport.py     # MAVLink serial/UDP transport (hardware only)
│
├── perception/          # Vision and identity tracking
│   ├── identity_lock.py # Single-subject lock state machine
│   ├── live_camera.py   # OpenCV bench-test tracker (camera/video input)
│   ├── adapter.py       # LockResult → TargetTrack conversion
│   ├── geometry.py      # Monocular ground projection
│   ├── schemas.py       # CandidateObservation, LockResult, IdentityCues, …
│   └── parameters.py    # IdentityLockParams
│
├── product/             # Operator-facing product layer
│   ├── operator_panel.py    # Desktop start/stop/mode control GUI
│   ├── adapters.py          # Autonomy ↔ product-layer translation
│   ├── config_profiles.py   # Shot profiles (walk, standup, interview)
│   ├── session_log.py       # Durable event + snapshot logging
│   └── schemas.py           # Product-layer data types
│
├── sim/                 # Simulation and testing tools
│   ├── sim_harness.py       # Full physics sim (drag, battery, wind, ground effect)
│   ├── interactive_sim.py   # Live matplotlib dashboard with controls
│   ├── visualize_sim.py     # 2D playback
│   ├── visualize_sim_3d.py  # 3D playback / GIF export
│   ├── regression_runner.py # YAML scenario → run → assert metrics
│   └── param_sweep.py       # Grid search over parameter space
│
├── scenarios/           # YAML regression test definitions
│   ├── smoke.yaml
│   ├── quick_start.yaml
│   ├── heavy_wind.yaml
│   ├── lock_loss_recovery.yaml
│   └── walking_follow.yaml
│
├── tests/               # pytest unit + integration tests
│
├── DESIGN.md            # Full technical design document
├── VISION_TRACKING_IDENTITY.md  # Perception/identity contract spec
├── VIDEO_APP_INFRASTRUCTURE.md  # Video/app/infra target architecture
├── CONTRIBUTING.md      # ← you are here
└── README.md            # Project overview and quick start
```

## Architecture at a Glance

The system runs a **50 Hz main loop** on the companion computer:

```
Every tick (50 Hz):
  ├── Read telemetry from autopilot
  ├── SafetyModule.update()                    ← always runs
  │
  ├── Every 2nd tick (25 Hz):
  │     ├── StabilitySupervisor.update()       ← wind/jitter → speed scale
  │     ├── MissionStateMachine.update()       ← lifecycle transitions
  │     └── ShotController.update()            ← desired position/velocity
  │
  ├── SafetyModule.clamp_setpoint()            ← enforce hard limits
  └── FlightInterface.send() → MAVLink out     ← to autopilot
```

**Data flows in one direction**: Perception → Mission → Shot → Safety → Flight Interface → Autopilot.
Safety can override anything downstream of it.

## Coordinate System

Everything uses **NED** (North-East-Down):
- `x` = North (metres)
- `y` = East (metres)
- `z` = Down (metres) — **negative z = up** (altitude = −z)
- Yaw = 0 pointing North, positive clockwise

This matches PX4/ArduPilot convention.

## Writing Tests

- Each module has a corresponding `tests/test_<module>.py`
- Tests should be **fast** (no I/O, no sleep, no GUI)
- Use the existing test patterns — construct the module with parameters, feed it inputs, assert outputs
- Integration tests in `test_integration.py` run the full sim loop
- Regression tests in `test_regression.py` load YAML scenarios from `scenarios/`

Example unit test pattern:

```python
def test_something():
    params = ShotParams(standup_distance=5.0)
    shot = ShotController(params)
    # … feed it a TargetTrack and DroneTelemetry …
    sp = shot.update(track, telem, MissionState.FILM, ShotMode.STANDUP, SafetyOverride.NONE, dt_s=0.04)
    assert sp.position.x == pytest.approx(expected, abs=0.5)
```

## Pull Request Checklist

- [ ] All tests pass: `python -m pytest`
- [ ] Lint clean: `ruff check .`
- [ ] No new type errors: `mypy config flight interfaces perception product sim`
- [ ] If you added a parameter, add it to `config/parameters.py` with a docstring-style comment
- [ ] If you changed a module's public API, update its tests
- [ ] If you added a new file, add it to the relevant `__init__.py`

## Key Design Rules

1. **No magic numbers** — every constant lives in `config/parameters.py` with a unit comment.
2. **NED everywhere** — never use lat/lon or ENU inside the flight stack.
3. **Fail safe, not fail silent** — if something is wrong, degrade to a known-safe state (hover, return, land).
4. **One subject only** — the identity lock never tracks two people at once.
5. **Modules are stateless-ish** — state is in dataclass fields, reset via `__init__`. No global variables.
