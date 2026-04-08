# Autonomy & Flight Systems — Complete Design Document

## Table of Contents
1. [System Architecture](#1-system-architecture)
2. [Mission State Machine](#2-mission-state-machine)
3. [Shot Controller](#3-shot-controller)
4. [Stability Supervisor](#4-stability-supervisor)
5. [Safety & Geofence Module](#5-safety--geofence-module)
6. [Interfaces — Exact Schemas](#6-interfaces--exact-schemas)
7. [Tunable Parameters](#7-tunable-parameters)
8. [Implementation Plan](#8-implementation-plan)

---

## 1. System Architecture

### 1.1 Module Breakdown

```
┌────────────────────────────────────────────────────────────────────────┐
│                    COMPANION COMPUTER  (Jetson / Pi 5)                 │
│                                                                        │
│   ┌─────────────┐                                                      │
│   │ Perception  │  (EXTERNAL — not our scope)                          │
│   │ Pipeline    │──┐                                                   │
│   └─────────────┘  │  TargetTrack @ 25 Hz                             │
│                     ▼                                                  │
│   ┌────────────────────────────────┐    ┌──────────────────────────┐   │
│   │     Mission State Machine      │◄──►│    Stability Supervisor  │   │
│   │  (lifecycle, transitions,      │    │  (wind, jitter, drift    │   │
│   │   ready_to_record)             │    │   → speed_scale, hover)  │   │
│   └──────────┬─────────────────────┘    └────────────┬─────────────┘   │
│              │ MissionState, ShotMode                 │ StabilityLevel │
│              ▼                                        ▼                │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │                      Shot Controller                           │   │
│   │  (relative positioning, smoothing, 4 shot modes,               │   │
│   │   prediction, jerk limiting)                                   │   │
│   └──────────────────────────┬─────────────────────────────────────┘   │
│                              │ FlightSetpoint                         │
│                              ▼                                        │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │                    Safety & Geofence Module                    │   │
│   │  (geofence, battery, link-loss, proximity, emergency stop)     │   │
│   │  ► clamps / overrides setpoints                                │   │
│   └──────────────────────────┬─────────────────────────────────────┘   │
│                              │ clamped FlightSetpoint                  │
│                              ▼                                        │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │                    Flight Interface Module                     │   │
│   │  (MAVLink serialisation, watchdog, arm/disarm)                 │   │
│   └──────────────────────────┬─────────────────────────────────────┘   │
│                              │ MAVLink SET_POSITION_TARGET_LOCAL_NED   │
└──────────────────────────────┼────────────────────────────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │   PX4 / ArduPilot   │
                    │  (low-level stab.)  │
                    └─────────────────────┘
```

### 1.2 Data Flow (per tick)

```
1. Perception publishes TargetTrack         (25 Hz)
2. Autopilot telemetry is read              (50 Hz)
3. SafetyModule.update()                    (50 Hz)  → SafetyStatus
4. StabilitySupervisor.update()             (25 Hz)  → StabilityLevel
5. MissionStateMachine.update()             (25 Hz)  → MissionState, ShotMode
6. ShotController.update()                  (25 Hz)  → FlightSetpoint
7. SafetyModule.clamp_setpoint()            (50 Hz)  → clamped FlightSetpoint
8. FlightInterface.send() + .tick()         (50 Hz)  → MAVLink to autopilot
```

### 1.3 What Runs Where

| Component              | Hardware           | Rate  | Priority   |
|------------------------|--------------------|-------|------------|
| Flight Interface       | Companion CPU      | 50 Hz | RT (high)  |
| Safety Module          | Companion CPU      | 50 Hz | RT (high)  |
| Shot Controller        | Companion CPU      | 25 Hz | Normal     |
| Mission State Machine  | Companion CPU      | 25 Hz | Normal     |
| Stability Supervisor   | Companion CPU      | 25 Hz | Normal     |
| PX4/ArduPilot          | Flight controller  | 400 Hz| Firmware   |

---

## 2. Mission State Machine

### 2.1 State Diagram

```
                    ┌──────┐
                    │ IDLE │◄────────────────────────────────────┐
                    └──┬───┘                                     │
                       │ operator "start"                        │
                       │ (GPS 3D fix + batt > 20%)               │
                       ▼                                         │
                    ┌──────────┐                                 │
                    │ TAKEOFF  │                                 │
                    └──┬───────┘                                 │
                       │ alt ≥ 90% target  (or timeout → LAND)   │
                       ▼                                         │
                    ┌──────────┐                                 │
                    │ ACQUIRE  │─── timeout ──► RETURN           │
                    └──┬───────┘                    │            │
                       │ lock_state ∈ {LOCKED,      │            │
                       │   CANDIDATE}               │            │
                       ▼                            │            │
                    ┌──────┐                        │            │
                    │ LOCK │─── lock LOST ──►ACQUIRE│            │
                    └──┬───┘                        │            │
                       │ stable lock ≥ 1.5 s        │            │
                       ▼                            │            │
              ┌────────────────┐                    │            │
              │     FILM       │                    │            │
              │ ready_to_record│                    │            │
              └──┬──────┬──────┘                    │            │
                 │      │                           │            │
     weak lock   │      │ lost lock                 │            │
     > 5 s       │      │ > 3 s                     │            │
                 ▼      ▼                           │            │
              ┌────────────┐                        │            │
              │  DEGRADE   │── re-lock ──► FILM     │            │
              └──┬─────────┘                        │            │
                 │ timeout 10 s or max retries      │            │
                 ▼                                  │            │
              ┌────────┐                            │            │
              │ RETURN │◄───────────────────────────┘            │
              └──┬─────┘                                         │
                 │ home reached (< 2 m)                          │
                 ▼                                               │
              ┌──────┐                                           │
              │ LAND │───────────────────────────────────────────┘
              └──────┘         (in_air == false → IDLE)

     ┌───────────┐
     │ EMERGENCY │  ◄── emergency_stop (any state)
     └─────┬─────┘      or SAFETY: EMERGENCY_STOP
           │ in_air == false
           └──────────────► IDLE
```

### 2.2 Transition Rules Summary

| From      | To        | Condition                                   | Timer / Threshold        |
|-----------|-----------|---------------------------------------------|--------------------------|
| IDLE      | TAKEOFF   | cmd="start" ∧ GPS≥3D ∧ batt>20%            | —                        |
| TAKEOFF   | ACQUIRE   | altitude ≥ 90% of takeoff_altitude          | —                        |
| TAKEOFF   | LAND      | timeout                                     | 15 s                     |
| ACQUIRE   | LOCK      | lock ∈ {LOCKED, CANDIDATE}                  | —                        |
| ACQUIRE   | RETURN    | timeout                                     | 20 s                     |
| LOCK      | FILM      | lock=LOCKED continuously                    | 1.5 s                    |
| LOCK      | ACQUIRE   | lock=LOST                                   | —                        |
| FILM      | DEGRADE   | lock=WEAK too long OR lock=LOST too long    | 5 s weak / 3 s lost      |
| FILM      | RETURN    | operator "stop"                             | —                        |
| DEGRADE   | FILM      | lock recovers (max 3 retries)               | —                        |
| DEGRADE   | RETURN    | timeout or max retries exhausted            | 10 s                     |
| DEGRADE   | ACQUIRE   | operator "relock"                           | —                        |
| RETURN    | LAND      | distance to home < 2 m                      | —                        |
| LAND      | IDLE      | in_air == false                             | —                        |
| ANY       | EMERGENCY | operator "emergency_stop"                   | —                        |
| ANY       | LAND      | safety override = LAND_NOW                  | —                        |
| ANY       | RETURN    | safety override = RETURN_HOME               | —                        |

---

## 3. Shot Controller

### 3.1 Shot Modes

#### STANDUP (default)
```
        Subject
           ●
           │  ← standup_distance (4 m)
           │
         ╔═╗
         ║D║  at standup_altitude (1.8 m AGL)
         ╚═╝
```
- Hover at fixed offset from subject
- Face the subject (yaw towards)
- Smooth position holding with deadband

#### WALK_AND_TALK
```
        Subject  ───velocity──►
           ●
            \
             \  ← walktalk_distance (5 m)
              \
             ╔═╗
             ║D║  behind + slightly above (2 m AGL)
             ╚═╝
```
- Follow behind the subject's velocity vector
- Predict future position (0.3 s horizon)
- Fall back to STANDUP geometry if subject is stationary

#### WIDE SAFETY
```
        Subject
           ●
           │
           │  ← wide_distance (10 m)
           │
         ╔═╗
         ║D║  at wide_altitude (4 m AGL)
         ╚═╝
```
- Increased standoff for degraded conditions
- Higher altitude for wider framing

#### ORBIT
```
            ╔═╗
           / ║D║ \
          /  ╚═╝  \
         /    │    \
        │  Subject  │   orbit_radius = 6 m
        │     ●     │   orbit_speed = 15 °/s
         \         /    orbit_altitude = 3 m
          \       /
           \_____/
```
- Constant radius circle around subject
- Speed scaled by stability supervisor

### 3.2 Smoothing Pipeline

```
Raw target pos ──► LPF (α=0.15) ──► Prediction ──► Desired pos
                                                        │
                        ┌───────────────────────────────┘
                        ▼
                   Error vector
                        │
                        ▼
                   P-controller (kp = max_vel / 3m)
                        │
                        ▼
                   Velocity cap (5 m/s)
                        │
                        ▼
                   Acceleration limiter (2 m/s²)
                        │
                        ▼
                   FlightSetpoint (pos + vel + yaw)
```

- **Position LPF**: α=0.15 (conservative, ~7 frames to converge)
- **Velocity LPF**: α=0.10 (even smoother for feed-forward)
- **Yaw LPF**: α=0.12
- **Deadband**: 0.15 m position, 1.7° yaw (suppresses micro-corrections)
- **Jerk limiter**: max acceleration change = 2.0 m/s² per tick

---

## 4. Stability Supervisor

### 4.1 Three Assessment Axes

| Axis           | Signal                        | Window      |
|----------------|-------------------------------|-------------|
| **Wind**       | EKF wind estimate magnitude   | Instantaneous |
| **Jitter**     | Velocity std-dev over N ticks | 25 samples (1 s) |
| **Drift**      | Max position deviation        | 2 s window  |

### 4.2 Level Thresholds

| Level     | Wind (m/s) | Jitter σ (m/s²) | Drift (m)  |
|-----------|-----------|------------------|------------|
| NOMINAL   | < 5       | < 1.5            | < 0.8      |
| MARGINAL  | 5–8       | 1.5–3.0          | 0.8–1.5    |
| DEGRADED  | 8–12      | 3.0–5.0          | 1.5–3.0    |
| CRITICAL  | > 12      | > 5.0            | > 3.0      |

### 4.3 Degradation Response

| Level     | Speed Scale | Extra Distance | Hover? |
|-----------|-------------|----------------|--------|
| NOMINAL   | 100%        | +0 m           | No     |
| MARGINAL  | 70%         | +0 m           | No     |
| DEGRADED  | 40%         | +2 m           | No     |
| CRITICAL  | 0%          | —              | Yes    |

---

## 5. Safety & Geofence Module

### 5.1 Geofence
- **Shape**: Cylinder around home position
- **Radius**: 100 m (configurable)
- **Ceiling**: 30 m AGL
- **Floor**: 0.5 m AGL
- **Warning margin**: 10 m inside boundary → REDUCE_SPEED
- **Breach**: → RETURN_HOME

### 5.2 Battery Failsafe (staged)
| Level      | Battery %  | Action         |
|------------|-----------|----------------|
| Warning    | ≤ 30%     | REDUCE_SPEED   |
| Return     | ≤ 20%     | RETURN_HOME    |
| Critical   | ≤ 10%     | LAND_NOW       |

### 5.3 Link Loss (staged)
| Duration     | Action       |
|-------------|-------------|
| > 3 s       | RETURN_HOME  |
| > 8 s       | LAND_NOW     |

### 5.4 Subject Proximity
| Distance       | Action          |
|---------------|-----------------|
| < 2.0 m       | INCREASE_DIST   |
| < 1.0 m       | HOVER (brake)   |

### 5.5 Emergency Stop
- Immediate descent at 1.0 m/s
- Disarm at 0.3 m AGL
- Motor kill path available

### 5.6 Override Priority (lowest → highest)
```
NONE < REDUCE_SPEED < INCREASE_DIST < HOVER < RETURN_HOME < LAND_NOW < EMERGENCY_STOP
```
Only the highest-priority override propagates.

---

## 6. Interfaces — Exact Schemas

All schemas are defined in `interfaces/schemas.py` as Python dataclasses.
Below are the JSON-equivalent representations:

### 6.1 Input: TargetTrack (from Perception)

```json
{
  "timestamp": 1712500000.123,
  "target_position_image": [0.52, 0.48],
  "target_position_world": {"x": 5.2, "y": -0.3, "z": 0.0},
  "target_velocity_world": {"x": 1.1, "y": 0.2, "z": 0.0},
  "identity_confidence": 0.94,
  "tracking_confidence": 0.91,
  "lock_state": "locked",
  "bounding_box": [0.35, 0.25, 0.30, 0.50]
}
```

### 6.2 Input: DroneTelemetry (from Autopilot)

```json
{
  "timestamp": 1712500000.123,
  "position": {"x": 1.0, "y": 0.5, "z": -2.5},
  "velocity": {"x": 0.1, "y": -0.05, "z": 0.0},
  "attitude_euler": {"x": 0.01, "y": -0.02, "z": 1.57},
  "gps": {"lat": 59.9, "lon": 10.7, "alt_msl": 102.5},
  "battery_voltage": 15.2,
  "battery_percent": 72.0,
  "armed": true,
  "in_air": true,
  "gps_fix": 3,
  "satellites": 14,
  "wind_estimate": {"x": 2.1, "y": -1.3, "z": 0.0},
  "home_position": {"x": 0.0, "y": 0.0, "z": 0.0}
}
```

### 6.3 Input: AppCommand (from Mobile App)

```json
{
  "timestamp": 1712500000.123,
  "action": "set_shot_mode",
  "shot_mode": "orbit",
  "desired_distance": null,
  "desired_altitude": null
}
```
Valid actions: `"start"`, `"stop"`, `"pause"`, `"resume"`, `"relock"`,
`"set_shot_mode"`, `"set_distance"`, `"emergency_stop"`

### 6.4 Output: FlightSetpoint (to Autopilot)

```json
{
  "timestamp": 1712500000.123,
  "position": {"x": 1.2, "y": 0.3, "z": -2.5},
  "velocity": {"x": 0.5, "y": 0.1, "z": 0.0},
  "yaw": 1.32,
  "yaw_rate": null,
  "is_body_frame": false
}
```

### 6.5 Output: SafetyStatus

```json
{
  "timestamp": 1712500000.123,
  "active_override": "none",
  "geofence_ok": true,
  "battery_ok": true,
  "link_ok": true,
  "min_distance_ok": true,
  "reasons": []
}
```

### 6.6 Output: MissionStatus (to App + Logger)

```json
{
  "timestamp": 1712500000.123,
  "state": "film",
  "shot_mode": "walk_and_talk",
  "ready_to_record": true,
  "stability": "nominal",
  "safety": { "...SafetyStatus..." },
  "message": "film"
}
```

### 6.7 Events Emitted (SystemEvent)

```json
{"timestamp": 1712500000.1, "source": "mission",   "event": "state_change",     "payload": {"old": "lock", "new": "film", "reason": "lock confirmed"}}
{"timestamp": 1712500001.2, "source": "stability",  "event": "stability_change", "payload": {"old": "nominal", "new": "marginal"}}
{"timestamp": 1712500002.3, "source": "safety",     "event": "safety_override",  "payload": {"override": "return_home", "reasons": ["battery low: 18%"]}}
```

---

## 7. Tunable Parameters

All parameters are in `config/parameters.py`.  Complete listing:

### Shot Controller
| Parameter                  | Default | Unit    | Description                            |
|----------------------------|---------|---------|----------------------------------------|
| standup_distance           | 4.0     | m       | Hover distance for standup shot        |
| standup_altitude           | 1.8     | m AGL   | Hover height for standup shot          |
| walktalk_distance          | 5.0     | m       | Follow distance for walk-and-talk      |
| walktalk_altitude          | 2.0     | m AGL   | Follow height                          |
| wide_distance              | 10.0    | m       | Standoff for wide safety shot          |
| wide_altitude              | 4.0     | m AGL   | Height for wide safety shot            |
| orbit_radius               | 6.0     | m       | Circle radius                          |
| orbit_altitude             | 3.0     | m AGL   | Circle height                          |
| orbit_speed_deg_s          | 15.0    | °/s     | Circle angular speed                   |
| min_distance               | 2.0     | m       | Hard minimum to subject                |
| max_distance               | 30.0    | m       | Hard maximum from subject              |
| min_altitude_agl           | 1.0     | m       | Floor                                  |
| max_altitude_agl           | 25.0    | m       | Ceiling                                |
| position_lpf_alpha         | 0.15    | —       | Position filter (0=slow, 1=instant)    |
| velocity_lpf_alpha         | 0.10    | —       | Velocity filter                        |
| yaw_lpf_alpha              | 0.12    | —       | Yaw filter                             |
| max_velocity               | 5.0     | m/s     | Absolute speed cap                     |
| max_acceleration           | 2.0     | m/s²    | Jerk limiter                           |
| max_yaw_rate               | 0.8     | rad/s   | Yaw rate cap                           |
| position_deadband          | 0.15    | m       | Ignore errors below this               |
| yaw_deadband               | 0.03    | rad     | Ignore yaw errors below this           |
| target_prediction_horizon  | 0.3     | s       | Feed-forward prediction window         |

### Stability Supervisor
| Parameter                  | Default | Unit    | Description                            |
|----------------------------|---------|---------|----------------------------------------|
| wind_marginal              | 5.0     | m/s     | → MARGINAL                             |
| wind_degraded              | 8.0     | m/s     | → DEGRADED                             |
| wind_critical              | 12.0    | m/s     | → CRITICAL                             |
| accel_jitter_window        | 25      | samples | Rolling window                         |
| accel_jitter_marginal      | 1.5     | m/s²    | σ threshold                            |
| accel_jitter_degraded      | 3.0     | m/s²    | σ threshold                            |
| accel_jitter_critical      | 5.0     | m/s²    | σ threshold                            |
| drift_window_s             | 2.0     | s       | Drift assessment window                |
| drift_marginal             | 0.8     | m       | Drift threshold                        |
| drift_degraded             | 1.5     | m       | Drift threshold                        |
| drift_critical             | 3.0     | m       | Drift threshold                        |
| marginal_speed_scale       | 0.7     | —       | Speed multiplier                       |
| degraded_speed_scale       | 0.4     | —       | Speed multiplier                       |
| degraded_distance_add      | 2.0     | m       | Extra standoff                         |

### Mission State Machine
| Parameter                  | Default | Unit    | Description                            |
|----------------------------|---------|---------|----------------------------------------|
| takeoff_altitude           | 2.5     | m AGL   | Target takeoff height                  |
| takeoff_timeout_s          | 15.0    | s       | Abort if not airborne                  |
| acquire_timeout_s          | 20.0    | s       | Time to find subject                   |
| lock_confirm_time_s        | 1.5     | s       | Stable lock before FILM                |
| weak_lock_timeout_s        | 5.0     | s       | Weak lock → DEGRADE                    |
| lost_lock_timeout_s        | 3.0     | s       | Lost lock → DEGRADE                    |
| degrade_hold_time_s        | 10.0    | s       | DEGRADE → auto RETURN                  |
| return_speed               | 3.0     | m/s     | RTH velocity                           |
| land_descent_rate          | 0.5     | m/s     | Descent speed                          |
| relock_attempts            | 3       | count   | Max re-locks before RETURN             |

### Safety & Geofence
| Parameter                  | Default | Unit    | Description                            |
|----------------------------|---------|---------|----------------------------------------|
| geofence_radius            | 100.0   | m       | Keep-in cylinder radius                |
| geofence_ceiling           | 30.0    | m AGL   | Keep-in ceiling                        |
| geofence_floor             | 0.5     | m AGL   | Keep-in floor                          |
| geofence_warn_margin       | 10.0    | m       | Warning distance inside fence          |
| battery_warn_percent       | 30.0    | %       | REDUCE_SPEED trigger                   |
| battery_return_percent     | 20.0    | %       | RETURN_HOME trigger                    |
| battery_land_percent       | 10.0    | %       | LAND_NOW trigger                       |
| heartbeat_timeout_s        | 3.0     | s       | Link-loss → RETURN                     |
| critical_link_timeout_s    | 8.0     | s       | Link-loss → LAND                       |
| subject_min_distance       | 2.0     | m       | Proximity floor                        |
| subject_emergency_dist     | 1.0     | m       | Proximity → HOVER                      |
| emergency_descent_rate     | 1.0     | m/s     | Emergency landing speed                |
| motor_kill_altitude        | 0.3     | m AGL   | Disarm height                          |

### Flight Interface
| Parameter                  | Default | Unit    | Description                            |
|----------------------------|---------|---------|----------------------------------------|
| setpoint_rate_hz           | 50.0    | Hz      | Autopilot command rate                 |
| timeout_no_setpoint_s      | 0.5     | s       | Watchdog → hover if no new setpoint    |

---

## 8. Implementation Plan

### 8.1 Language & Runtime

| Component           | Language   | Runtime           | Why                           |
|---------------------|-----------|-------------------|-------------------------------|
| All modules         | Python 3.11+ | asyncio / threaded | Rapid iteration, rich ecosystem |
| Flight Interface    | Python + pymavlink | Direct serial/UDP | MAVLink native support        |
| Production hardening| Consider Rust/C++ for shot controller | — | If 25 Hz is too slow in Python |
| Config              | Python dataclasses | — | Type-safe, IDE-friendly       |

### 8.2 Update Loop Timing

```
Main loop thread (50 Hz — 20 ms period):
  ├── Read telemetry from autopilot          (every tick)
  ├── Read latest TargetTrack from perception (every tick, latest)
  ├── SafetyModule.update()                  (every tick — 50 Hz)
  │
  ├── [every 2nd tick — 25 Hz]:
  │     ├── StabilitySupervisor.update()
  │     ├── MissionStateMachine.update()
  │     └── ShotController.update()
  │
  ├── SafetyModule.clamp_setpoint()          (every tick)
  ├── FlightInterface.send()                 (every tick)
  └── FlightInterface.tick()                 (every tick — watchdog)
```

### 8.3 Testing Strategy

| Level          | Tool    | What's tested                                | Files                           |
|----------------|---------|----------------------------------------------|---------------------------------|
| Unit           | pytest  | Each module in isolation with mock inputs     | `tests/test_*.py`              |
| Integration    | pytest  | Full pipeline with sim physics                | `tests/test_integration.py`    |
| Simulation     | Custom  | 70 s scenario with events                     | `sim/sim_harness.py`           |
| SITL           | PX4 SITL| Full MAVLink with Gazebo                     | (future)                        |
| Field          | Manual  | Real hardware, controlled environment         | (future)                        |

### 8.4 Development Phases

| Phase | Milestone                          | Duration  |
|-------|------------------------------------|-----------|
| 1     | Schemas + state machine + tests    | ✅ Done   |
| 2     | Shot controller + smoothing        | ✅ Done   |
| 3     | Safety + geofence + stability      | ✅ Done   |
| 4     | Simulation harness                 | ✅ Done   |
| 5     | PX4 SITL integration               | 1 week    |
| 6     | Real hardware bringup              | 2 weeks   |
| 7     | Field testing + parameter tuning   | 2 weeks   |

---

## Summary

This system provides a **complete, testable, reliable** autonomy stack for a journalist follow-drone:

- **9 states** with deterministic transitions and timing rules
- **4 shot modes** with smooth, jerk-limited motion
- **4-level stability** assessment with progressive degradation
- **5 safety checks** running at 50 Hz with priority-based overrides
- **~60 tunable parameters** — no magic numbers in code
- **Full simulation** with a 70-second scenario exercising wind, lock loss, shot changes
- **37+ unit tests** covering every module boundary
