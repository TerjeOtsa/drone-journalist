"""
Tunable Parameters — single source of truth for every threshold, gain,
distance, delay, and limit in the autonomy stack.

Organised by module.  Every value has a comment with unit and rationale.
Adjust these for flight testing; the code never contains magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from flight.geofence import GeofenceConfig


# ═══════════════════════════════════════════════════════════════════════════════
#  SHOT CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ShotParams:
    """Camera framing, smoothing, and face-aware positioning for each shot mode."""

    # ── Distance & altitude (metres) ────────────────────────────────────
    standup_distance: float     = 4.0    # metres in front of subject
    standup_altitude: float     = 1.8    # metres AGL (≈ eye level)
    walktalk_distance: float    = 5.0    # metres behind/beside
    walktalk_altitude: float    = 2.0
    wide_distance: float        = 10.0   # metres, wide safety shot
    wide_altitude: float        = 4.0
    orbit_radius: float         = 6.0    # metres
    orbit_altitude: float       = 3.0
    orbit_speed_deg_s: float    = 15.0   # degrees per second around subject

    # ── Distance limits ─────────────────────────────────────────────────
    min_distance: float         = 2.0    # never closer than this (metres)
    max_distance: float         = 30.0   # never farther than this
    min_altitude_agl: float     = 1.0    # metres above ground
    max_altitude_agl: float     = 25.0   # metres above ground

    # ── Smoothing ───────────────────────────────────────────────────────
    position_lpf_alpha: float   = 0.15   # low-pass filter (0=slow, 1=instant)
    velocity_lpf_alpha: float   = 0.10
    yaw_lpf_alpha: float        = 0.12
    max_velocity: float         = 5.0    # m/s (capped in any axis)
    max_acceleration: float     = 2.0    # m/s² (jerk limiter)
    max_yaw_rate: float         = 0.8    # rad/s ≈ 45 °/s

    # ── Deadband ────────────────────────────────────────────────────────
    position_deadband: float    = 0.15   # metres — ignore errors smaller than this
    yaw_deadband: float         = 0.03   # radians ≈ 1.7°

    # ── Prediction ──────────────────────────────────────────────────────
    target_prediction_horizon: float = 0.3   # seconds into the future
    use_velocity_feed_forward: bool  = True

    # ── Face-aware framing ──────────────────────────────────────────────
    face_framing_enabled: bool        = True   # enable face-aware position bias
    face_low_threshold: float         = 0.2    # face score below this → bias position
    face_bias_lateral_m: float        = 1.0    # side-step metres when face not visible
    face_bias_altitude_m: float       = 0.3    # lower altitude by this when face not visible
    face_bias_alpha: float            = 0.03   # smoothing rate for face bias (slow)


# ═══════════════════════════════════════════════════════════════════════════════
#  STABILITY SUPERVISOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StabilityParams:
    """Wind, oscillation, and drift thresholds for the stability supervisor."""

    # ── Wind thresholds (m/s) ───────────────────────────────────────────
    wind_marginal: float        = 5.0    # above → MARGINAL
    wind_degraded: float        = 8.0    # above → DEGRADED
    wind_critical: float        = 12.0   # above → CRITICAL (land)

    # ── Oscillation detection ───────────────────────────────────────────
    accel_jitter_window: int    = 25     # samples (1 s at 25 Hz)
    accel_jitter_marginal: float = 1.5   # m/s² std-dev
    accel_jitter_degraded: float = 3.0
    accel_jitter_critical: float = 5.0

    # ── Position-hold drift ─────────────────────────────────────────────
    drift_window_s: float       = 2.0    # seconds
    drift_speed_gate: float     = 0.5    # m/s horizontal; only assess drift near hover
    drift_marginal: float       = 0.8    # metres
    drift_degraded: float       = 1.5
    drift_critical: float       = 3.0

    # ── Degradation responses ───────────────────────────────────────────
    marginal_speed_scale: float  = 0.7   # reduce max vel to 70 %
    degraded_speed_scale: float  = 0.4
    degraded_distance_add: float = 2.0   # metres extra standoff
    critical_action: str         = "hover"  # "hover" or "land"


# ═══════════════════════════════════════════════════════════════════════════════
#  MISSION STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MissionParams:
    """Timing and threshold knobs for the mission state machine."""

    takeoff_altitude: float          = 2.5   # metres AGL
    takeoff_timeout_s: float         = 15.0  # seconds before abort
    acquire_timeout_s: float         = 20.0  # seconds to find subject
    lock_confirm_time_s: float       = 1.5   # seconds of stable lock before FILM
    weak_lock_timeout_s: float       = 5.0   # seconds of weak lock → DEGRADE
    lost_lock_timeout_s: float       = 3.0   # seconds of LOST → begin return
    degrade_hold_time_s: float       = 10.0  # seconds in DEGRADE before auto-return
    return_speed: float              = 3.0   # m/s return-to-home speed
    land_descent_rate: float         = 0.5   # m/s
    relock_attempts: int             = 3     # max re-lock tries before return


# ═══════════════════════════════════════════════════════════════════════════════
#  SAFETY & GEOFENCE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SafetyParams:
    """Geofence, battery, link-loss, and proximity thresholds for the safety module."""

    # ── Geofence (legacy cylinder — used as fallback if no polygon given) ───
    geofence_radius: float       = 100.0   # metres from home (cylinder)
    geofence_ceiling: float      = 30.0    # metres AGL
    geofence_floor: float        = 0.5     # metres AGL (never below)
    geofence_warn_margin: float  = 10.0    # metres inside fence → warning

    # ── Polygon geofence (optional, overrides cylinder when set) ────────
    geofence_config: Optional["GeofenceConfig"] = None

    # ── Battery ─────────────────────────────────────────────────────────
    battery_warn_percent: float  = 30.0
    battery_return_percent: float = 20.0   # RTH trigger
    battery_land_percent: float  = 10.0    # immediate land

    # ── Link / heartbeat ────────────────────────────────────────────────
    heartbeat_timeout_s: float   = 3.0     # seconds with no app heartbeat → RTH
    critical_link_timeout_s: float = 8.0   # seconds → land now

    # ── Subject proximity ───────────────────────────────────────────────
    subject_min_distance: float  = 2.0     # metres (hard floor)
    subject_emergency_dist: float = 1.0    # metres → immediate brake

    # ── Emergency ───────────────────────────────────────────────────────
    emergency_descent_rate: float = 1.0    # m/s (controlled crash-land)
    motor_kill_altitude: float   = 0.3     # metres AGL → disarm


# ═══════════════════════════════════════════════════════════════════════════════
#  FLIGHT INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FlightInterfaceParams:
    """Rate and timeout knobs for the 50 Hz flight interface bridge."""

    setpoint_rate_hz: float      = 50.0
    mavlink_system_id: int       = 1
    mavlink_component_id: int    = 191    # COMP_ID_ONBOARD_COMPUTER
    position_control_mode: str   = "position"  # "position" or "velocity"
    timeout_no_setpoint_s: float = 0.5    # hover if no new setpoint


@dataclass
class SimulationParams:
    """Physics, battery, perception, and subject-motion model for the simulator."""

    random_seed: int            = 7
    home_lat: float             = 59.9
    home_lon: float             = 10.7
    home_alt_msl: float         = 100.0

    # Vehicle and autopilot response
    mass_kg: float              = 1.8
    gravity_mps2: float         = 9.81
    thrust_to_weight_ratio: float = 2.35
    nominal_battery_voltage: float = 16.8
    voltage_thrust_exponent: float = 1.2
    max_tilt_deg: float         = 30.0
    max_accel_xy: float         = 4.5
    max_accel_z: float          = 3.0
    max_speed_xy: float         = 8.0
    max_speed_z: float          = 2.5
    position_kp: float          = 0.9
    velocity_kp_xy: float       = 2.0
    velocity_kp_z: float        = 2.4
    motor_response_tau_s: float = 0.18
    attitude_response_tau_s: float = 0.12
    yaw_response_tau_s: float   = 0.25
    yaw_rate_limit: float       = 1.2
    drag_coeff_xy: float        = 0.12
    drag_coeff_z: float         = 0.22
    turbulence_accel_gain: float = 0.18
    ground_friction: float      = 3.0
    ground_effect_gain: float   = 0.14
    ground_effect_height_m: float = 1.6
    translational_lift_gain: float = 0.08
    translational_lift_speed_mps: float = 5.0
    vortex_ring_descent_rate_mps: float = 1.8
    vortex_ring_speed_threshold_mps: float = 2.2
    vortex_ring_max_penalty: float = 0.14

    # Wind and estimator fidelity
    wind_response_tau_s: float  = 2.5
    gust_response_tau_s: float  = 1.2
    turbulence_std: float       = 0.35
    wind_estimate_tau_s: float  = 0.8
    wind_estimate_noise: float  = 0.15

    # Telemetry and navigation estimate quality
    state_estimate_tau_s: float = 0.10
    position_noise_std: float   = 0.04
    velocity_noise_std: float   = 0.05
    attitude_noise_std: float   = 0.008
    yaw_rate_noise_std: float   = 0.015
    gps_noise_xy_m: float       = 0.9
    gps_noise_alt_m: float      = 1.4

    # Battery model
    battery_capacity_mah: float = 5000.0
    battery_cells: int          = 4
    battery_internal_resistance: float = 0.06
    hover_current_a: float      = 11.0
    max_current_a: float        = 28.0

    # Camera / perception model
    perception_latency_s: float = 0.08
    camera_hfov_deg: float      = 78.0
    camera_vfov_deg: float      = 52.0
    camera_down_tilt_deg: float = 12.0
    max_tracking_range_m: float = 45.0
    person_height_m: float      = 1.75
    person_width_m: float       = 0.55
    perception_pos_noise_m: float = 0.12
    perception_vel_noise_mps: float = 0.08
    perception_lock_confirm_s: float = 0.35

    # Subject motion model
    subject_accel_mps2: float   = 1.8
    subject_turn_rate_deg_s: float = 140.0
    subject_stride_hz: float    = 1.9


# ═══════════════════════════════════════════════════════════════════════════════
#  MASTER CONFIG (aggregates all sub-configs)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SystemConfig:
    """Master configuration aggregating all sub-configs for the autonomy stack."""

    shot: ShotParams = field(default_factory=ShotParams)
    stability: StabilityParams = field(default_factory=StabilityParams)
    mission: MissionParams = field(default_factory=MissionParams)
    safety: SafetyParams = field(default_factory=SafetyParams)
    flight_interface: FlightInterfaceParams = field(default_factory=FlightInterfaceParams)
    sim: SimulationParams = field(default_factory=SimulationParams)


# Default global config
DEFAULT_CONFIG = SystemConfig()
