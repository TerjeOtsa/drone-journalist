"""
Simulation Harness
==================
A deterministic, higher-fidelity simulation loop that wires together every
module with:

- a simplified but plausible quadrotor/autopilot response model
- dynamic wind with gusts and turbulence
- battery sag and current draw
- filtered, noisy telemetry
- camera-geometry-based perception with latency and confidence changes

Usage:
    python -m sim.sim_harness
"""

from __future__ import annotations

import logging
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

from config.parameters import SimulationParams, SystemConfig
from flight.flight_interface import FlightInterface
from flight.mission_state_machine import MissionStateMachine
from flight.safety_module import SafetyModule
from flight.shot_controller import ShotController
from flight.stability_supervisor import StabilitySupervisor
from interfaces.clock import SimClock
from interfaces.schemas import (AppCommand, DroneTelemetry, FlightSetpoint,
                                GeoPoint, LockState, SafetyStatus, ShotMode,
                                TargetTrack, Vec3)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(name)-28s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sim")


def clamp(value: float, low: float, high: float) -> float:
    """Clamp *value* to the ``[low, high]`` interval."""
    return max(low, min(high, value))


def wrap_angle(angle: float) -> float:
    """Wrap an angle to the ``(-π, π]`` range."""
    return math.atan2(math.sin(angle), math.cos(angle))


def alpha_from_tau(dt: float, tau: float) -> float:
    """Compute exponential low-pass filter coefficient from time constant *tau*."""
    if tau <= 1e-6:
        return 1.0
    return 1.0 - math.exp(-dt / tau)


def lpf_scalar(prev: float, new: float, alpha: float) -> float:
    """First-order low-pass filter step on a scalar value."""
    return prev + alpha * (new - prev)


def lpf_vec(prev: Vec3, new: Vec3, alpha: float) -> Vec3:
    """First-order low-pass filter step on a Vec3."""
    return Vec3(
        lpf_scalar(prev.x, new.x, alpha),
        lpf_scalar(prev.y, new.y, alpha),
        lpf_scalar(prev.z, new.z, alpha),
    )


def add_noise(rng: random.Random, vec: Vec3, std_xy: float, std_z: float | None = None) -> Vec3:
    """Add Gaussian noise to a Vec3 (separate XY and Z standard deviations)."""
    std_z = std_xy if std_z is None else std_z
    return Vec3(
        vec.x + rng.gauss(0.0, std_xy),
        vec.y + rng.gauss(0.0, std_xy),
        vec.z + rng.gauss(0.0, std_z),
    )


def geo_from_local(home_lat: float, home_lon: float, home_alt_msl: float, ned_pos: Vec3) -> GeoPoint:
    """Convert a local NED position to a geodetic point relative to the home."""
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = max(1.0, meters_per_deg_lat * math.cos(math.radians(home_lat)))
    return GeoPoint(
        lat=home_lat + ned_pos.x / meters_per_deg_lat,
        lon=home_lon + ned_pos.y / meters_per_deg_lon,
        alt_msl=home_alt_msl - ned_pos.z,
    )


def ned_to_body(vec: Vec3, attitude: Vec3) -> Vec3:
    """Rotate a NED-frame vector into the body frame given Euler angles."""
    roll = attitude.x
    pitch = attitude.y
    yaw = attitude.z

    cphi = math.cos(roll)
    sphi = math.sin(roll)
    ctheta = math.cos(pitch)
    stheta = math.sin(pitch)
    cpsi = math.cos(yaw)
    spsi = math.sin(yaw)

    return Vec3(
        ctheta * cpsi * vec.x + ctheta * spsi * vec.y - stheta * vec.z,
        (sphi * stheta * cpsi - cphi * spsi) * vec.x
        + (sphi * stheta * spsi + cphi * cpsi) * vec.y
        + sphi * ctheta * vec.z,
        (cphi * stheta * cpsi + sphi * spsi) * vec.x
        + (cphi * stheta * spsi - sphi * cpsi) * vec.y
        + cphi * ctheta * vec.z,
    )


def body_to_camera(vec: Vec3, camera_down_tilt_rad: float) -> Vec3:
    """Rotate a body-frame vector into the camera frame given a downward tilt."""
    ctilt = math.cos(camera_down_tilt_rad)
    stilt = math.sin(camera_down_tilt_rad)
    return Vec3(
        ctilt * vec.x + stilt * vec.z,
        vec.y,
        -stilt * vec.x + ctilt * vec.z,
    )


def body_z_axis_ned(attitude: Vec3) -> Vec3:
    """Return the body-frame Z axis expressed in NED for the given Euler angles."""
    roll = attitude.x
    pitch = attitude.y
    yaw = attitude.z

    cphi = math.cos(roll)
    sphi = math.sin(roll)
    ctheta = math.cos(pitch)
    stheta = math.sin(pitch)
    cpsi = math.cos(yaw)
    spsi = math.sin(yaw)

    return Vec3(
        cphi * stheta * cpsi + sphi * spsi,
        cphi * stheta * spsi - sphi * cpsi,
        cphi * ctheta,
    )


@dataclass
class DroneTruth:
    """Ground-truth drone state before sensor noise and filtering."""

    position: Vec3
    velocity: Vec3
    attitude: Vec3
    body_rates: Vec3
    wind: Vec3
    airspeed: Vec3


@dataclass
class PerceptionFrame:
    """Single timestamped snapshot used for perception-delay modelling."""

    timestamp: float
    subject_pos: Vec3
    subject_vel: Vec3
    drone: DroneTruth


@dataclass
class ScenarioEvent:
    """Timed action injected into a simulation run (e.g. wind gust, lock loss)."""

    time: float
    action: str
    params: dict = field(default_factory=dict)


class WindField:
    """Dynamic wind model with mean, gust, turbulence, and a noisy estimator."""

    def __init__(self, params: SimulationParams, seed: int) -> None:
        self.p = params
        self.rng = random.Random(seed)
        self.target = Vec3()
        self.mean = Vec3()
        self.gust = Vec3()
        self.estimate = Vec3()

    def set_target(self, target: Vec3) -> None:
        """Set the wind target that the mean will track toward."""
        self.target = target

    def step(self, dt: float) -> Vec3:
        """Advance the wind model by *dt* seconds and return the true wind."""
        mean_alpha = alpha_from_tau(dt, self.p.wind_response_tau_s)
        self.mean = lpf_vec(self.mean, self.target, mean_alpha)

        gust_decay = math.exp(-dt / max(self.p.gust_response_tau_s, 1e-6))
        gust_noise = self.p.turbulence_std * (1.0 + 0.12 * self.mean.norm()) * math.sqrt(max(dt, 1e-6))
        self.gust = Vec3(
            self.gust.x * gust_decay + self.rng.gauss(0.0, gust_noise),
            self.gust.y * gust_decay + self.rng.gauss(0.0, gust_noise),
            self.gust.z * gust_decay + self.rng.gauss(0.0, gust_noise * 0.35),
        )

        true_wind = self.mean + self.gust

        estimate_alpha = alpha_from_tau(dt, self.p.wind_estimate_tau_s)
        noisy_wind = add_noise(
            self.rng,
            true_wind,
            self.p.wind_estimate_noise,
            self.p.wind_estimate_noise * 0.5,
        )
        self.estimate = lpf_vec(self.estimate, noisy_wind, estimate_alpha)
        return true_wind


class BatteryModel:
    """State-of-charge, voltage sag, and current-draw model for a LiPo pack."""

    def __init__(self, params: SimulationParams) -> None:
        self.p = params
        self.remaining_mah = params.battery_capacity_mah
        self.last_current_a = params.hover_current_a * 0.35

    @property
    def soc(self) -> float:
        return clamp(self.remaining_mah / max(self.p.battery_capacity_mah, 1e-6), 0.0, 1.0)

    @property
    def percent(self) -> float:
        return 100.0 * self.soc

    @property
    def voltage(self) -> float:
        soc = self.soc
        cell_ocv = 3.30 + 0.55 * soc + 0.35 * math.sqrt(max(soc, 0.0))
        sag = self.last_current_a * self.p.battery_internal_resistance
        pack_voltage = self.p.battery_cells * cell_ocv - sag
        return max(self.p.battery_cells * 3.15, pack_voltage)

    def update(self, dt: float, throttle_fraction: float, airspeed_mps: float, climb_rate_mps: float) -> None:
        """Drain battery by estimated current over *dt* seconds."""
        maneuver_load = 0.9 * airspeed_mps + 1.8 * max(0.0, climb_rate_mps)
        throttle_load = self.p.hover_current_a + throttle_fraction * (self.p.max_current_a - self.p.hover_current_a)
        self.last_current_a = clamp(0.6 * throttle_load + maneuver_load, 0.0, self.p.max_current_a)
        self.remaining_mah = max(
            0.0,
            self.remaining_mah - self.last_current_a * dt * 1000.0 / 3600.0,
        )


class RealisticPhysics:
    """Simplified but plausible quadrotor dynamics with thrust, drag, and ground effects."""

    def __init__(self, params: SimulationParams, seed: int) -> None:
        self.p = params
        self.rng = random.Random(seed)

        self.pos = Vec3(0.0, 0.0, 0.0)
        self.vel = Vec3(0.0, 0.0, 0.0)
        self.attitude = Vec3(0.0, 0.0, 0.0)
        self.body_rates = Vec3(0.0, 0.0, 0.0)
        self.command_accel = Vec3(0.0, 0.0, 0.0)
        self.last_total_accel = Vec3()
        self.last_throttle_fraction = 0.0
        self.last_thrust_n = 0.0
        self.last_available_thrust_n = self.p.mass_kg * self.p.gravity_mps2 * self.p.thrust_to_weight_ratio
        self.last_ground_effect_gain = 1.0
        self.last_translational_lift_gain = 1.0
        self.last_vortex_ring_penalty = 0.0

        self.armed = False
        self.in_air = False

        self._telem_pos = Vec3()
        self._telem_vel = Vec3()
        self._telem_att = Vec3()
        self._last_dt = 0.02

    def truth(self, true_wind: Vec3) -> DroneTruth:
        """Return the ground-truth state snapshot for the current timestep."""
        return DroneTruth(
            position=Vec3(self.pos.x, self.pos.y, self.pos.z),
            velocity=Vec3(self.vel.x, self.vel.y, self.vel.z),
            attitude=Vec3(self.attitude.x, self.attitude.y, self.attitude.z),
            body_rates=Vec3(self.body_rates.x, self.body_rates.y, self.body_rates.z),
            wind=Vec3(true_wind.x, true_wind.y, true_wind.z),
            airspeed=Vec3(
                self.vel.x - true_wind.x,
                self.vel.y - true_wind.y,
                self.vel.z - true_wind.z,
            ),
        )

    def step(
        self,
        sp: Optional[FlightSetpoint],
        dt: float,
        true_wind: Vec3,
        gust: Vec3,
        battery: BatteryModel,
    ) -> None:
        """Advance the physics model by *dt* seconds given a setpoint and environment."""
        self._last_dt = dt

        if sp is None or not self.armed:
            self.last_throttle_fraction = 0.0
            self.last_thrust_n = 0.0
            self.last_total_accel = Vec3()
            self.last_ground_effect_gain = 1.0
            self.last_translational_lift_gain = 1.0
            self.last_vortex_ring_penalty = 0.0
            self._settle_to_ground(dt)
            return

        desired_velocity = self._desired_velocity(sp)
        velocity_error = desired_velocity - self.vel
        desired_accel = Vec3(
            self.p.velocity_kp_xy * velocity_error.x,
            self.p.velocity_kp_xy * velocity_error.y,
            self.p.velocity_kp_z * velocity_error.z,
        )

        horiz_accel = math.hypot(desired_accel.x, desired_accel.y)
        if horiz_accel > self.p.max_accel_xy and horiz_accel > 1e-6:
            scale = self.p.max_accel_xy / horiz_accel
            desired_accel = Vec3(
                desired_accel.x * scale,
                desired_accel.y * scale,
                desired_accel.z,
            )
        desired_accel = Vec3(
            desired_accel.x,
            desired_accel.y,
            clamp(desired_accel.z, -self.p.max_accel_z, self.p.max_accel_z),
        )

        accel_alpha = alpha_from_tau(dt, self.p.motor_response_tau_s)
        self.command_accel = lpf_vec(self.command_accel, desired_accel, accel_alpha)
        self._update_attitude(sp, dt)

        airspeed = Vec3(
            self.vel.x - true_wind.x,
            self.vel.y - true_wind.y,
            self.vel.z - true_wind.z,
        )
        drag_accel = Vec3(
            -self.p.drag_coeff_xy * airspeed.x * abs(airspeed.x),
            -self.p.drag_coeff_xy * airspeed.y * abs(airspeed.y),
            -self.p.drag_coeff_z * airspeed.z * abs(airspeed.z),
        )
        gust_accel = Vec3(
            gust.x * self.p.turbulence_accel_gain,
            gust.y * self.p.turbulence_accel_gain,
            gust.z * self.p.turbulence_accel_gain * 0.6,
        )

        ground_effect_gain = self._ground_effect_gain()
        translational_lift_gain = self._translational_lift_gain(airspeed)
        vortex_ring_penalty = self._vortex_ring_penalty(airspeed)

        available_thrust_n = self._available_thrust_n(battery) * ground_effect_gain * translational_lift_gain
        available_thrust_n *= max(0.25, 1.0 - vortex_ring_penalty)
        self.last_available_thrust_n = available_thrust_n

        thrust_axis = self._thrust_axis_ned()
        vertical_lift_component = max(0.35, -thrust_axis.z)
        desired_thrust_accel = max(
            0.0,
            (self.p.gravity_mps2 - self.command_accel.z) / vertical_lift_component,
        )
        thrust_cmd_n = clamp(
            self.p.mass_kg * desired_thrust_accel,
            0.0,
            available_thrust_n,
        )
        thrust_alpha = alpha_from_tau(dt, self.p.motor_response_tau_s)
        self.last_thrust_n = lpf_scalar(self.last_thrust_n, thrust_cmd_n, thrust_alpha)

        thrust_accel = thrust_axis * (self.last_thrust_n / max(self.p.mass_kg, 1e-6))
        gravity_accel = Vec3(0.0, 0.0, self.p.gravity_mps2)
        total_accel = gravity_accel + thrust_accel + drag_accel + gust_accel
        self.last_total_accel = total_accel
        self.last_ground_effect_gain = ground_effect_gain
        self.last_translational_lift_gain = translational_lift_gain
        self.last_vortex_ring_penalty = vortex_ring_penalty

        self.vel = Vec3(
            self.vel.x + total_accel.x * dt,
            self.vel.y + total_accel.y * dt,
            self.vel.z + total_accel.z * dt,
        )

        horiz_speed = math.hypot(self.vel.x, self.vel.y)
        if horiz_speed > self.p.max_speed_xy and horiz_speed > 1e-6:
            scale = self.p.max_speed_xy / horiz_speed
            self.vel = Vec3(self.vel.x * scale, self.vel.y * scale, self.vel.z)
        self.vel = Vec3(
            self.vel.x,
            self.vel.y,
            clamp(self.vel.z, -self.p.max_speed_z, self.p.max_speed_z),
        )

        self.pos = Vec3(
            self.pos.x + self.vel.x * dt,
            self.pos.y + self.vel.y * dt,
            self.pos.z + self.vel.z * dt,
        )

        if self.pos.z > 0.0:
            self.pos = Vec3(self.pos.x, self.pos.y, 0.0)
            if self.vel.z > 0.0:
                self.vel = Vec3(self.vel.x, self.vel.y, 0.0)
            ground_alpha = clamp(self.p.ground_friction * dt, 0.0, 1.0)
            self.vel = Vec3(
                lpf_scalar(self.vel.x, 0.0, ground_alpha),
                lpf_scalar(self.vel.y, 0.0, ground_alpha),
                self.vel.z,
            )

        self.in_air = self.pos.z < -0.05 or abs(self.vel.z) > 0.1

        self.last_throttle_fraction = clamp(
            self.last_thrust_n / max(available_thrust_n, 1e-6),
            0.0,
            1.0,
        )

    def build_telemetry(
        self,
        battery: BatteryModel,
        wind_estimate: Vec3,
        sim_time: float,
    ) -> DroneTelemetry:
        """Build a noisy, filtered telemetry message mimicking a real autopilot."""
        est_alpha = alpha_from_tau(self._last_dt, self.p.state_estimate_tau_s)

        noisy_pos = add_noise(self.rng, self.pos, self.p.position_noise_std, self.p.position_noise_std * 0.7)
        noisy_vel = add_noise(self.rng, self.vel, self.p.velocity_noise_std, self.p.velocity_noise_std * 0.8)
        noisy_att = add_noise(self.rng, self.attitude, self.p.attitude_noise_std)

        self._telem_pos = lpf_vec(self._telem_pos, noisy_pos, est_alpha)
        self._telem_vel = lpf_vec(self._telem_vel, noisy_vel, est_alpha)
        self._telem_att = Vec3(
            lpf_scalar(self._telem_att.x, noisy_att.x, est_alpha),
            lpf_scalar(self._telem_att.y, noisy_att.y, est_alpha),
            wrap_angle(lpf_scalar(self._telem_att.z, noisy_att.z, est_alpha)),
        )

        gps_local = add_noise(self.rng, self.pos, self.p.gps_noise_xy_m, self.p.gps_noise_alt_m)
        gps = geo_from_local(self.p.home_lat, self.p.home_lon, self.p.home_alt_msl, gps_local)

        satellites = int(round(clamp(14.0 - 0.12 * wind_estimate.norm() + self.rng.gauss(0.0, 0.4), 10.0, 16.0)))
        gps_fix = 3 if satellites >= 10 else 2

        return DroneTelemetry(
            timestamp=sim_time,
            position=Vec3(self._telem_pos.x, self._telem_pos.y, self._telem_pos.z),
            velocity=Vec3(self._telem_vel.x, self._telem_vel.y, self._telem_vel.z),
            attitude_euler=Vec3(self._telem_att.x, self._telem_att.y, self._telem_att.z),
            gps=gps,
            battery_percent=battery.percent,
            battery_voltage=battery.voltage,
            armed=self.armed,
            in_air=self.in_air,
            gps_fix=gps_fix,
            satellites=satellites,
            wind_estimate=Vec3(wind_estimate.x, wind_estimate.y, wind_estimate.z),
            home_position=Vec3(0.0, 0.0, 0.0),
        )

    def _available_thrust_n(self, battery: BatteryModel) -> float:
        voltage_ratio = battery.voltage / max(self.p.nominal_battery_voltage, 1e-6)
        voltage_factor = clamp(voltage_ratio**self.p.voltage_thrust_exponent, 0.55, 1.05)
        return self.p.mass_kg * self.p.gravity_mps2 * self.p.thrust_to_weight_ratio * voltage_factor

    def _ground_effect_gain(self) -> float:
        altitude_agl = max(0.0, -self.pos.z)
        if altitude_agl <= 1e-6:
            return 1.0 + self.p.ground_effect_gain
        return 1.0 + self.p.ground_effect_gain * math.exp(
            -altitude_agl / max(self.p.ground_effect_height_m, 1e-6)
        )

    def _translational_lift_gain(self, airspeed: Vec3) -> float:
        horiz_airspeed = math.hypot(airspeed.x, airspeed.y)
        speed_ratio = clamp(
            horiz_airspeed / max(self.p.translational_lift_speed_mps, 1e-6),
            0.0,
            1.5,
        )
        return 1.0 + self.p.translational_lift_gain * (1.0 - math.exp(-(speed_ratio**2)))

    def _vortex_ring_penalty(self, airspeed: Vec3) -> float:
        descent_rate = max(0.0, self.vel.z)
        horiz_airspeed = math.hypot(airspeed.x, airspeed.y)
        descent_fraction = clamp(
            (descent_rate - self.p.vortex_ring_descent_rate_mps)
            / max(self.p.vortex_ring_descent_rate_mps, 1e-6),
            0.0,
            1.0,
        )
        low_speed_fraction = clamp(
            1.0 - horiz_airspeed / max(self.p.vortex_ring_speed_threshold_mps, 1e-6),
            0.0,
            1.0,
        )
        thrust_fraction = clamp(
            self.last_thrust_n / max(self.last_available_thrust_n, 1e-6),
            0.0,
            1.0,
        )
        return self.p.vortex_ring_max_penalty * descent_fraction * low_speed_fraction * max(0.35, thrust_fraction)

    def _thrust_axis_ned(self) -> Vec3:
        body_z = body_z_axis_ned(self.attitude)
        return Vec3(-body_z.x, -body_z.y, -body_z.z)

    def _desired_velocity(self, sp: FlightSetpoint) -> Vec3:
        desired = sp.velocity or Vec3()
        if sp.position is not None:
            position_error = sp.position - self.pos
            desired = desired + Vec3(
                self.p.position_kp * position_error.x,
                self.p.position_kp * position_error.y,
                self.p.position_kp * 0.8 * position_error.z,
            )

        horiz_speed = math.hypot(desired.x, desired.y)
        if horiz_speed > self.p.max_speed_xy and horiz_speed > 1e-6:
            scale = self.p.max_speed_xy / horiz_speed
            desired = Vec3(desired.x * scale, desired.y * scale, desired.z)

        return Vec3(desired.x, desired.y, clamp(desired.z, -self.p.max_speed_z, self.p.max_speed_z))

    def _settle_to_ground(self, dt: float) -> None:
        if self.pos.z < 0.0:
            self.vel = Vec3(self.vel.x, self.vel.y, min(self.vel.z + self.p.max_accel_z * dt, 0.5))
            self.pos = Vec3(self.pos.x, self.pos.y, min(self.pos.z + self.vel.z * dt, 0.0))
        if self.pos.z >= 0.0:
            self.pos = Vec3(self.pos.x, self.pos.y, 0.0)
            ground_alpha = clamp(self.p.ground_friction * dt, 0.0, 1.0)
            self.vel = Vec3(
                lpf_scalar(self.vel.x, 0.0, ground_alpha),
                lpf_scalar(self.vel.y, 0.0, ground_alpha),
                0.0,
            )
            self.in_air = False

    def _update_attitude(self, sp: FlightSetpoint, dt: float) -> None:
        yaw = self.attitude.z
        forward_accel = math.cos(yaw) * self.command_accel.x + math.sin(yaw) * self.command_accel.y
        right_accel = -math.sin(yaw) * self.command_accel.x + math.cos(yaw) * self.command_accel.y
        max_tilt = math.radians(self.p.max_tilt_deg)

        roll_cmd = clamp(math.atan2(right_accel, self.p.gravity_mps2), -max_tilt, max_tilt)
        pitch_cmd = clamp(-math.atan2(forward_accel, self.p.gravity_mps2), -max_tilt, max_tilt)

        att_alpha = alpha_from_tau(dt, self.p.attitude_response_tau_s)
        new_roll = lpf_scalar(self.attitude.x, roll_cmd, att_alpha)
        new_pitch = lpf_scalar(self.attitude.y, pitch_cmd, att_alpha)

        desired_yaw = sp.yaw
        yaw_error = wrap_angle(desired_yaw - self.attitude.z)
        yaw_rate_cmd = clamp(yaw_error / max(self.p.yaw_response_tau_s, 1e-6), -self.p.yaw_rate_limit, self.p.yaw_rate_limit)
        if sp.yaw_rate is not None:
            yaw_rate_cmd = clamp(sp.yaw_rate, -self.p.yaw_rate_limit, self.p.yaw_rate_limit)
        new_yaw = wrap_angle(self.attitude.z + yaw_rate_cmd * dt)

        self.body_rates = Vec3(
            (new_roll - self.attitude.x) / max(dt, 1e-6),
            (new_pitch - self.attitude.y) / max(dt, 1e-6),
            wrap_angle(new_yaw - self.attitude.z) / max(dt, 1e-6),
        )
        self.attitude = Vec3(new_roll, new_pitch, new_yaw)


class FakePerception:
    """Camera-geometry-based perception model with latency and lock state."""

    def __init__(self, params: SimulationParams, seed: int) -> None:
        self.p = params
        self.rng = random.Random(seed)
        self.subject_pos = Vec3(5.0, 0.0, 0.0)
        self.subject_vel = Vec3()
        self.subject_heading = 0.0
        self.subject_stride_phase = 0.0
        self.walking = False
        self.walk_speed = 1.2
        self.walk_dir = Vec3(1.0, 0.0, 0.0)

        self.lock = LockState.LOST
        self._was_locked = False
        self._lost_until = 0.0
        self._assist_lock_until = 0.0
        self._candidate_since: float | None = None
        self._history: Deque[PerceptionFrame] = deque()

    def advance(self, dt: float) -> None:
        """Update subject motion (position, velocity, heading, stride phase)."""
        desired_vel = Vec3()
        if self.walking:
            desired_vel = Vec3(
                self.walk_dir.x * self.walk_speed,
                self.walk_dir.y * self.walk_speed,
                0.0,
            )

        vel_error = desired_vel - self.subject_vel
        max_dv = max(self.p.subject_accel_mps2, 1e-6) * max(dt, 1e-6)
        error_mag = vel_error.norm()
        if error_mag > max_dv and error_mag > 1e-6:
            vel_error = vel_error * (max_dv / error_mag)

        self.subject_vel = Vec3(
            self.subject_vel.x + vel_error.x,
            self.subject_vel.y + vel_error.y,
            0.0,
        )

        speed = math.hypot(self.subject_vel.x, self.subject_vel.y)
        if not self.walking and speed < 0.03:
            self.subject_vel = Vec3()
            speed = 0.0

        if speed > 0.05:
            desired_heading = math.atan2(self.subject_vel.y, self.subject_vel.x)
            heading_error = wrap_angle(desired_heading - self.subject_heading)
            max_turn = math.radians(self.p.subject_turn_rate_deg_s) * dt
            self.subject_heading = wrap_angle(
                self.subject_heading + clamp(heading_error, -max_turn, max_turn)
            )

        stride_rate = self.p.subject_stride_hz * clamp(speed / max(self.walk_speed, 0.6), 0.0, 1.8)
        self.subject_stride_phase = (self.subject_stride_phase + 2.0 * math.pi * stride_rate * dt) % (2.0 * math.pi)
        self.subject_pos = Vec3(
            self.subject_pos.x + self.subject_vel.x * dt,
            self.subject_pos.y + self.subject_vel.y * dt,
            0.0,
        )

    def observe(self, sim_t: float, drone_truth: DroneTruth) -> TargetTrack:
        """Produce a delayed, noisy TargetTrack from the current perception state."""
        self._history.append(
            PerceptionFrame(
                timestamp=sim_t,
                subject_pos=Vec3(self.subject_pos.x, self.subject_pos.y, self.subject_pos.z),
                subject_vel=Vec3(self.subject_vel.x, self.subject_vel.y, self.subject_vel.z),
                drone=drone_truth,
            )
        )

        oldest_allowed = sim_t - max(0.5, self.p.perception_latency_s * 4.0)
        while self._history and self._history[0].timestamp < oldest_allowed:
            self._history.popleft()

        delayed_t = sim_t - self.p.perception_latency_s
        frame = self._history[0]
        for sample in self._history:
            if sample.timestamp <= delayed_t:
                frame = sample
            else:
                break

        return self._build_track(sim_t, frame)

    def force_lock(self, sim_t: float) -> None:
        """Immediately lock onto the subject (operator assist)."""
        self.lock = LockState.LOCKED
        self._was_locked = True
        self._assist_lock_until = sim_t + 3.0
        self._candidate_since = sim_t

    def force_lost(self, duration: float, sim_t: float) -> None:
        """Force a lock-loss lasting *duration* seconds."""
        self.lock = LockState.LOST
        self._lost_until = sim_t + duration
        self._candidate_since = None

    def _build_track(self, sim_t: float, frame: PerceptionFrame) -> TargetTrack:
        rel_ned = frame.subject_pos - frame.drone.position
        rel_body = ned_to_body(rel_ned, frame.drone.attitude)
        rel_cam = body_to_camera(rel_body, math.radians(self.p.camera_down_tilt_deg))

        hfov = math.radians(self.p.camera_hfov_deg)
        vfov = math.radians(self.p.camera_vfov_deg)
        max_h = hfov * 0.5
        max_v = vfov * 0.5

        range_m = rel_ned.norm()
        az = math.atan2(rel_cam.y, max(rel_cam.x, 1e-6))
        el = math.atan2(rel_cam.z, max(rel_cam.x, 1e-6))

        visible = (
            rel_cam.x > 0.3
            and range_m <= self.p.max_tracking_range_m
            and abs(az) <= max_h
            and abs(el) <= max_v
        )

        edge_ratio = 0.0
        if visible:
            edge_ratio = max(abs(az) / max(max_h, 1e-6), abs(el) / max(max_v, 1e-6))
        edge_score = clamp(1.0 - edge_ratio, 0.0, 1.0)
        range_score = clamp(1.0 - max(0.0, range_m - 7.0) / max(self.p.max_tracking_range_m - 7.0, 1e-6), 0.0, 1.0)

        motion_blur = (
            abs(frame.drone.body_rates.z) * 0.35
            + abs(frame.drone.body_rates.x) * 0.08
            + abs(frame.drone.body_rates.y) * 0.08
            + frame.drone.airspeed.norm() * 0.03
        )
        bank_penalty = clamp(
            (abs(frame.drone.attitude.x) + abs(frame.drone.attitude.y)) / math.radians(35.0),
            0.0,
            0.35,
        )

        tracking_conf = 0.0
        if visible:
            tracking_conf = clamp(
                0.18
                + 0.52 * edge_score
                + 0.30 * range_score
                - clamp(motion_blur, 0.0, 0.55)
                - bank_penalty
                + self.rng.gauss(0.0, 0.025),
                0.0,
                1.0,
            )

        face_score = 0.0
        if visible:
            to_drone = Vec3(
                frame.drone.position.x - frame.subject_pos.x,
                frame.drone.position.y - frame.subject_pos.y,
                0.0,
            )
            to_drone_norm = math.hypot(to_drone.x, to_drone.y)
            if to_drone_norm > 1e-6:
                subject_forward = Vec3(
                    math.cos(self.subject_heading),
                    math.sin(self.subject_heading),
                    0.0,
                )
                facing_alignment = clamp(
                    (subject_forward.x * to_drone.x + subject_forward.y * to_drone.y) / to_drone_norm,
                    -1.0,
                    1.0,
                )
                face_score = clamp(
                    (0.5 + 0.5 * facing_alignment)
                    * (0.35 + 0.65 * edge_score)
                    * (0.55 + 0.45 * tracking_conf),
                    0.0,
                    1.0,
                )

        h_norm = 0.0
        w_norm = 0.0
        u = 0.5
        v = 0.5
        bbox = None

        if visible:
            u = clamp(0.5 + az / hfov, 0.0, 1.0)
            v = clamp(0.5 + el / vfov, 0.0, 1.0)
            h_norm = clamp(
                self.p.person_height_m / max(2.0 * range_m * math.tan(vfov * 0.5), 1e-6),
                0.02,
                0.95,
            )
            w_norm = clamp(
                self.p.person_width_m / max(2.0 * range_m * math.tan(hfov * 0.5), 1e-6),
                0.02,
                0.90,
            )
            bbox = (
                clamp(u - 0.5 * w_norm, 0.0, 1.0),
                clamp(v - 0.5 * h_norm, 0.0, 1.0),
                w_norm,
                h_norm,
            )

        identity_conf = clamp(
            tracking_conf
            - max(0.0, 0.11 - h_norm) * 1.6
            + self.rng.gauss(0.0, 0.02),
            0.0,
            1.0,
        )

        if sim_t < self._lost_until:
            self.lock = LockState.LOST
            tracking_conf = 0.0
            identity_conf = 0.0
            bbox = None
            face_score = 0.0
            self._candidate_since = None
        else:
            if sim_t < self._assist_lock_until and visible:
                tracking_conf = max(tracking_conf, 0.92)
                identity_conf = max(identity_conf, 0.88)
                face_score = max(face_score, 0.75)
                self._candidate_since = sim_t

            if not visible or tracking_conf < 0.12:
                self.lock = LockState.LOST
                self._candidate_since = None
            elif self.lock == LockState.LOCKED or self._was_locked:
                if tracking_conf >= 0.55:
                    self.lock = LockState.LOCKED
                elif tracking_conf >= 0.28:
                    self.lock = LockState.WEAK
                else:
                    self.lock = LockState.LOST
                    self._candidate_since = None
            else:
                if tracking_conf >= 0.72 and identity_conf >= 0.52:
                    if self._candidate_since is None:
                        self._candidate_since = sim_t
                    if sim_t - self._candidate_since >= self.p.perception_lock_confirm_s:
                        self.lock = LockState.LOCKED
                    else:
                        self.lock = LockState.CANDIDATE
                elif tracking_conf >= 0.28:
                    if self._candidate_since is None:
                        self._candidate_since = sim_t
                    self.lock = LockState.CANDIDATE
                else:
                    self.lock = LockState.LOST
                    self._candidate_since = None

        if self.lock == LockState.LOCKED:
            self._was_locked = True

        target_world = None
        target_velocity = None
        if self.lock != LockState.LOST and tracking_conf >= 0.22:
            pos_noise = self.p.perception_pos_noise_m * (1.0 + 0.8 * (1.0 - edge_score) + max(0.0, range_m - 10.0) / 25.0)
            vel_noise = self.p.perception_vel_noise_mps * (1.0 + 0.5 * (1.0 - edge_score))
            target_world = add_noise(self.rng, frame.subject_pos, pos_noise, pos_noise * 0.4)
            target_velocity = add_noise(self.rng, frame.subject_vel, vel_noise, vel_noise * 0.5)

        if self.lock == LockState.LOST:
            target_world = None
            target_velocity = None

        return TargetTrack(
            timestamp=sim_t,
            target_position_image=(u, v) if visible else (0.0, 0.0),
            target_position_world=target_world,
            target_velocity_world=target_velocity,
            identity_confidence=identity_conf,
            tracking_confidence=tracking_conf,
            lock_state=self.lock,
            bounding_box=bbox,
            face_score=face_score,
        )


DEFAULT_SCENARIO: List[ScenarioEvent] = [
    ScenarioEvent(0.0, "start"),
    ScenarioEvent(3.0, "lock"),
    ScenarioEvent(8.0, "walk", {"speed": 1.2, "dir": [1, 0.3, 0]}),
    ScenarioEvent(14.0, "shot", {"mode": "walk_and_talk"}),
    ScenarioEvent(20.0, "wind", {"speed": [3.0, 2.0, 0.0]}),
    ScenarioEvent(28.0, "lost", {"duration": 4.0}),
    ScenarioEvent(36.0, "shot", {"mode": "orbit"}),
    ScenarioEvent(45.0, "wind", {"speed": [9.0, 4.0, 0.0]}),
    ScenarioEvent(52.0, "wind", {"speed": [0.0, 0.0, 0.0]}),
    ScenarioEvent(58.0, "stop"),
]


class SimulationSession:
    """Live simulation session that can be stepped interactively."""

    def __init__(
        self,
        *,
        scenario: List[ScenarioEvent] | None = None,
        dt: float = 0.02,
        config: SystemConfig | None = None,
        seed: int | None = None,
        log_interval_s: float | None = 5.0,
    ) -> None:
        self.cfg = config or SystemConfig()
        self.sim_cfg = self.cfg.sim
        self.dt = dt
        self.scenario = sorted(
            [ScenarioEvent(ev.time, ev.action, dict(ev.params)) for ev in (scenario or DEFAULT_SCENARIO)],
            key=lambda ev: ev.time,
        )
        self.base_seed = self.sim_cfg.random_seed if seed is None else seed
        self.log_interval_s = log_interval_s

        self.physics = RealisticPhysics(self.sim_cfg, self.base_seed + 1)
        self.wind_field = WindField(self.sim_cfg, self.base_seed + 2)
        self.battery = BatteryModel(self.sim_cfg)
        self.percep = FakePerception(self.sim_cfg, self.base_seed + 3)
        self.sim_clock = SimClock(start_time=0.0)

        self.fi = FlightInterface(self.cfg.flight_interface, clock=self.sim_clock)
        self.shot = ShotController(
            self.cfg.shot,
            mission_params=self.cfg.mission,
            safety_params=self.cfg.safety,
            clock=self.sim_clock,
        )
        self.stab = StabilitySupervisor(self.cfg.stability, clock=self.sim_clock)
        self.mission = MissionStateMachine(self.cfg.mission, clock=self.sim_clock)
        self.safety = SafetyModule(self.cfg.safety, clock=self.sim_clock)

        self.track = TargetTrack(timestamp=0.0)
        self.safety_status = SafetyStatus()
        self.stability = self.stab.level
        self.current_sp = FlightSetpoint()
        self.ready_to_record = False
        self.latest_truth = self.physics.truth(Vec3())
        self.latest_telem = DroneTelemetry()
        self.latest_record: Optional[dict] = None
        self.last_applied_setpoint: Optional[FlightSetpoint] = None

        self.control_period = 1.0 / 25.0
        self.heartbeat_period = 0.5
        self.next_control_t = 0.0
        self.next_heartbeat_t = 0.0
        self.next_log_t = 0.0 if log_interval_s is not None else float("inf")
        self.last_control_t: Optional[float] = None

        self.sim_t = 0.0
        self.tick = 0
        self.scenario_idx = 0
        self.pending_commands: List[AppCommand] = []
        self.records: List[dict] = []

    def step(self, steps: int = 1) -> dict:
        """Advance the simulation by *steps* ticks and return the latest record."""
        for _ in range(max(1, steps)):
            self._step_once()
        assert self.latest_record is not None
        return self.latest_record

    def start(self) -> None:
        """Inject a mission-start command."""
        self.apply_event("start")

    def stop(self) -> None:
        """Inject a mission-stop command."""
        self.apply_event("stop")

    def force_lock(self) -> None:
        """Force an immediate target lock (operator assist)."""
        self.apply_event("lock")

    def force_lost(self, duration: float = 3.0) -> None:
        """Force a lock-loss lasting *duration* seconds."""
        self.apply_event("lost", {"duration": duration})

    def set_shot_mode(self, mode: ShotMode | str) -> None:
        """Switch the active shot mode."""
        mode_value = mode.value if isinstance(mode, ShotMode) else str(mode)
        self.apply_event("shot", {"mode": mode_value})

    def set_distance(self, meters: float) -> None:
        """Set the desired follow distance in metres."""
        self.apply_event("distance", {"meters": float(meters)})

    def set_wind_target(self, x: float, y: float, z: float = 0.0) -> None:
        """Set the target mean wind vector (NED, m/s)."""
        self.wind_field.set_target(Vec3(float(x), float(y), float(z)))

    def set_subject_motion(self, *, walking: bool, speed: float, heading_deg: float) -> None:
        """Configure subject walking state, speed, and heading."""
        heading_rad = math.radians(heading_deg)
        self.percep.walking = walking
        self.percep.walk_speed = max(0.0, speed)
        self.percep.walk_dir = Vec3(math.cos(heading_rad), math.sin(heading_rad), 0.0).normalized()

    def apply_event(self, action: str, params: dict | None = None) -> None:
        """Apply a named scenario event immediately."""
        self._handle_event(ScenarioEvent(self.sim_t, action, params or {}))

    def _queue_command(self, command: AppCommand) -> None:
        if command.action in {"set_distance", "set_shot_mode"}:
            self.pending_commands = [
                item for item in self.pending_commands if item.action != command.action
            ]
        self.pending_commands.append(command)

    def _handle_event(self, ev: ScenarioEvent) -> None:
        log.info("SCENARIO t=%.1f  %s  %s", self.sim_t, ev.action, ev.params)

        if ev.action == "start":
            self.physics.armed = True
            self.fi.arm()
            self.fi.set_offboard_mode()
            self._queue_command(AppCommand(timestamp=self.sim_t, action="start"))
            self.safety.heartbeat(now_s=self.sim_t)
        elif ev.action == "lock":
            self.percep.force_lock(self.sim_t)
        elif ev.action == "walk":
            speed = float(ev.params.get("speed", 1.2))
            direction = Vec3(*ev.params.get("dir", [1.0, 0.0, 0.0])).normalized()
            heading_deg = math.degrees(math.atan2(direction.y, direction.x))
            self.set_subject_motion(walking=True, speed=speed, heading_deg=heading_deg)
        elif ev.action == "walk_stop":
            self.set_subject_motion(
                walking=False,
                speed=self.percep.walk_speed,
                heading_deg=math.degrees(math.atan2(self.percep.walk_dir.y, self.percep.walk_dir.x)),
            )
        elif ev.action == "wind":
            self.wind_field.set_target(Vec3(*ev.params.get("speed", [0.0, 0.0, 0.0])))
        elif ev.action == "lost":
            self.percep.force_lost(float(ev.params.get("duration", 3.0)), self.sim_t)
        elif ev.action == "shot":
            self._queue_command(AppCommand(
                timestamp=self.sim_t,
                action="set_shot_mode",
                shot_mode=ShotMode(ev.params.get("mode", "standup")),
            ))
        elif ev.action in ("distance", "set_distance"):
            self._queue_command(AppCommand(
                timestamp=self.sim_t,
                action="set_distance",
                desired_distance=float(ev.params.get("meters", ev.params.get("distance", 5.0))),
            ))
        elif ev.action == "stop":
            self._queue_command(AppCommand(timestamp=self.sim_t, action="stop"))

    def _build_record(self, sample_time: float, sample_telem: DroneTelemetry, true_wind: Vec3) -> dict:
        truth = self.latest_truth
        return {
            "t": round(sample_time, 3),
            "state": self.mission.state.value,
            "shot": self.mission.shot_mode.value,
            "desired_distance": (
                round(self.mission.desired_distance, 2)
                if self.mission.desired_distance is not None
                else None
            ),
            "ready": self.ready_to_record,
            "stability": self.stability.value,
            "safety": self.safety_status.active_override.value,
            "lock": self.track.lock_state.value,
            "drone_x": round(sample_telem.position.x, 2),
            "drone_y": round(sample_telem.position.y, 2),
            "drone_z": round(sample_telem.position.z, 2),
            "roll_deg": round(math.degrees(sample_telem.attitude_euler.x), 1),
            "pitch_deg": round(math.degrees(sample_telem.attitude_euler.y), 1),
            "yaw_deg": round(math.degrees(sample_telem.attitude_euler.z), 1),
            "airspeed": round(truth.airspeed.norm(), 2),
            "subj_x": round(self.percep.subject_pos.x, 2),
            "subj_y": round(self.percep.subject_pos.y, 2),
            "subject_heading_deg": round(math.degrees(self.percep.subject_heading), 1),
            "subject_speed": round(math.hypot(self.percep.subject_vel.x, self.percep.subject_vel.y), 2),
            "batt": round(self.battery.percent, 1),
            "voltage": round(self.battery.voltage, 2),
            "battery_v": round(self.battery.voltage, 2),
            "battery_current_a": round(self.battery.last_current_a, 2),
            "wind": round(true_wind.norm(), 1),
            "wind_est": round(self.wind_field.estimate.norm(), 1),
            "tracking_conf": round(self.track.tracking_confidence, 2),
            "identity_conf": round(self.track.identity_confidence, 2),
            "face_score": round(self.track.face_score, 2),
            "throttle": round(self.physics.last_throttle_fraction, 3),
            "thrust_n": round(self.physics.last_thrust_n, 1),
            "max_thrust_n": round(self.physics.last_available_thrust_n, 1),
            "ground_effect_gain": round(self.physics.last_ground_effect_gain, 3),
            "trans_lift_gain": round(self.physics.last_translational_lift_gain, 3),
            "vortex_ring_penalty": round(self.physics.last_vortex_ring_penalty, 3),
            "accel_xy": round(math.hypot(self.physics.last_total_accel.x, self.physics.last_total_accel.y), 2),
            "accel_z": round(self.physics.last_total_accel.z, 2),
        }

    def _step_once(self) -> None:
        while self.scenario_idx < len(self.scenario) and self.scenario[self.scenario_idx].time <= self.sim_t + 1e-9:
            ev = self.scenario[self.scenario_idx]
            self.scenario_idx += 1
            self._handle_event(ev)

        while self.sim_t + 1e-9 >= self.next_heartbeat_t:
            self.safety.heartbeat(now_s=self.sim_t)
            self.next_heartbeat_t += self.heartbeat_period

        true_wind = self.wind_field.step(self.dt)
        self.percep.advance(self.dt)

        telem = self.physics.build_telemetry(self.battery, self.wind_field.estimate, self.sim_t)
        self.latest_telem = telem

        control_due = self.sim_t + 1e-9 >= self.next_control_t
        if control_due:
            self.track = self.percep.observe(self.sim_t, self.physics.truth(true_wind))

        self.safety_status = self.safety.update(
            telem,
            self.current_sp,
            target_world=self.track.target_position_world,
            now_s=self.sim_t,
        )

        if control_due:
            control_dt = self.control_period if self.last_control_t is None else max(self.sim_t - self.last_control_t, 1e-6)
            cmd = self.pending_commands.pop(0) if self.pending_commands else None
            self.stability = self.stab.update(telem, now_s=self.sim_t)
            mission_status = self.mission.update(
                self.track,
                telem,
                self.safety_status,
                self.stability,
                cmd,
                now_s=self.sim_t,
            )
            self.ready_to_record = mission_status.ready_to_record

            self.shot.set_speed_scale(self.stab.speed_scale)
            self.current_sp = self.shot.update(
                self.track,
                telem,
                self.mission.state,
                self.mission.shot_mode,
                self.safety_status.active_override,
                desired_distance=self.mission.desired_distance,
                extra_distance=self.stab.extra_distance,
                dt_s=control_dt,
            )

            self.last_control_t = self.sim_t
            self.next_control_t += self.control_period

        self.current_sp = self.safety.clamp_setpoint(self.current_sp, telem, self.safety_status)

        self.fi.send(self.current_sp, now_s=self.sim_t)
        applied_sp = self.fi.tick(telem, now_s=self.sim_t)
        self.last_applied_setpoint = applied_sp

        self.physics.step(applied_sp, self.dt, true_wind, self.wind_field.gust, self.battery)
        self.latest_truth = self.physics.truth(true_wind)
        self.battery.update(
            self.dt,
            self.physics.last_throttle_fraction,
            self.latest_truth.airspeed.norm(),
            max(0.0, -self.latest_truth.velocity.z),
        )

        sample_time = self.sim_t + self.dt
        sample_telem = self.physics.build_telemetry(self.battery, self.wind_field.estimate, sample_time)
        rec = self._build_record(sample_time, sample_telem, true_wind)
        self.records.append(rec)
        self.latest_record = rec

        if sample_time + 1e-9 >= self.next_log_t:
            log.info(
                "t=%5.1f  state=%-10s lock=%-10s stab=%-8s safety=%-15s  "
                "drone=(%.1f,%.1f,%.1f)  subj=(%.1f,%.1f)  wind=%.1f  batt=%.1f%%",
                sample_time,
                rec["state"],
                rec["lock"],
                rec["stability"],
                rec["safety"],
                rec["drone_x"],
                rec["drone_y"],
                rec["drone_z"],
                rec["subj_x"],
                rec["subj_y"],
                rec["wind"],
                rec["batt"],
            )
            if self.log_interval_s is not None:
                self.next_log_t += self.log_interval_s

        self.sim_t += self.dt
        self.sim_clock.advance(self.dt)
        self.tick += 1


def run_simulation(
    scenario: List[ScenarioEvent] | None = None,
    duration: float = 70.0,
    dt: float = 0.02,
    config: SystemConfig | None = None,
    seed: int | None = None,
) -> List[dict]:
    """Run the complete simulation and return per-tick records."""
    session = SimulationSession(
        scenario=scenario,
        dt=dt,
        config=config,
        seed=seed,
        log_interval_s=5.0,
    )
    log.info("=== Simulation start - %.0f s, dt=%.3f s ===", duration, dt)
    while session.sim_t < duration:
        session.step()
    log.info("=== Simulation complete - %d ticks ===", session.tick)
    return session.records


if __name__ == "__main__":
    records = run_simulation()
    states_seen = sorted({r["state"] for r in records})
    filming_ticks = sum(1 for r in records if r["ready"])
    total_ticks = len(records)

    print("\n" + "=" * 60)
    print(f"  States visited : {states_seen}")
    print(f"  Filming ticks  : {filming_ticks} / {total_ticks} ({100 * filming_ticks / max(total_ticks, 1):.1f}%)")
    print(f"  Final state    : {records[-1]['state']}")
    print(f"  Final battery  : {records[-1]['batt']:.1f}%")
    print(f"  Final voltage  : {records[-1]['voltage']:.2f} V")
    print("=" * 60 + "\n")
