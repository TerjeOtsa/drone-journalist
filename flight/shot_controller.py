"""
Shot Controller
===============
Computes the desired drone position/velocity to maintain a stable, smooth
filming position relative to the tracked subject.

Supports four shot modes:
  • STANDUP       — hover in front, eye level
  • WALK_AND_TALK — follow behind/beside a moving subject
  • WIDE_SAFETY   — farther away, higher altitude
  • ORBIT         — circle the subject at constant radius

Design notes
------------
*  All output is in NED frame (metres from home).
*  Positions are low-pass filtered to eliminate jitter.
*  Velocity feed-forward anticipates subject motion.
*  A jerk limiter caps acceleration changes frame-to-frame.
*  The controller never produces setpoints that violate min/max distance.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from config.parameters import MissionParams, SafetyParams, ShotParams
from interfaces.clock import Clock, SystemClock
from interfaces.schemas import (
    FlightSetpoint,
    LockState,
    MissionState,
    SafetyOverride,
    ShotMode,
    TargetTrack,
    DroneTelemetry,
    Vec3,
)

log = logging.getLogger(__name__)


class ShotController:
    """25 Hz controller producing flight setpoints for the autopilot."""

    def __init__(
        self,
        params: ShotParams | None = None,
        *,
        mission_params: MissionParams | None = None,
        safety_params: SafetyParams | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.p = params or ShotParams()
        self.mission_params = mission_params or MissionParams()
        self.safety_params = safety_params or SafetyParams()
        self.clock = clock or SystemClock()

        # ── internal state ───────────────────────────────────────────────
        self._filtered_target: Optional[Vec3] = None
        self._filtered_velocity: Vec3 = Vec3()
        self._prev_setpoint: Optional[Vec3] = None
        self._prev_velocity_cmd: Vec3 = Vec3()
        self._orbit_angle: float = 0.0  # radians, current angle for orbit
        self._last_t: float = self.clock.monotonic()
        self._speed_scale: float = 1.0  # reduced by stability supervisor
        self._face_bias: float = 0.0    # smoothed face-framing bias [0,1]

    # ════════════════════════════════════════════════════════════════════
    #  PUBLIC
    # ════════════════════════════════════════════════════════════════════

    def set_speed_scale(self, scale: float) -> None:
        """Called by stability supervisor to throttle speed."""
        self._speed_scale = max(0.0, min(1.0, scale))

    def update(
        self,
        track: TargetTrack,
        telem: DroneTelemetry,
        mission_state: MissionState,
        shot_mode: ShotMode,
        safety_override: SafetyOverride,
        desired_distance: float | None = None,
        extra_distance: float = 0.0,
        dt_s: float | None = None,
    ) -> FlightSetpoint:
        """Call once per tick. Returns setpoint for flight interface."""
        if dt_s is None:
            now = self.clock.monotonic()
            dt = min(max(now - self._last_t, 0.0), 0.2)
            self._last_t = now
        else:
            dt = min(max(dt_s, 0.0), 0.2)
            self._last_t = self.clock.monotonic()

        timestamp_s = self.clock.time()

        # ── non-filming states → special setpoints ───────────────────────
        if mission_state == MissionState.TAKEOFF:
            return self._takeoff_setpoint(telem)
        if mission_state in (MissionState.RETURN, MissionState.LAND):
            return self._return_setpoint(telem, mission_state)
        if mission_state == MissionState.EMERGENCY:
            return self._emergency_setpoint(telem)
        if mission_state in (MissionState.IDLE,):
            return FlightSetpoint(timestamp=timestamp_s)
        if safety_override == SafetyOverride.HOVER:
            return self._hover_setpoint(telem)

        # ── filter target position ───────────────────────────────────────
        target_pos = self._get_target_pos(track, telem)
        if target_pos is None:
            return self._hover_setpoint(telem)

        self._filtered_target = self._lpf_vec3(
            self._filtered_target, target_pos, self.p.position_lpf_alpha
        )

        # ── filter target velocity ───────────────────────────────────────
        target_vel = track.target_velocity_world or Vec3()
        self._filtered_velocity = self._lpf_vec3(
            self._filtered_velocity, target_vel, self.p.velocity_lpf_alpha
        )

        # ── compute desired position based on shot mode ──────────────────
        desired = self._compute_desired(
            shot_mode, self._filtered_target, self._filtered_velocity,
            telem, dt, desired_distance, extra_distance,
        )

        # ── face-aware framing bias ─────────────────────────────────────
        if self.p.face_framing_enabled and track.lock_state == LockState.LOCKED:
            face_need = 1.0 if track.face_score < self.p.face_low_threshold else 0.0
            self._face_bias += self.p.face_bias_alpha * (face_need - self._face_bias)
            if self._face_bias > 0.05:
                desired = self._apply_face_bias(
                    desired, self._filtered_target, telem, self._face_bias,
                )
        else:
            # Decay bias when not filming
            self._face_bias *= (1.0 - self.p.face_bias_alpha)

        # ── clamp distance ───────────────────────────────────────────────
        desired = self._clamp_distance(desired, self._filtered_target)
        desired = self._clamp_altitude(desired)

        # ── compute velocity command ─────────────────────────────────────
        error = desired - telem.position
        if error.norm() < self.p.position_deadband:
            vel_cmd = Vec3()
        else:
            vel_cmd = self._velocity_from_error(error, dt)

        # ── yaw towards subject ──────────────────────────────────────────
        yaw = self._yaw_towards(telem.position, self._filtered_target)

        return FlightSetpoint(
            timestamp=timestamp_s,
            position=desired,
            velocity=vel_cmd,
            yaw=yaw,
        )

    # ════════════════════════════════════════════════════════════════════
    #  SHOT MODE GEOMETRY
    # ════════════════════════════════════════════════════════════════════

    def _compute_desired(
        self,
        mode: ShotMode,
        target: Vec3,
        target_vel: Vec3,
        telem: DroneTelemetry,
        dt: float,
        desired_distance: float | None,
        extra_dist: float,
    ) -> Vec3:
        if mode == ShotMode.STANDUP:
            return self._standup(target, telem, desired_distance, extra_dist)
        elif mode == ShotMode.WALK_AND_TALK:
            return self._walk_and_talk(target, target_vel, telem, desired_distance, extra_dist)
        elif mode == ShotMode.WIDE_SAFETY:
            return self._wide_safety(target, telem, desired_distance, extra_dist)
        elif mode == ShotMode.ORBIT:
            return self._orbit(target, dt, desired_distance, extra_dist)
        else:
            return self._standup(target, telem, desired_distance, extra_dist)

    def _standup(
        self,
        target: Vec3,
        telem: DroneTelemetry,
        desired_distance: float | None,
        extra: float,
    ) -> Vec3:
        """Fixed position in front of subject, eye level."""
        # "in front" = from subject towards drone current horizontal direction
        dx = telem.position.x - target.x
        dy = telem.position.y - target.y
        horiz = (dx**2 + dy**2) ** 0.5
        if horiz < 0.1:
            dx, dy = 1.0, 0.0
            horiz = 1.0
        dist = self._resolve_distance(self.p.standup_distance, desired_distance, extra)
        scale = dist / horiz
        return Vec3(
            x=target.x + dx * scale,
            y=target.y + dy * scale,
            z=-(self.p.standup_altitude),  # NED up is negative z
        )

    def _walk_and_talk(
        self,
        target: Vec3,
        target_vel: Vec3,
        telem: DroneTelemetry,
        desired_distance: float | None,
        extra: float,
    ) -> Vec3:
        """Follow behind / beside the subject as they walk."""
        dist = self._resolve_distance(self.p.walktalk_distance, desired_distance, extra)

        # Direction: behind the subject's velocity vector
        speed = (target_vel.x**2 + target_vel.y**2) ** 0.5
        if speed > 0.3:
            # Place drone behind the direction of travel
            behind = Vec3(-target_vel.x, -target_vel.y, 0.0).normalized()
        else:
            # Subject stationary → fall back to standup geometry
            return self._standup(target, telem, desired_distance, extra)

        # Predict future target position
        if self.p.use_velocity_feed_forward:
            pred = Vec3(
                target.x + target_vel.x * self.p.target_prediction_horizon,
                target.y + target_vel.y * self.p.target_prediction_horizon,
                target.z,
            )
        else:
            pred = target

        return Vec3(
            x=pred.x + behind.x * dist,
            y=pred.y + behind.y * dist,
            z=-(self.p.walktalk_altitude),
        )

    def _wide_safety(
        self,
        target: Vec3,
        telem: DroneTelemetry,
        desired_distance: float | None,
        extra: float,
    ) -> Vec3:
        """Farther away, higher — safe fallback shot."""
        dx = telem.position.x - target.x
        dy = telem.position.y - target.y
        horiz = (dx**2 + dy**2) ** 0.5
        if horiz < 0.1:
            dx, dy = 1.0, 0.0
            horiz = 1.0
        dist = self._resolve_distance(self.p.wide_distance, desired_distance, extra)
        scale = dist / horiz
        return Vec3(
            x=target.x + dx * scale,
            y=target.y + dy * scale,
            z=-(self.p.wide_altitude),
        )

    def _orbit(
        self,
        target: Vec3,
        dt: float,
        desired_distance: float | None,
        extra: float,
    ) -> Vec3:
        """Circle the subject at a constant radius."""
        self._orbit_angle += math.radians(self.p.orbit_speed_deg_s) * dt * self._speed_scale
        self._orbit_angle %= 2 * math.pi
        r = self._resolve_distance(self.p.orbit_radius, desired_distance, extra)
        return Vec3(
            x=target.x + r * math.cos(self._orbit_angle),
            y=target.y + r * math.sin(self._orbit_angle),
            z=-(self.p.orbit_altitude),
        )

    def _apply_face_bias(
        self, desired: Vec3, target: Vec3, telem: DroneTelemetry, bias: float
    ) -> Vec3:
        """Nudge the drone position to improve face visibility.

        When the face score is persistently low, this shifts the drone:
          - laterally (perpendicular to the drone→subject line) so it's
            slightly to one side rather than directly behind
          - slightly lower, to reduce the downward camera angle

        The bias ramps smoothly via _face_bias [0..1] so the drone doesn't
        jerk when a face disappears.
        """
        dx = desired.x - target.x
        dy = desired.y - target.y
        horiz = math.hypot(dx, dy)
        if horiz < 0.2:
            return desired
        # Perpendicular direction (rotate 90° CCW)
        perp_x = -dy / horiz
        perp_y = dx / horiz
        lateral = self.p.face_bias_lateral_m * bias
        alt_drop = self.p.face_bias_altitude_m * bias
        return Vec3(
            x=desired.x + perp_x * lateral,
            y=desired.y + perp_y * lateral,
            z=desired.z + alt_drop,  # NED: more positive z = lower
        )

    # ════════════════════════════════════════════════════════════════════
    #  SPECIAL-STATE SETPOINTS
    # ════════════════════════════════════════════════════════════════════

    def _takeoff_setpoint(self, telem: DroneTelemetry) -> FlightSetpoint:
        alt = self.mission_params.takeoff_altitude
        return FlightSetpoint(
            timestamp=self.clock.time(),
            position=Vec3(telem.position.x, telem.position.y, -alt),
            yaw=telem.attitude_euler.z,
        )

    def _return_setpoint(self, telem: DroneTelemetry, state: MissionState) -> FlightSetpoint:
        mp = self.mission_params
        home = telem.home_position
        if state == MissionState.LAND:
            target = Vec3(home.x, home.y, 0.0)
        else:
            target = Vec3(home.x, home.y, -mp.takeoff_altitude)
        error = target - telem.position
        if error.norm() < 0.3:
            vel = Vec3()
        else:
            vel = error.normalized() * min(mp.return_speed, error.norm())
        yaw = self._yaw_towards(telem.position, Vec3(home.x, home.y, 0.0))
        return FlightSetpoint(
            timestamp=self.clock.time(), position=target, velocity=vel, yaw=yaw,
        )

    def _hover_setpoint(self, telem: DroneTelemetry) -> FlightSetpoint:
        return FlightSetpoint(
            timestamp=self.clock.time(),
            position=Vec3(telem.position.x, telem.position.y, telem.position.z),
            velocity=Vec3(),
            yaw=telem.attitude_euler.z,
        )

    def _emergency_setpoint(self, telem: DroneTelemetry) -> FlightSetpoint:
        sp = self.safety_params
        return FlightSetpoint(
            timestamp=self.clock.time(),
            position=Vec3(telem.position.x, telem.position.y, 0.0),
            velocity=Vec3(0, 0, sp.emergency_descent_rate),  # NED: +z = down
            yaw=telem.attitude_euler.z,
        )

    # ════════════════════════════════════════════════════════════════════
    #  UTILITIES
    # ════════════════════════════════════════════════════════════════════

    def _get_target_pos(self, track: TargetTrack, telem: DroneTelemetry) -> Optional[Vec3]:
        """Extract best available world-frame target position."""
        if track.target_position_world is not None:
            return track.target_position_world
        # If only image coords available, we can't safely compute world pos
        # without depth — return None and let caller hover.
        return None

    def _lpf_vec3(self, prev: Optional[Vec3], new: Vec3, alpha: float) -> Vec3:
        if prev is None:
            return new
        return Vec3(
            prev.x + alpha * (new.x - prev.x),
            prev.y + alpha * (new.y - prev.y),
            prev.z + alpha * (new.z - prev.z),
        )

    def _velocity_from_error(self, error: Vec3, dt: float) -> Vec3:
        """P-controller with velocity cap and acceleration limiter."""
        max_v = self.p.max_velocity * self._speed_scale
        max_a = self.p.max_acceleration

        # Proportional gain = max_v / 3m (reach max speed at 3 m error)
        kp = max_v / 3.0
        desired_vel = Vec3(error.x * kp, error.y * kp, error.z * kp)

        # Cap velocity magnitude
        mag = desired_vel.norm()
        if mag > max_v:
            desired_vel = desired_vel.normalized() * max_v

        # Acceleration limiter (jerk limit)
        if dt > 1e-6:
            dv = desired_vel - self._prev_velocity_cmd
            accel = dv.norm() / dt
            if accel > max_a:
                dv = dv.normalized() * (max_a * dt)
                desired_vel = self._prev_velocity_cmd + dv

        self._prev_velocity_cmd = desired_vel
        return desired_vel

    def _clamp_distance(self, pos: Vec3, target: Vec3) -> Vec3:
        """Ensure distance to target is within [min_distance, max_distance]."""
        dx = pos.x - target.x
        dy = pos.y - target.y
        horiz = (dx**2 + dy**2) ** 0.5
        if horiz < 1e-6:
            return pos
        if horiz < self.p.min_distance:
            scale = self.p.min_distance / horiz
            return Vec3(target.x + dx * scale, target.y + dy * scale, pos.z)
        if horiz > self.p.max_distance:
            scale = self.p.max_distance / horiz
            return Vec3(target.x + dx * scale, target.y + dy * scale, pos.z)
        return pos

    def _clamp_altitude(self, pos: Vec3) -> Vec3:
        """Enforce altitude limits (NED: z negative = up)."""
        alt_agl = -pos.z
        alt_agl = max(self.p.min_altitude_agl, min(self.p.max_altitude_agl, alt_agl))
        return Vec3(pos.x, pos.y, -alt_agl)

    def _resolve_distance(
        self,
        default_distance: float,
        desired_distance: float | None,
        extra: float,
    ) -> float:
        base = default_distance if desired_distance is None else desired_distance
        return max(self.p.min_distance, min(self.p.max_distance, base + extra))

    @staticmethod
    def _yaw_towards(from_pos: Vec3, to_pos: Vec3) -> float:
        """Yaw angle (NED, clockwise from north) to face target."""
        dx = to_pos.x - from_pos.x
        dy = to_pos.y - from_pos.y
        return math.atan2(dy, dx)
