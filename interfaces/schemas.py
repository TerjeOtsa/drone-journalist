"""
Autonomy & Flight Systems — Interfaces / Schemas
=================================================
Canonical data-classes for every message that crosses a module boundary.
All modules import from here — never from each other's internals.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

# ─────────────────────────── Enumerations ────────────────────────────────────

class LockState(Enum):
    """Target lock lifecycle reported by perception."""
    CANDIDATE = "candidate"
    LOCKED    = "locked"
    WEAK      = "weak"
    LOST      = "lost"


class MissionState(Enum):
    """Top-level mission phase."""
    IDLE        = "idle"
    TAKEOFF     = "takeoff"
    ACQUIRE     = "acquire"
    LOCK        = "lock"
    FILM        = "film"
    DEGRADE     = "degrade"
    RETURN      = "return"
    LAND        = "land"
    EMERGENCY   = "emergency"


class ShotMode(Enum):
    """Camera-relative shot type."""
    STANDUP       = "standup"        # hovering, fixed position facing subject
    WALK_AND_TALK = "walk_and_talk"  # follow behind/beside while subject moves
    WIDE_SAFETY   = "wide_safety"    # increased distance, wider framing
    ORBIT         = "orbit"          # circling the subject


class SafetyOverride(Enum):
    """Safety system can impose these overrides."""
    NONE          = "none"
    REDUCE_SPEED  = "reduce_speed"
    INCREASE_DIST = "increase_distance"
    HOVER         = "hover"
    RETURN_HOME   = "return_home"
    LAND_NOW      = "land_now"
    EMERGENCY_STOP = "emergency_stop"


class StabilityLevel(Enum):
    """How stable the platform currently is."""
    NOMINAL  = "nominal"
    MARGINAL = "marginal"
    DEGRADED = "degraded"
    CRITICAL = "critical"


# ─────────────────────────── Geometry ────────────────────────────────────────

@dataclass
class Vec3:
    """3-D vector (NED frame unless noted)."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, o: "Vec3") -> "Vec3":
        return Vec3(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o: "Vec3") -> "Vec3":
        return Vec3(self.x - o.x, self.y - o.y, self.z - o.z)

    def __mul__(self, s: float) -> "Vec3":
        return Vec3(self.x * s, self.y * s, self.z * s)

    def norm(self) -> float:
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5

    def normalized(self) -> "Vec3":
        n = self.norm()
        if n < 1e-9:
            return Vec3()
        return Vec3(self.x / n, self.y / n, self.z / n)


@dataclass
class GeoPoint:
    """WGS-84 position."""
    lat: float = 0.0      # degrees
    lon: float = 0.0      # degrees
    alt_msl: float = 0.0  # meters above mean sea level


# ─────────────────────── Perception Inputs ───────────────────────────────────

@dataclass
class TargetTrack:
    """
    Consolidated tracking output from the perception pipeline.
    This is the PRIMARY input the autonomy stack consumes every frame.
    """
    timestamp: float                        # UNIX epoch seconds
    target_position_image: tuple[float, float] = (0.0, 0.0)  # (u, v) normalised 0..1
    target_position_world: Optional[Vec3] = None              # NED metres, may be None
    target_velocity_world: Optional[Vec3] = None              # NED m/s, may be None
    identity_confidence: float = 0.0        # 0..1
    tracking_confidence: float = 0.0        # 0..1
    lock_state: LockState = LockState.LOST
    bounding_box: Optional[tuple[float, float, float, float]] = None  # (x, y, w, h) norm
    face_score: float = 0.0                 # 0..1  face visibility / quality signal


# ─────────────────────── Telemetry Input ─────────────────────────────────────

@dataclass
class DroneTelemetry:
    """
    Autopilot telemetry snapshot consumed every tick.
    """
    timestamp: float = 0.0
    position: Vec3 = field(default_factory=Vec3)       # NED metres from home
    velocity: Vec3 = field(default_factory=Vec3)        # NED m/s
    attitude_euler: Vec3 = field(default_factory=Vec3)  # roll, pitch, yaw (rad)
    gps: GeoPoint = field(default_factory=GeoPoint)
    battery_voltage: float = 16.8    # V (4S nominal)
    battery_percent: float = 100.0   # 0..100
    armed: bool = False
    in_air: bool = False
    gps_fix: int = 0                 # 0=none, 3=3-D
    satellites: int = 0
    wind_estimate: Vec3 = field(default_factory=Vec3)   # NED m/s (from EKF)
    home_position: Vec3 = field(default_factory=Vec3)


# ──────────────────── App / Operator Commands ────────────────────────────────

@dataclass
class AppCommand:
    """Command from the mobile app / operator."""
    timestamp: float = 0.0
    action: str = ""          # "start", "stop", "pause", "resume", "relock",
                              # "set_shot_mode", "set_distance", "emergency_stop"
    shot_mode: Optional[ShotMode] = None
    desired_distance: Optional[float] = None   # metres
    desired_altitude: Optional[float] = None   # metres AGL


# ──────────────────── Flight Setpoint Output ─────────────────────────────────

@dataclass
class FlightSetpoint:
    """
    The SOLE output sent to the autopilot every cycle.
    Only one of position / velocity should be active at a time.
    """
    timestamp: float = 0.0
    # Position setpoint (NED, metres from home)
    position: Optional[Vec3] = None
    # Velocity setpoint (NED, m/s) — used for smooth tracking
    velocity: Optional[Vec3] = None
    # Yaw (radians, NED, clockwise from north)
    yaw: float = 0.0
    yaw_rate: Optional[float] = None   # rad/s (if yaw-rate control preferred)
    # Coordinate frame flag
    is_body_frame: bool = False


# ──────────────────── Safety Output ──────────────────────────────────────────

@dataclass
class SafetyStatus:
    """Published by the safety module every tick."""
    timestamp: float = 0.0
    active_override: SafetyOverride = SafetyOverride.NONE
    geofence_ok: bool = True
    battery_ok: bool = True
    link_ok: bool = True
    min_distance_ok: bool = True
    reasons: List[str] = field(default_factory=list)


# ──────────────────── Mission Status Output ──────────────────────────────────

@dataclass
class MissionStatus:
    """Broadcast to app & logger every state change."""
    timestamp: float = 0.0
    state: MissionState = MissionState.IDLE
    shot_mode: ShotMode = ShotMode.STANDUP
    desired_distance: Optional[float] = None
    ready_to_record: bool = False
    stability: StabilityLevel = StabilityLevel.NOMINAL
    safety: SafetyStatus = field(default_factory=SafetyStatus)
    message: str = ""


# ──────────────────── Events ─────────────────────────────────────────────────

@dataclass
class SystemEvent:
    """Lightweight event for the internal event bus."""
    timestamp: float = field(default_factory=time.time)
    source: str = ""
    event: str = ""
    payload: dict = field(default_factory=dict)
