"""
Product-side schemas derived from the autonomy and perception contracts.

These objects are the app/video/logging view of the system. They do not replace
the existing `interfaces.schemas` classes; they adapt them into a shape that is
useful for mobile UX, recording state, and durable logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Optional

from interfaces.schemas import (
    LockState,
    MissionState,
    SafetyOverride,
    ShotMode,
    StabilityLevel,
)


def _serialize(value: Any) -> Any:
    """Recursively convert dataclasses and enums into JSON-safe primitives."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            item.name: _serialize(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


class RecordingState(Enum):
    """Operator-visible recorder lifecycle."""

    IDLE = "idle"
    ARMED = "armed"
    STARTING = "starting"
    RECORDING_CONFIRMED = "recording_confirmed"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass(frozen=True)
class ConfidenceSnapshot:
    """Identity and tracking confidence packaged for UI and logs."""

    identity: float = 0.0
    tracking: float = 0.0
    effective: float = 0.0

    @classmethod
    def from_values(cls, identity: float, tracking: float) -> "ConfidenceSnapshot":
        effective = max(0.0, min(identity, tracking))
        return cls(identity=identity, tracking=tracking, effective=effective)

    def to_dict(self) -> dict[str, float]:
        return _serialize(self)


@dataclass(frozen=True)
class LockStatusView:
    """Product-layer view of the current subject lock."""

    timestamp: float
    lock_state: LockState = LockState.LOST
    confidence: ConfidenceSnapshot = field(default_factory=ConfidenceSnapshot)
    target_world_valid: bool = False
    target_velocity_valid: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class MissionView:
    """Product-layer view of autonomy state relevant to the app."""

    timestamp: float
    mission_state: MissionState = MissionState.IDLE
    shot_mode: ShotMode = ShotMode.STANDUP
    desired_distance: Optional[float] = None
    ready_to_record: bool = False
    stability: StabilityLevel = StabilityLevel.NOMINAL
    safety_override: SafetyOverride = SafetyOverride.NONE
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class LinkStats:
    """Phone-link health exposed to the app and logs."""

    connected: bool = True
    rssi_dbm: Optional[int] = None
    rtt_ms: Optional[int] = None
    preview_latency_ms: Optional[int] = None
    preview_bitrate_kbps: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class ProductSnapshot:
    """Cold-start and reconnect snapshot exposed to the app."""

    timestamp: float
    drone_id: str
    session_id: str
    profile: str
    mission: MissionView
    lock: LockStatusView
    recording_state: RecordingState = RecordingState.IDLE
    active_clip_id: Optional[str] = None
    battery_percent: Optional[float] = None
    storage_minutes_remaining: Optional[int] = None
    link: LinkStats = field(default_factory=LinkStats)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class ProductEvent:
    """Replay-friendly event emitted by the product layer."""

    seq: int
    timestamp: float
    monotonic_time: float
    session_id: str
    source: str
    event: str
    payload: dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)
