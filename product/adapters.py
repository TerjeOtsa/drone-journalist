"""
Adapters between the current autonomy/perception prototype and product-side
video/app/logging contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from interfaces.schemas import (AppCommand, DroneTelemetry, MissionStatus,
                                SafetyOverride, ShotMode, SystemEvent,
                                TargetTrack)
from product.schemas import (ConfidenceSnapshot, LinkStats, LockStatusView,
                             MissionView, ProductEvent, ProductSnapshot,
                             RecordingState)


@dataclass(frozen=True)
class LaunchTranslation:
    """
    Launch command mapping for the current autonomy prototype.

    The present mission-state machine only accepts `start` in `IDLE` and applies
    `set_shot_mode` while filming. Requested launch shot modes therefore become
    deferred commands rather than immediate pre-launch commands.
    """

    immediate: tuple[AppCommand, ...]
    deferred_until_state: Optional[str] = None
    deferred: tuple[AppCommand, ...] = field(default_factory=tuple)


class ProductStateAdapter:
    """Build product-facing state from current repo schemas."""

    def __init__(self, drone_id: str = "drn_sim", profile: str = "walk") -> None:
        self.drone_id = drone_id
        self.profile = profile

    def make_lock_view(self, track: TargetTrack) -> LockStatusView:
        """Build a product-facing lock status view from a perception track."""
        return LockStatusView(
            timestamp=track.timestamp,
            lock_state=track.lock_state,
            confidence=ConfidenceSnapshot.from_values(
                track.identity_confidence,
                track.tracking_confidence,
            ),
            target_world_valid=track.target_position_world is not None,
            target_velocity_valid=track.target_velocity_world is not None,
        )

    def make_mission_view(self, status: MissionStatus) -> MissionView:
        """Build a product-facing mission view from a mission status."""
        safety_override = (
            status.safety.active_override
            if status.safety is not None
            else SafetyOverride.NONE
        )
        return MissionView(
            timestamp=status.timestamp,
            mission_state=status.state,
            shot_mode=status.shot_mode,
            desired_distance=status.desired_distance,
            ready_to_record=status.ready_to_record,
            stability=status.stability,
            safety_override=safety_override,
            message=status.message,
        )

    def make_snapshot(
        self,
        session_id: str,
        mission_status: MissionStatus,
        track: TargetTrack,
        *,
        recording_state: RecordingState = RecordingState.IDLE,
        active_clip_id: Optional[str] = None,
        telem: Optional[DroneTelemetry] = None,
        storage_minutes_remaining: Optional[int] = None,
        link: Optional[LinkStats] = None,
    ) -> ProductSnapshot:
        """Assemble a full product snapshot for cold-start / reconnect."""
        return ProductSnapshot(
            timestamp=mission_status.timestamp,
            drone_id=self.drone_id,
            session_id=session_id,
            profile=self.profile,
            mission=self.make_mission_view(mission_status),
            lock=self.make_lock_view(track),
            recording_state=recording_state,
            active_clip_id=active_clip_id,
            battery_percent=telem.battery_percent if telem is not None else None,
            storage_minutes_remaining=storage_minutes_remaining,
            link=link or LinkStats(),
        )


class ProductCommandAdapter:
    """Translate product-layer API intents into current autonomy commands."""

    def launch_plan(
        self,
        timestamp: float,
        requested_shot_mode: ShotMode | None = None,
    ) -> LaunchTranslation:
        """Build a launch plan with optional deferred shot-mode switch."""
        immediate = (AppCommand(timestamp=timestamp, action="start"),)
        deferred: tuple[AppCommand, ...] = ()
        if requested_shot_mode is not None:
            deferred = (
                AppCommand(
                    timestamp=timestamp,
                    action="set_shot_mode",
                    shot_mode=requested_shot_mode,
                ),
            )
        return LaunchTranslation(
            immediate=immediate,
            deferred_until_state="film" if deferred else None,
            deferred=deferred,
        )

    def launch_commands(
        self,
        timestamp: float,
        requested_shot_mode: ShotMode | None = None,
    ) -> list[AppCommand]:
        """Immediate launch commands only, for callers that do not use plans."""
        return list(self.launch_plan(timestamp, requested_shot_mode).immediate)

    def set_mode(self, timestamp: float, shot_mode: ShotMode) -> AppCommand:
        """Create a ``set_shot_mode`` command."""
        return AppCommand(
            timestamp=timestamp,
            action="set_shot_mode",
            shot_mode=shot_mode,
        )

    def set_distance(self, timestamp: float, desired_distance: float) -> AppCommand:
        """Create a ``set_distance`` command."""
        return AppCommand(
            timestamp=timestamp,
            action="set_distance",
            desired_distance=desired_distance,
        )

    def relock(self, timestamp: float) -> AppCommand:
        """Create a ``relock`` command."""
        return AppCommand(timestamp=timestamp, action="relock")

    def stop(self, timestamp: float) -> AppCommand:
        """Create a ``stop`` command."""
        return AppCommand(timestamp=timestamp, action="stop")

    def emergency_stop(self, timestamp: float) -> AppCommand:
        """Create an ``emergency_stop`` command."""
        return AppCommand(timestamp=timestamp, action="emergency_stop")


class EventMirror:
    """Assign product-event sequence numbers to the current lightweight bus."""

    def __init__(self, start_seq: int = 0) -> None:
        self._next_seq = start_seq

    def mirror_system_event(
        self,
        event: SystemEvent,
        *,
        session_id: str,
        monotonic_time: float | None = None,
        correlation_id: str | None = None,
    ) -> ProductEvent:
        """Wrap a low-level ``SystemEvent`` as a sequenced ``ProductEvent``."""
        self._next_seq += 1
        return ProductEvent(
            seq=self._next_seq,
            timestamp=event.timestamp,
            monotonic_time=event.timestamp if monotonic_time is None else monotonic_time,
            session_id=session_id,
            source=event.source,
            event=event.event,
            payload=dict(event.payload),
            correlation_id=correlation_id,
        )
