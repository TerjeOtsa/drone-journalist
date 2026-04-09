"""
interfaces — Shared data types and infrastructure used across all packages.

Modules:
    schemas     Core dataclasses: Vec3, DroneTelemetry, TargetTrack, FlightSetpoint,
                MissionState, LockState, ShotMode, SafetyOverride, and more.
    event_bus   Simple in-process pub/sub for SystemEvent distribution.
    clock       Wall-clock abstraction (mockable for deterministic tests).
"""
