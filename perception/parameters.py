"""
Parameters for the perception-side identity and lock state machine.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IdentityLockParams:
    """Weights, thresholds, and timing for the perception-side identity lock FSM."""

    # Identity fusion weights
    weight_body: float = 0.35
    weight_temporal: float = 0.20
    weight_pose: float = 0.10
    weight_phone: float = 0.15
    weight_face: float = 0.20

    # Tracking fusion weights
    weight_detection: float = 0.30
    weight_association: float = 0.30
    weight_motion: float = 0.20
    weight_visibility: float = 0.10
    weight_quality: float = 0.10

    # State thresholds
    strong_identity_threshold: float = 0.75
    strong_tracking_threshold: float = 0.70
    weak_identity_threshold: float = 0.35
    weak_tracking_threshold: float = 0.25
    candidate_margin_threshold: float = 0.15
    relock_identity_threshold: float = 0.75
    relock_tracking_threshold: float = 0.70

    # Stability timing
    candidate_stable_time_s: float = 0.50
    relock_stable_time_s: float = 0.40
    weak_to_lost_timeout_s: float = 1.50

    # Confidence decay
    identity_decay_tau_s: float = 2.0
    tracking_decay_tau_s: float = 0.6
    positive_identity_threshold: float = 0.60

    # Ambiguity handling
    ambiguity_cap_delta: float = 0.10
    ambiguity_identity_cap: float = 0.55

    # Safety caps
    image_only_tracking_cap: float = 0.69
    mismatch_identity_cap: float = 0.25
    mismatch_tracking_cap: float = 0.20

    # App-assist timers
    candidate_help_after_s: float = 5.0
    weak_help_after_s: float = 2.0
    lost_help_after_s: float = 3.0
