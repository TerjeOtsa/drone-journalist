"""
Identity lock manager for a conservative single-subject perception pipeline.
"""

from __future__ import annotations

import math
from typing import Optional

from interfaces.schemas import LockState
from perception.parameters import IdentityLockParams
from perception.schemas import (
    CandidateObservation,
    LockResult,
    PerceptionDiagnostics,
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


class IdentityLockManager:
    """
    Fuse identity and tracking evidence into a conservative target lock state.

    This class assumes an upstream perception front-end has already selected the
    best current candidate and supplied cue scores. It owns:

    - identity / tracking confidence fusion
    - lock state hysteresis
    - conservative anti-switch behavior
    - adapters' source data for autonomy and app-facing lock summaries
    """

    def __init__(self, params: IdentityLockParams | None = None) -> None:
        self.p = params or IdentityLockParams()

        self.lock_state: LockState = LockState.LOST
        self.protected_target_id: Optional[str] = None

        self._last_update_ts: Optional[float] = None
        self._last_positive_identity_ts: Optional[float] = None
        self._last_fresh_detection_ts: Optional[float] = None
        self._last_candidate_id: Optional[str] = None
        self._last_candidate_ts: Optional[float] = None

        self._candidate_state_since: Optional[float] = None
        self._weak_since: Optional[float] = None
        self._lost_since: Optional[float] = None

        self._stable_candidate_id: Optional[str] = None
        self._stable_candidate_since: Optional[float] = None

    def update(self, obs: CandidateObservation) -> LockResult:
        """Update the lock state from the latest best-candidate observation."""
        ts = obs.timestamp
        self._last_update_ts = ts

        temporal_cue = (
            obs.identity.temporal
            if obs.identity.temporal is not None
            else self._derive_temporal_cue(obs)
        )
        identity_base = self._weighted_average(
            values={
                "body": obs.identity.body,
                "temporal": temporal_cue,
                "pose": obs.identity.pose,
                "phone": obs.identity.phone,
                "face": obs.identity.face,
            },
            weights={
                "body": self.p.weight_body,
                "temporal": self.p.weight_temporal,
                "pose": self.p.weight_pose,
                "phone": self.p.weight_phone,
                "face": self.p.weight_face,
            },
        )
        tracking_base = self._weighted_average(
            values={
                "detection": obs.tracking.detection,
                "association": obs.tracking.association,
                "motion": obs.tracking.motion,
                "visibility": obs.tracking.visibility,
                "quality": obs.tracking.quality,
            },
            weights={
                "detection": self.p.weight_detection,
                "association": self.p.weight_association,
                "motion": self.p.weight_motion,
                "visibility": self.p.weight_visibility,
                "quality": self.p.weight_quality,
            },
        )

        best_score = _clamp01(
            obs.best_candidate_score if obs.best_candidate_score is not None
            else identity_base
        )
        second_best = _clamp01(obs.second_best_score)
        score_margin = max(0.0, best_score - second_best)
        ambiguous = (
            obs.candidate_id is not None
            and score_margin <= self.p.ambiguity_cap_delta
        )

        if (
            obs.candidate_id is not None
            and identity_base >= self.p.positive_identity_threshold
            and not self._is_subject_mismatch(obs)
        ):
            self._last_positive_identity_ts = ts

        if obs.has_fresh_detection and obs.image_target is not None:
            self._last_fresh_detection_ts = ts

        identity_conf = identity_base * self._identity_decay(ts)
        if ambiguous:
            identity_conf = min(identity_conf, self.p.ambiguity_identity_cap)

        tracking_conf = tracking_base * self._tracking_decay(ts, obs.has_fresh_detection)
        if obs.image_target is None:
            tracking_conf = 0.0
        elif obs.world_target is None:
            # Image-only tracking is useful, but not strong enough for active follow.
            tracking_conf = min(tracking_conf, self.p.image_only_tracking_cap)

        failure_reason = obs.failure_reason
        suppress_positions = False
        mismatch = self._is_subject_mismatch(obs)
        if mismatch and self.lock_state != LockState.LOST:
            identity_conf = min(identity_conf, self.p.mismatch_identity_cap)
            tracking_conf = min(tracking_conf, self.p.mismatch_tracking_cap)
            failure_reason = failure_reason or "identity_mismatch"
            suppress_positions = True

        if suppress_positions or self.lock_state == LockState.LOST and obs.candidate_id is None:
            image_target = None
            world_target = None
        else:
            image_target = obs.image_target
            world_target = obs.world_target

        strong_candidate = (
            obs.candidate_id is not None
            and not mismatch
            and identity_conf >= self.p.strong_identity_threshold
            and tracking_conf >= self.p.strong_tracking_threshold
            and score_margin >= self.p.candidate_margin_threshold
        )
        strong_relock = (
            obs.candidate_id is not None
            and obs.candidate_id == self.protected_target_id
            and not mismatch
            and identity_conf >= self.p.relock_identity_threshold
            and tracking_conf >= self.p.relock_tracking_threshold
            and score_margin >= self.p.candidate_margin_threshold
        )

        stable_duration = self.p.candidate_stable_time_s
        if strong_relock:
            stable_duration = self.p.relock_stable_time_s
        self._update_stable_candidate(
            obs.candidate_id,
            obs.candidate_id is not None and not mismatch,
            ts,
        )

        if self.lock_state == LockState.LOST:
            if obs.candidate_id is None:
                self._enter_lost(ts)
            elif strong_candidate and self._candidate_is_stable(obs.candidate_id, ts, stable_duration):
                self.protected_target_id = obs.candidate_id
                self._enter_locked(ts)
            else:
                self._enter_candidate(ts)
        elif self.lock_state == LockState.CANDIDATE:
            if obs.candidate_id is None:
                self._enter_lost(ts)
            elif strong_candidate and self._candidate_is_stable(obs.candidate_id, ts, stable_duration):
                self.protected_target_id = obs.candidate_id
                self._enter_locked(ts)
            else:
                self._enter_candidate(ts)
        elif self.lock_state == LockState.LOCKED:
            if mismatch or obs.candidate_id is None:
                self._enter_weak(ts)
            elif (
                identity_conf >= self.p.strong_identity_threshold
                and tracking_conf >= self.p.strong_tracking_threshold
            ):
                self._enter_locked(ts)
            else:
                self._enter_weak(ts)
        elif self.lock_state == LockState.WEAK:
            if strong_relock and self._candidate_is_stable(obs.candidate_id, ts, stable_duration):
                self._enter_locked(ts)
            elif (
                mismatch
                or obs.candidate_id is None
                or tracking_conf < self.p.weak_tracking_threshold
            ):
                self._enter_weak(ts)
                if self._weak_since is not None and ts - self._weak_since >= self.p.weak_to_lost_timeout_s:
                    self._enter_lost(ts)
                    image_target = None
                    world_target = None
            else:
                self._enter_weak(ts)

        if self.lock_state == LockState.LOST:
            image_target = None
            world_target = None

        assist_request = obs.assist_request or self._assist_request(
            ts=ts,
            ambiguous=ambiguous,
            mismatch=mismatch,
            has_candidate=obs.candidate_id is not None,
        )

        diagnostics = PerceptionDiagnostics(
            target_track_id=obs.track_id,
            best_candidate_score=best_score,
            second_best_score=second_best,
            occlusion_ratio=_clamp01(obs.occlusion_ratio),
            blur_score=_clamp01(obs.blur_score),
            face_used=obs.face_used,
            phone_used=obs.phone_used,
            failure_reason=failure_reason,
            assist_request=assist_request,
        )

        output_target_id = self.protected_target_id or obs.candidate_id
        result = LockResult(
            timestamp_ms=int(ts * 1000.0),
            frame_id=obs.frame_id,
            lock_state=self.lock_state,
            identity_confidence=_clamp01(identity_conf),
            tracking_confidence=_clamp01(tracking_conf),
            target_id=output_target_id,
            target_position_image=image_target,
            target_position_world=world_target,
            diagnostics=diagnostics,
            face_score=_clamp01(obs.identity.face) if obs.identity.face is not None else 0.0,
        )
        self._last_candidate_id = obs.candidate_id
        self._last_candidate_ts = ts
        return result

    def _derive_temporal_cue(self, obs: CandidateObservation) -> Optional[float]:
        if obs.candidate_id is None:
            return None
        if self.protected_target_id is not None and obs.candidate_id == self.protected_target_id:
            if self.lock_state == LockState.LOCKED:
                return 1.0
            if self.lock_state == LockState.WEAK:
                return 0.75
            return 0.60
        if (
            self._last_candidate_id == obs.candidate_id
            and self._last_candidate_ts is not None
            and self._last_update_ts is not None
            and self._last_update_ts - self._last_candidate_ts <= 1.0
        ):
            return 0.60
        return 0.15

    def _identity_decay(self, ts: float) -> float:
        if self._last_positive_identity_ts is None:
            return 1.0
        dt = max(0.0, ts - self._last_positive_identity_ts)
        return math.exp(-dt / self.p.identity_decay_tau_s)

    def _tracking_decay(self, ts: float, has_fresh_detection: bool) -> float:
        if has_fresh_detection or self._last_fresh_detection_ts is None:
            return 1.0
        dt = max(0.0, ts - self._last_fresh_detection_ts)
        return math.exp(-dt / self.p.tracking_decay_tau_s)

    def _weighted_average(
        self,
        values: dict[str, Optional[float]],
        weights: dict[str, float],
    ) -> float:
        total_weight = 0.0
        weighted_sum = 0.0
        for key, value in values.items():
            if value is None:
                continue
            weight = weights[key]
            total_weight += weight
            weighted_sum += weight * _clamp01(value)
        if total_weight <= 1e-9:
            return 0.0
        return weighted_sum / total_weight

    def _update_stable_candidate(
        self,
        candidate_id: Optional[str],
        qualifies: bool,
        ts: float,
    ) -> None:
        if not qualifies or candidate_id is None:
            self._stable_candidate_id = None
            self._stable_candidate_since = None
            return
        if self._stable_candidate_id != candidate_id:
            self._stable_candidate_id = candidate_id
            self._stable_candidate_since = ts

    def _candidate_is_stable(
        self,
        candidate_id: Optional[str],
        ts: float,
        required_s: float,
    ) -> bool:
        if candidate_id is None:
            return False
        if self._stable_candidate_id != candidate_id:
            return False
        if self._stable_candidate_since is None:
            return False
        return ts - self._stable_candidate_since >= required_s

    def _is_subject_mismatch(self, obs: CandidateObservation) -> bool:
        if self.lock_state == LockState.LOST:
            return False
        if self.protected_target_id is None or obs.candidate_id is None:
            return False
        return obs.candidate_id != self.protected_target_id

    def _enter_candidate(self, ts: float) -> None:
        self.lock_state = LockState.CANDIDATE
        if self._candidate_state_since is None:
            self._candidate_state_since = ts
        self._weak_since = None
        self._lost_since = None

    def _enter_locked(self, ts: float) -> None:
        self.lock_state = LockState.LOCKED
        self._candidate_state_since = None
        self._weak_since = None
        self._lost_since = None

    def _enter_weak(self, ts: float) -> None:
        if self.lock_state != LockState.WEAK or self._weak_since is None:
            self._weak_since = ts
        self.lock_state = LockState.WEAK
        self._candidate_state_since = None
        self._lost_since = None

    def _enter_lost(self, ts: float) -> None:
        if self.lock_state != LockState.LOST or self._lost_since is None:
            self._lost_since = ts
        self.lock_state = LockState.LOST
        self._candidate_state_since = None
        self._weak_since = None
        self._stable_candidate_id = None
        self._stable_candidate_since = None

    def _assist_request(
        self,
        ts: float,
        ambiguous: bool,
        mismatch: bool,
        has_candidate: bool,
    ) -> Optional[str]:
        if self.lock_state == LockState.CANDIDATE and self._candidate_state_since is not None:
            if ts - self._candidate_state_since >= self.p.candidate_help_after_s:
                return "need_face_view" if has_candidate else "need_full_body_view"
        if self.lock_state == LockState.WEAK and self._weak_since is not None:
            if ts - self._weak_since >= self.p.weak_help_after_s:
                if ambiguous or mismatch:
                    return "identity_ambiguous"
                return "need_face_view"
        if self.lock_state == LockState.LOST and self._lost_since is not None:
            if ts - self._lost_since >= self.p.lost_help_after_s:
                if self.protected_target_id is None:
                    return "need_full_body_view"
                return "identity_ambiguous"
        return None
