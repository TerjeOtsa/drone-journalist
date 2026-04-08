"""
Unit tests for the perception-side identity and lock state logic.
"""

from interfaces.schemas import LockState, Vec3
from perception.identity_lock import IdentityLockManager
from perception.parameters import IdentityLockParams
from perception.schemas import (
    CandidateObservation,
    IdentityCues,
    ImageTarget,
    TrackingCues,
    WorldTarget,
)


def _image() -> ImageTarget:
    return ImageTarget(cx_norm=0.50, cy_norm=0.55, w_norm=0.20, h_norm=0.40)


def _world(x: float = 5.0, y: float = 0.0) -> WorldTarget:
    return WorldTarget(
        position_m=Vec3(x, y, 0.0),
        velocity_mps=Vec3(0.2, 0.0, 0.0),
    )


def _obs(
    ts: float,
    candidate_id: str | None = "journalist",
    second_best: float = 0.10,
    body: float = 0.92,
    face: float | None = None,
    phone: float | None = 0.90,
    pose: float | None = 0.85,
    detection: float = 0.95,
    association: float = 0.95,
    motion: float = 0.90,
    visibility: float = 0.90,
    quality: float = 0.90,
    image: bool = True,
    world: bool = True,
    fresh: bool = True,
    track_id: str = "trk_1",
) -> CandidateObservation:
    return CandidateObservation(
        timestamp=ts,
        frame_id=int(ts * 100),
        candidate_id=candidate_id,
        track_id=track_id,
        image_target=_image() if image else None,
        world_target=_world() if world else None,
        identity=IdentityCues(body=body, face=face, phone=phone, pose=pose),
        tracking=TrackingCues(
            detection=detection,
            association=association,
            motion=motion,
            visibility=visibility,
            quality=quality,
        ),
        second_best_score=second_best,
        has_fresh_detection=fresh,
        face_used=face is not None,
        phone_used=phone is not None,
    )


class TestIdentityLockManager:
    def test_candidate_becomes_locked_after_stable_strong_observations(self):
        params = IdentityLockParams(candidate_stable_time_s=0.20)
        mgr = IdentityLockManager(params)

        first = mgr.update(_obs(0.00))
        second = mgr.update(_obs(0.25))

        assert first.lock_state == LockState.CANDIDATE
        assert second.lock_state == LockState.LOCKED
        assert second.target_id == "journalist"

    def test_ambiguous_runner_up_caps_identity_and_prevents_lock(self):
        params = IdentityLockParams(candidate_stable_time_s=0.10)
        mgr = IdentityLockManager(params)

        out1 = mgr.update(_obs(0.00, second_best=0.86))
        out2 = mgr.update(_obs(0.20, second_best=0.86))

        assert out1.identity_confidence <= params.ambiguity_identity_cap
        assert out2.lock_state == LockState.CANDIDATE
        assert out2.identity_confidence <= params.ambiguity_identity_cap

    def test_locked_target_will_not_switch_directly_to_new_subject(self):
        params = IdentityLockParams(candidate_stable_time_s=0.10, weak_to_lost_timeout_s=0.20)
        mgr = IdentityLockManager(params)

        mgr.update(_obs(0.00, candidate_id="alice"))
        locked = mgr.update(_obs(0.15, candidate_id="alice"))
        mismatch = mgr.update(_obs(0.20, candidate_id="bob", track_id="trk_2"))

        assert locked.lock_state == LockState.LOCKED
        assert mismatch.lock_state == LockState.WEAK
        assert mismatch.target_id == "alice"
        assert mismatch.target_position_world is None
        assert mismatch.diagnostics.failure_reason == "identity_mismatch"

    def test_prediction_only_tracking_decays_to_lost(self):
        params = IdentityLockParams(
            candidate_stable_time_s=0.05,
            weak_to_lost_timeout_s=0.50,
            tracking_decay_tau_s=0.20,
        )
        mgr = IdentityLockManager(params)

        mgr.update(_obs(0.00))
        mgr.update(_obs(0.10))
        weak = mgr.update(
            _obs(
                0.25,
                world=False,
                fresh=False,
                detection=0.20,
                association=0.15,
                motion=0.20,
                visibility=0.10,
                quality=0.20,
            )
        )
        lost = mgr.update(
            _obs(
                0.90,
                candidate_id=None,
                image=False,
                world=False,
                fresh=False,
                detection=0.0,
                association=0.0,
                motion=0.0,
                visibility=0.0,
                quality=0.0,
            )
        )

        assert weak.lock_state == LockState.WEAK
        assert lost.lock_state == LockState.LOST
        assert lost.lost_target is True

    def test_world_missing_caps_tracking_confidence(self):
        params = IdentityLockParams(candidate_stable_time_s=0.10, image_only_tracking_cap=0.60)
        mgr = IdentityLockManager(params)

        out = mgr.update(_obs(0.00, world=False))

        assert out.tracking_confidence <= 0.60
        assert out.lock_state == LockState.CANDIDATE

    def test_weak_state_requests_help_after_timeout(self):
        params = IdentityLockParams(
            candidate_stable_time_s=0.05,
            weak_help_after_s=0.20,
        )
        mgr = IdentityLockManager(params)

        mgr.update(_obs(0.00))
        mgr.update(_obs(0.10))
        weak1 = mgr.update(_obs(0.20, second_best=0.50, body=0.60, phone=None, pose=0.60))
        weak2 = mgr.update(_obs(0.50, second_best=0.55, body=0.60, phone=None, pose=0.60))

        assert weak1.lock_state == LockState.WEAK
        assert weak2.diagnostics.assist_request == "identity_ambiguous"
