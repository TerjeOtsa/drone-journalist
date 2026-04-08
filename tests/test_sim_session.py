"""
Tests for the step-based simulation session and force outputs.
"""

import pytest

from interfaces.schemas import ShotMode
from sim.sim_harness import SimulationSession


class TestSimulationSession:
    def test_session_exposes_force_terms_and_interactive_controls(self):
        session = SimulationSession(scenario=[], dt=0.04, seed=11, log_interval_s=None)
        session.start()
        session.force_lock()
        session.set_subject_motion(walking=True, speed=1.4, heading_deg=20.0)

        for _ in range(220):
            session.step()

        session.set_distance(8.0)
        session.set_shot_mode(ShotMode.ORBIT)
        for _ in range(12):
            session.step()

        rec = session.latest_record
        assert rec is not None
        assert rec["max_thrust_n"] > 0.0
        assert rec["thrust_n"] >= 0.0
        assert rec["ground_effect_gain"] >= 1.0
        assert rec["trans_lift_gain"] >= 1.0
        assert 0.0 <= rec["vortex_ring_penalty"] <= 1.0
        assert 0.0 <= rec["throttle"] <= 1.0
        assert session.mission.desired_distance == pytest.approx(8.0)

    def test_ground_effect_is_stronger_near_ground(self):
        session = SimulationSession(scenario=[], dt=0.04, seed=3, log_interval_s=None)
        physics = session.physics

        physics.pos.z = -0.1
        near_ground = physics._ground_effect_gain()

        physics.pos.z = -8.0
        far_from_ground = physics._ground_effect_gain()

        assert near_ground > far_from_ground
        assert near_ground > 1.0

    def test_subject_motion_is_rate_limited(self):
        session = SimulationSession(scenario=[], dt=0.04, seed=21, log_interval_s=None)
        session.set_subject_motion(walking=True, speed=2.0, heading_deg=0.0)

        session.step()

        rec = session.latest_record
        assert rec is not None
        assert 0.0 < rec["subject_speed"] < 2.0

    def test_face_score_is_published_when_subject_is_visible(self):
        session = SimulationSession(scenario=[], dt=0.04, seed=9, log_interval_s=None)
        session.start()
        session.force_lock()

        for _ in range(240):
            session.step()

        face_scores = [record["face_score"] for record in session.records]
        assert max(face_scores) > 0.15
