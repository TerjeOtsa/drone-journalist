"""
Unit tests for the Stability Supervisor.
"""

import time
import pytest

from config.parameters import StabilityParams
from flight.stability_supervisor import StabilitySupervisor
from interfaces.schemas import DroneTelemetry, StabilityLevel, Vec3


def _telem(wind_x=0, wind_y=0, vx=0, vy=0, px=0, py=0) -> DroneTelemetry:
    return DroneTelemetry(
        wind_estimate=Vec3(wind_x, wind_y, 0),
        velocity=Vec3(vx, vy, 0),
        position=Vec3(px, py, -2.5),
        in_air=True,
    )


class TestWindAssessment:
    def test_low_wind_nominal(self):
        sup = StabilitySupervisor()
        level = sup.update(_telem(wind_x=2.0))
        assert level == StabilityLevel.NOMINAL

    def test_moderate_wind_marginal(self):
        sup = StabilitySupervisor()
        level = sup.update(_telem(wind_x=6.0))
        assert level == StabilityLevel.MARGINAL

    def test_strong_wind_degraded(self):
        sup = StabilitySupervisor()
        level = sup.update(_telem(wind_x=9.0))
        assert level == StabilityLevel.DEGRADED

    def test_critical_wind(self):
        sup = StabilitySupervisor()
        level = sup.update(_telem(wind_x=13.0))
        assert level == StabilityLevel.CRITICAL


class TestSpeedScaleApplication:
    def test_nominal_full_speed(self):
        sup = StabilitySupervisor()
        sup.update(_telem(wind_x=1.0))
        assert sup.speed_scale == 1.0

    def test_marginal_reduced_speed(self):
        sup = StabilitySupervisor()
        sup.update(_telem(wind_x=6.0))
        assert sup.speed_scale == 0.7

    def test_degraded_extra_distance(self):
        sup = StabilitySupervisor()
        sup.update(_telem(wind_x=9.0))
        assert sup.extra_distance == 2.0

    def test_critical_hover(self):
        sup = StabilitySupervisor()
        sup.update(_telem(wind_x=13.0))
        assert sup.should_hover is True


class TestHysteresis:
    def test_no_rapid_flapping(self):
        """Level should NOT immediately drop back when wind briefly dips."""
        sup = StabilitySupervisor()
        # Push to DEGRADED with strong wind
        sup.update(_telem(wind_x=9.0), now_s=0.0)
        assert sup.level == StabilityLevel.DEGRADED
        # Wind drops to nominal on next tick — but hysteresis holds
        sup.update(_telem(wind_x=1.0), now_s=0.1)
        assert sup.level == StabilityLevel.DEGRADED  # held by hysteresis

    def test_immediate_escalation(self):
        """Escalation (worse) should happen immediately, no hold."""
        sup = StabilitySupervisor()
        sup.update(_telem(wind_x=6.0), now_s=0.0)
        assert sup.level == StabilityLevel.MARGINAL
        # Wind jumps to critical — should escalate immediately
        sup.update(_telem(wind_x=13.0), now_s=0.1)
        assert sup.level == StabilityLevel.CRITICAL

    def test_de_escalation_after_hold(self):
        """De-escalation should happen once hold timer expires."""
        sup = StabilitySupervisor()
        sup.update(_telem(wind_x=9.0), now_s=0.0)
        assert sup.level == StabilityLevel.DEGRADED
        # Wind drops — wait past hysteresis period
        sup.update(_telem(wind_x=1.0), now_s=2.0)
        assert sup.level == StabilityLevel.NOMINAL
