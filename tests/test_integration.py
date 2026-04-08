"""
Integration test: run the full simulation and verify key properties.
"""

import pytest
from sim.sim_harness import run_simulation, DEFAULT_SCENARIO, ScenarioEvent
from config.parameters import SystemConfig


class TestFullSimulation:
    def test_default_scenario_completes(self):
        records = run_simulation(duration=70.0, dt=0.04)
        assert len(records) > 100
        # Should visit FILM state at some point
        states = set(r["state"] for r in records)
        assert "film" in states

    def test_filming_occurs(self):
        records = run_simulation(duration=70.0, dt=0.04)
        filming = [r for r in records if r["ready"]]
        assert len(filming) > 10

    def test_ends_with_return_or_land(self):
        records = run_simulation(duration=70.0, dt=0.04)
        final = records[-1]["state"]
        assert final in ("return", "land", "idle")

    def test_battery_above_zero(self):
        records = run_simulation(duration=70.0, dt=0.04)
        assert records[-1]["batt"] > 0

    def test_quick_start_scenario(self):
        """Minimal scenario: start → lock → film briefly → stop."""
        scenario = [
            ScenarioEvent(0.0, "start"),
            ScenarioEvent(1.5, "lock"),
            ScenarioEvent(8.0, "stop"),
        ]
        records = run_simulation(scenario=scenario, duration=15.0, dt=0.04)
        states = set(r["state"] for r in records)
        assert "takeoff" in states

    def test_deterministic_with_same_seed(self):
        a = run_simulation(duration=20.0, dt=0.04, seed=123)
        b = run_simulation(duration=20.0, dt=0.04, seed=123)
        assert a == b

    def test_realistic_outputs_include_voltage_and_wind_estimate(self):
        records = run_simulation(duration=20.0, dt=0.04, seed=5)
        assert "voltage" in records[0]
        assert "wind_est" in records[0]
        assert min(r["voltage"] for r in records) < records[0]["voltage"]

    def test_wind_triggers_stability_degradation(self):
        """Heavy wind scenario should push stability to marginal or worse."""
        scenario = [
            ScenarioEvent(0.0, "start"),
            ScenarioEvent(1.5, "lock"),
            ScenarioEvent(4.0, "wind", {"speed": [10.0, 5.0, 0.0]}),
            ScenarioEvent(15.0, "stop"),
        ]
        records = run_simulation(scenario=scenario, duration=20.0, dt=0.04, seed=42)
        stability_levels = {r["stability"] for r in records}
        # Heavy wind should produce at least one non-nominal reading
        assert stability_levels != {"nominal"}, (
            f"Expected stability degradation under 10 m/s wind, got only {stability_levels}"
        )

    def test_battery_drains_over_long_flight(self):
        """Over a 120 s flight battery should drop appreciably."""
        scenario = [
            ScenarioEvent(0.0, "start"),
            ScenarioEvent(2.0, "lock"),
            ScenarioEvent(115.0, "stop"),
        ]
        records = run_simulation(scenario=scenario, duration=120.0, dt=0.04, seed=99)
        initial_batt = records[0]["batt"]
        final_batt = records[-1]["batt"]
        drain = initial_batt - final_batt
        assert drain > 3.0, f"Expected >3 % drain over 120 s, got {drain:.1f}%"

    def test_lock_lost_triggers_degrade_state(self):
        """Explicit perception loss should push mission into DEGRADE."""
        scenario = [
            ScenarioEvent(0.0, "start"),
            ScenarioEvent(1.5, "lock"),
            ScenarioEvent(5.0, "lost", {"duration": 15.0}),
            ScenarioEvent(30.0, "stop"),
        ]
        records = run_simulation(scenario=scenario, duration=35.0, dt=0.04, seed=77)
        states = {r["state"] for r in records}
        assert "degrade" in states or "return" in states, (
            f"Expected DEGRADE or RETURN after long lock loss, saw only {states}"
        )

    def test_different_seeds_produce_different_results(self):
        a = run_simulation(duration=20.0, dt=0.04, seed=1)
        b = run_simulation(duration=20.0, dt=0.04, seed=2)
        # At least some positions should differ
        diffs = [
            abs(ra["drone_x"] - rb["drone_x"]) + abs(ra["drone_y"] - rb["drone_y"])
            for ra, rb in zip(a[-50:], b[-50:])
        ]
        assert max(diffs) > 0.01, "Different seeds should produce different trajectories"
