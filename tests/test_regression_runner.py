"""
Unit tests for regression-runner helpers.
"""

import pytest

from sim.regression_runner import ScenarioSpec, extract_metrics, run_scenario


def _record(t: float, state: str) -> dict:
    return {
        "t": t,
        "state": state,
        "ready": state == "film",
        "batt": 80.0,
        "wind": 2.0,
        "lock": "locked",
        "drone_x": 0.0,
        "drone_y": 0.0,
        "drone_z": -2.5,
    }


class TestRegressionRunner:
    def test_extract_metrics_uses_scenario_dt_for_state_time(self):
        records = [
            _record(0.05, "lock"),
            _record(0.10, "film"),
            _record(0.15, "film"),
        ]

        metrics = extract_metrics(records, dt=0.05)

        assert metrics["state_time"]["lock"] == pytest.approx(0.05)
        assert metrics["state_time"]["film"] == pytest.approx(0.10)

    def test_run_scenario_applies_config_overrides(self, monkeypatch):
        captured = {}

        def _fake_run_simulation(*, config, scenario, duration, dt, seed):
            captured["config"] = config
            captured["scenario"] = scenario
            captured["duration"] = duration
            captured["dt"] = dt
            captured["seed"] = seed
            return [_record(0.05, "idle")]

        monkeypatch.setattr("sim.regression_runner.run_simulation", _fake_run_simulation)

        spec = ScenarioSpec(
            name="override-check",
            duration=1.0,
            dt=0.05,
            seed=9,
            config_overrides={
                "shot": {"standup_distance": 7.5},
                "mission": {"takeoff_altitude": 3.8},
            },
        )

        result = run_scenario(spec)

        assert result.passed is True
        assert captured["config"].shot.standup_distance == 7.5
        assert captured["config"].mission.takeoff_altitude == 3.8
        assert captured["dt"] == 0.05
        assert captured["seed"] == 9
