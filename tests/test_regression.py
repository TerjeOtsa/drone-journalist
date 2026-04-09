"""
Pytest wrapper for regression scenarios.

Discovers all .yaml scenario files under ``scenarios/`` and runs each one
as a parameterised test case.
"""

import pathlib

import pytest

from sim.regression_runner import discover_scenarios, load_scenario, run_scenario

SCENARIO_DIR = pathlib.Path(__file__).resolve().parent.parent / "scenarios"


def _scenario_ids() -> list[str]:
    return [p.stem for p in discover_scenarios(SCENARIO_DIR)]


def _scenario_paths() -> list[pathlib.Path]:
    return discover_scenarios(SCENARIO_DIR)


@pytest.mark.parametrize(
    "scenario_path",
    _scenario_paths(),
    ids=_scenario_ids(),
)
def test_regression_scenario(scenario_path: pathlib.Path) -> None:
    spec = load_scenario(scenario_path)
    result = run_scenario(spec)
    if not result.passed:
        detail = "\n".join(result.failed_assertions)
        if result.error:
            detail += f"\nError: {result.error}"
        pytest.fail(f"Scenario '{spec.name}' failed:\n{detail}")
