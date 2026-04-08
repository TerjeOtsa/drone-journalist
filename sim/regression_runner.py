"""
Regression Scenario Harness
============================
Run a library of YAML-defined scenarios and assert expected metrics.

Each scenario file specifies:
  - A sequence of ScenarioEvents
  - Duration, dt, seed
  - Pass/fail assertions on the simulation output

Usage:
    python -m sim.regression_runner                       # run all scenarios
    python -m sim.regression_runner scenarios/smoke.yaml  # run one scenario
    python -m pytest tests/test_regression.py             # via pytest
"""

from __future__ import annotations

import logging
import pathlib
import sys
from dataclasses import dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional

import yaml  # type: ignore[import-untyped]

from config.parameters import SystemConfig
from interfaces.schemas import ShotMode
from sim.sim_harness import ScenarioEvent, run_simulation

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  YAML schema
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Assertion:
    """A single pass/fail check on the simulation output."""
    metric: str            # e.g. "filming_percent", "final_state", "min_battery"
    op: str = "gte"        # gte, lte, eq, neq, in, contains
    value: Any = None

    def evaluate(self, actual: Any) -> bool:
        if self.op == "gte":
            return actual >= self.value
        if self.op == "lte":
            return actual <= self.value
        if self.op == "eq":
            return actual == self.value
        if self.op == "neq":
            return actual != self.value
        if self.op == "in":
            return actual in self.value
        if self.op == "contains":
            return self.value in actual
        raise ValueError(f"Unknown op: {self.op}")

    def describe(self, actual: Any) -> str:
        return f"{self.metric} {self.op} {self.value!r}  (actual={actual!r})"


@dataclass
class ScenarioSpec:
    """Parsed scenario specification from YAML."""
    name: str
    description: str = ""
    duration: float = 70.0
    dt: float = 0.02
    seed: int = 42
    events: List[ScenarioEvent] = field(default_factory=list)
    assertions: List[Assertion] = field(default_factory=list)
    config_overrides: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
#  Metric extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_metrics(records: List[dict], dt: float | None = None) -> Dict[str, Any]:
    """Compute standard metrics from simulation records."""
    if not records:
        return {}

    total = len(records)
    filming_ticks = sum(1 for r in records if r["ready"])
    states_seen = sorted({r["state"] for r in records})
    min_batt = min(r["batt"] for r in records)
    max_wind = max(r["wind"] for r in records)
    final = records[-1]

    # Time in each state
    state_time: Dict[str, float] = {}
    default_dt = dt
    if default_dt is None and len(records) >= 2:
        default_dt = max(
            0.0,
            float(records[1].get("t", 0.0)) - float(records[0].get("t", 0.0)),
        )
    if default_dt is None:
        default_dt = 0.0

    prev_t: float | None = None
    for r in records:
        st = r["state"]
        current_t = float(r.get("t", 0.0))
        sample_dt = default_dt if prev_t is None else max(0.0, current_t - prev_t)
        state_time[st] = state_time.get(st, 0.0) + sample_dt
        prev_t = current_t

    # Tracking statistics
    locked_ticks = sum(1 for r in records if r["lock"] == "locked")
    lost_ticks = sum(1 for r in records if r["lock"] == "lost")

    return {
        "total_ticks": total,
        "filming_ticks": filming_ticks,
        "filming_percent": 100.0 * filming_ticks / max(total, 1),
        "states_seen": states_seen,
        "state_count": len(states_seen),
        "final_state": final["state"],
        "final_battery": final["batt"],
        "min_battery": min_batt,
        "max_wind": max_wind,
        "locked_percent": 100.0 * locked_ticks / max(total, 1),
        "lost_percent": 100.0 * lost_ticks / max(total, 1),
        "state_time": state_time,
        "final_drone_x": final["drone_x"],
        "final_drone_y": final["drone_y"],
        "final_drone_z": final["drone_z"],
    }


def _apply_config_overrides(config: SystemConfig, overrides: Dict[str, Any]) -> SystemConfig:
    for section_name, section_overrides in overrides.items():
        if not hasattr(config, section_name):
            raise ValueError(f"Unknown config section: {section_name}")

        section = getattr(config, section_name)
        if not isinstance(section_overrides, dict):
            setattr(config, section_name, section_overrides)
            continue

        if not is_dataclass(section):
            raise ValueError(f"Config section is not overridable: {section_name}")

        for field_name, value in section_overrides.items():
            if not hasattr(section, field_name):
                raise ValueError(f"Unknown config field: {section_name}.{field_name}")
            setattr(section, field_name, value)
    return config


# ═══════════════════════════════════════════════════════════════════════════
#  YAML loading
# ═══════════════════════════════════════════════════════════════════════════

def _parse_event(raw: dict) -> ScenarioEvent:
    return ScenarioEvent(
        time=float(raw["time"]),
        action=raw["action"],
        params=raw.get("params", {}),
    )


def _parse_assertion(raw: dict) -> Assertion:
    return Assertion(
        metric=raw["metric"],
        op=raw.get("op", "gte"),
        value=raw["value"],
    )


def load_scenario(path: str | pathlib.Path) -> ScenarioSpec:
    """Load a single scenario from a YAML file."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return ScenarioSpec(
        name=data.get("name", pathlib.Path(path).stem),
        description=data.get("description", ""),
        duration=data.get("duration", 70.0),
        dt=data.get("dt", 0.02),
        seed=data.get("seed", 42),
        events=[_parse_event(e) for e in data.get("events", [])],
        assertions=[_parse_assertion(a) for a in data.get("assertions", [])],
        config_overrides=data.get("config_overrides", {}),
    )


def discover_scenarios(directory: str | pathlib.Path) -> List[pathlib.Path]:
    """Discover all .yaml scenario files in a directory."""
    d = pathlib.Path(directory)
    if not d.is_dir():
        return []
    return sorted(d.glob("*.yaml"))


# ═══════════════════════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ScenarioResult:
    name: str
    passed: bool
    metrics: Dict[str, Any]
    failed_assertions: List[str] = field(default_factory=list)
    error: Optional[str] = None


def run_scenario(spec: ScenarioSpec) -> ScenarioResult:
    """Run a single scenario and check its assertions."""
    try:
        config = _apply_config_overrides(SystemConfig(), spec.config_overrides)
        records = run_simulation(
            config=config,
            scenario=spec.events or None,
            duration=spec.duration,
            dt=spec.dt,
            seed=spec.seed,
        )
        metrics = extract_metrics(records, dt=spec.dt)
    except Exception as exc:
        return ScenarioResult(
            name=spec.name,
            passed=False,
            metrics={},
            error=str(exc),
        )

    failures: List[str] = []
    for a in spec.assertions:
        actual = metrics.get(a.metric)
        if actual is None:
            failures.append(f"UNKNOWN metric: {a.metric}")
        elif not a.evaluate(actual):
            failures.append(f"FAIL: {a.describe(actual)}")

    return ScenarioResult(
        name=spec.name,
        passed=len(failures) == 0,
        metrics=metrics,
        failed_assertions=failures,
    )


def run_all_scenarios(
    directory: str | pathlib.Path = "scenarios",
) -> List[ScenarioResult]:
    """Discover and run all scenarios in a directory."""
    paths = discover_scenarios(directory)
    if not paths:
        log.warning("No scenarios found in %s", directory)
        return []

    results: List[ScenarioResult] = []
    for path in paths:
        spec = load_scenario(path)
        log.info("Running scenario: %s (%s)", spec.name, path.name)
        result = run_scenario(spec)
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        log.info("  %s  %s", status, spec.name)
        for f in result.failed_assertions:
            log.info("    %s", f)
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-5s  %(message)s",
    )

    if len(sys.argv) > 1:
        # Run specific scenario file(s)
        for path in sys.argv[1:]:
            spec = load_scenario(path)
            result = run_scenario(spec)
            _print_result(result)
    else:
        # Run all from scenarios/
        results = run_all_scenarios()
        print("\n" + "=" * 60)
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        print(f"  Regression results: {passed}/{total} passed")
        for r in results:
            mark = "✓" if r.passed else "✗"
            print(f"    {mark}  {r.name}")
            for f in r.failed_assertions:
                print(f"        {f}")
        print("=" * 60)

        if passed < total:
            sys.exit(1)


def _print_result(result: ScenarioResult) -> None:
    mark = "PASS" if result.passed else "FAIL"
    print(f"\n{mark}: {result.name}")
    if result.error:
        print(f"  Error: {result.error}")
    for f in result.failed_assertions:
        print(f"  {f}")
    if result.metrics:
        print(f"  Metrics: filming={result.metrics.get('filming_percent', 0):.1f}%  "
              f"final_state={result.metrics.get('final_state')}  "
              f"min_batt={result.metrics.get('min_battery', 0):.1f}%  "
              f"states={result.metrics.get('state_count', 0)}")


if __name__ == "__main__":
    main()
