"""
Parameter Sensitivity Sweep
============================
Systematically vary one or more SimulationParams / SystemConfig fields and
measure how key metrics respond.  Produces a tabular report and optional CSV.

Use this to:
  - Find safe operating envelopes (e.g. max wind vs filming %)
  - Validate that tuning a parameter doesn't break other metrics
  - Compare performance across seeds for robustness

Usage:
    python -m sim.param_sweep                 # default sweep
    python -m sim.param_sweep --csv out.csv   # write CSV
"""

from __future__ import annotations

import argparse
import copy
import csv
import io
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from config.parameters import SimulationParams, SystemConfig
from sim.regression_runner import extract_metrics
from sim.sim_harness import ScenarioEvent, run_simulation

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Sweep specification
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SweepAxis:
    """One parameter axis to vary."""
    param_path: str           # dot-separated path, e.g. "sim.wind_response_tau_s"
    values: List[Any]         # list of values to try
    label: str = ""           # display label

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.param_path


@dataclass
class SweepSpec:
    """Full sweep definition."""
    axes: List[SweepAxis]
    seeds: List[int] = field(default_factory=lambda: [42])
    duration: float = 70.0
    dt: float = 0.02
    scenario: Optional[List[ScenarioEvent]] = None
    metrics_of_interest: List[str] = field(default_factory=lambda: [
        "filming_percent", "min_battery", "locked_percent",
        "state_count", "final_state",
    ])


@dataclass
class SweepResult:
    """One row of sweep results."""
    param_values: Dict[str, Any]
    seed: int
    metrics: Dict[str, Any]


# ═══════════════════════════════════════════════════════════════════════════
#  Config manipulation
# ═══════════════════════════════════════════════════════════════════════════

def _set_param(config: SystemConfig, path: str, value: Any) -> None:
    """Set a nested parameter on a SystemConfig by dot-path.

    Examples:
        "sim.wind_response_tau_s"   -> config.sim.wind_response_tau_s = value
        "safety.geofence_radius"    -> config.safety.geofence_radius = value
        "shot.follow_distance"      -> config.shot.follow_distance = value
    """
    parts = path.split(".")
    obj: Any = config
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _get_param(config: SystemConfig, path: str) -> Any:
    parts = path.split(".")
    obj: Any = config
    for part in parts:
        obj = getattr(obj, part)
    return obj


# ═══════════════════════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════════════════════

def _cartesian(axes: List[SweepAxis]) -> List[Dict[str, Any]]:
    """Cartesian product of all axis values."""
    if not axes:
        return [{}]
    combos: List[Dict[str, Any]] = [{}]
    for axis in axes:
        new_combos = []
        for combo in combos:
            for val in axis.values:
                c = dict(combo)
                c[axis.param_path] = val
                new_combos.append(c)
        combos = new_combos
    return combos


def run_sweep(spec: SweepSpec) -> List[SweepResult]:
    """Run the full parameter sweep and return results."""
    combos = _cartesian(spec.axes)
    total = len(combos) * len(spec.seeds)
    results: List[SweepResult] = []

    log.info("Parameter sweep: %d combinations × %d seeds = %d runs",
             len(combos), len(spec.seeds), total)

    for i, combo in enumerate(combos):
        for seed in spec.seeds:
            cfg = SystemConfig()
            for path, val in combo.items():
                _set_param(cfg, path, val)

            log.info("  Run %d/%d  seed=%d  %s",
                     i * len(spec.seeds) + seed + 1, total, seed,
                     {k.split(".")[-1]: v for k, v in combo.items()})

            records = run_simulation(
                scenario=spec.scenario,
                duration=spec.duration,
                dt=spec.dt,
                config=cfg,
                seed=seed,
            )
            metrics = extract_metrics(records)
            results.append(SweepResult(
                param_values=dict(combo),
                seed=seed,
                metrics=metrics,
            ))

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Reporting
# ═══════════════════════════════════════════════════════════════════════════

def format_table(
    results: List[SweepResult],
    spec: SweepSpec,
) -> str:
    """Format results as a readable text table."""
    lines: List[str] = []

    # Header
    param_cols = [a.label for a in spec.axes]
    metric_cols = spec.metrics_of_interest
    header = ["seed"] + param_cols + metric_cols
    lines.append("  ".join(f"{h:>16s}" for h in header))
    lines.append("-" * len(lines[0]))

    for r in results:
        row = [str(r.seed)]
        for a in spec.axes:
            v = r.param_values.get(a.param_path, "")
            row.append(f"{v}")
        for m in metric_cols:
            val = r.metrics.get(m, "")
            if isinstance(val, float):
                row.append(f"{val:.1f}")
            else:
                row.append(str(val))
        lines.append("  ".join(f"{c:>16s}" for c in row))

    return "\n".join(lines)


def write_csv(
    results: List[SweepResult],
    spec: SweepSpec,
    path: str | None = None,
) -> str:
    """Write results as CSV (to file or return as string)."""
    buf = io.StringIO()
    param_keys = [a.param_path for a in spec.axes]
    metric_keys = spec.metrics_of_interest
    fieldnames = ["seed"] + param_keys + metric_keys
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        row: Dict[str, Any] = {"seed": r.seed}
        row.update(r.param_values)
        for m in metric_keys:
            row[m] = r.metrics.get(m, "")
        writer.writerow(row)

    text = buf.getvalue()
    if path:
        with open(path, "w", newline="") as f:
            f.write(text)
        log.info("CSV written to %s", path)
    return text


# ═══════════════════════════════════════════════════════════════════════════
#  Built-in sweep definitions
# ═══════════════════════════════════════════════════════════════════════════

def wind_sweep() -> SweepSpec:
    """Sweep wind response tau vs filming performance."""
    return SweepSpec(
        axes=[
            SweepAxis(
                param_path="sim.wind_response_tau_s",
                values=[1.0, 3.0, 5.0, 8.0, 12.0],
                label="wind_tau",
            ),
        ],
        seeds=[42, 100],
        duration=70.0,
    )


def perception_noise_sweep() -> SweepSpec:
    """Sweep perception noise vs filming and locked percentages."""
    return SweepSpec(
        axes=[
            SweepAxis(
                param_path="sim.perception_pos_noise_m",
                values=[0.1, 0.3, 0.5, 1.0, 2.0],
                label="percep_noise",
            ),
        ],
        seeds=[42, 55, 100],
        duration=60.0,
    )


def wind_vs_follow_distance() -> SweepSpec:
    """2-D sweep: wind tau × follow distance."""
    return SweepSpec(
        axes=[
            SweepAxis("sim.wind_response_tau_s", [2.0, 6.0, 10.0], "wind_tau"),
            SweepAxis("shot.follow_distance", [5.0, 8.0, 12.0], "follow_dist"),
        ],
        seeds=[42],
        duration=60.0,
    )


BUILTIN_SWEEPS = {
    "wind": wind_sweep,
    "perception": perception_noise_sweep,
    "wind_vs_follow": wind_vs_follow_distance,
}


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)-5s  %(message)s")

    parser = argparse.ArgumentParser(description="Parameter sensitivity sweep")
    parser.add_argument(
        "sweep",
        nargs="?",
        default="wind",
        choices=list(BUILTIN_SWEEPS.keys()),
        help="Built-in sweep to run (default: wind)",
    )
    parser.add_argument("--csv", type=str, default=None, help="Write CSV output")
    args = parser.parse_args()

    spec = BUILTIN_SWEEPS[args.sweep]()
    results = run_sweep(spec)

    print("\n" + format_table(results, spec))
    print()

    if args.csv:
        write_csv(results, spec, args.csv)


if __name__ == "__main__":
    main()
