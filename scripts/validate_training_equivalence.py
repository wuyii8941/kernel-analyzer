#!/usr/bin/env python3
"""Validate false-equivalence behavior for the frozen engineering margins."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.training_equivalence import (  # noqa: E402
    BRANCHES,
    classify_training_equivalence,
)


MARGINS = {"additive": 0.001, "repair_aligned": 0.01, "residual_direction": 0.001}


def t_critical(df: int, probability: float) -> float:
    z = NormalDist().inv_cdf(probability)
    f = float(df)
    return (
        z
        + (z**3 + z) / (4.0 * f)
        + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * f**2)
        + (3.0 * z**7 + 19.0 * z**5 + 17.0 * z**3 - 15.0 * z) / (384.0 * f**3)
    )


def simultaneous_intervals(samples: dict[str, np.ndarray]) -> dict[str, list[float]]:
    count = len(next(iter(samples.values())))
    # Bonferroni intervals give at least 95% simultaneous coverage across the
    # three predeclared update effects, without assuming independence.
    critical = t_critical(count - 1, 1.0 - 0.05 / (2.0 * len(BRANCHES)))
    result = {}
    for name, values in samples.items():
        mean = float(values.mean())
        half = critical * float(values.std(ddof=1) / math.sqrt(count))
        result[name] = [mean - half, mean + half]
    return result


def run_scenario(name: str, trials: int, units: int) -> dict:
    decisions: dict[str, int] = {}
    for trial in range(trials):
        rng = np.random.default_rng(20261201 + 100_000 * SCENARIOS.index(name) + trial)
        means = {branch: 0.0 for branch in BRANCHES}
        noise = {branch: 0.30 * MARGINS[branch] for branch in BRANCHES}
        if name == "small_centered":
            pass
        elif name == "detectable_inside_margin":
            means["additive"] = 0.45 * MARGINS["additive"]
            noise["additive"] = 0.20 * MARGINS["additive"]
        elif name == "just_outside_additive_margin":
            means["additive"] = 1.10 * MARGINS["additive"]
        elif name == "material_additive":
            means["additive"] = 1.50 * MARGINS["additive"]
        elif name == "material_aligned":
            means["repair_aligned"] = -1.50 * MARGINS["repair_aligned"]
        elif name == "one_uncertain_branch":
            means["residual_direction"] = 0.90 * MARGINS["residual_direction"]
            noise["residual_direction"] = 0.70 * MARGINS["residual_direction"]
        else:
            raise ValueError(name)
        samples = {
            branch: rng.normal(means[branch], noise[branch], size=units)
            for branch in BRANCHES
        }
        decision = classify_training_equivalence(
            simultaneous_intervals(samples), MARGINS,
        )["decision"]
        decisions[decision] = decisions.get(decision, 0) + 1
    return {
        "trials": trials,
        "decision_counts": decisions,
        "decision_rates": {key: value / trials for key, value in decisions.items()},
    }


SCENARIOS = [
    "small_centered",
    "detectable_inside_margin",
    "just_outside_additive_margin",
    "material_additive",
    "material_aligned",
    "one_uncertain_branch",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--units", type=int, default=16)
    args = parser.parse_args()
    rows = {name: run_scenario(name, args.trials, args.units) for name in SCENARIOS}
    false_equivalence = rows["just_outside_additive_margin"]["decision_rates"].get(
        "EQUIVALENT_UNDER_PROTOCOL", 0.0
    )
    material_power = min(
        rows[name]["decision_rates"].get("MATERIAL_EFFECT", 0.0)
        for name in ("material_additive", "material_aligned")
    )
    centered_equivalence = rows["small_centered"]["decision_rates"].get(
        "EQUIVALENT_UNDER_PROTOCOL", 0.0
    )
    gates = {
        "false_equivalence_lte_0_05": false_equivalence <= 0.05,
        "material_effect_power_gte_0_90": material_power >= 0.90,
        "small_centered_equivalence_gte_0_90": centered_equivalence >= 0.90,
    }
    payload = {
        "schema": "kernel-analyzer-training-equivalence-validation-v1",
        "status": "GO" if all(gates.values()) else "NO_GO",
        "independent_unit_count": args.units,
        "trials_per_scenario": args.trials,
        "margins": MARGINS,
        "interval_rule": "95% Bonferroni simultaneous intervals across three update effects",
        "rows": rows,
        "gates": gates,
        "claim_boundary": (
            "This validates decision behavior under declared synthetic distributions. "
            "The margins are an engineering contract for the next held-out benchmark, "
            "not universal constants for all LLM training."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "gates": gates}))


if __name__ == "__main__":
    main()
