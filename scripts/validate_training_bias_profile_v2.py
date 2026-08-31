#!/usr/bin/env python3
"""Validate the production v2 profile rule before empirical recapture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.training_bias_profile import (  # noqa: E402
    BRANCHES,
    holm_adjusted_p,
    matched_training_bias_profile,
)


CALIBRATION_UNITS = 16
CONFIRMATION_UNITS = 16
STATES_PER_UNIT = 4
DIMENSION = 64


def _unit_ids() -> list[str]:
    return [
        f"cal-{unit}"
        for unit in range(CALIBRATION_UNITS)
        for _ in range(STATES_PER_UNIT)
    ] + [
        f"confirm-{unit}"
        for unit in range(CONFIRMATION_UNITS)
        for _ in range(STATES_PER_UNIT)
    ]


def _base_repair(rng: np.random.Generator) -> np.ndarray:
    count = (CALIBRATION_UNITS + CONFIRMATION_UNITS) * STATES_PER_UNIT
    repair = rng.normal(size=(count, DIMENSION))
    repair /= np.linalg.norm(repair, axis=1, keepdims=True)
    return repair


def _unit_noise(rng: np.random.Generator, *, heavy_tail: bool = False) -> np.ndarray:
    rows = []
    for _ in range(CALIBRATION_UNITS + CONFIRMATION_UNITS):
        latent = 0.025 * rng.normal(size=DIMENSION)
        innovations = (
            0.02 * rng.standard_t(df=3.0, size=(STATES_PER_UNIT, DIMENSION))
            if heavy_tail
            else 0.02 * rng.normal(size=(STATES_PER_UNIT, DIMENSION))
        )
        rows.extend(latent + innovations)
    return np.asarray(rows)


def generate(name: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, set[str]]:
    repair = _base_repair(rng)
    noise = _unit_noise(rng, heavy_tail=name == "clustered_heavy_tail_centered")
    expected: set[str] = set()
    if name in {"clustered_centered", "clustered_heavy_tail_centered"}:
        effect = noise
    elif name == "clustered_skew_centered":
        effect = _unit_noise(rng) + 0.015 * (rng.exponential(size=repair.shape) - 1.0)
    elif name == "fixed_additive":
        direction = np.zeros(DIMENSION)
        direction[:8] = 0.05
        effect = direction + noise
        expected.update(("additive", "residual_direction"))
    elif name == "rotating_repair_aligned":
        effect = 0.08 * repair + noise
        expected.add("repair_aligned")
    elif name == "alternating_by_unit":
        direction = np.zeros(DIMENSION)
        direction[:8] = 0.08
        signs = np.repeat(
            np.asarray([1.0 if unit % 2 == 0 else -1.0 for unit in range(32)]),
            STATES_PER_UNIT,
        )
        effect = signs[:, None] * direction + noise
    elif name == "tiny_repair":
        repair *= 1e-14
        effect = noise
    else:
        raise ValueError(name)
    return effect, repair, expected


def evaluate(name: str, trial: int) -> tuple[dict, set[str]]:
    rng = np.random.default_rng(20261001 + 10000 * list(SCENARIOS).index(name) + trial)
    effect, repair, expected = generate(name, rng)
    split = CALIBRATION_UNITS * STATES_PER_UNIT
    result = matched_training_bias_profile(
        effect,
        repair,
        calibration_indices=range(split),
        confirmation_indices=range(split, len(effect)),
        inference_unit_ids=_unit_ids(),
        minimum_independent_units=8,
        signflip_draws=999,
        seed=20261031 + trial,
    )
    if result["status"] != "POPULATION_INFERENCE_COMPLETE":
        return result, expected
    branches = result["population_inference"]["branches"]
    adjusted = holm_adjusted_p({
        name: branch["raw_studentized_signflip_p"]
        for name, branch in branches.items()
    })
    for branch_name, branch in branches.items():
        interval = branch["confidence_interval_95"]
        excludes_zero = interval[0] > 0.0 or interval[1] < 0.0
        branch["holm_adjusted_p"] = adjusted[branch_name]
        branch["confirmed_after_holm"] = bool(
            adjusted[branch_name] <= 0.05
            and excludes_zero
            and branch["confirmation_direction_matches_calibration"]
        )
    return result, expected


SCENARIOS = (
    "clustered_centered",
    "clustered_heavy_tail_centered",
    "clustered_skew_centered",
    "alternating_by_unit",
    "fixed_additive",
    "rotating_repair_aligned",
    "tiny_repair",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=200)
    args = parser.parse_args()
    rows = {}
    for scenario in SCENARIOS:
        detections = {branch: 0 for branch in BRANCHES}
        coverage = {branch: 0 for branch in BRANCHES}
        complete = abstain = any_detection = 0
        for trial in range(args.trials):
            result, expected = evaluate(scenario, trial)
            if result["status"] != "POPULATION_INFERENCE_COMPLETE":
                abstain += 1
                continue
            complete += 1
            branches = result["population_inference"]["branches"]
            confirmed = []
            for branch_name, branch in branches.items():
                decision = bool(branch["confirmed_after_holm"])
                detections[branch_name] += int(decision)
                confirmed.append(decision)
                if branch_name not in expected:
                    lo, hi = branch["confidence_interval_95"]
                    coverage[branch_name] += int(lo <= 0.0 <= hi)
            any_detection += int(any(confirmed))
        denominator = max(complete, 1)
        rows[scenario] = {
            "trials": args.trials,
            "complete": complete,
            "abstain": abstain,
            "family_detection_rate": any_detection / denominator,
            "branch_detection_rate": {
                branch: detections[branch] / denominator for branch in BRANCHES
            },
            "zero_effect_interval_coverage": {
                branch: coverage[branch] / denominator
                for branch in BRANCHES
                if branch not in ({"additive", "residual_direction"} if scenario == "fixed_additive" else {"repair_aligned"} if scenario == "rotating_repair_aligned" else set())
            },
        }

    nulls = (
        "clustered_centered",
        "clustered_heavy_tail_centered",
        "clustered_skew_centered",
        "alternating_by_unit",
    )
    false_positive_gate = all(rows[name]["family_detection_rate"] <= 0.075 for name in nulls)
    coverage_values = [
        value
        for name in nulls
        for value in rows[name]["zero_effect_interval_coverage"].values()
    ]
    coverage_gate = min(coverage_values) >= 0.90
    power_gate = (
        rows["fixed_additive"]["branch_detection_rate"]["additive"] >= 0.80
        and rows["rotating_repair_aligned"]["branch_detection_rate"]["repair_aligned"] >= 0.80
    )
    abstention_gate = rows["tiny_repair"]["abstain"] == args.trials
    status = "GO" if all((false_positive_gate, coverage_gate, power_gate, abstention_gate)) else "NO_GO"
    payload = {
        "schema": "kernel-analyzer-training-bias-profile-v2-synthetic-validation",
        "status": status,
        "production_rule_reused": True,
        "trials_per_scenario": args.trials,
        "inference_unit": f"independent run-like cluster with {STATES_PER_UNIT} correlated states",
        "rows": rows,
        "gates": {
            "family_false_positive_lte_0_075": false_positive_gate,
            "zero_effect_interval_coverage_gte_0_90": coverage_gate,
            "target_branch_power_gte_0_80": power_gate,
            "tiny_repair_always_abstains": abstention_gate,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": status, "output": str(args.output)}))


if __name__ == "__main__":
    main()
