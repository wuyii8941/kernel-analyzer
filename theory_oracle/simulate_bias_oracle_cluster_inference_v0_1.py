#!/usr/bin/env python
"""Coverage audit for trajectory-level versus incorrectly state-level t intervals."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.stats import t


SCHEMA_VERSION = "forkcert.bias-oracle-cluster-coverage.v0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def t_interval(values: np.ndarray) -> tuple[float, float]:
    count = values.size
    if count < 2:
        return -math.inf, math.inf
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / math.sqrt(count))
    critical = float(t.ppf(0.975, count - 1))
    return mean - critical * standard_error, mean + critical * standard_error


def normal_effect(rng: np.random.Generator, trajectories: int) -> np.ndarray:
    return rng.normal(size=trajectories)


def heteroskedastic_normal_effect(rng: np.random.Generator, trajectories: int) -> np.ndarray:
    scales = np.linspace(0.25, 2.0, trajectories)
    return rng.normal(size=trajectories) * scales


def heavy_tail_effect(rng: np.random.Generator, trajectories: int) -> np.ndarray:
    return rng.standard_t(df=3, size=trajectories) / math.sqrt(3.0)


def skew_effect(rng: np.random.Generator, trajectories: int) -> np.ndarray:
    raw = rng.lognormal(mean=0.0, sigma=1.0, size=trajectories)
    mean = math.exp(0.5)
    variance = (math.exp(1.0) - 1.0) * math.exp(1.0)
    return (raw - mean) / math.sqrt(variance)


def rare_effect(rng: np.random.Generator, trajectories: int) -> np.ndarray:
    probability = 0.05
    high = math.sqrt((1.0 - probability) / probability)
    low = -math.sqrt(probability / (1.0 - probability))
    return np.where(rng.random(trajectories) < probability, high, low)


SCENARIOS: dict[str, Callable[[np.random.Generator, int], np.ndarray]] = {
    "gaussian": normal_effect,
    "heteroskedastic_gaussian": heteroskedastic_normal_effect,
    "student_t3": heavy_tail_effect,
    "centered_lognormal": skew_effect,
    "rare_five_percent_cluster": rare_effect,
}


def simulate(
    *,
    trials: int,
    trajectories: int,
    states_per_trajectory: int,
    effect_generator: Callable[[np.random.Generator, int], np.ndarray],
    rng: np.random.Generator,
) -> dict[str, float]:
    trajectory_covers = 0
    naive_state_covers = 0
    trajectory_half_width = 0.0
    naive_half_width = 0.0
    for _ in range(trials):
        trajectory_component = effect_generator(rng, trajectories)
        state_residual = rng.normal(scale=0.35, size=(trajectories, states_per_trajectory))
        state_means = trajectory_component[:, None] + state_residual
        trajectory_estimates = state_means.mean(axis=1)

        trajectory_lower, trajectory_upper = t_interval(trajectory_estimates)
        naive_lower, naive_upper = t_interval(state_means.reshape(-1))
        trajectory_covers += int(trajectory_lower <= 0.0 <= trajectory_upper)
        naive_state_covers += int(naive_lower <= 0.0 <= naive_upper)
        trajectory_half_width += (trajectory_upper - trajectory_lower) / 2.0
        naive_half_width += (naive_upper - naive_lower) / 2.0

    return {
        "trajectory_t_coverage": trajectory_covers / trials,
        "naive_state_t_coverage": naive_state_covers / trials,
        "trajectory_t_false_positive_rate": 1.0 - trajectory_covers / trials,
        "naive_state_t_false_positive_rate": 1.0 - naive_state_covers / trials,
        "trajectory_t_mean_half_width": trajectory_half_width / trials,
        "naive_state_t_mean_half_width": naive_half_width / trials,
    }


def main() -> None:
    args = parse_args()
    if args.trials < 100:
        raise ValueError("coverage audit requires at least 100 trials")
    rng = np.random.default_rng(args.seed)
    rows = []
    for trajectories in (8, 12, 16, 24):
        for scenario, generator in SCENARIOS.items():
            result = simulate(
                trials=args.trials,
                trajectories=trajectories,
                states_per_trajectory=24,
                effect_generator=generator,
                rng=rng,
            )
            rows.append(
                {
                    "scenario": scenario,
                    "trajectories": trajectories,
                    "states_per_trajectory": 24,
                    **result,
                }
            )

    output = {
        "schema_version": SCHEMA_VERSION,
        "seed": args.seed,
        "trials_per_cell": args.trials,
        "target_coverage": 0.95,
        "design": {
            "trajectory_component": "scenario-specific independent top-level effect",
            "within_trajectory_state_residual_sd": 0.35,
            "estimand": "zero population mean",
            "trajectory_interval": "Student t over trajectory means",
            "negative_control": "incorrect Student t over all state means as if independent",
        },
        "rows": rows,
        "interpretation_limits": [
            "simulation coverage is conditional on the declared generators",
            "rare-mixture failure cannot be repaired by adding states within the same trajectories",
            "the audit informs a minimum design but does not prove the real trajectory law",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()

