#!/usr/bin/env python3
"""Validate the exact state-population exceedance endpoint end to end."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from kernel_analyzer.training_equivalence import (
    population_statewise_rms_exceedance_equivalence,
)
from kernel_analyzer.training_numerical_analysis import (
    analyze_population_exceedance_artifact,
)


def _artifact(effect: np.ndarray, repair: np.ndarray) -> dict:
    count = len(effect)
    return {
        "case_id": "population-exceedance-synthetic",
        "contrast_id": "CONTROLLED_SYNTHETIC_UPDATE",
        "status": "COMPLETE",
        "runtime_boundary": {"kind": "SYNTHETIC_NOT_A_NATURAL_TRAINING_CASE"},
        "parameter_write_protocol": {
            "version": "adamw-readback-v2",
            "measurement": "parameter_after_step_minus_parameter_before_step",
            "synthetic": True,
        },
        "state_ids": [f"state-{index}" for index in range(count)],
        "inference_unit_ids": [f"independent-{index}" for index in range(count)],
        "original_coordinate_statistics": {
            "PARAMETER_WRITE": [
                {
                    "effect_energy": float(u @ u),
                    "repair_energy": float(r @ r),
                    "effect_repair_inner_product": float(u @ r),
                    "nonzero_effect_coordinates": int(np.count_nonzero(u)),
                }
                for u, r in zip(effect, repair)
            ]
        },
    }


def _protocol(*, margin: float, maximum_probability: float) -> dict:
    return {
        "schema": "population-statewise-exceedance-validation-v1",
        "primary_stage": "PARAMETER_WRITE",
        "claim_scope": "DECLARED_STATE_POPULATION_UPDATE",
        "population_estimand": "STATEWISE_RMS_EXCEEDANCE_PROBABILITY",
        "statewise_rms_margin": margin,
        "maximum_exceedance_probability": maximum_probability,
        "repair_energy_floor": 0.0,
        "alpha": 0.05,
        "data_use": "METHOD_VALIDATION",
    }


def _binomial_probability(count: int, trials: int, probability: float) -> float:
    return (
        math.comb(trials, count)
        * probability**count
        * (1.0 - probability) ** (trials - count)
    )


def _boundary_false_equivalence_probability(
    trials: int, maximum_probability: float, alpha: float
) -> float:
    probability = 0.0
    for violations in range(trials + 1):
        result = population_statewise_rms_exceedance_equivalence(
            [1.0] * violations + [0.0] * (trials - violations),
            [1.0] * trials,
            statewise_rms_margin=1.0,
            maximum_exceedance_probability=maximum_probability,
            alpha=alpha,
        )
        if result["decision"] == "EQUIVALENT":
            probability += _binomial_probability(
                violations, trials, maximum_probability
            )
    return probability


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.output.exists() or not output.is_relative_to(Path("/data1/tzh")):
        parser.error("use a new output below /data1/tzh")

    alpha = 0.05
    boundary_checks = [
        {
            "independent_units": count,
            "maximum_exceedance_probability": probability,
            "exact_false_equivalence_probability": _boundary_false_equivalence_probability(
                count, probability, alpha
            ),
        }
        for count, probability in ((16, 0.20), (64, 0.05), (299, 0.01))
    ]

    rng = np.random.default_rng(20260910)
    repair = rng.normal(size=(64, 31))
    repair /= np.linalg.norm(repair, axis=1, keepdims=True)
    clean_effect = np.zeros_like(repair)
    clean = analyze_population_exceedance_artifact(
        _artifact(clean_effect, repair),
        _protocol(margin=0.01, maximum_probability=0.05),
    )

    # A direction orthogonal to the first two coordinates exercises the full
    # original-coordinate energy path rather than any learned projection.
    large_effect = np.zeros_like(repair)
    large_effect[:, 2] = 0.2
    large = analyze_population_exceedance_artifact(
        _artifact(large_effect, repair),
        _protocol(margin=0.01, maximum_probability=0.05),
    )

    zero_failure_298 = population_statewise_rms_exceedance_equivalence(
        [0.0] * 298,
        [1.0] * 298,
        statewise_rms_margin=0.01,
        maximum_exceedance_probability=0.01,
        alpha=alpha,
    )
    zero_failure_299 = population_statewise_rms_exceedance_equivalence(
        [0.0] * 299,
        [1.0] * 299,
        statewise_rms_margin=0.01,
        maximum_exceedance_probability=0.01,
        alpha=alpha,
    )

    passed = all(
        row["exact_false_equivalence_probability"] <= alpha + 1e-12
        for row in boundary_checks
    )
    passed = passed and clean["equivalence_decision"] == "EQUIVALENT"
    passed = passed and large["equivalence_decision"] == "NON_EQUIVALENT"
    passed = passed and zero_failure_298["decision"] == "INCONCLUSIVE"
    passed = passed and zero_failure_299["decision"] == "EQUIVALENT"

    payload = {
        "schema": "population-statewise-exceedance-validation-v1",
        "status": "PASS" if passed else "FAIL",
        "production_functions": [
            "population_statewise_rms_exceedance_equivalence",
            "analyze_population_exceedance_artifact",
        ],
        "validation_path": "HIGH_DIMENSIONAL_VECTORS_TO_ORIGINAL_COORDINATE_STATISTICS_TO_PRODUCTION_DECISION",
        "nominal_alpha": alpha,
        "exact_boundary_checks": boundary_checks,
        "full_path_controls": {
            "zero_effect": clean,
            "large_orthogonal_coordinate_effect": large,
        },
        "sample_size_boundary": {
            "target_maximum_exceedance_probability": 0.01,
            "zero_violations_298_units": zero_failure_298,
            "zero_violations_299_units": zero_failure_299,
        },
        "claim_boundary": (
            "This validates a finite-sample iid Bernoulli prevalence endpoint. "
            "It does not validate population mean Q, dependence between units, "
            "or the magnitude of effects beyond the statewise threshold."
        ),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path("src/kernel_analyzer/training_equivalence.py"),
                Path("src/kernel_analyzer/training_numerical_analysis.py"),
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "status": payload["status"],
        "boundary_checks": boundary_checks,
        "sample_size_boundary": {
            "n298": zero_failure_298["decision"],
            "n299": zero_failure_299["decision"],
        },
    }, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
