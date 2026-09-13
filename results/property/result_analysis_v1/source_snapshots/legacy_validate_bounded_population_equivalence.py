#!/usr/bin/env python3
"""Exercise finite-sample population decisions from synthetic update vectors.

This validation deliberately includes rare, unobserved boundary events that
make the legacy studentized rule unsafe.  It calls the production bounded
energy and aligned functions after constructing original-coordinate X, B and
A from vectors; it does not generate already-decided confidence intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from kernel_analyzer.training_equivalence import (
    bounded_population_aligned_equivalence,
    bounded_population_total_energy_equivalence,
)
from kernel_analyzer.training_numerical_analysis import analyze_bounded_population_artifact


def _wilson(successes: int, repetitions: int) -> list[float]:
    z = 1.959963984540054
    rate = successes / repetitions
    denominator = 1.0 + z * z / repetitions
    center = (rate + z * z / (2.0 * repetitions)) / denominator
    half = z * math.sqrt(
        rate * (1.0 - rate) / repetitions + z * z / (4.0 * repetitions**2)
    ) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def _population_artifact(effect: np.ndarray, repair: np.ndarray) -> dict:
    count = len(effect)
    return {
        "case_id": "bounded-population-synthetic",
        "contrast_id": "CONTROLLED_SYNTHETIC_UPDATE",
        "status": "COMPLETE",
        "runtime_boundary": {"kind": "SYNTHETIC_NOT_A_NATURAL_TRAINING_CASE"},
        "parameter_write_protocol": {"version": "adamw-readback-v2", "synthetic": True},
        "state_ids": list(range(count)),
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


def _population_protocol(*, x_max: float, b_max: float) -> dict:
    return {
        "schema": "bounded-population-synthetic-protocol-v1",
        "claim_scope": "DECLARED_STATE_POPULATION_UPDATE",
        "primary_stage": "PARAMETER_WRITE",
        "data_use": "METHOD_VALIDATION",
        "mandatory_population_endpoints": ["TOTAL_RMS", "REPAIR_ALIGNED"],
        "population_margins": {"full_update_rms": 0.01, "repair_aligned": 0.01},
        "population_energy_bounds": {
            "effect_energy_upper_bound": x_max,
            "repair_energy_upper_bound": b_max,
            "provenance": "predeclared synthetic finite support",
            "fixed_before_observation": True,
        },
        "family_alpha": 0.05,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2000)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("use a new output below /data1/tzh")
    if args.repetitions < 100:
        parser.error("at least 100 repetitions are required")

    rng = np.random.default_rng(20260908)
    alpha = 0.05
    margin = 0.01
    n = 64
    rare_probability = 1.0 / (20.0 * n)
    boundary_counts = {"energy_false_equivalent": 0, "aligned_false_equivalent": 0}
    legacy_trap_count = 0

    for _ in range(args.repetitions):
        repair = np.ones((n, 1), dtype=np.float64)

        energy_event = rng.binomial(1, rare_probability, size=n).astype(np.float64)
        energy_effect = margin * np.sqrt(energy_event / rare_probability)[:, None]
        energy_x = np.einsum("ij,ij->i", energy_effect, energy_effect)
        energy_b = np.einsum("ij,ij->i", repair, repair)
        energy_result = bounded_population_total_energy_equivalence(
            energy_x,
            energy_b,
            rms_margin=margin,
            effect_energy_upper_bound=margin**2 / rare_probability,
            repair_energy_upper_bound=1.0,
            bound_provenance="predeclared synthetic finite support",
            alpha=alpha,
        )
        boundary_counts["energy_false_equivalent"] += energy_result["decision"] == "EQUIVALENT"

        aligned_event = rng.binomial(1, rare_probability, size=n).astype(np.float64)
        aligned_effect = margin * (aligned_event / rare_probability)[:, None]
        aligned_x = np.einsum("ij,ij->i", aligned_effect, aligned_effect)
        aligned_b = np.einsum("ij,ij->i", repair, repair)
        aligned_a = np.einsum("ij,ij->i", aligned_effect, repair)
        aligned_result = bounded_population_aligned_equivalence(
            aligned_a,
            aligned_b,
            margin=margin,
            effect_energy_upper_bound=(margin / rare_probability) ** 2,
            repair_energy_upper_bound=1.0,
            bound_provenance="predeclared synthetic finite support",
            alpha=alpha,
        )
        boundary_counts["aligned_false_equivalent"] += aligned_result["decision"] == "EQUIVALENT"
        legacy_trap_count += not np.any(energy_event)

    exact_repair = np.ones((2048, 1), dtype=np.float64)
    exact_effect = np.zeros_like(exact_repair)
    exact_x = np.einsum("ij,ij->i", exact_effect, exact_effect)
    exact_b = np.einsum("ij,ij->i", exact_repair, exact_repair)
    exact_a = np.einsum("ij,ij->i", exact_effect, exact_repair)
    exact_energy = bounded_population_total_energy_equivalence(
        exact_x,
        exact_b,
        rms_margin=margin,
        effect_energy_upper_bound=0.0,
        repair_energy_upper_bound=1.0,
        bound_provenance="protocol-enforced exact synthetic identity",
        alpha=alpha,
    )
    exact_aligned = bounded_population_aligned_equivalence(
        exact_a,
        exact_b,
        margin=margin,
        effect_energy_upper_bound=0.0,
        repair_energy_upper_bound=1.0,
        bound_provenance="protocol-enforced exact synthetic identity",
        alpha=alpha,
    )
    exact_analysis = analyze_bounded_population_artifact(
        _population_artifact(exact_effect, exact_repair),
        _population_protocol(x_max=0.0, b_max=1.0),
    )
    unseen_boundary_effect = np.zeros((n, 1), dtype=np.float64)
    unseen_boundary_repair = np.ones((n, 1), dtype=np.float64)
    unseen_boundary_analysis = analyze_bounded_population_artifact(
        _population_artifact(unseen_boundary_effect, unseen_boundary_repair),
        _population_protocol(x_max=(margin / rare_probability) ** 2, b_max=1.0),
    )

    rows = []
    for name, count in boundary_counts.items():
        rows.append(
            {
                "endpoint": name,
                "boundary_false_equivalence_count": int(count),
                "repetitions": args.repetitions,
                "rate": count / args.repetitions,
                "monte_carlo_interval_95": _wilson(count, args.repetitions),
                "nominal_alpha": alpha,
            }
        )
    passed = all(row["monte_carlo_interval_95"][1] <= alpha for row in rows)
    passed = passed and exact_energy["decision"] == exact_aligned["decision"] == "EQUIVALENT"
    passed = passed and exact_analysis["equivalence_decision"] == "EQUIVALENT"
    passed = passed and unseen_boundary_analysis["equivalence_decision"] == "INCONCLUSIVE"
    payload = {
        "schema": "bounded-population-equivalence-validation-v1",
        "status": "PASS" if passed else "FAIL",
        "production_functions": [
            "bounded_population_total_energy_equivalence",
            "bounded_population_aligned_equivalence",
        ],
        "vector_to_sufficient_statistics_path": True,
        "rare_event_probability": rare_probability,
        "rare_event_population_is_on_margin": True,
        "all_zero_observation_count": int(legacy_trap_count),
        "boundary_results": rows,
        "exact_identity_controls": {
            "energy": exact_energy,
            "aligned": exact_aligned,
        },
        "full_analysis_path_controls": {
            "exact_identity": exact_analysis,
            "unobserved_rare_boundary": unseen_boundary_analysis,
        },
        "limitations": (
            "The finite-sample guarantee requires independent units and protocol-level finite "
            "energy bounds fixed before observation. Sample extrema are not valid bounds."
        ),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path("src/kernel_analyzer/training_equivalence.py"))
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": payload["status"], "boundary_results": rows}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
