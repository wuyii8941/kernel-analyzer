#!/usr/bin/env python3
"""Run end-to-end synthetic checks through production analysis functions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kernel_analyzer.training_equivalence import (
    classify_fixed_suite_update_equivalence,
    fixed_suite_total_rms_from_joint_gram,
    simultaneous_intervals_from_joint_gram,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/property/training_numerical_analysis_v1/synthetic_validation.json"
MARGINS = {"additive": 0.001, "repair_aligned": 0.01, "residual_direction": 0.001}


def _evaluate(name: str, effects: np.ndarray, repairs: np.ndarray) -> dict:
    gram = {
        "effect_effect": effects @ effects.T,
        "repair_repair": repairs @ repairs.T,
        "effect_repair": effects @ repairs.T,
    }
    serial = {key: value.tolist() for key, value in gram.items()}
    try:
        intervals = simultaneous_intervals_from_joint_gram(serial)
        profile_status = "COMPLETE"
    except ValueError as error:
        intervals = None
        profile_status = f"NOT_ASSESSED: {error}"
    total = fixed_suite_total_rms_from_joint_gram(serial)
    decision = classify_fixed_suite_update_equivalence(
        intervals, MARGINS, total_rms=total, total_rms_margin=0.01,
        exact_identity_verified=bool(np.count_nonzero(effects) == 0),
    )
    effect_norms = np.linalg.norm(effects, axis=1)
    repair_norms = np.linalg.norm(repairs, axis=1)
    denominator = float(np.sqrt(np.sum(effect_norms * effect_norms)))
    old_direction_score = (
        float(np.linalg.norm(effects.sum(axis=0)) / denominator)
        if denominator > 0.0 else 0.0
    )
    return {
        "name": name, "total_rms": total, "intervals": intervals,
        "profile_status": profile_status, "decision": decision,
        "baseline_summaries": {
            "effect_rms": float(np.sqrt(np.mean(effect_norms * effect_norms))),
            "maximum_state_effect_over_repair_norm": float(np.max(np.divide(
                effect_norms, repair_norms,
                out=np.full_like(effect_norms, np.inf), where=repair_norms > 0.0,
            ))),
            "old_direction_score_A": old_direction_score,
            "interpretation": (
                "Descriptive baselines only; they do not by themselves establish "
                "equivalence, a bias mechanism, or a population confidence statement."
            ),
        },
    }


def main() -> None:
    rng = np.random.default_rng(20260905)
    repairs = rng.normal(size=(32, 64))
    repairs /= np.linalg.norm(repairs, axis=1, keepdims=True)
    identity = np.zeros_like(repairs)
    small = 0.002 * rng.normal(size=repairs.shape)
    scaling = 0.02 * repairs
    drift = np.zeros_like(repairs)
    drift[:16, 1] = 1e-4
    drift[16:, 2] = 1.0
    alternating = np.zeros_like(repairs)
    alternating[16:, 3] = np.tile([1.0, -1.0], 8)
    scenarios = [
        _evaluate("exact_identity", identity, repairs),
        _evaluate("small_nonzero", small, repairs),
        _evaluate("material_scaling", scaling, repairs),
        _evaluate("calibration_confirmation_orthogonal_drift", drift, repairs),
        _evaluate("large_zero_mean_energy", alternating, repairs),
    ]
    low_repair = repairs.copy()
    low_repair[16:, :] *= 1e-12
    heterogeneous = repairs * np.geomspace(1e-3, 1e3, 32)[:, None]
    heterogeneous_scaling = 0.02 * heterogeneous
    orthogonal = np.zeros_like(repairs)
    orthogonal[:, 5] = 0.02
    mixed = 0.015 * repairs
    mixed[:, 6] += 0.01
    heavy_tailed = 0.02 * rng.standard_t(df=3, size=repairs.shape)
    state_dependent_centered = np.zeros_like(repairs)
    state_dependent_centered[:, 7] = np.tile([0.02, -0.02], 16)
    scenarios.extend([
        _evaluate("repair_energy_near_zero", 0.002 * rng.normal(size=repairs.shape), low_repair),
        _evaluate("heterogeneous_repair_scaling", heterogeneous_scaling, heterogeneous),
        _evaluate("orthogonal_mean", orthogonal, repairs),
        _evaluate("mixed_scaling_and_orthogonal_mean", mixed, repairs),
        _evaluate("heavy_tailed_zero_mean", heavy_tailed, repairs),
        _evaluate("state_dependent_centered", state_dependent_centered, repairs),
    ])
    expected = {
        "exact_identity": "EXACT_UPDATE_IDENTITY_ON_FIXED_SUITE",
        "material_scaling": "MATERIAL_EFFECT",
        "calibration_confirmation_orthogonal_drift": "FIXED_SUITE_UPDATE_ENERGY_EXCEEDS_MARGIN",
        "large_zero_mean_energy": "FIXED_SUITE_UPDATE_ENERGY_EXCEEDS_MARGIN",
        "heterogeneous_repair_scaling": "MATERIAL_EFFECT",
        "orthogonal_mean": "MATERIAL_EFFECT",
        "mixed_scaling_and_orthogonal_mean": "MATERIAL_EFFECT",
        "state_dependent_centered": "FIXED_SUITE_UPDATE_ENERGY_EXCEEDS_MARGIN",
    }
    checks = {row["name"]: row["decision"]["decision"] == wanted for row in scenarios
              for wanted in [expected.get(row["name"])] if wanted is not None}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # A small repeated benchmark reports behavior near the energy boundary.  It
    # deliberately calls the same high-dimensional production path as above.
    monte_carlo = []
    for scale in (0.0, 0.005, 0.0095, 0.01, 0.0105, 0.02):
        counts = {}
        repetitions = 200
        for _ in range(repetitions):
            repeated_repairs = rng.normal(size=(32, 64))
            repeated_repairs /= np.linalg.norm(repeated_repairs, axis=1, keepdims=True)
            noise = rng.normal(size=(32, 64))
            noise /= np.sqrt(np.mean(np.sum(noise * noise, axis=1)))
            row = _evaluate("boundary", scale * noise, repeated_repairs)
            label = row["decision"]["decision"]
            counts[label] = counts.get(label, 0) + 1
        monte_carlo.append({
            "true_rms_scale": scale,
            "repetitions": repetitions,
            "decision_counts": counts,
            "monte_carlo_standard_error_upper_bound": 0.5 / np.sqrt(repetitions),
        })
    OUT.write_text(json.dumps({
        "schema": "kernel-analyzer-training-numerical-analysis-synthetic-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "production_path": [
            "high_dimensional_effect_and_repair", "joint_gram",
            "profile_intervals", "total_energy", "fixed_suite_decision",
        ],
        "checks": checks,
        "boundary_monte_carlo": monte_carlo,
        "scenarios": scenarios,
    }, indent=2, sort_keys=True) + "\n")
    if not all(checks.values()):
        raise SystemExit("synthetic production-path validation failed")


if __name__ == "__main__":
    main()
