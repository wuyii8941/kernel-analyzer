#!/usr/bin/env python3
"""Validate the corrected certificate through its full vector-to-decision path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kernel_analyzer.training_equivalence import (  # noqa: E402
    classify_fixed_suite_update_equivalence,
    fixed_suite_total_rms_from_joint_gram,
    simultaneous_intervals_from_joint_gram,
)


MARGINS = {"additive": 0.001, "repair_aligned": 0.01, "residual_direction": 0.001}
TOTAL_RMS_MARGIN = 0.01


def gram(effects: np.ndarray, repairs: np.ndarray) -> dict:
    return {
        "effect_effect": (effects @ effects.T).tolist(),
        "repair_repair": (repairs @ repairs.T).tolist(),
        "effect_repair": (effects @ repairs.T).tolist(),
    }


def classify(effects: np.ndarray, repairs: np.ndarray) -> dict:
    joint = gram(effects, repairs)
    intervals = simultaneous_intervals_from_joint_gram(joint)
    return classify_fixed_suite_update_equivalence(
        intervals,
        MARGINS,
        total_rms=fixed_suite_total_rms_from_joint_gram(joint),
        total_rms_margin=TOTAL_RMS_MARGIN,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repairs = np.tile(np.array([1.0, 0.0, 0.0]), (32, 1))

    direction_shift = np.zeros((32, 3))
    direction_shift[:16, 1] = 1e-4
    direction_shift[16:, 2] = 10.0
    direction_shift_result = classify(direction_shift, repairs)

    centered_high_energy = np.zeros((32, 3))
    centered_high_energy[:16, 1] = 1e-4
    centered_high_energy[16:, 2] = np.tile([1.0, -1.0], 8)
    centered_high_energy_result = classify(centered_high_energy, repairs)

    material_residual = np.tile(np.array([0.0, 0.002, 0.0]), (32, 1))
    material_residual_result = classify(material_residual, repairs)

    scales = np.tile([0.1, 10.0], 16)
    gains = np.tile([0.02, 0.002], 16)
    heterogeneous_repairs = np.stack([scales, np.zeros(32)], axis=1)
    heterogeneous_effects = gains[:, None] * heterogeneous_repairs
    heterogeneous_intervals = simultaneous_intervals_from_joint_gram(
        gram(heterogeneous_effects, heterogeneous_repairs)
    )
    observed_aligned = sum(heterogeneous_intervals["repair_aligned"]) / 2.0
    expected_aligned = float(
        np.sum(gains[16:] * scales[16:] ** 2) / np.sum(scales[16:] ** 2)
    )

    boundary_results = {}
    zero_intervals = {name: [0.0, 0.0] for name in MARGINS}
    for label, value in (("below", 0.0095), ("at", 0.01), ("above", 0.0105)):
        boundary_results[label] = classify_fixed_suite_update_equivalence(
            zero_intervals,
            MARGINS,
            total_rms=value,
            total_rms_margin=TOTAL_RMS_MARGIN,
        )["decision"]

    gates = {
        "orthogonal_direction_shift_rejected": (
            direction_shift_result["decision"]
            == "FIXED_SUITE_UPDATE_ENERGY_EXCEEDS_MARGIN"
        ),
        "large_centered_energy_rejected": (
            centered_high_energy_result["decision"]
            == "FIXED_SUITE_UPDATE_ENERGY_EXCEEDS_MARGIN"
        ),
        "material_residual_direction_rejected": (
            material_residual_result["decision"] == "MATERIAL_EFFECT"
        ),
        "aligned_effect_uses_ratio_of_sums": bool(
            np.isclose(observed_aligned, expected_aligned)
            and not np.isclose(observed_aligned, float(np.mean(gains[16:])))
        ),
        "energy_margin_boundary_is_fail_closed": boundary_results == {
            "below": "FIXED_SUITE_UPDATE_EQUIVALENT",
            "at": "INCONCLUSIVE",
            "above": "FIXED_SUITE_UPDATE_ENERGY_EXCEEDS_MARGIN",
        },
    }
    payload = {
        "schema": "kernel-analyzer-fixed-suite-update-equivalence-validation-v2",
        "status": "GO" if all(gates.values()) else "NO_GO",
        "gates": gates,
        "margins": {**MARGINS, "full_update_rms": TOTAL_RMS_MARGIN},
        "observations": {
            "orthogonal_direction_shift": direction_shift_result,
            "large_centered_unseen_direction": centered_high_energy_result,
            "material_residual_direction": material_residual_result,
            "heterogeneous_repair_energy": {
                "observed_aligned_center": observed_aligned,
                "ratio_of_sums_target": expected_aligned,
                "mean_of_statewise_ratios": float(np.mean(gains[16:])),
            },
            "energy_margin_boundary": boundary_results,
        },
        "claim_boundary": (
            "This deterministic validation exercises the high-dimensional Gram-to-decision "
            "path. It closes known geometric blind spots; it is not a population-power study."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "gates": gates}))


if __name__ == "__main__":
    main()
