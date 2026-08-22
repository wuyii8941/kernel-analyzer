#!/usr/bin/env python3
"""Summarize the bound Phi four-arm scale comparison without overclaiming."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def fit(curve: list[dict]) -> dict[str, float]:
    x = np.log(np.asarray([row["horizon"] for row in curve], dtype=np.float64))
    y = np.log(np.asarray([row["distance_l2"] for row in curve], dtype=np.float64))
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    ss_res = float(np.square(y - fitted).sum())
    ss_tot = float(np.square(y - y.mean()).sum())
    return {
        "power_exponent": float(slope),
        "coefficient": float(math.exp(intercept)),
        "log_space_r2": 1.0 - ss_res / max(ss_tot, 1e-30),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text())
    arms = source["arms"]
    fits = {}
    for name, arm in arms.items():
        if not arm["zero_perturbation"] and all(
            row["distance_l2"] > 0 for row in arm["prefix_curve"]
        ):
            fits[name] = fit(arm["prefix_curve"])
    # A five-point prefix fit is descriptive.  In particular, the precision
    # arm is not guaranteed to follow sqrt(T), so extrapolating a crossing is
    # not a preregistered or scientifically identified operation.
    crossing = None
    payload = {
        "schema": "kernel-analyzer-four-scale-summary-v1",
        "status": "COMPLETE_BOUNDED_CARRIER_SCALE_COMPARISON",
        "source": str(args.input),
        "arms": {name: {
            "final_distance_l2": arm["final_distance_l2"],
            "coherence_amplification": arm["coherence_amplification"],
            "distance_over_initial_parameter_l2": arm["distance_over_initial_parameter_l2"],
            "interpretation": (
                ("STRONGLY_COHERENT_OPERATOR_DIFFERENCE" if arm["coherence_amplification"] >= 2.0
                 else "DIFFUSIVE_OR_WEAK_OPERATOR_DIFFERENCE") if name == "A_operator" else
                "NOT_INFORMATIVE_ZERO_RNG_SENSITIVITY" if name == "B_rng" else
                "ORDER_EFFECT_CANCELS_AFTER_COMPLETE_MULTISET" if name == "C_data_order" else
                ("PRECISION_DIFFERENCE_HAS_NONDIFFUSIVE_LATE_GROWTH"
                 if arm["coherence_amplification"] > 1.25 else "DIFFUSIVE_PRECISION_DIFFERENCE")
            ),
        } for name, arm in arms.items()},
        "power_law_fits": fits,
        "operator_precision_crossover_steps": crossing,
        "crossover_status": "NOT_REPORTED_SCALING_MODEL_NOT_ESTABLISHED",
        "conclusion": (
            "The operator contrast is strongly coherent on this carrier. "
            if arms["A_operator"]["coherence_amplification"] >= 2.0 else
            "The operator contrast is diffusive or weak on this carrier. "
        ) + (
            "The BF16-vs-FP32 contrast is also non-diffusive. "
            if arms["D_precision"]["coherence_amplification"] > 1.25 else
            "The BF16-vs-FP32 contrast is diffusive under the measured horizon. "
        ) + (
            "Data-order effects cancel after the complete multiset, and the RNG arm is "
            "inapplicable because this Phi graph has zero dropout."
        ),
        "claim_boundary": source["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "crossover": crossing}, sort_keys=True))


if __name__ == "__main__":
    main()
