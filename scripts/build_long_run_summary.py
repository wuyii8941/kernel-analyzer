#!/usr/bin/env python3
"""Create a compact, claim-bounded summary of the Phi long-run experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def artifact(path: str) -> dict[str, str]:
    payload = (ROOT / path).read_bytes()
    return {"path": path, "sha256": hashlib.sha256(payload).hexdigest()}


def mean_and_std(values: list[float]) -> dict[str, float]:
    mean = fmean(values)
    variance = fmean([(value - mean) ** 2 for value in values])
    return {"mean": mean, "population_std": math.sqrt(variance)}


def linear_slope(xs: list[float], ys: list[float]) -> float:
    x_mean = fmean(xs)
    y_mean = fmean(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    if denominator == 0:
        return 0.0
    return sum(
        (x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)
    ) / denominator


def summarize_training_windows(result: dict[str, Any], window_steps: int = 128) -> list[dict[str, Any]]:
    rows = result["train_rows"]
    windows: list[dict[str, Any]] = []
    for start in range(0, len(rows), window_steps):
        block = rows[start : start + window_steps]
        if len(block) != window_steps:
            continue
        steps = [float(row["step"]) for row in block]
        candidate_losses = [float(row["candidate_loss"]) for row in block]
        repair_losses = [float(row["repair_loss"]) for row in block]
        loss_gaps = [float(row["loss_gap_candidate_minus_repair"]) for row in block]
        candidate_updates = [float(row["candidate_update_l2"]) for row in block]
        repair_updates = [float(row["repair_update_l2"]) for row in block]
        windows.append({
            "step_start": int(block[0]["step"]),
            "step_end": int(block[-1]["step"]),
            "candidate_train_loss": mean_and_std(candidate_losses),
            "repair_train_loss": mean_and_std(repair_losses),
            "paired_train_loss_gap": mean_and_std(loss_gaps),
            "candidate_train_loss_slope_per_step": linear_slope(steps, candidate_losses),
            "repair_train_loss_slope_per_step": linear_slope(steps, repair_losses),
            "candidate_update_l2": mean_and_std(candidate_updates),
            "repair_update_l2": mean_and_std(repair_updates),
            "parameter_distance_l2_start": float(block[0]["parameter_distance_l2"]),
            "parameter_distance_l2_end": float(block[-1]["parameter_distance_l2"]),
        })
    return windows


def summarize_convergence_attempt(
    result: dict[str, Any], *, schedule: str
) -> dict[str, Any]:
    final = result["final"]
    checkpoints = result["checkpoints"]
    convergence_points = int(result["protocol"]["convergence_points"])
    recent = checkpoints[-convergence_points:]
    windows = summarize_training_windows(result)
    return {
        "schedule": schedule,
        "steps": result["steps_completed"],
        "status": result["status"],
        "frozen_convergence_gate": {
            "validation_every_steps": result["protocol"]["validation_every"],
            "consecutive_checkpoints": convergence_points,
            "relative_loss_span_threshold": result["protocol"]["loss_relative_span_threshold"],
            "passed": result["loss_plateau_reached"],
        },
        "final": {
            "candidate_loss": final["candidate_loss"],
            "repair_loss": final["repair_loss"],
            "loss_gap": final["loss_gap_candidate_minus_repair"],
            "gradient_difference_l2": final["mean_gradient_difference_l2"],
            "parameter_distance_l2": final["parameter_distance_l2"],
            "candidate_recent_relative_loss_span": final["candidate_recent_relative_loss_span"],
            "repair_recent_relative_loss_span": final["repair_recent_relative_loss_span"],
        },
        "last_validation_checkpoints": [
            {
                "step": row["step"],
                "candidate_loss": row["candidate_loss"],
                "repair_loss": row["repair_loss"],
                "loss_gap": row["loss_gap_candidate_minus_repair"],
                "candidate_mean_gradient_l2": row["candidate_mean_gradient_l2"],
                "repair_mean_gradient_l2": row["repair_mean_gradient_l2"],
                "mean_gradient_difference_l2": row["mean_gradient_difference_l2"],
                "parameter_distance_l2": row["parameter_distance_l2"],
            }
            for row in recent
        ],
        "rolling_train_windows": windows,
        "last_512_train_steps": {
            "candidate_train_loss": mean_and_std(
                [float(row["candidate_loss"]) for row in result["train_rows"][-512:]]
            ),
            "repair_train_loss": mean_and_std(
                [float(row["repair_loss"]) for row in result["train_rows"][-512:]]
            ),
            "paired_train_loss_gap": mean_and_std(
                [float(row["loss_gap_candidate_minus_repair"]) for row in result["train_rows"][-512:]]
            ),
            "candidate_update_l2": mean_and_std(
                [float(row["candidate_update_l2"]) for row in result["train_rows"][-512:]]
            ),
            "repair_update_l2": mean_and_std(
                [float(row["repair_update_l2"]) for row in result["train_rows"][-512:]]
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    long_path = "results/property/long_horizon_v2/phi_warm128_measure512.json"
    cold_path = "results/property/long_horizon_v1/phi_same_states_cold32.json"
    reset_path = "results/property/long_horizon_v1/phi_warm_weights_reset_moments32.json"
    constant_path = "results/property/convergence_v1/phi_carrier_adamw_convergence.json"
    cosine_path = "results/property/convergence_v2/phi_carrier_adamw_cosine_convergence.json"
    long = load(long_path)
    cold = load(cold_path)
    reset = load(reset_path)
    constant = load(constant_path)
    cosine = load(cosine_path)

    first_window = long["windows"][0]
    payload = {
        "schema": "kernel-analyzer-long-run-summary-v2",
        "status": "MEDIUM_HORIZON_CONFIRMED_CONVERGENCE_UNRESOLVED",
        "case_id": "phi4_seq64_lmhead_dx",
        "warm_state_direct_result": {
            "warmup_steps": long["protocol"]["warmup_steps"],
            "measurement_steps": long["protocol"]["measurement_steps"],
            "A512": long["full"]["coherence_amplification"],
            "null95_A512": long["full"]["sign_flip_null"]["upper_95"],
            "disjoint_32_step_windows_above_null": sum(
                bool(row["above_sign_flip_95"]) for row in long["windows"]
            ),
            "disjoint_32_step_window_count": len(long["windows"]),
            "lag1_correlation": long["full"]["lag_correlation"][0]["normalized_correlation"],
            "lag64_correlation": long["full"]["lag_correlation"][63]["normalized_correlation"],
            "effective_rank": long["measurement_geometry"]["effective_rank_participation_ratio"],
            "top_direction_energy_fraction": long["measurement_geometry"]["top_16_spectrum_energy_fractions"][0],
            "measurement_geometry": long["measurement_geometry"]["kind"],
            "classification": "WARM_STATE_512_STEP_DIRECT_DIRECTIONAL",
        },
        "optimizer_state_contrast": {
            "same_measurement_states_cold_A32": cold["full"]["coherence_amplification"],
            "same_measurement_states_warm_A32": first_window["coherence_amplification"],
            "warm_weights_reset_moments_A32": reset["full"]["coherence_amplification"],
            "interpretation": (
                "The large warm-state amplification is not explained by the measurement text. "
                "Resetting moments while retaining warmed weights returns A32 near the cold result, "
                "so evolved AdamW memory is necessary for the measured amplification. Sufficiency "
                "is not claimed."
            ),
        },
        "convergence_attempts": [
            summarize_convergence_attempt(constant, schedule="constant_lr_1e-4"),
            summarize_convergence_attempt(cosine, schedule="cosine_decay_1e-4_to_0"),
        ],
        "scientific_conclusion": {
            "supported": (
                "The exact Phi direct effective-update difference remains directional for 512 fresh "
                "states after 128 AdamW warmup steps, including every late disjoint window."
            ),
            "not_supported": (
                "Neither 4096-step paired run passed the frozen validation-loss plateau rule. The "
                "current evidence therefore does not establish different converged loss, gradient, "
                "or parameter solutions."
            ),
            "long_run_label": "CONVERGENCE_OUTCOME_UNRESOLVED",
        },
        "source_artifacts": [
            artifact(long_path),
            artifact(cold_path),
            artifact(reset_path),
            artifact(constant_path),
            artifact(cosine_path),
            artifact(long["measurement_geometry"]["gram_artifact"]),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "status": payload["status"]}))


if __name__ == "__main__":
    main()
