#!/usr/bin/env python3
"""Compare information added by each analysis layer on shared controlled data."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from kernel_analyzer.training_equivalence import (
    classify_fixed_suite_update_equivalence,
    fixed_suite_total_rms_from_joint_gram,
    simultaneous_intervals_from_joint_gram,
)

ROOT = Path(__file__).resolve().parents[1]
MARGINS = {"additive": .001, "repair_aligned": .01, "residual_direction": .001}
TOTAL_MARGIN = .01


def load(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gram(effects: np.ndarray, repairs: np.ndarray) -> dict:
    return {
        "effect_effect": (effects @ effects.T).tolist(),
        "repair_repair": (repairs @ repairs.T).tolist(),
        "effect_repair": (effects @ repairs.T).tolist(),
    }


def rms(effects: np.ndarray, repairs: np.ndarray, indices=slice(None)) -> float:
    return math.sqrt(float(np.sum(effects[indices] ** 2) / np.sum(repairs[indices] ** 2)))


def profile_labels(intervals: dict | None) -> list[str]:
    if intervals is None:
        return []
    labels = []
    for name, interval in intervals.items():
        margin = MARGINS[name]
        if interval[0] > margin or interval[1] < -margin:
            labels.append(name)
    return labels


def evaluate(name: str, local: np.ndarray, update: np.ndarray, repairs: np.ndarray,
             structure: list[str], note: str) -> dict:
    joint = gram(update, repairs)
    update_rms = fixed_suite_total_rms_from_joint_gram(joint)
    try:
        intervals = simultaneous_intervals_from_joint_gram(joint)
    except ValueError:
        intervals = None
    full = classify_fixed_suite_update_equivalence(
        intervals, MARGINS, total_rms=update_rms,
        total_rms_margin=TOTAL_MARGIN,
        exact_identity_verified=bool(np.count_nonzero(update) == 0),
    )
    conf = slice(16, 32)
    local_rms = rms(local, repairs, conf)
    mean_severity = float(np.linalg.norm(update[conf].mean(axis=0)) /
                          np.sqrt(np.mean(np.sum(repairs[conf] ** 2, axis=1))))
    first_state_rms = float(np.linalg.norm(update[16]) / np.linalg.norm(repairs[16]))
    local_candidate = repairs + local
    local_allclose = bool(np.allclose(local_candidate, repairs, rtol=1e-5, atol=1e-8))
    ground_equivalent = update_rms < TOTAL_MARGIN
    expected_decision = "EQUIVALENT" if ground_equivalent else "NON_EQUIVALENT"
    return {
        "name": name, "note": note, "declared_structure": structure,
        "controlled_ground_truth": {
            "fixed_suite_update_equivalent": ground_equivalent,
            "basis": "directly constructed confirmation update RMS versus 1% margin",
        },
        "methods": {
            "LOCAL_ALLCLOSE": {"passed": local_allclose},
            "LOCAL_RMS": {"value": local_rms, "inside_1pct": local_rms < TOTAL_MARGIN},
            "UPDATE_RMS": {"value": update_rms, "decision": expected_decision},
            "UPDATE_MEAN_ONLY": {
                "value": mean_severity,
                "inside_additive_margin": mean_severity < MARGINS["additive"],
            },
            "ONE_CONFIRMATION_STATE": {
                "value": first_state_rms,
                "inside_1pct": first_state_rms < TOTAL_MARGIN,
            },
            "FULL_ANALYSIS": {
                "decision": full["decision"], "reason": full["reason"],
                "direction_intervals": intervals,
                "confirmed_structure": profile_labels(intervals),
            },
        },
    }


def scenarios() -> list[dict]:
    rng = np.random.default_rng(20260913)
    repairs = rng.normal(size=(32, 128))
    repairs /= np.linalg.norm(repairs, axis=1, keepdims=True)
    zero = np.zeros_like(repairs)
    noise = rng.normal(size=repairs.shape)
    noise /= np.sqrt(np.mean(np.sum(noise ** 2, axis=1)))

    alternating = np.zeros_like(repairs)
    alternating[:, 3] = np.tile([.02, -.02], 16)
    additive = np.zeros_like(repairs)
    additive[:, 4] = .02
    residual = np.empty_like(repairs)
    direction = np.zeros(repairs.shape[1]); direction[5] = 1.0
    for index, repair in enumerate(repairs):
        value = direction - np.dot(direction, repair) * repair
        residual[index] = .02 * value / np.linalg.norm(value)
    single_miss = .02 * noise
    single_miss[16] = 0.0
    drift = np.zeros_like(repairs)
    drift[:16, 6] = .001
    drift[16:, 7] = .02

    definitions = [
        ("exact_identity", zero, zero, [], "program identity check"),
        ("large_local_small_update", .02 * noise, .005 * noise, [],
         "local difference exceeds 1%, update remains below 1%"),
        ("small_local_material_update", zero, .02 * repairs, ["repair_aligned"],
         "local outputs match while downstream update has 2% rotating scaling"),
        ("zero_mean_material_energy", zero, alternating, [],
         "large update energy with exactly alternating signs"),
        ("fixed_additive_direction", zero, additive, ["additive", "residual_direction"],
         "shared parameter-space direction"),
        ("state_aligned_scaling", zero, .02 * repairs, ["repair_aligned"],
         "normal update direction rotates but is scaled in every state"),
        ("statewise_orthogonal_mean", zero, residual, ["additive", "residual_direction"],
         "coherent component after removing each state's repair-aligned part"),
        ("single_state_false_reassurance", zero, single_miss, [],
         "first confirmation state is zero but the suite has material energy"),
        ("calibration_confirmation_direction_change", zero, drift, [],
         "confirmation moves to a direction absent from calibration"),
    ]
    return [evaluate(name, local, update, repairs, structure, note)
            for name, local, update, structure, note in definitions]


def empirical_modification_evidence() -> list[dict]:
    block = load(ROOT / "results/property/numerical_coverage_v1/adamw8bit_full_model_block_probe_v1/result.json")
    block_train = load(ROOT / "results/property/numerical_coverage_v1/mamba_adamw8bit_training_confirmation_v1/verification.json")
    hybrid_component = load(ROOT / "results/property/numerical_coverage_v1/torchao_adamw8bit_component_mechanism_v1/summary.json")
    hybrid_train = load(ROOT / "results/property/numerical_coverage_v1/mamba_adamw8bit_hybrid_training_confirmation_v1/verification_v2.json")
    compensation = load(ROOT / "results/property/numerical_coverage_v1/adamw8bit_error_compensation_probe_v1/result.json")
    compensation_train = load(ROOT / "results/property/result_analysis_v1/same_path_training/verification.json")
    return [
        {
            "modification": "BLOCK64",
            "write_evidence": {"default_rms": block["mean_full_model_write_rms"]["block256"],
                               "modified_rms": block["mean_full_model_write_rms"]["block64"],
                               "improved_histories": "16/16"},
            "training_decision": block_train["modified_variant"]["decision"],
            "training_mean_default_minus_modified": block_train["modified_variant"]["mean"],
        },
        {
            "modification": "FP32_FIRST_MOMENT",
            "write_evidence": {
                "default_rms": hybrid_component["confirmation_relative_rms"]["DEFAULT_8BIT"]["PARAMETER_WRITE"],
                "modified_rms": hybrid_component["confirmation_relative_rms"]["FP32_FIRST_MOMENT"]["PARAMETER_WRITE"],
                "scope": "fixed x_proj gradient sequence",
            },
            "training_decision": hybrid_train["primary"]["decision"],
            "training_mean_default_minus_modified": hybrid_train["primary"]["mean"],
        },
        {
            "modification": "CROSS_STEP_RESIDUAL_COMPENSATION",
            "write_evidence": {"default_rms": compensation["mean_write_rms"]["default_block256"],
                               "modified_rms": compensation["mean_write_rms"]["compensated_block256"],
                               "improved_histories": "16/16"},
            "training_decision": compensation_train["primary"]["decision"],
            "training_mean_default_minus_modified": compensation_train["primary"]["mean"],
            "training_interval_95": compensation_train["primary"]["interval_95"],
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT):
        parser.error("choose a new output inside kernel-analyzer")
    source = Path(__file__).resolve()
    rows = scenarios()
    equivalence_methods = {
        "LOCAL_RMS": [], "UPDATE_RMS": [], "UPDATE_MEAN_ONLY": [],
        "ONE_CONFIRMATION_STATE": [], "FULL_ANALYSIS": [],
    }
    for row in rows:
        truth = row["controlled_ground_truth"]["fixed_suite_update_equivalent"]
        methods = row["methods"]
        equivalence_methods["LOCAL_RMS"].append(methods["LOCAL_RMS"]["inside_1pct"] == truth)
        equivalence_methods["UPDATE_RMS"].append((methods["UPDATE_RMS"]["decision"] == "EQUIVALENT") == truth)
        equivalence_methods["UPDATE_MEAN_ONLY"].append(methods["UPDATE_MEAN_ONLY"]["inside_additive_margin"] == truth)
        equivalence_methods["ONE_CONFIRMATION_STATE"].append(methods["ONE_CONFIRMATION_STATE"]["inside_1pct"] == truth)
        full_equivalent = methods["FULL_ANALYSIS"]["decision"] in {
            "FIXED_SUITE_UPDATE_EQUIVALENT", "EQUIVALENT_UNDER_PROTOCOL",
            "EXACT_UPDATE_IDENTITY_ON_FIXED_SUITE",
        }
        equivalence_methods["FULL_ANALYSIS"].append(full_equivalent == truth)
    result = {
        "schema": "method-incremental-value-benchmark-v1", "status": "COMPLETE",
        "data_use": "CONTROLLED_SHARED_DATA_AND_RETROSPECTIVE_EMPIRICAL_COMPARISON",
        "source_sha256": {str(source): sha(source)},
        "empirical_evidence_sha256": {
            str(path): sha(path) for path in [
                ROOT / "results/property/numerical_coverage_v1/adamw8bit_full_model_block_probe_v1/result.json",
                ROOT / "results/property/numerical_coverage_v1/mamba_adamw8bit_training_confirmation_v1/verification.json",
                ROOT / "results/property/numerical_coverage_v1/torchao_adamw8bit_component_mechanism_v1/summary.json",
                ROOT / "results/property/numerical_coverage_v1/mamba_adamw8bit_hybrid_training_confirmation_v1/verification_v2.json",
                ROOT / "results/property/numerical_coverage_v1/adamw8bit_error_compensation_probe_v1/result.json",
                ROOT / "results/property/result_analysis_v1/same_path_training/verification.json",
            ]
        },
        "controlled_scenarios": rows,
        "fixed_suite_equivalence_correct_count": {
            name: sum(values) for name, values in equivalence_methods.items()
        },
        "scenario_count": len(rows),
        "empirical_modification_evidence": empirical_modification_evidence(),
        "conclusions": [
            "For fixed-suite full-space equivalence, full analysis does not outperform update RMS because update RMS is its mandatory envelope.",
            "Stage comparison, multiple states, and direction/scaling diagnostics answer localization and effect-form questions that update RMS alone cannot answer.",
            "The empirical optimizer modifications show that reducing one-step update RMS is not sufficient to predict training loss improvement.",
            "Current saved allclose comparisons do not establish that local allclose missed a large write difference.",
        ],
        "limitations": [
            "Controlled labels are injected, not natural-kernel ground truth.",
            "Empirical modifications used different frozen data sets and cannot be ranked as one randomized comparison.",
            "This benchmark does not implement or claim RENDER/TTrace results.",
            "No AUROC is reported because the methods answer different declared tasks.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"],
                      "correct": result["fixed_suite_equivalence_correct_count"]}))


if __name__ == "__main__":
    main()
