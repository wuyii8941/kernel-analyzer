#!/usr/bin/env python3
"""Build the evidence audit for the bias/source/intervention/reuse plan."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from kernel_analyzer.bias_evidence import statewise_aligned_summary, total_rms_from_gram
from kernel_analyzer.training_bias_profile import matched_training_bias_profile

ROOT = Path(__file__).resolve().parents[1]


def read(path: str):
    return json.loads((ROOT / path).read_text())


def zero_mean_control() -> dict:
    repair = np.zeros((32, 8)); repair[:, 0] = 1
    effect = np.zeros_like(repair); effect[:16, 1] = .02
    effect[16::2, 1] = .02; effect[17::2, 1] = -.02
    profile = matched_training_bias_profile(
        effect, repair, calibration_indices=range(16), confirmation_indices=range(16, 32),
        inference_unit_ids=[f"unit-{index:02d}" for index in range(32)],
        minimum_independent_units=8, signflip_draws=9999, seed=20260914,
    )
    branches = profile["population_inference"]["branches"]
    q = math.sqrt(float(np.sum(effect[16:] ** 2) / np.sum(repair[16:] ** 2)))
    return {
        "construction": "confirmation effects alternate +2% and -2% in one fixed direction",
        "total_rms": q,
        "mean_effect_is_exactly_zero": bool(np.count_nonzero(effect[16:].sum(axis=0)) == 0),
        "branch_mean_confirmed": {
            name: bool(row.get("raw_confirmed", False)) for name, row in branches.items()
        },
        "result": "HIGH_ENERGY_NOT_MISLABELED_AS_MEAN_BIAS",
    }


def adamw8bit_evidence() -> dict:
    unit_dir = ROOT / "results/property/numerical_coverage_v1/adamw8bit_population_direction_v1/units"
    units = [json.loads(path.read_text()) for path in sorted(unit_dir.glob("unit-*.json"))]
    direction = read("results/property/numerical_coverage_v1/adamw8bit_population_direction_v1/result.json")
    reconstruction = read("results/property/result_analysis_v5/real_history_reconstruction/result.json")
    structure = read("results/property/result_analysis_v3/residual_structure/result.json")
    training = read("results/property/result_analysis_v4/iid_training_confirmation/verification.json")
    source_link = read("results/property/bias_proof_plan_v1/adamw8bit_source_link/result.json")
    if len(units) != 32 or direction["primary_endpoint"]["positive_count"] != 32:
        raise ValueError("AdamW8bit independent direction evidence is incomplete")
    aligned = statewise_aligned_summary(
        [row["effect_repair_inner_product"] for row in units],
        [row["repair_energy"] for row in units],
    )
    final_step = max(row["step"] for row in reconstruction["rows"])
    terminal = [row for row in reconstruction["rows"] if row["step"] == final_step]
    prediction = {}
    for mode in ("OFF", "ON"):
        selected = [row for row in terminal if row["mode"] == mode]
        prediction[mode] = {
            "row_count": len(selected),
            "maximum_storage_only_prediction_relative_error": max(
                math.sqrt(row["storage_only_prediction_error_energy"] / row["actual_error_energy"])
                if row["actual_error_energy"] else 0 for row in selected
            ),
        }
    primary = training["recomputed_primary"]
    return {
        "bias_result": {
            "classification": "SYSTEMATIC_REPAIR_ALIGNED_EFFECT_UNDER_DECLARED_HISTORY_POPULATION",
            "independent_history_count": len(units),
            "aligned": aligned,
            "scope": direction["claim_scope"],
            "not_claimed": ["nonzero full-vector mean", "mean bias as the unique cause of loss"],
        },
        "source_result": {
            "code_boundary": "blockwise first/second-moment save and later readback",
            "equation": "state error at t = beta * prior state error - saved residual + remaining evaluation error",
            "real_history_final_step": final_step,
            "storage_residual_prediction": prediction,
            "interpretation": "saved residual quantitatively explains moment-state differences on the measured real history; this is conditional propagation, not a population mean proof",
            "same_history_aligned_source_link": {
                "history_count": source_link["default_aligned"]["positive_direction_frequency"]["independent_unit_count"],
                "default_mean_statewise_gain": source_link["default_aligned"]["estimate"],
                "default_conditional_interval": source_link["default_aligned"]["conditional_student_interval"],
                "compensated_mean_statewise_gain": source_link["compensated_aligned"]["estimate"],
                "compensated_conditional_interval": source_link["compensated_aligned"]["conditional_student_interval"],
                "default_positive_count": source_link["default_aligned"]["positive_direction_frequency"]["positive_count"],
                "compensated_positive_count": source_link["compensated_aligned"]["positive_direction_frequency"]["positive_count"],
                "absolute_gain_reduction_count": source_link["absolute_aligned_reduction_prevalence"]["success_count"],
                "prediction_results": source_link["prediction_results"],
                "data_use": "post-hoc aligned source link on histories frozen for the earlier RMS probe",
            },
        },
        "intervention_result": {
            "fixed_history_count": structure["unit_count"],
            "correct_better_than_coordinate_count": structure["correct_better_than_coordinate_count"],
            "correct_better_than_time_count": structure["correct_better_than_time_count"],
            "mean_write_rms": structure["mean_write_rms"],
        },
        "training_result": {
            "independent_pair_count": primary["paired_finite_count"],
            "mean_off_minus_on": primary["mean"],
            "interval_95": primary["interval_95"],
            "decision": primary["decision"],
            "interpretation": "targeted readback improves loss under the declared experiment; it does not identify mean bias as the sole mediator",
        },
    }


def liger_reuse() -> dict:
    protocol = read("results/property/liger_fp32_chunk_order_v1/protocol.json")
    result = read("results/property/liger_fp32_chunk_order_v1/result.json")
    if protocol["status"] != "FROZEN_BEFORE_EMPIRICAL_RESULTS" or result["status"] != "COMPLETE":
        raise ValueError("Liger diagnostic reuse evidence is incomplete")
    rows = result["rows"]
    if not all(row["forward_and_hidden_gradient_bitwise_equal"] and row["matched_sham_exact"] for row in rows):
        raise ValueError("Liger source isolation invariant failed")
    stage_rms = {
        stage: total_rms_from_gram(profile["suite"]["joint_gram"], range(16, 32))
        for stage, profile in result["profiles"].items()
    }
    return {
        "operator_family": "LIGER_FUSED_LINEAR_CE_DW_ACCUMULATION_ORDER",
        "data_use": "FROZEN_BEFORE_EMPIRICAL_RESULTS_FIXED_SUITE",
        "independent_answer_basis": "implementations differ only in the declared order of the same 64 FP32 dW additions; sham and forward/hidden-gradient identity were required",
        "simple_local_check": "NO_DIFFERENCE_IN_FORWARD_OR_HIDDEN_GRADIENT",
        "confirmation_relative_rms": stage_rms,
        "full_diagnosis": {
            "parameter_gradient_direction_reproduced": result["profiles"]["PARAMETER_GRADIENT"]["population_inference"]["branches"]["additive"]["raw_confirmed"],
            "parameter_update_direction_reproduced": result["profiles"]["ADAMW_UPDATE"]["population_inference"]["branches"]["additive"]["raw_confirmed"],
            "predeclared_source_prediction": result["source_prediction"],
        },
        "incremental_value": "stage tracing and the predeclared order predictor locate a systematic, same-dtype source even though its update RMS is tiny; magnitude alone cannot provide that diagnosis",
        "limits": "one fixed suite and parameter; no population or material loss claim",
    }


def build_payload() -> dict:
    control = zero_mean_control()
    adam = adamw8bit_evidence()
    liger = liger_reuse()
    gates = {
        "main_case_bias_distinguished_from_energy": (
            adam["bias_result"]["aligned"]["positive_direction_frequency"]["decision"]
            == "DIRECTION_PREVALENCE_CONFIRMED" and not any(control["branch_mean_confirmed"].values())
        ),
        "source_propagation_and_aligned_effect_linked": (
            adam["source_result"]["storage_residual_prediction"]["OFF"]["maximum_storage_only_prediction_relative_error"] < 1e-4
            and adam["source_result"]["same_history_aligned_source_link"]["prediction_results"]["positive_default"]
            and adam["source_result"]["same_history_aligned_source_link"]["prediction_results"]["aligned_reduction"]
        ),
        "mechanism_and_training_claims_are_separate": (
            adam["training_result"]["decision"] == "MATERIAL_IMPROVEMENT"
        ),
        "second_operator_family_diagnostic_reuse": (
            liger["full_diagnosis"]["parameter_gradient_direction_reproduced"]
            and liger["full_diagnosis"]["predeclared_source_prediction"]["direction_repeated"]
        ),
    }
    return {
        "schema": "bias-proof-plan-audit-v1",
        "status": "COMPLETE" if all(gates.values()) else "INCOMPLETE",
        "gates": gates,
        "zero_mean_high_energy_control": control,
        "adamw8bit": adam,
        "second_family": liger,
        "overall_claim": "A reproducible repair-aligned numerical effect is directly reduced by residual readback on the same frozen histories; saved-state propagation, a targeted intervention with training benefit, and diagnostic reuse on an FP32 accumulation-order family are established under their respective protocols.",
        "overall_limit": "The evidence does not prove a nonzero full-vector mean, mean bias as the unique loss mediator, universal kernel coverage, or population-wide Liger materiality.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT):
        raise ValueError("choose a new output path inside kernel-analyzer")
    payload = build_payload()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": payload["status"], "gates": payload["gates"]}))


if __name__ == "__main__":
    main()
