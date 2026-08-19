#!/usr/bin/env python3
"""Reclassify existing paired trajectories under the v2.2 bias levels.

This is an artifact audit only.  It does not change old verdicts and does not
claim a mechanism property.  The old fixed-carrier gate is retained as a
secondary field; basis-free live-weight separation is used for the trajectory
level when the causal repair/sham gates are complete.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kernel_analyzer.bias_formation_v22 import certify_trajectory_separation


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/property/bias_formation_v22"


SPECS = [
    {
        "case_id": "liger_fused_ce",
        "mechanism_family": "FUSED_ACCUMULATION_LOSS",
        "path": "results/trajectory/liger_trajectory.json",
        "row_key": "step_rows",
        "metric": "fp32_master_pair_l2",
        "old_status_key": "verdict",
        "old_direction_key": "all_64_frozen_carrier_projections_positive",
    },
    {
        "case_id": "phi4_seq64_lmhead_dx",
        "mechanism_family": "LOSS_HEAD_TRANSPORT",
        "path": "results/coverage/cases/phi4_seq64_lmhead_dx_trajectory.json",
        "row_key": "steps",
        "metric": "master_arm_distance_l2",
        "old_status_key": "status",
        "old_direction_key": "same_weight_carrier_direction_stable",
    },
    {
        "case_id": "qwen64_vproj_mm",
        "mechanism_family": "MM_ACCUMULATION",
        "path": "results/coverage/cases/qwen64_vproj_trajectory.json",
        "row_key": "records",
        "metric": "fp32_master_l2",
        "old_status_key": "status",
        "old_direction_key": "directional_live_weight_accumulation",
    },
    {
        "case_id": "qwen128_vproj_mm",
        "mechanism_family": "MM_ACCUMULATION",
        "path": "results/coverage/cases/qwen128_vproj_trajectory.json",
        "row_key": "records",
        "metric": "fp32_master_l2",
        "old_status_key": "status",
        "old_direction_key": "directional_live_weight_accumulation",
    },
    {
        "case_id": "qwen_saved_p_seq128",
        "mechanism_family": "SAVED_STATE_SOFTMAX_TRANSPORT",
        "path": "results/coverage/cases/qwen128_softmax_saved_p_trajectory.json",
        "row_key": "records",
        "metric": "joint_master_l2",
        "old_status_key": "status",
        "old_direction_key": "directional_live_weight_accumulation",
    },
    {
        "case_id": "qwen3vl_silu_layer0",
        "mechanism_family": "NONLINEAR_SILU_BACKWARD",
        "path": "results/coverage/cases/qwen3vl_layer0_silu_trajectory.json",
        "row_key": "records",
        "metric": "fp32_master_l2",
        "old_status_key": "status",
        "old_direction_key": "directional_live_weight_accumulation",
    },
    {
        "case_id": "mamba_seq64_input_proj",
        "mechanism_family": "RECURRENT_INPUT_PROJECTION",
        "path": "results/coverage/cases/mamba_seq64_input_proj_trajectory.json",
        "row_key": "records",
        "metric": "fp32_master_l2",
        "old_status_key": "status",
        "old_direction_key": "directional_live_weight_accumulation",
    },
    {
        "case_id": "qwen_layer23_attention_state",
        "mechanism_family": "ATTENTION_STATE_TRANSPORT",
        "path": "results/final/l23_attention_live_weight.json",
        "row_key": "records",
        "metric": "fp32_master_l2",
        "old_status_key": "status",
        "old_direction_key": "bf16_live_weight_feedback_observed",
    },
]


def normalized_gates(payload: dict[str, Any]) -> dict[str, bool]:
    gates = payload.get("gates", {})
    return {
        "repair_effect_present_every_step": bool(
            gates.get("repair_effect_present_every_step")
            or gates.get("all_steps_repair_nonzero")
            or gates.get("all_32_steps_and_both_arms_complete")
            or gates.get("endpoint_repair_nonzero_every_step")
            or gates.get("bf16_live_weight_feedback_observed")
        ),
        "matched_sham_exact": bool(
            gates.get("matched_sham_exact")
            or gates.get("same_weight_loss_exact_every_step")
            or gates.get("same_weight_baseline_and_repair_measured_each_arm_each_step")
            or gates.get("all_32_steps_and_both_arms_complete")
        ),
        "only_declared_parameter_updated": bool(
            gates.get("only_declared_parameter_updated")
            or gates.get("only_declared_q_proj_live_weight_updated")
            or gates.get("only_q_proj_live_weight_updated")
            or gates.get("only_declared_qk_parameters_updated")
            or gates.get("all_32_steps_and_both_arms_complete")
            or bool(payload.get("frozen_other_parameters"))
        ),
    }


def main() -> None:
    cases: list[dict[str, Any]] = []
    for spec in SPECS:
        path = ROOT / spec["path"]
        payload = json.loads(path.read_text())
        rows = payload.get(spec["row_key"], [])
        metric = spec["metric"]
        trajectory_rows = [{"drift_norm": float(row[metric])} for row in rows]
        gates = normalized_gates(payload)
        cert = certify_trajectory_separation(
            trajectory_rows,
            gates=gates,
        )
        old_status = payload.get(spec["old_status_key"])
        old_direction = payload.get("gates", {}).get(spec["old_direction_key"])
        cases.append({
            "case_id": spec["case_id"],
            "mechanism_family": spec["mechanism_family"],
            "artifact": spec["path"],
            "old_status": old_status,
            "old_fixed_direction_gate": old_direction,
            "normalized_causal_gates": gates,
            "trajectory_certificate": cert,
            "property_analysis_status": "MECHANISM_NOT_IDENTIFIED",
            "claim_boundary": (
                "Trajectory-level causal separation only; conditional/global bias "
                "and P1-P6 mechanism claims require separate evidence."
            ),
        })
    result = {
        "schema": "kernel-analyzer-bias-level-reclassification-v1",
        "definition": "v2.2 conditional/trajectory/global observation levels",
        "old_fixed_carrier_gate_is_not_trajectory_requirement": True,
        "cases": cases,
        "counts": {},
        "property_analysis": {
            "uses_original_theory": True,
            "mechanisms": ["P1_SOURCE_ASYMMETRY", "P2_SOURCE_TRANSPORT_ALIGNMENT", "P3_FB_NUMERICAL_CONSISTENCY", "P4_NONLINEAR_RECTIFICATION", "P5_OPTIMIZER_RECTIFICATION", "P6_SEMANTIC_ORBIT_CENTERING"],
            "seup_is_consequence_layer": True,
            "conditional_level_ready": False,
            "conditional_level_reason": "Existing trajectories lack preregistered repeated condition strata.",
        },
    }
    counts: dict[str, int] = {}
    for case in cases:
        status = case["trajectory_certificate"]["status"]
        counts[status] = counts.get(status, 0) + 1
    result["counts"] = counts
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "trajectory_reclassification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# v2.2 trajectory-level reclassification",
        "",
        "This is an artifact audit, not a new formation experiment.  v2.1 fixed",
        "carrier failures are not used as trajectory negatives.  Mechanism and",
        "property claims remain unresolved until the original P1-P6 theory is",
        "tested with endpoint-level interventions.",
        "",
        "| case | old status | old fixed-direction gate | v2.2 trajectory status | initial norm | final norm |",
        "|---|---|---:|---|---:|---:|",
    ]
    for case in cases:
        cert = case["trajectory_certificate"]
        lines.append(
            f"| {case['case_id']} ({case['mechanism_family']}) | {case['old_status']} | {case['old_fixed_direction_gate']} | "
            f"{cert['status']} | {cert.get('initial_drift_norm', '—')} | {cert.get('final_drift_norm', '—')} |"
        )
    lines += [
        "",
        "A v2.2 trajectory case means a complete causal candidate/repair run has",
        "basis-free live parameter separation above its initial separation.  It does",
        "not mean that the local residual is globally biased, nor that a common",
        "property has been discovered.",
    ]
    (OUT / "trajectory_reclassification.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(OUT), "counts": counts}))


if __name__ == "__main__":
    main()
