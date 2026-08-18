#!/usr/bin/env python3
"""Build the fail-closed layer-23 q_proj semantic-region attribution certificate."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results/final"
OUTPUT = ROOT / "results/coverage/cases/l23_qproj_attention_state_region.json"

SOURCES = (
    "results/final/structured_carrier_confirmation.json",
    "results/final/l23_qproj_operand_decomposition.json",
    "results/final/l23_go_summary.json",
    "results/final/l23_attention_live_weight.json",
    "results/final/l23_go_step64.json",
    "results/final/l23_go_step256.json",
    "results/final/l23_go_step1024.json",
    "results/final/l23_go_step2048.json",
    "results/final/l23_go_step4096.json",
)


def load(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text())


def sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def clustered_metric(payloads: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    by_state = {index: [] for index in range(8, 40)}
    for payload in payloads:
        for row in payload["rows"]:
            by_state[int(row["state_index"])].append(float(row[metric]))
    values = [sum(rows) / len(rows) for rows in by_state.values()]
    rng = random.Random(20260805)
    draws = 20_000
    n = len(values)
    samples = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws)
    )
    return {
        "state_cluster_mean": sum(values) / n,
        "state_cluster_bootstrap_95": [samples[500], samples[19_500]],
        "positive_states": sum(value > 0 for value in values),
        "states": n,
    }


def main() -> None:
    carrier = load(SOURCES[0])
    operand = load(SOURCES[1])
    attention = load(SOURCES[2])
    trajectory = load(SOURCES[3])
    step_payloads = [load(relative) for relative in SOURCES[4:]]

    rows = operand["rows"]
    total = attention["metric_summary"]["candidate_minus_eager_projection"]
    s_term = attention["metric_summary"]["s_shapley_removal_projection"]
    k_term = attention["metric_summary"]["k_shapley_removal_projection"]
    residual = attention["metric_summary"]["reference_s_reference_k_residual_projection"]
    s_only_residual = clustered_metric(
        step_payloads, "reference_s_candidate_k_residual_projection"
    )
    k_only_residual = clustered_metric(
        step_payloads, "candidate_s_reference_k_residual_projection"
    )
    projections = [row["fp32_master_projection"] for row in trajectory["records"]]

    gates = {
        "complete_forward_backward_math": (
            operand["equation"]
            == "delta_dW=(delta_G)^T H_ref + G_ref^T(delta_H) + (delta_G)^T(delta_H)"
            and operand["binding"]["forward"] == "mm_161: Y=H W^T"
            and operand["binding"]["actual_weight_backward"] == "mm_267: dW=G^T H"
            and all(row["exact_mm_output_matches_parameter_gradient_tile"] for row in rows)
            and all(row["finite_mm_total_matches_actual_delta"] for row in rows)
        ),
        "real_directional_parameter_carrier": (
            carrier["confirmed_parameters"]
            == ["model.layers.23.self_attn.q_proj.weight"]
            and carrier["gate"]["cluster_bootstrap_lower_95_positive"]
            and total["state_cluster_bootstrap_95"][0] > 0
        ),
        "query_cotangent_is_local_carrier": (
            sum(row["g_effect_projection"] for row in rows) / len(rows) > 0
            and attention["ratios"]["s_shapley_over_total"] > 0.95
            and k_term["state_cluster_bootstrap_95"][0] <= 0
            and k_term["state_cluster_bootstrap_95"][1] >= 0
        ),
        "joint_attention_state_repair_closes_direction": (
            residual["state_cluster_bootstrap_95"][0] <= 0
            and residual["state_cluster_bootstrap_95"][1] >= 0
            and abs(attention["ratios"]["rr_residual_over_total"]) < 0.01
        ),
        "s_bwd_only_repair_closes_direction": (
            s_only_residual["state_cluster_bootstrap_95"][0] <= 0
            and s_only_residual["state_cluster_bootstrap_95"][1] >= 0
            and k_only_residual["state_cluster_bootstrap_95"][0] > 0
        ),
        "matched_sham_exact": attention["validation"]["max_candidate_restoration_sham_abs"] == 0,
        "paired_live_weight_trajectory": (
            trajectory["status"] == "COMPLETE"
            and trajectory["repair_boundary"] == "actual bmm_76: G_q = S_bwd @ K"
            and trajectory["steps"] == 32
            and all(trajectory["gates"].get(key) for key in (
                "bf16_live_weight_feedback_observed",
                "final_fp32_master_divergence_nonzero",
                "only_q_proj_live_weight_updated",
                "same_initial_fp32_master",
                "same_state_order",
                "same_weight_baseline_and_repair_measured_each_arm_each_step",
            ))
            and all(right > left for left, right in zip(projections, projections[1:]))
            and projections[-1] > 0
        ),
    }
    passed = all(gates.values())
    payload = {
        "schema": "kernel-analyzer-l23-semantic-region-attribution-v1",
        "status": "PASS_STRICT_SEMANTIC_REGION_FLASH_STYLE_CASE" if passed else "INCOMPLETE",
        "subject": {
            "model": "Qwen3-1.7B",
            "parameter": operand["parameter"],
            "tile": operand["tile"],
            "checkpoint_step": operand["checkpoint_step"],
        },
        "closed_math": {
            "forward": "Y = H W^T",
            "parameter_vjp": "dW = G_q^T H",
            "attention_vjp_boundary": "G_q = S_bwd K",
            "attention_state": "S_bwd = alpha * J_softmax(P)^T * (D V^T)",
            "difference_factorization": operand["equation"],
        },
        "causal_attribution": {
            "root_boundary": "actual bmm_76 left operand S_bwd",
            "mechanism_class": "ATTENTION_STATE_SEMANTIC_REGION",
            "single_kernel_attribution": False,
            "combination_necessary": False,
            "trajectory_repair_is_conservative_superset": True,
            "reason": (
                "The concrete q_proj F+B unit is exact and its final same-input GEMM is not "
                "the source. The directional carrier enters through G_q. Restoring S_bwd alone "
                "closes the direction; restoring K alone does not. The paired trajectory used "
                "the conservative S_bwd/K boundary, whose extra K contribution is independently "
                "non-directional. Deeper "
                "forward/backward contributors overlap and are not additively identifiable as "
                "one kernel. The attention-backward state S_bwd is therefore the causal root."
            ),
            "total_projection": total,
            "s_bwd_shapley": s_term,
            "k_shapley": k_term,
            "joint_repair_residual": residual,
            "s_bwd_only_repair_residual": s_only_residual,
            "k_only_repair_residual": k_only_residual,
            "joint_residual_over_total": attention["ratios"]["rr_residual_over_total"],
            "matched_sham_max_abs": attention["validation"]["max_candidate_restoration_sham_abs"],
        },
        "trajectory": {
            "optimizer": trajectory["optimizer"],
            "steps": trajectory["steps"],
            "repair_boundary": trajectory["repair_boundary"],
            "first_fp32_master_projection": projections[0],
            "final_fp32_master_projection": projections[-1],
            "final_fp32_master_l2": trajectory["records"][-1]["fp32_master_l2"],
            "projection_strictly_increases_each_step": all(
                right > left for left, right in zip(projections, projections[1:])
            ),
        },
        "gates": gates,
        "classification": {
            "valid_flash_style_case": passed,
            "strictness_level": "CLOSED_SEMANTIC_REGION",
            "single_kernel_property_eligible": False,
            "cross_operator_property_claimed": False,
        },
        "evidence_files": {relative: sha256(relative) for relative in SOURCES},
        "claim_boundary": (
            "This certificate proves a concrete closed F+B semantic-region mechanism and paired "
            "directional weight accumulation. It does not identify one defective kernel and must "
            "not be used as a single-kernel property-positive training label."
        ),
    }
    payload["result_sha256"] = canonical(payload)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUTPUT.relative_to(ROOT)), "status": payload["status"]}))


if __name__ == "__main__":
    main()
