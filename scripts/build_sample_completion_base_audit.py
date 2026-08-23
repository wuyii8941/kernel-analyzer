#!/usr/bin/env python3
"""Audit the eight base cases against the frozen uniform trace schema.

This is intentionally conservative.  It reports whether an artifact exists
for a layer; it never infers a missing layer from a T1/T4/SEUP verdict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/property/sample_completion_v1/base_case_audit.json"


REQUIRED = ("operator_output", "parameter_gradient", "optimizer_update", "trajectory_32", "feedback")


def load(path: Path) -> Any:
    if path.suffix == ".gz":
        import gzip
        with gzip.open(path, "rt", encoding="utf-8") as h:
            return json.load(h)
    return json.loads(path.read_text(encoding="utf-8"))


def exists(rel: str) -> bool:
    return (ROOT / rel).exists()


def row(case_id: str, role: str, family: str, evidence: dict[str, list[str]], note: str) -> dict[str, Any]:
    presence = {key: any(exists(path) for path in paths) for key, paths in evidence.items()}
    layer_complete = all(presence.get(key, False) for key in REQUIRED)
    return {
        "case_id": case_id,
        "role": role,
        "implementation_family": family,
        "evidence": evidence,
        "layer_artifact_present": presence,
        "missing_layers": [key for key in REQUIRED if not presence.get(key, False)],
        "layer_artifacts_complete": layer_complete,
        "uniform_trace_ready": False,
        "scientific_label": None,
        "note": note,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    rows = [
        row("liger_fused_ce_t128", "POSITIVE", "fused_reduction_accumulation", {
            "operator_output": ["results/property/joint_bias_formation_v1/liger_three_stage_summary.json"],
            "parameter_gradient": ["results/property/joint_bias_formation_v1/liger_three_stage_summary.json"],
            "optimizer_update": ["results/property/joint_bias_formation_v1/liger_three_stage_summary.json"],
            "trajectory_32": ["results/trajectory/liger_trajectory.json"],
            "feedback": ["results/property/joint_bias_formation_v1/three_stage_summary.json"],
        }, "Existing strict case; a uniform export still needs to be generated."),
        row("phi4_lm_head_dx_seq64", "POSITIVE", "lm_head_vjp_gemm", {
            "operator_output": ["results/coverage/cases/phi4_seq64_lmhead_dx.json"],
            "parameter_gradient": ["results/coverage/cases/phi4_seq64_lmhead_dx.json"],
            "optimizer_update": ["results/property/joint_bias_formation_v1/phi_three_stage_reference.json"],
            "trajectory_32": ["results/coverage/cases/phi4_seq64_lmhead_dx_trajectory.json"],
            "feedback": ["results/property/joint_bias_formation_v1/phi_fixed_update_propagation.json"],
        }, "Existing strict case; optimizer and feedback artifacts use older schemas."),
        row("qwen_seq256_lm_head_dx", "POSITIVE", "lm_head_vjp_gemm", {
            "operator_output": ["results/property/persistence_v1/confirmation/qwen256_lmhead.json"],
            "parameter_gradient": ["results/property/bias_oracle_recovery/confirmation/qwen/qwen_seq64_lm_head_dx.json"],
            "optimizer_update": ["results/property/persistence_v1/confirmation/qwen256_lmhead.json"],
            "trajectory_32": ["results/property/persistence_v1/confirmation/qwen256_lmhead.json"],
            "feedback": [],
        }, "Historical Qwen evidence lacks a separate feedback recurrence export."),
        row("qwen128_layer0_vproj_output", "CONTROL", "projection_rounding", {
            "operator_output": ["results/coverage/cases/qwen128_vproj.json"],
            "parameter_gradient": [],
            "optimizer_update": [],
            "trajectory_32": ["results/coverage/cases/qwen128_vproj_trajectory.json"],
            "feedback": ["results/coverage/cases/qwen128_vproj_trajectory.json"],
        }, "Trajectory exists, but the three update layers are not exported in the uniform format."),
        row("qwen_saved_p_seq128", "CONTROL", "saved_state_reconstruction", {
            "operator_output": ["results/property/joint_bias_formation_v1/qwen_three_stage_reference.json"],
            "parameter_gradient": ["results/property/joint_bias_formation_v1/qwen_three_stage_reference.json"],
            "optimizer_update": ["results/property/joint_bias_formation_v1/qwen_three_stage_reference.json"],
            "trajectory_32": ["results/coverage/cases/qwen128_softmax_saved_p_trajectory.json"],
            "feedback": ["results/property/joint_bias_formation_v1/qwen_three_stage_reference.json"],
        }, "All layers exist in older artifacts, but require schema-normalized export."),
        row("qwen3vl_silu_backward", "CONTROL", "pointwise_backward", {
            "operator_output": ["results/round2/vl_bias.json"],
            "parameter_gradient": ["results/round2/vl_bias.json"],
            "optimizer_update": ["results/property/joint_bias_formation_v1/vl_silu_optimizer_oddness_vectors_v3.json"],
            "trajectory_32": ["results/coverage/cases/qwen3vl_layer0_silu_trajectory.json"],
            "feedback": ["results/property/joint_bias_formation_v1/vl_silu_optimizer_oddness_vectors_v3.json"],
        }, "The historical natural trajectory has fewer than the new protocol's 32 uniform states."),
        row("gemma4_e2b_ple_rmsnorm", "CONTROL", "normalization_backward", {
            "operator_output": ["results/property/tcmp_allop_v1/heldout/gemma4_e2b_text128/norm_formation16.json"],
            "parameter_gradient": ["results/property/tcmp_allop_v1/heldout/gemma4_e2b_text128/norm_formation16.json"],
            "optimizer_update": ["results/property/tcmp_allop_v1/heldout/gemma4_e2b_text128/norm_consequence32.json"],
            "trajectory_32": ["results/property/tcmp_allop_v1/heldout/gemma4_e2b_text128/norm_consequence32.json"],
            "feedback": ["results/property/tcmp_allop_v1/heldout/gemma4_e2b_text128/norm_consequence32.json"],
        }, "Complete targeted control, not yet schema-normalized."),
        row("qwen_bmm_seq64", "CONTROL", "attention_bmm", {
            "operator_output": ["results/property/seup_geometry_followup/qwen_bmm_seq64_geometry.json"],
            "parameter_gradient": [],
            "optimizer_update": ["results/property/seup_mainline/qwen_bmm_seq64_seup.json.gz"],
            "trajectory_32": ["results/property/seup_mainline/qwen_bmm_seq64_seup.json.gz"],
            "feedback": [],
        }, "Existing geometry artifact does not prove the new uniform repair/sham chain."),
    ]
    result = {
        "schema": "kernel-analyzer-sample-completion-base-case-audit-v1",
        "status": "PRE_UNIFORM_REEXPORT_AUDIT",
        "required_layers": list(REQUIRED),
        "rows": rows,
        "layer_artifacts_complete_count": sum(r["layer_artifacts_complete"] for r in rows),
        "uniform_trace_ready_count": sum(r["uniform_trace_ready"] for r in rows),
        "uniform_control_count": sum(r["uniform_trace_ready"] and r["role"] == "CONTROL" for r in rows),
        "claim_boundary": "Existing artifacts are not silently promoted; all eight cases need a schema-normalized trace before they enter the new uniform denominator.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "uniform_trace_ready": result["uniform_trace_ready_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
