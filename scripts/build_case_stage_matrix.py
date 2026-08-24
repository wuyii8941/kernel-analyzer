#!/usr/bin/env python3
"""Build a protocol-aware evidence matrix from machine-readable artifacts only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text())


def digest(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def source(path: str) -> dict[str, str]:
    return {"path": path, "sha256": digest(path)}


def attach_comparison_identity(record: dict[str, Any]) -> None:
    identity = {
        "exact_contrast_id": record["exact_contrast_id"],
        "optimizer": record["optimizer"],
        "moment_state": record["moment_state"],
        "parameter_scope": record["parameter_scope"],
        "state_protocol": record["state_protocol"],
        "horizon": record["horizon"],
        "measurement_geometry": record["measurement_geometry"],
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    record["comparison_identity"] = identity
    record["comparison_identity_sha256"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


HISTORICAL_META: dict[str, dict[str, Any]] = {
    "seq128_lm_head_input_vjp_mm": {
        "case_id": "qwen_seq128_lmhead_dx",
        "model": "Qwen3-1.7B",
        "exact_contrast_id": "seq128 lm_head input-VJP compiled MM vs FP32 repair",
        "forward_endpoint": "lm_head logits",
        "backward_region": "lm_head dX",
        "optimizer": "stateless SGD with FP32 master",
        "moment_state": "NOT_APPLICABLE",
        "parameter_scope": "all trainable FP32-master parameters",
        "measurement_geometry": "FIXED_CARRIER_AND_FULL_PARAMETER_DISTANCE",
    },
    "liger_fused_linear_ce_dw": {
        "case_id": "liger_fused_ce_t128",
        "model": "Qwen3-1.7B + Liger fused CE",
        "exact_contrast_id": "fused CE BF16 dW accumulation vs FP32 accumulation",
        "forward_endpoint": "fused linear cross entropy",
        "backward_region": "tied embedding dW accumulation",
        "optimizer": "stateless SGD with FP32 master",
        "moment_state": "NOT_APPLICABLE",
        "parameter_scope": "310 trained parameters",
        "measurement_geometry": "FIXED_CARRIER_AND_FULL_PARAMETER_DISTANCE",
    },
    "phi4_seq64_lmhead_dx_mm": {
        "case_id": "phi4_seq64_lmhead_dx",
        "model": "Phi-4-mini",
        "exact_contrast_id": "backward:497:output_0 compiled BF16 MM vs FP32-cast repair",
        "forward_endpoint": "lm_head logits",
        "backward_region": "lm_head dX",
        "optimizer": "SGD with FP32 master",
        "moment_state": "NOT_APPLICABLE",
        "parameter_scope": "model.norm.weight only",
        "measurement_geometry": "FIXED_CARRIER",
    },
    "mamba64_layer0_input_proj_output": {
        "case_id": "mamba_seq64_in_proj",
        "model": "Mamba-130M",
        "exact_contrast_id": "layer0 in_proj GEMM arithmetic repair",
        "forward_endpoint": "layer0 in_proj output",
        "backward_region": "in_proj backward",
        "optimizer": "AdamW(beta1=0.9,beta2=0.95)",
        "moment_state": "ZERO_THEN_EVOLVED",
        "parameter_scope": "backbone.layers.0.mixer.in_proj.weight",
        "measurement_geometry": "FIXED_CARRIER",
    },
    "qwen128_layer27_softmax_saved_state": {
        "case_id": "qwen_saved_p_seq128",
        "model": "Qwen3-1.7B",
        "exact_contrast_id": "layer27 reconstructed saved-P vs typed true-forward-P VJP",
        "forward_endpoint": "attention probability state",
        "backward_region": "layer27 softmax backward",
        "optimizer": "AdamW(beta1=0.9,beta2=0.95)",
        "moment_state": "ZERO_THEN_EVOLVED",
        "parameter_scope": "layer27 q_proj.weight + k_proj.weight",
        "measurement_geometry": "FIXED_CARRIER",
    },
    "layer23_qproj_attention_state_region": {
        "case_id": "layer23_qproj_attention_state_region",
        "model": "Qwen3-1.7B",
        "exact_contrast_id": "joint S_bwd/K saved-state repair",
        "forward_endpoint": "layer23 attention state",
        "backward_region": "layer23 attention backward to q_proj",
        "optimizer": "AdamW(beta1=0.9,beta2=0.95)",
        "moment_state": "ZERO_THEN_EVOLVED",
        "parameter_scope": "predeclared q_proj.weight tile",
        "measurement_geometry": "FIXED_CARRIER",
    },
    "qwen3vl_silu_backward_decomposition": {
        "case_id": "qwen3vl_silu_backward",
        "model": "Qwen3-VL-Reranker-2B",
        "exact_contrast_id": "compiled SiLU backward decomposition vs reference",
        "forward_endpoint": "SiLU output",
        "backward_region": "SiLU backward",
        "optimizer": "historical paired trajectory",
        "moment_state": "SEE_SOURCE_ARTIFACT",
        "parameter_scope": "declared layer0 carrier",
        "measurement_geometry": "FIXED_CARRIER",
    },
    "qwen128_layer0_vproj_output": {
        "case_id": "qwen128_vproj_output",
        "model": "Qwen3-1.7B",
        "exact_contrast_id": "layer0 v_proj GEMM accumulation repair",
        "forward_endpoint": "layer0 v_proj output",
        "backward_region": "v_proj backward",
        "optimizer": "historical paired trajectory",
        "moment_state": "SEE_SOURCE_ARTIFACT",
        "parameter_scope": "declared v_proj carrier",
        "measurement_geometry": "FIXED_CARRIER",
    },
}


def historical_records() -> list[dict[str, Any]]:
    path = "results/coverage/existing_case_reaudit.json"
    audit = read(path)
    records: list[dict[str, Any]] = []
    for row in audit["rows"]:
        meta = HISTORICAL_META[row["case"]]
        verdict = row["flash_style"]["verdict"]
        records.append({
            "record_id": f"historical32::{meta['case_id']}",
            **meta,
            "state_protocol": "historical frozen 32-state live paired trajectory",
            "horizon": {"steps": 32, "class": "SHORT_HORIZON"},
            "stages": {
                "fb_formation": row["generalizable_bias"]["verdict"],
                "optimizer_transformation": "PROTOCOL_SPECIFIC_NOT_CROSS_COMPARABLE",
                "short_horizon": verdict,
                "long_run": "NOT_MEASURED",
                "feedback": "NOT_NORMALIZED_IN_THIS_AUDIT",
                "training_convergence": "NOT_MEASURED",
            },
            "status": verdict,
            "abstention_reason": row.get("note"),
            "source_artifacts": [source(path)],
        })
    return records


def case_plan_task_id(result: dict[str, Any]) -> str:
    plan_path = result.get("case_plan")
    if not plan_path or not (ROOT / plan_path).exists():
        return result["case_id"]
    plan = read(plan_path)
    cases = [row for row in plan.get("cases", []) if row.get("case_id") == result["case_id"]]
    return str(cases[0].get("task_id", result["case_id"])) if cases else result["case_id"]


def unified_adamw_records() -> list[dict[str, Any]]:
    cohort_path = "results/property/direct_persistence_v4/cohort.json"
    cohort = read(cohort_path)
    records: list[dict[str, Any]] = []
    for row in cohort["rows"]:
        result = read(row["source"])
        confirmed = row.get("confirmed_label")
        candidate = row["case_id"] in cohort.get("unresolved_candidates", [])
        if confirmed:
            status = "CONFIRMED_32_STEP_DIRECTIONAL"
        elif candidate:
            status = "UNRESOLVED_MULTIPLICITY_CANDIDATE"
        else:
            status = "NO_CONFIRMED_32_STEP_DIRECT_DIRECTION"
        statistics = result.get("statistics") or {}
        records.append({
            "record_id": f"unified_adamw32::{row['case_id']}",
            "case_id": row["case_id"],
            "exact_contrast_id": case_plan_task_id(result),
            "model": result.get("model", row.get("architecture")),
            "forward_endpoint": "see exact task binding",
            "backward_region": result.get("case_id"),
            "optimizer": row["optimizer"],
            "moment_state": "ZERO_THEN_EVOLVED_NORMALLY",
            "state_protocol": "32-step result-blind/common-optimizer cohort",
            "parameter_scope": result.get("carrier", "declared carrier"),
            "measurement_geometry": "FULL_VECTOR_GRAM_ON_DECLARED_CARRIER",
            "horizon": {"steps": 32, "class": "SHORT_HORIZON"},
            "stages": {
                "fb_formation": result.get("formation_point", "NOT_EVALUATED"),
                "optimizer_transformation": "MEASURED_ADAMW_EFFECTIVE_UPDATE",
                "short_horizon": status,
                "long_run": "NOT_MEASURED",
                "feedback": result.get("trajectory_status", statistics.get("classification", "MEASURED_SEPARATELY")),
                "training_convergence": "NOT_MEASURED",
            },
            "metrics": {
                "A16": row["prefix16_local_A"],
                "A32": row["full32_local_A"],
                "null95_A32": row.get("full32_sign_flip_upper95"),
                "raw_p_A32": row.get("full32_sign_flip_one_sided_p"),
                "holm": row.get("holm"),
            },
            "status": status,
            "abstention_reason": "multiplicity correction" if candidate else None,
            "source_artifacts": [source(cohort_path), source(row["source"])],
        })
    return records


def optional_new_records() -> list[dict[str, Any]]:
    specs = [
        (
            "results/property/long_horizon_v2/phi_warm128_measure512.json",
            "warm_adamw512::phi4_seq64_lmhead_dx",
            "WARM_STATE_MEDIUM_HORIZON_DIRECT",
        ),
        (
            "results/property/convergence_v1/phi_carrier_adamw_convergence.json",
            "constant_lr4096::phi4_seq64_lmhead_dx",
            "CONTROLLED_CONVERGENCE_ATTEMPT",
        ),
        (
            "results/property/convergence_v2/phi_carrier_adamw_cosine_convergence.json",
            "cosine_lr4096::phi4_seq64_lmhead_dx",
            "CONTROLLED_CONVERGENCE_ATTEMPT",
        ),
    ]
    records: list[dict[str, Any]] = []
    for path, record_id, kind in specs:
        if not (ROOT / path).exists():
            continue
        result = read(path)
        if kind == "WARM_STATE_MEDIUM_HORIZON_DIRECT":
            full = result["full"]
            records.append({
                "record_id": record_id,
                "case_id": result["case_id"],
                "exact_contrast_id": result["protocol"]["contrast"],
                "model": "Phi-4-mini",
                "forward_endpoint": "lm_head logits",
                "backward_region": "lm_head dX",
                "optimizer": result["protocol"]["optimizer"],
                "moment_state": result["protocol"]["optimizer"]["initial_moments"],
                "state_protocol": "128 warmup + 512 fresh natural measurement states",
                "parameter_scope": result["protocol"]["carrier"],
                "measurement_geometry": result["measurement_geometry"]["kind"],
                "horizon": {"steps": 512, "class": "MEDIUM_HORIZON_WARM_STATE"},
                "stages": {
                    "fb_formation": "EXACT_REPAIR_BOUND",
                    "optimizer_transformation": "ADAMW_WARM_STATE_DIRECT_EFFECT",
                    "short_horizon": "ALL_16_DISJOINT_32_STEP_WINDOWS_ABOVE_NULL",
                    "long_run": "512_STEP_DIRECT_DIRECTIONAL",
                    "feedback": "UNMEASURED_BY_THIS_DIRECT_ONLY_PROTOCOL",
                    "training_convergence": "NOT_MEASURED",
                },
                "metrics": {
                    "A512": full["coherence_amplification"],
                    "null95_A512": full["sign_flip_null"]["upper_95"],
                    "late_windows_above_null": result["late_half_windows_above_null"],
                    "late_window_count": result["late_half_window_count"],
                    "effective_rank": result["measurement_geometry"]["effective_rank_participation_ratio"],
                    "moving_frame": result["reference_relative"],
                },
                "status": "WARM_STATE_512_STEP_DIRECT_DIRECTIONAL",
                "abstention_reason": None,
                "source_artifacts": [source(path), source(result["measurement_geometry"]["gram_artifact"])],
            })
        else:
            final = result["final"]
            records.append({
                "record_id": record_id,
                "case_id": result["case_id"],
                "exact_contrast_id": "compiled BF16 lm_head dX vs FP32-cast repair",
                "model": "Phi-4-mini",
                "forward_endpoint": "lm_head logits",
                "backward_region": "lm_head dX",
                "optimizer": result["protocol"]["optimizer"],
                "moment_state": "SEPARATE_LIVE_MOMENTS_PER_ARM",
                "state_protocol": "32-state frozen train cycle + 8 held-out validation states",
                "parameter_scope": result["protocol"]["carrier"],
                "measurement_geometry": "PAIRED_LIVE_FIXED_CARRIER",
                "horizon": {"steps": result["steps_completed"], "class": "CONVERGENCE_ATTEMPT"},
                "stages": {
                    "fb_formation": "NOT_REESTIMATED",
                    "optimizer_transformation": "LIVE_ADAMW_PER_ARM",
                    "short_horizon": "NOT_PRIMARY_OUTCOME",
                    "long_run": "PARAMETER_SEPARATION_MEASURED",
                    "feedback": "PRESENT_BUT_NOT_DECOMPOSED",
                    "training_convergence": result["status"],
                },
                "metrics": {
                    "candidate_loss": final["candidate_loss"],
                    "repair_loss": final["repair_loss"],
                    "loss_gap": final["loss_gap_candidate_minus_repair"],
                    "gradient_difference_l2": final["mean_gradient_difference_l2"],
                    "parameter_distance_l2": final["parameter_distance_l2"],
                },
                "status": result["status"],
                "abstention_reason": None if result["loss_plateau_reached"] else "frozen loss plateau gate not reached",
                "source_artifacts": [source(path)],
            })
    return records


def prospective_gemma_records() -> list[dict[str, Any]]:
    summary_path = "results/property/direct_persistence_v4/heldout_confirmation_v2.json"
    if not (ROOT / summary_path).exists():
        return []
    summary = read(summary_path)
    records: list[dict[str, Any]] = []
    for row in summary["rows"]:
        consequence_path = row["artifacts"]["consequence.json"]["path"]
        consequence = read(consequence_path)
        local_a = row["consequence"]["local_A32"]
        if row["status"].startswith("NOT_APPLICABLE"):
            short = "NOT_APPLICABLE_NO_OBSERVED_CARRIER_EFFECT"
        elif local_a > 1.1:
            short = "32_STEP_DIRECT_DIRECTIONAL"
        else:
            short = "NO_32_STEP_DIRECT_DIRECTION"
        records.append({
            "record_id": f"prospective_gemma32::{row['case_id']}",
            "case_id": row["case_id"],
            "exact_contrast_id": row["target_region"],
            "model": row["model"],
            "forward_endpoint": "fresh in-process compiled target",
            "backward_region": row["target_region"],
            "optimizer": "AdamW; see consequence artifact",
            "moment_state": "ZERO_THEN_EVOLVED_NORMALLY",
            "state_protocol": "prediction frozen before independent 32-step trajectory",
            "parameter_scope": row["carrier"],
            "measurement_geometry": "FULL_VECTOR_STATISTICS_NO_RAW_GRAM",
            "horizon": {"steps": 32, "class": "SHORT_HORIZON_PROSPECTIVE"},
            "stages": {
                "fb_formation": row["prediction"],
                "optimizer_transformation": "MEASURED_IN_LIVE_ADAMW_TRAJECTORY",
                "short_horizon": short,
                "long_run": "NOT_MEASURED",
                "feedback": (
                    "FEEDBACK_SUSTAINED"
                    if row["consequence"]["feedback_A32"] > 1.1
                    else "NO_FEEDBACK_DIRECTION_DETECTED"
                ),
                "training_convergence": "NOT_MEASURED",
            },
            "metrics": row["consequence"],
            "status": row["status"],
            "abstention_reason": row.get("claim_boundary"),
            "source_artifacts": [source(summary_path), source(consequence_path)],
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    args = parser.parse_args()
    records = (
        historical_records()
        + unified_adamw_records()
        + prospective_gemma_records()
        + optional_new_records()
    )
    for record in records:
        attach_comparison_identity(record)
    case_index: dict[str, list[str]] = {}
    for record in records:
        case_index.setdefault(record["case_id"], []).append(record["record_id"])
    payload = {
        "schema": "kernel-analyzer-case-stage-matrix-v2",
        "status": "COMPLETE_PROTOCOL_AWARE_EVIDENCE_INDEX",
        "record_count": len(records),
        "case_count": len(case_index),
        "comparison_rule": (
            "Numerical scores may be compared only when exact contrast, optimizer, moment "
            "state, parameter scope, state protocol, horizon and measurement geometry match."
        ),
        "geometry_vocabulary": [
            "FULL_VECTOR_GRAM_ON_DECLARED_CARRIER",
            "FULL_VECTOR_GRAM",
            "FULL_VECTOR_STATISTICS_NO_RAW_GRAM",
            "FIXED_CARRIER",
            "FIXED_CARRIER_AND_FULL_PARAMETER_DISTANCE",
            "PAIRED_LIVE_FIXED_CARRIER",
        ],
        "case_index": case_index,
        "unindexed_evidence_gaps": [
            {
                "case_family": "Llama/Ministral lm_head dX held-out runs",
                "reason": "No standalone machine-readable result JSON was found under results/; do not reconstruct it from Markdown.",
            },
            {
                "case_family": "DeepSeek layer35 attention dV",
                "reason": "No standalone machine-readable result JSON was found under results/; conditional-bias prose is not imported.",
            },
        ],
        "records": records,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    fields = [
        "record_id", "case_id", "model", "exact_contrast_id", "optimizer",
        "moment_state", "state_protocol", "parameter_scope", "measurement_geometry",
        "comparison_identity_sha256",
        "horizon_steps", "horizon_class", "fb_formation", "optimizer_transformation",
        "short_horizon", "long_run", "feedback", "training_convergence", "status",
        "abstention_reason", "source_paths",
    ]
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({
                "record_id": record["record_id"],
                "case_id": record["case_id"],
                "model": record["model"],
                "exact_contrast_id": record["exact_contrast_id"],
                "optimizer": json.dumps(record["optimizer"], sort_keys=True) if isinstance(record["optimizer"], dict) else record["optimizer"],
                "moment_state": record["moment_state"],
                "state_protocol": record["state_protocol"],
                "parameter_scope": record["parameter_scope"],
                "measurement_geometry": record["measurement_geometry"],
                "comparison_identity_sha256": record["comparison_identity_sha256"],
                "horizon_steps": record["horizon"]["steps"],
                "horizon_class": record["horizon"]["class"],
                **record["stages"],
                "status": record["status"],
                "abstention_reason": record.get("abstention_reason"),
                "source_paths": ";".join(item["path"] for item in record["source_artifacts"]),
            })
    print(json.dumps({
        "json": str(args.json_output),
        "csv": str(args.csv_output),
        "records": len(records),
        "cases": len(case_index),
    }))


if __name__ == "__main__":
    main()
