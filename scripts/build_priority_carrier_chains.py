#!/usr/bin/env python3
"""Preserve exact F+B chains worth prioritizing in the next carrier replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/full_step_inventory"
FINAL = ROOT / "results/final"


def metric(records: dict[str, dict], region_id: str, endpoint: str = "out_ptr0") -> dict:
    value = records.get(region_id, {}).get("same_precision_result", {})
    if endpoint in value:
        return value[endpoint]
    return value


def main() -> None:
    binding = json.loads((RAW / "key_rmsnorm_forward_vjp_exact_binding_v1.json").read_text())
    replay = json.loads((RAW / "all_attention_norm_sum_region_replay.json").read_text())
    carrier_replay_path = FINAL / "priority_carrier_replay.json"
    carrier_replay = (
        json.loads(carrier_replay_path.read_text())
        if carrier_replay_path.exists()
        else None
    )
    carrier_by_region = {}
    if carrier_replay is not None:
        for record in carrier_replay.get("records", []):
            carrier_by_region.setdefault(str(record["region_id"]), []).append(record)
    records = {
        str(row["region"]["region_id"]): row
        for row in replay["records"]
    }
    rows = []
    for row in binding["rows"]:
        identity = row["exact_saved_tensor_identity"]
        partial = str(row["backward_weight_partial_region_id"])
        dot = str(row["backward_dot_region_id"])
        carrier_records = carrier_by_region.get(dot, [])
        rows.append({
            "forward_region_id": str(row["forward_region_id"]),
            "weight_partial_region_id": partial,
            "dot_region_id": dot,
            "pointwise_region_id": str(row["backward_pointwise_region_id"]),
            "weight_final_region_id": str(row["backward_weight_final_region_id"]),
            "weight_parameter_abi_root": str(identity["weight_backward_abi_root"]),
            "projected_tensor_abi_root": str(identity["projected_tensor_backward_abi_root"]),
            "mechanisms": list(row["non_bf16_mechanisms_retained"]),
            "one_state_partial_metric": metric(records, partial),
            "one_state_dot_metric": metric(records, dot),
            "one_state_weight_final_metric": metric(
                records, str(row["backward_weight_final_region_id"])
            ),
            "candidate_correctness_granted": bool(row["candidate_correctness_granted"]),
            "evolving_carrier_replay": (
                "COMPLETE"
                if carrier_records
                and all(record["exact_boundary"] for record in carrier_records)
                else "PENDING_GPU_REMEASUREMENT"
            ),
            "carrier_replay_summary": {
                "record_count": len(carrier_records),
                "initial_positive_direction": dot in (
                    carrier_replay.get("initial_all_positive_regions", [])
                    if carrier_replay is not None
                    else []
                ),
                "focused_followup_lost_direction": (
                    dot == "backward:852"
                    and bool(carrier_replay and carrier_replay.get("focused_candidate_lost_direction"))
                ),
                "verdict": (
                    carrier_replay.get("verdict")
                    if carrier_replay is not None and carrier_records
                    else "PENDING_GPU_REMEASUREMENT"
                ),
            },
        })
    rows.sort(key=lambda x: float(x["one_state_dot_metric"].get("max_abs", 0.0)), reverse=True)
    output = {
        "schema": "kernel-analyzer-priority-carrier-chains-v1",
        "subject": "Qwen3-1.7B exact key-RMSNorm forward/VJP chains",
        "source_binding": "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/full_step_inventory/key_rmsnorm_forward_vjp_exact_binding_v1.json",
        "source_replay": "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/full_step_inventory/all_attention_norm_sum_region_replay.json",
        "candidate_values_used_to_select_or_classify": False,
        "chain_count": len(rows),
        "one_state_weight_final_exact_count": sum(
            bool(row["one_state_weight_final_metric"].get("exact"))
            for row in rows
        ),
        "rows": rows,
        "interpretation": (
            "These are exact mathematical and pointer chains selected because they end in a "
            "parameter weight-gradient reduction. The candidate-blind carrier screen records "
            "repeat-stable downstream deltas; its checkpoint direction test does not support a "
            "FlashAttention-style coherent carrier bias."
            if carrier_replay is not None
            else "These are exact mathematical and pointer chains selected because they end in a parameter weight-gradient reduction. One-state residuals are prioritization evidence only; no evolving directional verdict is assigned."
        ),
        "natural_bias_case_added": False,
        "property_claim": False,
    }
    output["result_sha256"] = hashlib.sha256(
        json.dumps(output, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = FINAL / "priority_carrier_chains.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(path), "chain_count": len(rows)}))


if __name__ == "__main__":
    main()
