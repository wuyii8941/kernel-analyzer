#!/usr/bin/env python3
"""Summarise unisolated natural-training candidates for the next capture round.

This is a triage artifact, not a root-cause verdict.  Signature campaigns are
region substitutions and may contain upstream differences; they are therefore
grouped by semantic clues only and remain outside the active problem count.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/property/case_causal_audit_v1/signature_causal_links.json"
OUTPUT = ROOT / "results/property/case_causal_audit_v1/natural_candidate_frontier.json"
ALIGNED_SCREEN = ROOT / "results/property/case_causal_audit_v1/signature_candidate_aligned_screen.json"


def _candidate_family(symbol: str, carrier: str) -> str:
    text = f"{symbol} {carrier}".lower()
    if "embedding" in text or "embed_tokens" in text:
        return "embedding_dense_backward"
    if "nll" in text:
        return "nll_backward"
    if "softmax" in text:
        return "softmax_attention_backward"
    if "silu" in text:
        return "silu_gating"
    if "softplus" in text or "a_log" in text:
        return "recurrence_softplus"
    if "arange" in text and ("cos" in text or "sin" in text):
        return "rope_attention_rotation"
    if any(token in text for token in ("clone", "squeeze", "transpose", "layout")):
        return "layout_transpose_materialization"
    if any(token in text for token in ("q_norm", "k_norm", "layernorm", "input_layernorm", "post_attention_layernorm")):
        return "normalization_reduction"
    if "sum" in text and any(token in text for token in ("pow", "mul_sum", "rsqrt")):
        return "normalization_reduction"
    return "other_fused_arithmetic"


def _rms(raw_artifact: str) -> float | None:
    try:
        data = json.loads((ROOT / raw_artifact).read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    stats = data.get("original_coordinate_statistics") or {}
    write = stats.get("PARAMETER_WRITE") or {}
    if isinstance(write, list):
        effect = sum(float(row.get("effect_energy", 0) or 0) for row in write)
        repair = sum(float(row.get("repair_energy", 0) or 0) for row in write)
    elif isinstance(write, dict):
        effect = float(write.get("effect_energy", 0) or 0)
        repair = float(write.get("repair_energy", 0) or 0)
    else:
        return None
    if repair <= 0:
        return None
    return math.sqrt(max(effect, 0.0) / repair)


def build() -> dict[str, Any]:
    source = json.loads(SOURCE.read_text())
    ledger_path = ROOT / "results/property/case_causal_audit_v1/root_cause_closure_current.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.is_file() else {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source.get("rows", []):
        symbol = str(row.get("symbol") or "")
        carrier = str(row.get("carrier") or "")
        family = _candidate_family(symbol, carrier)
        grouped[family].append(
            {
                "carrier": carrier,
                "symbol": symbol,
                "case_id": row.get("case_id"),
                "task_id": (row.get("runtime_boundary") or {}).get("task_id"),
                "exact_aot_endpoint_id": (row.get("runtime_boundary") or {}).get(
                    "exact_aot_endpoint_id"
                ),
                "write_rms": _rms(str(row.get("raw_artifact") or "")),
                "same_local_operands": (row.get("reference_comparison_scope") or {}).get(
                    "same_local_operands"
                ),
                "upstream_differences_possible": (
                    row.get("reference_comparison_scope") or {}
                ).get("includes_possible_upstream_differences"),
            }
        )

    overlap = {
        "embedding_dense_backward": "not yet represented as an isolated natural embedding-dense root; may overlap the NLL loss path",
        "nll_backward": "overlaps selected-NLL and softcapped-NLL candidates",
        "softmax_attention_backward": "overlaps softmax/saved-state and attention-region groups",
        "normalization_reduction": "overlaps RMSNorm and normalization records; source may still split into a new LayerNorm/q_norm family",
        "rope_attention_rotation": "overlaps RoPE and attention-region groups",
        "silu_gating": "overlaps the existing SiLU group",
        "layout_transpose_materialization": "not yet represented as a closed natural root; the specific DeepSeek/Qwen3 GQA repeat_kv boundary was screened as exact identity, so remaining priority is the other fused transpose/materialization paths",
        "recurrence_softplus": "overlaps closed Mamba component roots and the RWKV time-decay root; remaining fused recurrence regions still need source isolation",
        "other_fused_arithmetic": "unresolved; requires semantic inspection before promotion",
    }
    priority = {
        "embedding_dense_backward": "HIGH",
        "layout_transpose_materialization": "HIGH",
        "recurrence_softplus": "HIGH",
        "normalization_reduction": "HIGH_IF_LAYER_NORM_OR_QK_NORM_ISOLATES",
        "nll_backward": "MEDIUM",
        "other_fused_arithmetic": "MEDIUM",
        "softmax_attention_backward": "LOW_OVERLAP_WITH_EXISTING_GROUP",
        "rope_attention_rotation": "LOW_OVERLAP_WITH_EXISTING_GROUP",
        "silu_gating": "LOW_OVERLAP_WITH_EXISTING_GROUP",
    }
    adapter = {
        "embedding_dense_backward": "NEW_EMBEDDING_BOUNDARY_ADAPTER",
        "nll_backward": "REUSE_NLL_CAPTURE_AFTER_RUNTIME_BINDING",
        "softmax_attention_backward": "REUSE_SAVED_STATE_OR_ATTENTION_PROBE",
        "normalization_reduction": "REUSE_ROW_REDUCTION_AFTER_RUNTIME_BINDING",
        "rope_attention_rotation": "REUSE_ROPE_PATH_PROBE",
        "layout_transpose_materialization": "NEW_LAYOUT_BOUNDARY_ADAPTER",
        "silu_gating": "REUSE_SILU_FACTORIAL_PROBE",
        "recurrence_softplus": "NEW_RECURRENCE_BOUNDARY_ADAPTER",
        "other_fused_arithmetic": "NEW_SEMANTIC_ADAPTER_REQUIRED",
    }

    aligned_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    if ALIGNED_SCREEN.is_file():
        try:
            screen = json.loads(ALIGNED_SCREEN.read_text())
            for item in screen.get("records", []):
                aligned_by_key[(str(item.get("carrier")), str(item.get("symbol")))] = item
        except (OSError, json.JSONDecodeError, TypeError):
            aligned_by_key = {}

    groups = []
    for family, rows in sorted(
        grouped.items(),
        key=lambda item: max((row["write_rms"] or 0.0) for row in item[1]),
        reverse=True,
    ):
        rms_values = [row["write_rms"] for row in rows if row["write_rms"] is not None]
        screen_rows = [
            aligned_by_key[(str(row["carrier"]), str(row["symbol"]))]
            for row in rows
            if (str(row["carrier"]), str(row["symbol"])) in aligned_by_key
        ]
        groups.append(
            {
                "candidate_family": family,
                "record_count": len(rows),
                "max_fixed_suite_write_rms": max(rms_values) if rms_values else None,
                "median_fixed_suite_write_rms": (
                    sorted(rms_values)[len(rms_values) // 2] if rms_values else None
                ),
                "carriers": sorted({row["carrier"] for row in rows if row["carrier"]}),
                "symbols": sorted({row["symbol"] for row in rows if row["symbol"]}),
                "all_same_local_operands": all(
                    row["same_local_operands"] is True for row in rows
                ),
                "all_allow_no_upstream_difference": all(
                    row["upstream_differences_possible"] is False for row in rows
                ),
                "deduplication_note": overlap[family],
                "next_capture_priority": priority[family],
                "adapter_plan": adapter[family],
                "recommended_records": sorted(
                    rows,
                    key=lambda row: row["write_rms"] or 0.0,
                    reverse=True,
                )[:3],
                "promotion_status": "CANDIDATE_ONLY_REGION_SUBSTITUTION",
                "aligned_screen": {
                    "available": bool(screen_rows),
                    "record_count": len(screen_rows),
                    "all_aligned_negative": bool(screen_rows)
                    and all(
                        item["aligned_negative_count"] == item["state_count"]
                        for item in screen_rows
                    ),
                    "strongest_absolute_aligned_mean": (
                        max(abs(float(item["aligned_mean"])) for item in screen_rows)
                        if screen_rows else None
                    ),
                    "scope": "DESCRIPTIVE_REGION_SUBSTITUTION_ONLY",
                },
                "required_next_observations": [
                    "same-local-operand replay at the selected boundary",
                    "single-source path-preserving intervention",
                    "parameter-write confirmation over a predeclared state bank",
                    "deduplicate against the existing problem-group ledger",
                ],
            }
        )

    return {
        "schema": "natural-training-candidate-frontier-v1",
        "status": "TRIAGE_ONLY_NOT_A_ROOT_CAUSE_COUNT",
        "source_scope": "31 valid real-model signature-region records retained by the causal audit",
        "record_count": sum(len(rows) for rows in grouped.values()),
        "candidate_family_count": len(groups),
        "active_problem_group_count_unchanged": ledger.get("summary", {}).get(
            "active_problem_group_count"
        ),
        "negative_control_count_unchanged": ledger.get("summary", {}).get(
            "negative_control_count"
        ),
        "groups": groups,
        "interpretation": (
            "These groups rank where a new natural problem group could be found. "
            "They do not increase the scientific count until the region is isolated "
            "and reaches a real training parameter under a declared state protocol."
        ),
    }


if __name__ == "__main__":
    result = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "record_count": result["record_count"],
        "candidate_family_count": result["candidate_family_count"],
        "families": [
            {
                "candidate_family": group["candidate_family"],
                "record_count": group["record_count"],
                "max_fixed_suite_write_rms": group["max_fixed_suite_write_rms"],
            }
            for group in result["groups"]
        ],
    }, ensure_ascii=False, indent=2))
