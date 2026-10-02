#!/usr/bin/env python3
"""Rank natural-training candidates without promoting them to problem groups.

This report is intentionally conservative.  A real-model endpoint, a large
fixed-suite effect, or a stable projection is only a *candidate* until a
same-input source intervention and a parameter-reachable confirmation are
available.  The report is used to decide what to isolate next; it does not
change the root-cause ledger or manufacture new groups from coverage counts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "results/property/case_causal_audit_v1/root_cause_closure_current.json"
FRONTIER = ROOT / "results/property/case_causal_audit_v1/natural_candidate_frontier.json"
SEMANTIC = ROOT / "results/property/case_causal_audit_v1/natural_semantic_gap_frontier.json"
OUT = ROOT / "results/property/case_causal_audit_v1/natural_problem_group_discovery_current.json"
PROBE_ROOT = ROOT / "results/property/new_problem_group_search_v1"

RECENT_SCREENS = [
    (
        "bert_gelu_natural_32_20260920.json",
        "BERT MLP GELU evaluation/materialization",
        "BERT native model-dtype GELU versus exact FP32 GELU/writeback",
    ),
    (
        "llama_swiglu_gate_product_natural_16_20260920.json",
        "Llama SwiGLU gate-times-up product materialization",
        "native BF16 gate product versus FP32 product/writeback",
    ),
    (
        "olmoe_router_score_natural_16_20260920.json",
        "OLMoE router score materialization",
        "native router-score dtype versus FP32 router score with native top-k frozen",
    ),
    (
        "olmoe_router_score_natural_32_20260920.json",
        "OLMoE router score materialization",
        "native router-score dtype versus FP32 router score with native top-k frozen",
    ),
    (
        "olmoe_router_score_natural_64_20260920.json",
        "OLMoE router score materialization",
        "native router-score dtype versus FP32 router score with native top-k frozen",
    ),
    (
        "olmoe_router_score_natural_128_20260920.json",
        "OLMoE router score materialization",
        "native router-score dtype versus FP32 router score with native top-k frozen",
    ),
    (
        "qwen3_attention_output_layout_layer13_kproj_16_20260920.json",
        "Qwen3 attention output layout materialization (k_proj)",
        "native transpose(1,2).contiguous() versus equivalent permute(...).reshape",
    ),
    (
        "mamba_fused_x_proj_layer0_natural_24_20260920.json",
        "Mamba fused recurrence x_proj projection",
        "official fused recurrence versus sequential reference on a fixed layer-0 x_proj boundary",
    ),
    (
        "gemma4_audio_attention_softmax_natural_16_20260920.json",
        "Gemma4 audio attention softmax materialization",
        "native FP32 softmax versus FP64 softmax with logits and softcap fixed",
    ),
    (
        "gemma4_audio_attention_softmax_layer1_natural_16_20260920.json",
        "Gemma4 audio attention softmax materialization",
        "same isolated softmax boundary in audio-attention layer 1",
    ),
    (
        "rwkv_channel_mix_readme_128_20260920.json",
        "RWKV channel-mix square/ReLU materialization",
        "native model-dtype square(ReLU(key)) versus FP32 square/ReLU with original-dtype write-back",
    ),
    (
        "rwkv_channel_mix_block1_readme_64_20260920.json",
        "RWKV channel-mix square/ReLU materialization",
        "same isolated boundary on RWKV block 1",
    ),
    (
        "../root_cause_closure_v1/silu_single_source_intervention_v4_16_20260920.json",
        "DeepSeek generated SiLU backward source intervention",
        "six path-preserving arithmetic variants on a 16-state natural bank",
    ),
]


def _rank(row: Dict[str, Any]) -> int:
    """Priority for source isolation, not a probability of finding a root."""

    priority = row.get("next_capture_priority", "LOW")
    if priority == "HIGH":
        return 3
    if priority == "HIGH_IF_LAYER_NORM_OR_QK_NORM_ISOLATES":
        return 3
    if priority == "MEDIUM":
        return 2
    return 1


def _collect_relative_result_paths(value: Any, found: Set[str]) -> None:
    """Collect retained result paths without treating unlisted files as cases.

    The closure ledger contains active groups, open groups, controls, and
    coverage records.  An exhaustive inventory is useful for finding a saved
    high-effect probe which was never attached to one of those records, but it
    must remain an audit signal rather than a promotion mechanism.
    """

    if isinstance(value, dict):
        for item in value.values():
            _collect_relative_result_paths(item, found)
    elif isinstance(value, list):
        for item in value:
            _collect_relative_result_paths(item, found)
    elif isinstance(value, str) and value.startswith("results/") and value.endswith(".json"):
        found.add(value)


def _probe_write_rms(summary: Any) -> List[float]:
    values: List[float] = []
    if isinstance(summary, dict):
        for key, value in summary.items():
            if "write_effect_rms_mean" in str(key) and isinstance(value, (int, float)):
                values.append(float(value))
            else:
                values.extend(_probe_write_rms(value))
    elif isinstance(summary, list):
        for value in summary:
            values.extend(_probe_write_rms(value))
    return values


def _build_saved_probe_inventory(ledger: Dict[str, Any]) -> Dict[str, Any]:
    """Audit every saved natural probe, without promoting any probe.

    This closes a bookkeeping gap in the search frontier: the frontier is
    built from signature-region records, while several family-specific probes
    live in the same result directory.  A file absent from the ledger is not
    automatically a new problem group; it is simply surfaced for review.
    """

    referenced: Set[str] = set()
    _collect_relative_result_paths(ledger, referenced)
    known_duplicate_groups = {
        "results/property/new_problem_group_search_v1/bert_linear_fused_addmm_cuda32_20260919.json": "bert_fused_addmm_bias_materialization",
        "results/property/new_problem_group_search_v1/bert_softmax_materialization_cuda_32_20260919.json": "bert_attention_softmax_materialization",
        "results/property/new_problem_group_search_v1/bert_tiny_layernorm_natural_20260918.json": "bert_layernorm_compiled_materialization",
        "results/property/new_problem_group_search_v1/deberta_disentangled_attention_materialization_natural_16_20260919.json": "deberta_disentangled_relative_attention_materialization",
        "results/property/new_problem_group_search_v1/deberta_disentangled_attention_materialization_natural_32_20260919.json": "deberta_disentangled_relative_attention_materialization",
        "results/property/new_problem_group_search_v1/gemma4_audio_language_output_projection_natural_8_20260919.json": "gemma4_audio_output_projection_materialization",
        "results/property/new_problem_group_search_v1/gemma4_audio_subsample_conv_layer0_natural_8_20260919.json": "gemma4_audio_subsampling_convolution_materialization",
        "results/property/new_problem_group_search_v1/gemma4_audio_subsample_conv_layer1_natural_16_20260920.json": "gemma4_audio_subsampling_convolution_materialization",
        "results/property/new_problem_group_search_v1/gemma4_audio_subsample_conv_layer1_natural_8_20260919.json": "gemma4_audio_subsampling_convolution_materialization",
        "results/property/new_problem_group_search_v1/mamba_official_local_scan_layer3_8_20260920.json": "mamba_fused_selective_scan_reassociation",
        "results/property/new_problem_group_search_v1/gemma4_audio_attention_softmax_layer0_16_repeat_20260920.json": "gemma4_audio_attention_softmax_probability_materialization",
    }
    reviewed_nonpromotion_files = {
        "results/property/new_problem_group_search_v1/rwkv_channel_mix_readme_128_20260920.json",
        "results/property/new_problem_group_search_v1/rwkv_channel_mix_block1_readme_64_20260920.json",
    }
    rows: List[Dict[str, Any]] = []
    for path in sorted(PROBE_ROOT.rglob("*.json")):
        relative = str(path.relative_to(ROOT))
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict):
            continue
        summary = raw.get("summary")
        values = _probe_write_rms(summary)
        if not values:
            continue
        rows.append(
            {
                "file": relative,
                "operator": raw.get("operator"),
                "status": raw.get("status"),
                "max_write_effect_rms_mean": max(abs(value) for value in values),
                "ledger_referenced": relative in referenced,
                "known_duplicate_group": known_duplicate_groups.get(relative),
            }
        )
    high_effect = [row for row in rows if row["max_write_effect_rms_mean"] >= 0.01]
    unreferenced_high_effect = [
        row
        for row in high_effect
        if not row["ledger_referenced"] and row["file"] not in reviewed_nonpromotion_files
    ]
    return {
        "scope": "results/property/new_problem_group_search_v1/**/*.json",
        "write_effect_triage_threshold": 0.01,
        "probe_count_with_write_effect": len(rows),
        "high_effect_probe_count": len(high_effect),
        "unreferenced_high_effect_probe_count": len(unreferenced_high_effect),
        "unreferenced_high_effect_probes": unreferenced_high_effect,
        "unreferenced_high_effects_all_classified": all(
            row.get("known_duplicate_group") for row in unreferenced_high_effect
        ),
        "reviewed_nonpromotion_files": sorted(reviewed_nonpromotion_files),
        "interpretation": (
            "This inventory is a bookkeeping audit only. A saved probe can be "
            "unreferenced because it is an alternate run, a negative/control "
            "record, or a duplicate. It never promotes a problem group without "
            "the source-isolation, parameter-reachability, and deduplication gate."
        ),
    }


FAMILY_SEARCH_PLAN = {
    "RECURRENCE": (3, "fused recurrence component intervention", "large region effects exist but source components are not all isolated"),
    "DATA_MOVEMENT_LAYOUT": (3, "same-input layout/materialization intervention", "large coverage family with no closed numerical source"),
    "ELEMENTWISE": (2, "select an unseen real activation boundary", "many positions are measured but most are not source-isolated"),
    "FUSED_MIXED": (1, "repair runtime binding before scientific promotion", "execution path is incomplete; do not interpret its measurements as bias"),
    "SOFTPLUS": (1, "deduplicate against the existing Mamba softplus group", "the family has a closed Mamba boundary; remaining records are not automatically new"),
}


def build() -> Dict[str, Any]:
    ledger = json.loads(LEDGER.read_text())
    frontier = json.loads(FRONTIER.read_text())
    semantic = json.loads(SEMANTIC.read_text())

    open_groups = [x["problem_group"] for x in ledger.get("open_root_cause_groups", [])]
    active_groups = [x["problem_group"] for x in ledger.get("active_problem_groups", [])]
    family_rows: List[Dict[str, Any]] = []
    for row in frontier.get("groups", []):
        family_rows.append(
            {
                "candidate_family": row["candidate_family"],
                "record_count": row["record_count"],
                "max_fixed_suite_write_rms": row.get("max_fixed_suite_write_rms"),
                "median_fixed_suite_write_rms": row.get("median_fixed_suite_write_rms"),
                "next_capture_priority": row.get("next_capture_priority", "LOW"),
                "priority_rank": _rank(row),
                "adapter_plan": row.get("adapter_plan"),
                "promotion_status": row.get("promotion_status"),
                "deduplication_note": row.get("deduplication_note"),
                "required_next_observations": row.get("required_next_observations", []),
                "natural_problem_group_eligible_now": False,
                "eligibility_reason": (
                    "candidate records are region substitutions and may contain upstream "
                    "differences; source isolation and parameter-write confirmation are missing"
                ),
            }
        )
    family_rows.sort(
        key=lambda x: (
            -x["priority_rank"],
            -(x["max_fixed_suite_write_rms"] or 0.0),
            x["candidate_family"],
        )
    )

    remaining_search_frontier: List[Dict[str, Any]] = []
    for family in ledger.get("operator_family_frontier", {}).get("families", []):
        family_id = str(family.get("family_id"))
        plan = FAMILY_SEARCH_PLAN.get(family_id)
        status = str(family.get("root_cause_status", ""))
        if plan is None and "MEASUREMENT_ONLY" not in status:
            continue
        rank, next_action, reason = plan or (
            2,
            "inspect family-specific source boundary",
            "measurement exists without a distinct source intervention",
        )
        remaining_search_frontier.append(
            {
                "family_id": family_id,
                "classified_positions": family.get("classified_positions"),
                "root_cause_status": status,
                "priority_rank": rank,
                "next_action": next_action,
                "reason": reason,
                "eligible_for_final_group_now": False,
            }
        )
    remaining_search_frontier.sort(key=lambda x: (-x["priority_rank"], x["family_id"]))

    # A semantic gap is a scheduling signal only.  It is deliberately not
    # interpreted as a missing bias family or a new problem group.
    semantic_gaps = [x for x in semantic.get("gaps", []) if x.get("status") not in ("EXISTING_FAMILY_NEEDS_NEW_ROOT_ONLY",)]

    # These are the only rows that can be promoted by the next cycle.  Keeping
    # the gate explicit prevents a stable screen from silently becoming a
    # scientific result.
    promotion_gate = [
        "real model/checkpoint and real training-derived input states",
        "same-input or same-local-operand candidate/reference comparison",
        "one declared numerical source intervention, with all other paths held fixed",
        "actual gradient and parameter-write endpoint on a predeclared state bank",
        "deduplication against the current root-cause ledger",
    ]

    recent_screens: List[Dict[str, Any]] = []
    for filename, operator, intervention in RECENT_SCREENS:
        path = ROOT / "results/property/new_problem_group_search_v1" / filename
        if not path.is_file():
            continue
        raw = json.loads(path.read_text())
        summary = raw.get("summary", {})
        metric_summary = summary
        if isinstance(summary, dict) and isinstance(summary.get("modes"), dict):
            # Some family probes expose one summary per intervention mode.  A
            # recent screen is only a triage record, so report the first mode
            # as a compact representative while preserving the raw artifact.
            first_mode = next(iter(summary["modes"].values()), {})
            if isinstance(first_mode, dict):
                metric_summary = first_mode
        interval = (
            metric_summary.get("confirmation_aligned_write_interval_normal_95")
            or metric_summary.get("aligned_write_interval_normal_95")
            or metric_summary.get("write_aligned_interval_normal_95")
        )
        exact_zero = (
            metric_summary.get("gradient_effect_rms_mean") == 0.0
            and metric_summary.get("write_effect_rms_mean") == 0.0
            and metric_summary.get("loss_difference_interval_normal_95") == [0.0, 0.0]
        )
        if exact_zero:
            status = "SCREENED_EXACT_IDENTITY"
            reason = "all observed gradient/write/loss effects are exactly zero"
        elif operator == "OLMoE router score materialization":
            status = "MERGED_WITH_EXISTING_ROUTER_SCORE_GROUP"
            reason = (
                "the source boundary is isolated with native top-k indices held fixed, but "
                "the 16, 32 and 64-state screens were high-variance; the 128-state replay has "
                "a negative aligned-write interval and confirms the same router-score/weight "
                "materialization source already closed for Granite, so it is merged rather than "
                "counted as a second problem group"
            )
        elif operator == "Mamba fused recurrence x_proj projection":
            status = "SCREENED_FUSED_RECURRENCE_NO_CONFIRMED_ALIGNED_BIAS"
            reason = (
                "the official fused recurrence replay is source-localized to the layer-0 x_proj "
                "write boundary, but 11/5 held-out projections have an aligned-gradient interval "
                "crossing zero; the large write RMS is retained as a recurrence candidate, not "
                "promoted to a signed-bias problem group"
            )
        elif operator == "DeepSeek generated SiLU backward source intervention":
            status = "SCREENED_ARITHMETIC_VARIANTS_DO_NOT_CLOSE_SOURCE"
            reason = (
                "the 16-state path-preserving replay of all six tested arithmetic variants "
                "produces only about 6.94e-5 write RMS with zero loss difference, far below "
                "the retained native/reference profile; it excludes these variants as sufficient "
                "causes but does not identify the remaining fused materialization or instruction source"
            )
        elif operator == "Gemma4 audio attention softmax materialization":
            status = "SCREENED_HIGH_VARIANCE_NO_DIRECTIONAL_BIAS"
            reason = (
                "the FP32-versus-FP64 softmax boundary is isolated in two real audio-attention "
                "layers, but both aligned-write and held-out direction intervals cross zero; "
                "the large RMS effect is retained as a variance candidate rather than a signed-bias group"
            )
        elif operator == "RWKV channel-mix square/ReLU materialization":
            status = "SCREENED_SPARSE_EFFECT_NO_DIRECTIONAL_BIAS"
            reason = (
                "the square/ReLU boundary is isolated on real README-text states in two blocks, but "
                "only one state in each run changes and the remaining states are exact identity; "
                "the aligned intervals cross zero, so this is retained as a sparse screen rather "
                "than a signed-bias group"
            )
        else:
            status = "REQUIRES_LEDGER_REVIEW"
            reason = "new screen is not promoted automatically; source and deduplication review required"
        recent_screens.append(
            {
                "file": str(path.relative_to(ROOT)),
                "operator": operator,
                "intervention": intervention,
                "status": status,
                "aligned_interval": interval,
                "write_effect_rms_mean": metric_summary.get("write_effect_rms_mean"),
                "natural_problem_group_promoted": False,
                "reason": reason,
            }
        )

    result = {
        "schema": "natural-problem-group-discovery-v1",
        "status": "TRIAGE_ONLY_NO_PROMOTION",
        "counting_rule": (
            "A candidate becomes a natural problem group only after source isolation, "
            "parameter reachability and deduplication. Coverage records and stable "
            "region screens never promote themselves."
        ),
        "ledger_snapshot": {
            "problem_group_count": ledger["summary"]["problem_group_count"],
            "active_problem_group_count": ledger["summary"]["active_problem_group_count"],
            "root_cause_closed_final_count": ledger["summary"]["root_cause_closed_final_count"],
            "root_cause_unresolved_count": ledger["summary"]["root_cause_unresolved_count"],
            "open_problem_groups": open_groups,
            "active_problem_groups": active_groups,
        },
        "candidate_frontier": {
            "record_count": frontier.get("record_count"),
            "candidate_family_count": frontier.get("candidate_family_count"),
            "families": family_rows,
        },
        "saved_probe_inventory": _build_saved_probe_inventory(ledger),
        "remaining_operator_family_frontier": remaining_search_frontier,
        "recent_natural_screens": recent_screens,
        "semantic_gap_audit": {
            "model_count": semantic.get("model_count"),
            "semantic_gap_count": semantic.get("semantic_gap_count"),
            "unresolved_scheduling_gaps": semantic_gaps,
            "interpretation": (
                "No semantic gap means the current model inventory has no obvious "
                "unrepresented feature; it does not prove that every implementation "
                "boundary has been tested."
            ),
        },
        "next_source_isolation_order": [x["candidate_family"] for x in family_rows],
        "promotion_gate": promotion_gate,
        "non_promotions": [
            {
                "kind": "open_root_cause",
                "items": open_groups,
                "reason": "source competition remains; keep separate from new groups",
            },
            {
                "kind": "candidate_region",
                "items": [x["candidate_family"] for x in family_rows],
                "reason": "region substitution can include upstream differences",
            },
            {
                "kind": "conditional_operator_only",
                "items": ["indexed_accumulation_reduction_order"],
                "reason": "no natural training boundary or training-state population",
            },
        ],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


if __name__ == "__main__":
    report = build()
    print(
        json.dumps(
            {
                "output": str(OUT),
                "candidate_family_count": report["candidate_frontier"]["candidate_family_count"],
                "semantic_gap_count": report["semantic_gap_audit"]["semantic_gap_count"],
                "promoted_now": 0,
                "root_cause_closed_final_count": report["ledger_snapshot"]["root_cause_closed_final_count"],
            },
            ensure_ascii=False,
        )
    )
