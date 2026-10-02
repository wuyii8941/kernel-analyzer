#!/usr/bin/env python3
"""Build a conservative audit separating source, natural bias, and loss claims.

This report is intentionally not a new statistical test.  It joins the current
root-cause ledger with an explicit, reviewable claim boundary for each problem
group.  A source-closed row is not silently promoted to a natural-population
bias or a training-quality result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "results/property/case_causal_audit_v1/root_cause_closure_current.json"
OUT_JSON = ROOT / "results/property/root_cause_closure_v1/natural_bias_consequence_audit_v1.json"
MEAN_REPORT = ROOT / "results/property/root_cause_closure_v1/all_case_mean_bias_audit_v1.json"


def read(path: Path) -> Any:
    return json.loads(path.read_text())


# These are claim-scope annotations, not labels inferred from a nonzero norm.
# They are deliberately conservative and must be reviewed whenever the ledger
# changes.  The supporting evidence and the precise source statement always
# come from the ledger row itself.
LOSS = {
    "adamw8bit_moment_quantization": "CONFIRMED_DECLARED_SETTING_IMPROVEMENT",
    "liger_fused_linear_ce_dw_accumulation": "TRAJECTORY_NON_IDENTITY_NO_MONOTONE_QUALITY",
    "mm_gemm_output_and_accumulation": "HISTORICAL_SEPARATION_CASE_DEPENDENT",
    "softmax_saved_state_backward": "TRAJECTORY_NON_IDENTITY_NO_MONOTONE_QUALITY",
    "silu_backward_evaluation": "SMALL_FEEDBACK_SUSTAINED_SEPARATION",
    "attention_state_to_q_projection_region": "HISTORICAL_PAIRED_TRAJECTORY_ONLY",
    "fused_rope_position_scaling": "NOT_MEASURED",
    "rmsnorm_cast_materialization": "NOT_MEASURED",
    "bert_layernorm_compiled_materialization": "NOT_MEASURED",
    "gemma3_vision_patch_convolution": "NO_LOSS_DIRECTION_CONFIRMED_FIXED_SUITE",
    "deepseek_embedding_backward_accumulation": "FORWARD_LOSS_IDENTICAL_SINGLE_STEP_ONLY",
    "gemma4_causal_nll_loss_evaluation": "NO_LOSS_DIRECTION_CONFIRMED_FIXED_SUITE",
    "gemma_gelu_backward_evaluation": "NOT_MEASURED",
    "gemma_rms_feature_reduction_order": "NOT_MEASURED",
    "granite_router_topk_selection": "NO_DIFFERENCE_IN_FIXED_SUITE",
    "granite_moe_expert_contribution_order": "NOT_MEASURED",
}


def main() -> None:
    ledger = read(LEDGER)
    mean_report = read(MEAN_REPORT)
    mean_statuses = {row["problem_group"]: row["mean_bias_status"] for row in mean_report["rows"]}
    if set(mean_statuses) != {row["problem_group"] for row in ledger["rows"]}:
        raise RuntimeError("mean-bias report and root-cause ledger have different problem groups")
    rows: list[dict[str, Any]] = []
    for row in ledger["rows"]:
        group = row["problem_group"]
        if group not in LOSS:
            raise RuntimeError(f"missing explicit claim scope for {group}")
        closure = row["closure"]
        rows.append({
            "problem_group": group,
            "closure": closure,
            "source_statement": row["what_is_proven"],
            "natural_bias_status": mean_statuses[group],
            "loss_status": LOSS[group],
            "claim_boundary": row["next_needed_observation"],
            "evidence": row["evidence"],
        })

    active_rows = [row for row in rows if not row["closure"].startswith("NEGATIVE")]

    report = {
        "schema": "kernel-analyzer-natural-bias-consequence-audit-v1",
        "status": "CONSERVATIVE_CLAIM_SCOPE_AUDIT",
        "source_ledger": str(LEDGER.relative_to(ROOT)),
        "rows": rows,
        "retired_records": [{
            "record_id": "gemma_bound_square_sum",
            "status": "RETIRED_EXECUTION_SOURCE_MISMATCH",
            "active_mainline": False,
            "reason": "The live execution source did not match the declared square-sum endpoint, so no valid same-endpoint measurement was formed.",
            "evidence": [
                "scripts/recover_gemma_square_sum_boundary.py",
                "scripts/diagnose_gemma_live_source.py",
                "results/property/training_numerical_analysis_v2/",
            ],
        }],
        "summary": {
            "active_problem_groups": len(active_rows),
            "source_or_local_closed": sum(
                row["closure"].startswith((
                    "END_TO_END", "SOURCE_CLOSED", "SOURCE_CHOICE_RESPONSE_CLOSED",
                    "LOCAL_", "CASE_SPECIFIC",
                ))
                for row in active_rows
            ),
            "semantic_region_closed": sum(
                row["closure"].startswith("SEMANTIC_REGION")
                for row in active_rows
            ),
            "history_aligned_frequency_supported": sum(
                row["natural_bias_status"].startswith("SUPPORTED_ALIGNED_MEAN")
                for row in active_rows
            ),
            "loss_improvement_confirmed": sum(
                row["loss_status"] == "CONFIRMED_DECLARED_SETTING_IMPROVEMENT"
                for row in active_rows
            ),
            "negative_controls": sum(row["closure"].startswith("NEGATIVE") for row in rows),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"json": str(OUT_JSON), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
