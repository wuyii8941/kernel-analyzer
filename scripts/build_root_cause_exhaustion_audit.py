#!/usr/bin/env python3
"""Record which root-cause conclusions are derivable from retained evidence."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/property/root_cause_closure_v1/exhaustion_audit.json"


def read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def main() -> None:
    softmax = read("results/property/root_cause_closure_v1/softmax_saved_state.json")
    liger = read("results/property/root_cause_closure_v1/liger_chunk_order_reanalysis.json")
    silu = read("results/property/numerical_coverage_v1/deepseek128_silu_batch000/raw/mapped_backward_667_in_out_ptr0-silu-common-input.json")
    rotary = read("results/property/numerical_coverage_v1/ministral_fused_rotary_optimizer_condition_summary_v1.json")
    mm = read("results/property/case_causal_audit_v1/mm_conditional_sources.json")
    assert softmax["status"] == "FIXED_CALL_LOCAL_ROOT_CONFIRMED" and softmax["row_count"] == 114688
    assert liger["status"] == "COMPLETE"
    assert liger["update_equivalence"]["decision"] == "FIXED_SUITE_UPDATE_EQUIVALENT"
    assert silu["status"] == "COMPLETE" and "original_coordinate_statistics" in silu
    assert "conditions" in rotary and mm["cases"]

    rows = [
        ("adamw8bit_moment_quantization", "end-to-end declared-protocol causal chain", "mean bias as the unique mediator of loss", "an intervention independently changing mean, variance, and coordinate/time structure while preserving the others"),
        ("liger_fused_linear_ce_dw_accumulation", "FP32 addition-order source plus nonzero profile means over observed declared units", "fresh population generality or causal sufficiency for loss", "new independently sampled states and a matched training intervention changing only addition order"),
        ("softmax_saved_state_backward", "saved-score/statistic inconsistency causes row-mass defect on every retained row", "natural population mean update bias or loss consequence", "independent states with consistent-saved-state restoration propagated through gradient and parameter write"),
        ("attention_state_to_q_projection_region", "one delayed-BF16-materialization contributor and its transport path", "one unique source for the whole attention region", "separate interventions for remaining upstream-logit and residual-stream contributors"),
        ("mm_gemm_output_and_accumulation", "conditional arithmetic and output-rounding effects in selected comparisons", "exact main effects and interaction on common coordinates", "candidate, arithmetic-only, cast-only, and joint output vectors on identical operands"),
        ("silu_backward_evaluation", "candidate/reference derivative evaluation differs and optimizer response has an even component", "which sigmoid, expression-order, or final-cast choice creates the natural residual", "common operands and componentwise intermediate outputs"),
        ("fused_rope_position_scaling", "same-input implementation difference and optimizer-state dependence", "low-level arithmetic source or moments separately from step counter", "same-operands materialization variants and a fixed-step-counter optimizer-state comparison"),
    ]
    payload = {
        "schema": "kernel-analyzer-root-cause-exhaustion-audit-v1",
        "status": "NO_ADDITIONAL_ROOT_CAUSE_CONCLUSION_FROM_RETAINED_EVIDENCE",
        "meaning": "Every stronger listed conclusion requires a new distinguishing observation; missing sufficient statistics are not treated as a negative result.",
        "scientific_problem_group_count": len(rows),
        "rows": [{
            "problem_group": group,
            "deducible_now": current,
            "not_deducible_from_retained_data": unavailable,
            "missing_observation": missing,
            "further_offline_inference": False,
        } for group, current, unavailable, missing in rows],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
