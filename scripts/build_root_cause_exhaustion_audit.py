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
    silu_factorial = read(
        "results/property/numerical_coverage_v1/silu_factorial_explicit_source_run3/raw/"
        "mapped_backward_667_in_out_ptr0-silu-common-input.json"
    )
    rotary = read("results/property/numerical_coverage_v1/ministral_fused_rotary_optimizer_condition_summary_v1.json")
    mm = read("results/property/case_causal_audit_v1/mm_conditional_sources.json")
    assert softmax["status"] == "FIXED_CALL_LOCAL_ROOT_CONFIRMED" and softmax["row_count"] == 114688
    assert liger["status"] == "COMPLETE"
    assert liger["update_equivalence"]["decision"] == "FIXED_SUITE_UPDATE_EQUIVALENT"
    assert silu["status"] == "COMPLETE" and "original_coordinate_statistics" in silu
    assert silu_factorial["status"] == "COMPLETE" and silu_factorial["reference_comparison_scope"]["same_local_operands"]
    assert "conditions" in rotary and mm["cases"]

    rows = [
        ("adamw8bit_moment_quantization", "end-to-end declared-protocol causal chain", "mean bias as the unique mediator of loss", "an intervention independently changing mean, variance, and coordinate/time structure while preserving the others"),
        ("liger_fused_linear_ce_dw_accumulation", "FP32 addition-order source plus nonzero profile means reproduced on disjoint length-64 and length-256 confirmation banks", "actual-write generality or causal sufficiency for loss", "original-coordinate writes or a predeclared longer paired loss comparison"),
        ("softmax_saved_state_backward", "saved-score/statistic inconsistency causes row-mass defect on every retained row; a 1024-step declared trajectory confirms the restoration reaches gradient/write and produces trajectory non-identity", "natural population mean update bias or material loss consequence", "independent state-bank sampling with a predeclared loss endpoint and consistent-saved-state restoration propagated through gradient and parameter write"),
        ("attention_state_to_q_projection_region", "one delayed-BF16-materialization contributor and its transport path", "one unique source for the whole attention region", "separate interventions for remaining upstream-logit and residual-stream contributors"),
        ("mm_gemm_output_and_accumulation", "case-specific same-operands decompositions: output rounding only for Qwen128, kernel plus output rounding for Qwen64/Mamba, and kernel arithmetic for Phi", "one universal MM root or a common natural-population mean bias", "a new shared factorial only if a cross-case MM mechanism claim is needed"),
        ("silu_backward_evaluation", "an AST-checked explicit-exponential source variant changes the selected gate-gradient endpoint through local, gradient, moment, update and write stages", "natural population mean bias and the separate sigmoid, expression-order, or final-cast main effects", "componentwise intermediate outputs and an independent natural-state confirmation"),
        ("fused_rope_position_scaling", "same-input implementation difference and optimizer-state dependence", "low-level arithmetic source or moments separately from step counter", "same-operands materialization variants and a fixed-step-counter optimizer-state comparison"),
        ("gemma_gelu_backward_evaluation", "source-choice-sensitive response for the bound GELU backward endpoint; native tanh and fused multiply-add agree in the recorded profile", "a natural-input mean bias or a universal GELU source", "an independent natural-state bank and componentwise source intervention if that stronger claim is needed"),
        ("gemma_rms_feature_reduction_order", "the tested FP32 feature-reduction order is a tiny or exact-identity source control with no confirmed direction", "a different RMS source being responsible for the earlier region effect", "a new predeclared RMS intervention; no further inference is available for this tested order"),
        ("granite_router_topk_selection", "the tested equal-score tie-order variant leaves selected experts, values, gradients, writes and loss unchanged", "all selection implementations being safe or every non-tie/NaN case being equivalent", "a separately declared changed-selection or non-tie intervention"),
        ("granite_moe_expert_contribution_order", "reversing FP32 expert-contribution accumulation creates a small fixed-suite write difference", "a natural-population mean bias or a training-quality consequence", "an independent state-bank confirmation that isolates expert accumulation from routing"),
        ("gemma_bound_square_sum", "no scientific source is identifiable because the attempted endpoint did not match the live executed source", "whether the intended square-sum comparison has any numerical effect", "bind the live endpoint and repeat the declared comparison"),
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
