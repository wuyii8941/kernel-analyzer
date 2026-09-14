#!/usr/bin/env python3
"""Build the deduplicated scientific-case causal-closure table."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/property/case_causal_audit_v1/scientific_case_closure.json"

ROWS = [
    {
        "problem_group": "adamw8bit_moment_quantization",
        "implementation_boundary": "TorchAO AdamW8bit compiled moment save/read and parameter write",
        "numerical_source": "blockwise quantization discards a state residual which otherwise enters the next moment recurrence",
        "bias_formation": "the saved residual is fed back with beta-weighted history; measured residuals reconstruct later moment differences on fixed real gradient histories",
        "decisive_interventions": ["same-path residual readback off versus on", "coordinate-permuted equal-energy residual", "time-permuted residual", "key-parameter-only versus complement versus full compensation"],
        "training_outcome": "two eight-pair 1024-step comparisons support a material loss improvement for correct residual readback; coordinate permutation causes constructed numerical failure",
        "closure": "END_TO_END_CAUSAL_CHAIN_UNDER_DECLARED_PROTOCOL",
        "remaining_limit": "one checkpoint/model setting; constructed failure is not a natural collapse; a production-quality optimizer claim is not made",
        "evidence": ["docs/optimizer_update_family_audit.md", "docs/adamw8bit_residual_structure_20260913.md", "docs/priority_analysis_1_2_20260913.md"],
    },
    {
        "problem_group": "liger_fused_linear_ce_dw_accumulation",
        "implementation_boundary": "Liger fused linear cross entropy plus external chunk-gradient accumulation",
        "numerical_source": "finite-precision addition order and accumulation precision of identical dW chunk products",
        "bias_formation": "one late training state is reconstructed exactly from ordered chunk additions; historical protocols show persistent directions, but an expectation-level sign condition is not proved universally",
        "decisive_interventions": ["higher-precision accumulation", "same-FP32 different addition order"],
        "training_outcome": "paired trajectories separate; 4096/10000-step records do not support monotone degradation or collapse",
        "closure": "LOCAL_ARITHMETIC_EXACT_GLOBAL_CAUSAL_SUFFICIENCY_OPEN",
        "remaining_limit": "the local identity does not prove a nonzero population mean or that this term is sufficient for the observed loss difference",
        "evidence": ["docs/liger_language_mechanism_followup.md", "docs/liger_single_boundary_collapse_experiment.md", "results/property/bias_formation_final/liger_formation_analysis.json"],
    },
    {
        "problem_group": "mm_gemm_output_and_accumulation",
        "implementation_boundary": "training MM/GEMM at lm-head, projection, and input-gradient locations",
        "numerical_source": "kernel arithmetic residual and final output rounding are separable for selected fixed-input comparisons",
        "bias_formation": "output-rounding conditional effects reproduce for Qwen128; other MM locations show different or noncoherent geometry, so one common cross-model root is not established",
        "decisive_interventions": ["rounding-only replacement", "joint arithmetic-plus-rounding replacement", "row-permutation coupling control"],
        "training_outcome": "historical trajectories separate, but robust cross-state direction is case-dependent",
        "closure": "CONDITIONAL_SOURCE_ISOLATION_NOT_UNIVERSAL_MM_ROOT",
        "remaining_limit": "Qwen128 rounding-only preservation has seven tiny nonzero delivery remainders in 128 repeats; Phi transport reconstruction is approximate",
        "evidence": ["docs/source_aligned_repair.md", "results/property/case_causal_audit_v1/mm_conditional_sources.json", "results/property/bias_formation_final/phi_transport_mechanism.json"],
    },
    {
        "problem_group": "softmax_saved_state_backward",
        "implementation_boundary": "Qwen layer-27 softmax backward using reconstructed versus higher-precision saved probability",
        "numerical_source": "saved/reconstructed probability changes the analytic softmax VJP input",
        "bias_formation": "the semantic-region replacement reaches q/k gradients; positive/negative residual replay demonstrates an even AdamW response, not a nonzero natural source mean",
        "decisive_interventions": ["saved-probability replacement", "head permutation", "positive/negative gradient-residual replay"],
        "training_outcome": "trajectory non-identity exists; the historical persistent-direction gate is not passed",
        "closure": "SEMANTIC_SOURCE_AND_RESPONSE_SEPARATED_NATURAL_BIAS_OPEN",
        "remaining_limit": "head-specific pairing was rejected and the natural error distribution has not been shown to have the required nonzero mean component",
        "evidence": ["results/coverage/cases/qwen128_softmax_saved_p_trajectory.json", "docs/effective_antithetic_symmetry.md"],
    },
    {
        "problem_group": "silu_backward_evaluation",
        "implementation_boundary": "Qwen3-VL decomposed AOT SiLU VJP versus aten.silu_backward",
        "numerical_source": "different finite-precision evaluation of the same derivative expression",
        "bias_formation": "positive/negative residual replay reveals a strongly even early optimizer response; it does not establish why the natural VJP residual has nonzero expectation",
        "decisive_interventions": ["native backward replacement", "sham replacement", "positive/negative gradient-residual replay"],
        "training_outcome": "direct persistence is weak; later separation is feedback-sustained and the loss difference is small",
        "closure": "OPTIMIZER_RESPONSE_CAUSAL_NATURAL_SOURCE_BIAS_OPEN",
        "remaining_limit": "the natural source distribution and its expectation-level asymmetry are not isolated",
        "evidence": ["docs/effective_antithetic_symmetry.md", "results/round2/vl_silu_cause.json"],
    },
    {
        "problem_group": "attention_state_to_q_projection_region",
        "implementation_boundary": "Qwen layer-23 S_bwd/K to q-projection gradient semantic region",
        "numerical_source": "the S_bwd input to bmm_76 carries the measured regional difference",
        "bias_formation": "Gq=S_bwd K and dW=Gq^T H identify transport; S-only and joint restoration remove the selected direction while K-only does not",
        "decisive_interventions": ["S-only", "K-only", "joint", "sham"],
        "training_outcome": "a paired trajectory exists under the historical protocol",
        "closure": "CLOSED_SEMANTIC_REGION_SINGLE_OPERATION_ROOT_OPEN",
        "remaining_limit": "the upstream operation that creates the S_bwd difference is not uniquely isolated",
        "evidence": ["results/coverage/cases/l23_qproj_attention_state_region.json", "docs/l23_qproj_tile.md"],
    },
    {
        "problem_group": "fused_rope_position_scaling",
        "implementation_boundary": "Ministral compiled Triton fused RoPE/position-scaling calculation",
        "numerical_source": "same-input implementation evaluation difference; position scaling alone is ruled out as the unique explanation",
        "bias_formation": "identical gradient differences produce very different writes under warm versus reset optimizer state; reset also changes the step counter",
        "decisive_interventions": ["high versus low position", "warm state", "warm parameters with optimizer reset"],
        "training_outcome": "not measured",
        "closure": "STATE_RESPONSE_ESTABLISHED_SOURCE_ROOT_OPEN",
        "remaining_limit": "the low-level arithmetic source is not isolated and moments are not separated from step-counter effects",
        "evidence": ["docs/fused_rotary_position_scaling_audit.md", "docs/findings_and_competing_explanations.md"],
    },
    {
        "problem_group": "common_input_silu_and_rms_backward_families",
        "implementation_boundary": "99 same-input compiled SiLU and RMS-backward output locations",
        "numerical_source": "candidate/reference evaluation differs, but each FP32 reference changes multiple arithmetic choices together",
        "bias_formation": "fixed-suite total difference is measurable; nonzero mean bias and a unique arithmetic source are not established",
        "decisive_interventions": ["whole audited output replacement"],
        "training_outcome": "not measured per position",
        "closure": "MEASUREMENT_ONLY_FOR_ROOT_CAUSE",
        "remaining_limit": "requires targeted source-isolating variants before root-cause claims",
        "evidence": ["docs/silu_backward_common_input.md", "docs/new_family_source_audit.md"],
    },
    {
        "problem_group": "reference_graph_regions",
        "implementation_boundary": "31 compiled training regions measured by reference-graph endpoint substitution",
        "numerical_source": "region-level candidate/reference difference may include upstream differences",
        "bias_formation": "the framework measures actual update effects but cannot assign them to one computation",
        "decisive_interventions": ["region endpoint substitution"],
        "training_outcome": "not measured per region",
        "closure": "REGION_MEASUREMENT_NO_SINGLE_KERNEL_ROOT",
        "remaining_limit": "same local operands and unique source attribution are absent",
        "evidence": ["docs/attribution_derivation_20260913.md", "docs/result_analysis_20260913.md"],
    },
]


def main() -> None:
    payload = {
        "status": "COMPLETE_AS_EVIDENCE_CLASSIFICATION_NOT_COMPLETE_ROOT_CAUSE_FOR_EVERY_CASE",
        "counting_rule": "one computation/problem group is counted once across models, layers, shapes, and repeated protocols",
        "summary": {"problem_groups": len(ROWS), "end_to_end_causal_chains": sum(r["closure"].startswith("END_TO_END") for r in ROWS), "all_groups_have_unique_root": False},
        "rows": ROWS,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
