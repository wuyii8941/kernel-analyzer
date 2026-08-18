#!/usr/bin/env python3
"""Freeze the scientific denominator independently of execution results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/coverage/coverage_contract.json"


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    payload = {
        "schema": "kernel-analyzer-coverage-contract-v2",
        "status": "FROZEN_BEFORE_ADDITIONAL_CASE_SEARCH",
        "scientific_unit": (
            "minimal closed connected component of actual forward invocations and "
            "their actual backward program invocations"
        ),
        "denominators": {
            "execution_census": (
                "every actual dispatcher invocation in each frozen complete loss F+B witness"
            ),
            "primary_fb_proof_units": (
                "one ordinary forward/VJP pair, or the minimal closed semantic region when "
                "an actual backward invocation is shared by multiple forward invocations"
            ),
            "auxiliary_backward_accounting_units": (
                "backward-only gradient accumulation, detach, control, or AD-program "
                "invocations; retained in execution accounting but excluded from the "
                "primary F+B scientific denominator"
            ),
            "candidate_runtime_regions": (
                "every actual generated Triton, external-library, and direct-ATen compute region"
            ),
        },
        "models": {
            "qwen3_1p7b": {
                "scope": "FULL_STEP",
                "ledger": "results/coverage/qwen_invocation_ledger.json.gz",
                "primary_candidate": "bf16_inductor_full_step",
                "candidate_configurations": [
                    "bf16_inductor_full_step", "pytorch_fused", "liger_fused",
                ],
            },
            "mamba_130m": {
                "scope": "FULL_STEP",
                "ledger": "results/coverage/mamba_invocation_ledger.json.gz",
                "primary_candidate": "compiled_explicit_recurrence",
                "candidate_configurations": [
                    "compiled_explicit_recurrence", "official_fused",
                ],
            },
            "phi4_mini_3p8b": {
                "scope": "FULL_STEP",
                "ledger": "results/coverage/phi4_seq64_invocation_ledger.json.gz",
                "primary_candidate": "bf16_inductor_full_step",
                "candidate_configurations": ["bf16_inductor_full_step"],
            },
            "deepseek_r1_0528_qwen3_8b": {
                "scope": "FULL_STEP",
                "ledger": "results/coverage/deepseek8b_seq64_invocation_ledger.json.gz",
                "primary_candidate": "bf16_inductor_sharded_full_step",
                "candidate_configurations": ["bf16_inductor_sharded_full_step"],
            },
            "granite_3p1_1b_a400m": {
                "scope": "PAUSED_OUT_OF_SCOPE",
                "ledger": "results/coverage/moe_invocation_ledger.json.gz",
                "primary_candidate": "bf16_inductor_full_step",
                "candidate_configurations": ["bf16_inductor_full_step"],
            },
        },
        "execution_strata": {
            "batch_size": [1],
            "sequence_lengths": [64, 128, 256],
            "reference_arms": ["fp32_eager", "bf16_eager"],
            "primary_candidate_dtype": "bf16",
            "secondary_nonblocking_arms": ["fp16", "fp32_strict", "fp32_tf32"],
            "state_role": (
                "states and repeats estimate numerical statistics and never multiply the "
                "primary F+B proof-unit denominator"
            ),
        },
        "denominator_axes": {
            "primary_fb_snapshot_key": [
                "model", "batch_size", "sequence_length", "reference_program",
            ],
            "reference_program": "bf16_eager_complete_loss_forward_backward",
            "candidate_configuration_does_not_multiply_primary_fb_units": True,
            "candidate_configuration_has_its_own_runtime_region_denominator": True,
            "different_dtype_topology_requires_alignment_or_a_separate_snapshot": True,
            "no_shape_may_inherit_another_shapes_invocation_witness": True,
        },
        "unit_rules": {
            "no_operator_family_deduplication": True,
            "repeated_layers_remain_distinct": True,
            "empty_stop_gradient_and_elided_vjp_are_explicit_units": True,
            "primary_fb_unit_must_contain_at_least_one_forward_invocation": True,
            "backward_only_component_is_auxiliary_not_primary_fb": True,
            "shared_backward_forms_one_minimal_closed_region": True,
            "unresolved_invalid_and_abstained_remain_in_denominator": True,
            "many_to_one_candidate_fusion_never_reduces_fb_units": True,
            "unchanged_eager_candidate_paths_are_identical_controls": True,
            "new_dynamic_path_adds_units_only_when_new_invocations_execute": True,
        },
        "ordered_gates": [
            "EXECUTED", "MATH_CLOSED", "CANDIDATE_BOUND", "NUMERIC_MEASURED",
            "T1_LOCAL", "T2_CAUSAL", "T3_COHERENT", "T4_ACCUMULATION",
        ],
        "terminal_statuses": [
            "PASS", "EQUIVALENT", "UNRESOLVED", "INVALID_BOUNDARY", "NOT_APPLICABLE",
        ],
        "completion_rules": {
            "complete_mathematical_coverage_requires": (
                "every census invocation belongs to exactly one primary F+B proof unit or "
                "auxiliary backward accounting unit"
            ),
            "complete_candidate_certificate_requires": (
                "every F+B proof unit has an exact mapping or explicit identical-control or "
                "not-applicable disposition for that candidate configuration"
            ),
            "unmeasured_is_never_equivalent": True,
            "coverage_reports_confirmed_tested_total": True,
        },
    }
    payload["contract_sha256"] = digest(payload)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUTPUT.relative_to(ROOT)), "sha256": payload["contract_sha256"]}))


if __name__ == "__main__":
    main()
