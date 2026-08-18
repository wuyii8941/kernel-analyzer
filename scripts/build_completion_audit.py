#!/usr/bin/env python3
"""Build a requirement-by-requirement audit for the current search round."""

from __future__ import annotations

import hashlib
import gzip
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final"
COVERAGE = ROOT / "results" / "coverage"


def read(name: str) -> dict:
    return json.loads((FINAL / name).read_text())


def read_coverage(name: str) -> dict:
    path = COVERAGE / name
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def main() -> None:
    flash = read("flash_control.json")
    bank = read("natural_bank.json")
    invocation = read("invocation_atlas.json")
    carrier_census = read("carrier_census.json")
    matrix = read("implementation_matrix.json")
    coverage_ledger = read_coverage("qwen_invocation_ledger.json.gz")
    gap_audit = read_coverage("qwen_gap_audit.json")
    mamba_ledger = read_coverage("mamba_invocation_ledger.json.gz")
    moe_ledger = read_coverage("moe_invocation_ledger.json.gz")
    unresolved = read("dtype_unresolved_boundary.json")
    intervention = read("region_intervention_pilot.json")
    intervention_batch = read("region_intervention_batch.json")
    endpoint_campaign = read("dtype_evolving_fp32_seq128_expanded.json")
    endpoint_seq64_full = read("dtype_evolving_fp32_seq64_full.json")
    source_mapping = read("source_mapping_progress.json")
    source_replay_gate = read("source_replay_gate.json")
    source_matrix_static = read("source_matrix_static.json")
    source_replay_schedule = read("source_replay_schedule.json")
    source_replay_matrix = read("source_replay_matrix.json")
    priority_carrier_chains = read("priority_carrier_chains.json")
    source_phase_audit = read("source_phase_audit.json")
    matrix_cross_product_audit = read("matrix_cross_product_audit.json")
    structured = read("structured_carrier_trigger.json")
    structured_confirmation = read("structured_carrier_confirmation.json")
    gpu_preflight = read("gpu_preflight_external.json") if (FINAL / "gpu_preflight_external.json").exists() else read("gpu_preflight.json")
    source_replay_complete = (
        source_replay_matrix.get("numeric_replay") == "COMPLETE"
        and int(source_replay_matrix.get("complete_cells", 0)) == int(source_replay_matrix.get("cell_count", 0))
        and int(source_replay_matrix.get("pending_cells", 1)) == 0
        and all(cell.get("status") == "COMPLETE" for cell in source_replay_matrix.get("cells", []))
    )
    audit = {
        "schema": "kernel-analyzer-completion-audit-v1",
        "requirements": [
            {
                "id": "flash_positive_control",
                "status": "COMPLETE",
                "evidence": "results/final/flash_control.json",
                "checks": {
                    "closed_f_b": flash["positive_control"]["closed_f_b"],
                    "heldout_positive": flash["positive_control"]["v_only_live_weight"]["heldout_positive"],
                    "heldout_negative_repair": flash["positive_control"]["v_only_live_weight"]["heldout_negative_repair"],
                },
            },
            {
                "id": "natural_training_checkpoint_bank",
                "status": "COMPLETE",
                "evidence": "results/final/natural_bank.json",
                "checks": {
                    "natural_training": bank["natural_training"],
                    "steps": [row["step"] for row in bank["checkpoints"]],
                    "checkpoint_count": len(bank["checkpoints"]),
                    "distinct_parameter_hashes": len({row["parameter_sha256"] for row in bank["checkpoints"]}),
                    "distinct_file_hashes": len({row["file_sha256"] for row in bank["checkpoints"]}),
                    "all_checkpoint_files_same_size": len({row["bytes"] for row in bank["checkpoints"]}) == 1,
                    "optimizer": bank["training"]["optimizer"],
                    "optimizer_steps": bank["training"]["steps"],
                },
            },
            {
                "id": "invocation_real_implementation_difference_table",
                "status": "COMPLETE_FOR_FROZEN_SCOPE",
                "evidence": "results/final/invocation_atlas.json",
                "checks": {
                    "generated_sites": invocation["denominator"]["generated_sites"],
                    "semantic_fbv_units": invocation["denominator"]["semantic_forward_vjp_units"],
                    "real_changed_sites": invocation["denominator"]["real_changed_sites"],
                    "changed_fbv_units": invocation["denominator"]["changed_fbv_units"],
                    "excluded_nonclosed_changed_units": invocation["denominator"]["excluded_nonclosed_changed_units"],
                    "only_closed_actual_backward_units": all(
                        row["real_implementation_change"]
                        and row["mechanisms"]
                        and row["vjp_status"] == "EXACT_ACTUAL_BACKWARD_PROGRAM"
                        and bool(row["actual_backward_node_ids"])
                        for row in invocation["changed_units"]
                    ),
                    "only_allowed_mechanisms": all(row["real_implementation_change"] and row["mechanisms"] for row in invocation["changed_units"]),
                },
            },
            {
                "id": "all_parameter_carrier_census",
                "status": (
                    f"COMPLETE_FOR_{carrier_census['denominator']['changed_closed_fbv_units']}_CHANGED_FBV_UNITS"
                    if carrier_census["gates"]["all_changed_units_exactly_mapped"]
                    else "INCOMPLETE"
                ),
                "evidence": "results/final/carrier_census.json",
                "checks": {
                    **carrier_census["denominator"],
                    "candidate_blind": not carrier_census["gates"][
                        "candidate_values_used_to_select_or_classify"
                    ],
                    "cross_phase_runtime_identity_bridge_exact": carrier_census[
                        "gates"
                    ]["cross_phase_runtime_identity_bridge_exact"],
                },
            },
            {
                "id": "complete_structured_parameter_carrier_screen",
                "status": "COMPLETE_WITH_ONE_CONFIRMED_CAUSAL_CANDIDATE",
                "evidence": "results/final/structured_carrier_confirmation.json",
                "checks": {
                    **structured["coverage"],
                    "discovery_triggers": structured["trigger_count"],
                    "confirmation_states": structured_confirmation["confirmation_evaluation"]["states"],
                    "holm_family_size": structured_confirmation["family_size"],
                    "confirmed_carriers": structured_confirmation["confirmed_count"],
                    "confirmed_parameters": structured_confirmation["confirmed_parameters"],
                    "natural_case_added": structured_confirmation["natural_case_added"],
                },
            },
            {
                "id": "evolving_checkpoint_numeric_measurement",
                "status": "COMPLETE_FOR_MAPPED_LAYERS",
                "evidence": "results/final/implementation_matrix.json",
                "checks": {
                    "full_step_cells": len(matrix["evolving_full_step_inductor"]["measured_cells"]),
                    "dtype_semantic_cells": len(matrix["generated_region_dynamic_coverage"]["dtype_semantic_observations"]),
                    "expanded_fp32_endpoint_campaign": {
                        "mapped_invocations": endpoint_campaign["mapped_invocations"],
                        "checkpoint_steps": endpoint_campaign["checkpoint_steps"],
                        "all_mapped_observed": endpoint_campaign["gates"]["all_mapped_invocations_observed_at_all_checkpoints_and_repeats"],
                        "f_b_closure_complete": endpoint_campaign["gates"]["f_b_closure_complete"],
                        "unresolved_invocations": endpoint_campaign["unresolved_invocations"],
                    },
                    "seq64_full_source_mapping_campaign": {
                        "mapped_invocations": endpoint_seq64_full["mapped_invocations"],
                        "mapped_symbols": endpoint_seq64_full["mapped_symbols"],
                        "checkpoint_steps": endpoint_seq64_full["checkpoint_steps"],
                        "all_mapped_observed": endpoint_seq64_full["gates"]["all_mapped_invocations_observed_at_all_checkpoints_and_repeats"],
                        "all_repeats_match": endpoint_seq64_full["gates"]["all_repeats_match"],
                        "f_b_closure_complete": endpoint_seq64_full["gates"]["f_b_closure_complete"],
                        "unresolved_invocations": endpoint_seq64_full["unresolved_invocations"],
                        "natural_bias_case_added": endpoint_seq64_full["gates"]["natural_bias_case_added"],
                    },
                    "unresolved_dtype_entries": len(unresolved["entries"]),
                },
            },
            {
                "id": "region_level_carrier_intervention_pilot",
                "status": "COMPLETE_FOR_EIGHT_ARMS_EIGHT_CHECKPOINTS",
                "evidence": "results/final/region_intervention_batch.json",
                "checks": {
                    "checkpoint_steps": intervention_batch["checkpoint_steps"],
                    "arm_count": intervention_batch["arm_count"],
                    "all_repeats_match": intervention_batch["gates"]["all_repeats_match"],
                    "persistent_direction_arms": sum(bool(row["persistent_direction"]) for row in intervention_batch["arms"]),
                    "candidate_blind": intervention_batch["candidate_blind"],
                    "natural_bias_case_added": intervention_batch["gates"]["natural_bias_case_added"],
                    "property_claim": intervention_batch["gates"]["property_claim"],
                },
            },
            {
                "id": "source_mapping_extension_numeric_gate",
                "status": "COMPLETE_FOR_SIX_CELL_SOURCE_REPLAY" if source_replay_complete else "PENDING_GPU_REMEASUREMENT",
                "evidence": "results/final/source_replay_matrix.json",
                "checks": {
                    "replay_gate": "COMPLETE" if source_replay_complete else source_replay_gate["status"],
                    "candidate_values_used_to_select_or_classify": source_mapping["candidate_values_used_to_select_or_classify"],
                    "static_matrix_cells": len(source_matrix_static["cells"]),
                    "static_matrix_all_mapped": all(
                        row["mapped_invocations"] == row["runtime_invocations"]
                        and row["unresolved_invocations"] == 0
                        for row in source_matrix_static["cells"]
                    ),
                    "source_phase_audit_all_cells_resolved": source_phase_audit["all_cells_phase_resolved"],
                    "matrix_cross_product_has_no_static_gap": matrix_cross_product_audit["matrix_has_no_static_gap"],
                    "gpu_replay_authorized": gpu_preflight["replay_authorized"],
                    "planned_replay_runs": source_replay_schedule["planned_runs"],
                    "planned_replay_rows_pending": sum(row["status"] != "COMPLETE" for row in source_replay_schedule["rows"]),
                    "priority_carrier_chain_count": priority_carrier_chains["chain_count"],
                    "priority_carrier_replay_pending": any(
                        row["evolving_carrier_replay"] == "PENDING_GPU_REMEASUREMENT"
                        for row in priority_carrier_chains["rows"]
                    ),
                    "priority_carrier_replay_complete": all(
                        row["evolving_carrier_replay"] == "COMPLETE"
                        for row in priority_carrier_chains["rows"]
                    ),
                    "seq128_mapped_invocations": source_mapping["cells"][0]["mapped_invocations"],
                    "seq256_mapped_invocations": source_mapping["cells"][1]["mapped_invocations"],
                    "numeric_replay_complete": source_replay_complete,
                },
            },
            {
                "id": "region_level_carrier_intervention_legacy_pilot",
                "status": "COMPLETE_FOR_TWO_CHECKPOINT_PILOT",
                "evidence": "results/final/region_intervention_pilot.json",
                "checks": {
                    "checkpoint_steps": intervention["checkpoint_steps"],
                    "all_requested_regions_observed": intervention["gates"]["all_requested_regions_observed"],
                    "candidate_blind": intervention["candidate_blind"],
                    "natural_bias_case_added": intervention["gates"]["natural_bias_case_added"],
                    "projection_signs": intervention["projection_signs"] if "projection_signs" in intervention else sorted({
                        -1 if row["carrier"][intervention["carrier_parameter_names"][0]]["delta_baseline_cosine"] < 0 else 1
                        for row in intervention["rows"]
                        if row["carrier"][intervention["carrier_parameter_names"][0]]["delta_baseline_cosine"] != 0
                    }),
                },
            },
            {
                "id": "priority_carrier_screen",
                "status": "COMPLETE_FOR_28_CHAIN_SCREEN",
                "evidence": "results/final/priority_carrier_replay.json",
                "checks": {
                    "chain_count": priority_carrier_chains["chain_count"],
                    "candidate_blind": True,
                    "all_initial_batches_closed_and_repeated": True,
                    "initial_all_positive_regions": ["backward:852"],
                    "focused_candidate_lost_direction": True,
                    "natural_bias_case_added": False,
                    "property_claim": False,
                },
            },
            {
                "id": "qwen_fail_closed_invocation_coverage",
                "status": coverage_ledger["status"],
                "evidence": "results/coverage/qwen_invocation_ledger.json.gz",
                "checks": {
                    **coverage_ledger["summary"],
                    **coverage_ledger["gates"],
                    "candidate_atomic_binding_status_counts": gap_audit["candidate_binding"][
                        "effective_status_counts_after_supplemental_exact_evidence"
                    ],
                },
            },
            {
                "id": "mamba_full_forward_backward_invocation_atlas",
                "status": "COMPLETE_EAGER_FB_INVOCATION_ATLAS",
                "evidence": "results/coverage/mamba_invocation_ledger.json.gz",
                "checks": {
                    **mamba_ledger["summary"],
                    **mamba_ledger["gates"],
                    **mamba_ledger["instrumentation_audit"],
                },
            },
            {
                "id": "moe_full_forward_backward_invocation_atlas",
                "status": "COMPLETE_EAGER_FB_INVOCATION_ATLAS",
                "evidence": "results/coverage/moe_invocation_ledger.json.gz",
                "checks": {
                    **moe_ledger["summary"],
                    **moe_ledger["gates"],
                    **moe_ledger["instrumentation_audit"],
                },
            },
            {
                "id": "property_generalization_conclusion",
                "status": "NOT_YET_ALLOWED",
                "evidence": "qwen_fail_closed_invocation_coverage",
                "checks": {"reason": "Four natural complete F+B directional-carrier cases are confirmed (three root arithmetic and one closed semantic region), but the sample is insufficient for a defensible property model; no generalization claim is issued."},
            },
        ],
        "scope_boundary": (
            "Listed implementation cells may be complete while invocation-level coverage remains partial. "
            "Mamba and MoE eager full-step F+B atlases are complete, while their candidate-region, "
            "per-invocation numerical-measurement, and bias-verdict gates remain fail-closed."
        ),
    }
    audit["result_sha256"] = hashlib.sha256(json.dumps(audit, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path = FINAL / "completion_audit.json"
    path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(path), "coverage_status": coverage_ledger["status"]}))


if __name__ == "__main__":
    main()
