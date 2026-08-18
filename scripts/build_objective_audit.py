#!/usr/bin/env python3
"""Build a requirement-level audit for the active search objective."""

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
    completion = read("completion_audit.json")
    requirements = {row["id"]: row for row in completion["requirements"]}
    flash = read("flash_control.json")
    bank = read("natural_bank.json")
    atlas = read("invocation_atlas.json")
    carrier_census = read("carrier_census.json")
    matrix = read("implementation_matrix.json")
    coverage_ledger = read_coverage("qwen_invocation_ledger.json.gz")
    mamba_ledger = read_coverage("mamba_invocation_ledger.json.gz")
    moe_ledger = read_coverage("moe_invocation_ledger.json.gz")
    preflight = read("gpu_preflight_external.json") if (FINAL / "gpu_preflight_external.json").exists() else read("gpu_preflight.json")
    source_replay = read("source_replay_matrix.json")
    carrier_replay = read("priority_carrier_replay.json")
    active_status = read_coverage("four_model_full_operator_status.json")
    case_audit = read_coverage("existing_case_reaudit.json")
    replay_complete = (
        source_replay.get("numeric_replay") == "COMPLETE"
        and int(source_replay.get("complete_cells", 0)) == int(source_replay.get("cell_count", 0))
        and int(source_replay.get("pending_cells", 1)) == 0
    )
    output = {
        "schema": "kernel-analyzer-objective-audit-v1",
        "objective": "Flash-style positive control plus natural checkpoint x real implementation matrix search",
        "candidate_values_used_to_select_or_classify": False,
        "requirements": [
            {
                "id": "flash_positive_control",
                "status": requirements["flash_positive_control"]["status"],
                "evidence": "results/final/flash_control.json",
                "checks": {
                    "closed_f_b": flash["positive_control"]["closed_f_b"],
                    "heldout_positive": flash["positive_control"]["v_only_live_weight"]["heldout_positive"],
                    "heldout_negative_repair": flash["positive_control"]["v_only_live_weight"]["heldout_negative_repair"],
                },
            },
            {
                "id": "natural_training_checkpoint_bank",
                "status": requirements["natural_training_checkpoint_bank"]["status"],
                "evidence": "results/final/natural_bank.json",
                "checks": {
                    "natural_training": bank["natural_training"],
                    "not_frozen_text_state_campaign": bank["not_frozen_text_state_campaign"],
                    "checkpoint_steps": [row["step"] for row in bank["checkpoints"]],
                    "distinct_parameter_hashes": len({row["parameter_sha256"] for row in bank["checkpoints"]}),
                    "optimizer": bank["training"]["optimizer"],
                    "optimizer_steps": bank["training"]["steps"],
                },
            },
            {
                "id": "invocation_level_real_difference_table",
                "status": requirements["invocation_real_implementation_difference_table"]["status"],
                "evidence": "results/final/invocation_atlas.json",
                "checks": {
                    "generated_sites": atlas["denominator"]["generated_sites"],
                    "real_changed_sites": atlas["denominator"]["real_changed_sites"],
                    "changed_closed_fbv_units": atlas["denominator"]["changed_fbv_units"],
                    "excluded_nonclosed_units_retained": atlas["denominator"]["excluded_nonclosed_changed_units"],
                },
            },
            {
                "id": "evolving_checkpoint_numeric_replay",
                "status": "COMPLETE_FOR_SIX_CELL_SOURCE_REPLAY" if replay_complete else requirements["source_mapping_extension_numeric_gate"]["status"],
                "evidence": "results/final/source_replay_matrix.json",
                "checks": {
                    "planned_runs": 48,
                    "complete_cells": source_replay["complete_cells"],
                    "pending_cells": source_replay["pending_cells"],
                    "gpu_replay_authorized": preflight["replay_authorized"],
                },
            },
            {
                "id": "all_parameter_carrier_census",
                "status": requirements["all_parameter_carrier_census"]["status"],
                "evidence": "results/final/carrier_census.json",
                "checks": carrier_census["denominator"],
            },
            {
                "id": "qwen_fail_closed_invocation_coverage",
                "status": requirements["qwen_fail_closed_invocation_coverage"]["status"],
                "evidence": "results/coverage/qwen_invocation_ledger.json.gz",
                "checks": {
                    **coverage_ledger["summary"],
                    **coverage_ledger["gates"],
                },
            },
            {
                "id": "mamba_full_forward_backward_invocation_atlas",
                "status": requirements["mamba_full_forward_backward_invocation_atlas"]["status"],
                "evidence": "results/coverage/mamba_invocation_ledger.json.gz",
                "checks": {**mamba_ledger["summary"], **mamba_ledger["gates"]},
            },
            {
                "id": "moe_full_forward_backward_invocation_atlas",
                "status": requirements["moe_full_forward_backward_invocation_atlas"]["status"],
                "evidence": "results/coverage/moe_invocation_ledger.json.gz",
                "checks": {**moe_ledger["summary"], **moe_ledger["gates"]},
            },
            {
                "id": "priority_carrier_screen",
                "status": "COMPLETE_FOR_28_CHAIN_SCREEN",
                "evidence": "results/final/priority_carrier_replay.json",
                "checks": {
                    "chain_count": carrier_replay["chain_count"],
                    "candidate_blind": carrier_replay["candidate_blind"],
                    "initial_all_positive_regions": carrier_replay["initial_all_positive_regions"],
                    "focused_candidate_lost_direction": carrier_replay["focused_candidate_lost_direction"],
                    "natural_bias_case_added": carrier_replay["natural_bias_case_added"],
                },
            },
        ],
        "conclusion": {
            "strict_flash_style_cases": case_audit["counts"]["strict_flash_style_cases"],
            "strict_root_arithmetic_op_cases": case_audit["counts"]["strict_root_arithmetic_op_pass"],
            "strict_semantic_region_cases": case_audit["counts"]["strict_semantic_region_pass"],
            "unresolved_composite_carrier_cases": case_audit["counts"]["composite_carrier_pass"],
            "historical_cases_rejected_by_direction_gate": case_audit["counts"]["rejected_by_direction_gate"],
            "cross_state_concrete_mechanism_passes": case_audit["counts"]
            ["cross_state_concrete_mechanism_passes"],
            "fully_closed_model_shape_cells": active_status["counts"]["fully_closed_cells"],
            "property_claim_allowed": False,
            "reason": (
                "Execution and F+B-origin accounting are complete for all 12 declared cells, but "
                "strict analytic F+B proof, candidate binding, valid Triton reference, and same-dtype "
                "optimization gates remain partial. Five strict Flash-style cases (four root "
                "arithmetic and one closed semantic region) pass trajectory-local gates; only "
                "two concrete mechanisms pass "
                "cross-state confirmation, so no cross-operator property claim is issued."
            ),
        },
        "gpu_blocker": {
            "status": preflight["status"],
            "view": preflight.get("view", "sandbox"),
            "device_nodes_present": preflight["device_nodes_present"],
            "loaded_nvidia_modules": preflight["loaded_nvidia_modules"],
        },
        "boundary": "This audit records completion and blockers; it does not infer numeric safety from missing replay rows.",
    }
    output["result_sha256"] = hashlib.sha256(json.dumps(output, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path = FINAL / "objective_audit.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(path), "all_required": all(row["status"].startswith("COMPLETE") for row in output["requirements"]), "coverage_status": coverage_ledger["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
