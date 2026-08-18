#!/usr/bin/env python3
"""Build a compact, fail-closed explanation for the current case count.

This is a diagnostic index, not a correctness verdict.  It records which
links of the Flash-style chain have already been measured and which remain
blocked by the pending real-kernel replay.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final"
COVERAGE = ROOT / "results" / "coverage"


def read(name: str) -> dict:
    return json.loads((FINAL / name).read_text())


def main() -> None:
    atlas = read("invocation_atlas.json")
    matrix = read("implementation_matrix.json")
    intervention = read("region_intervention_batch.json")
    static = read("source_matrix_static.json")
    schedule = read("source_replay_schedule.json")
    replay = read("source_replay_matrix.json")
    chains = read("priority_carrier_chains.json")
    structured = read("structured_carrier_trigger.json")
    confirmation = read("structured_carrier_confirmation.json")
    cases = json.loads((COVERAGE / "existing_case_reaudit.json").read_text())
    live = json.loads((COVERAGE / "live_contrast_dispositions.json").read_text())

    shape_rows = []
    for seq_len, expected in ((64, 506), (128, 452), (256, 450)):
        dynamic = read(f"evolving_triton_seq{seq_len}.json")
        shape_rows.append({
            "seq_len": seq_len,
            "changed_sites": dynamic["denominator"]["changed_generated_region_sites"],
            "observed_at_all_checkpoints_and_repeats": dynamic["denominator"]["changed_sites_observed_at_all_steps_and_repeats"],
            "positive_signed_records": sum(row["positive_signed_mean_count"] for row in dynamic["rows"]),
            "negative_signed_records": sum(row["negative_signed_mean_count"] for row in dynamic["rows"]),
            "expected_changed_sites": expected,
            "all_worker_gates_pass": dynamic["gates"]["all_worker_gates_pass"],
        })

    carrier_cells = matrix["evolving_full_step_inductor"].get("carrier_screen", [])
    output = {
        "schema": "kernel-analyzer-missing-case-diagnosis-v1",
        "subject": "why local implementation differences rarely become coherent weight bias",
        "candidate_values_used_to_select_or_classify": False,
        "flash_chain": [
            "directional_local_error",
            "persistent_gradient_carrier",
            "weight_accumulation",
        ],
        "measured_links": {
            "local_difference": {
                "generated_sites": atlas["denominator"]["generated_sites"],
                "real_changed_sites": atlas["denominator"]["real_changed_sites"],
                "changed_closed_fbv_units": atlas["denominator"]["changed_fbv_units"],
                "shape_rows": shape_rows,
            },
            "local_to_carrier_intervention": {
                "arms": intervention["arm_count"],
                "persistent_direction_arms": sum(bool(row["persistent_direction"]) for row in intervention["arms"]),
                "candidate_blind": intervention["candidate_blind"],
            },
            "full_step_carrier_screen": {
                "cells": len(carrier_cells),
                "persistent_positive_cells": sum(bool(row.get("persistent_positive")) for row in carrier_cells),
                "persistent_negative_cells": sum(bool(row.get("persistent_negative")) for row in carrier_cells),
            },
            "complete_structured_carrier_screen": {
                "parameter_coordinates": structured["coverage"]["unique_parameter_coordinates"],
                "partition_blocks": structured["coverage"]["partition_blocks"],
                "sampled_coordinate_subset": structured["coverage"]["sampled_coordinate_subset"],
                "discovery_triggers": structured["trigger_count"],
                "independent_confirmation_states": confirmation["confirmation_evaluation"]["states"],
                "holm_family_size": confirmation["family_size"],
                "confirmed_carriers": confirmation["confirmed_count"],
                "confirmed_parameters": confirmation["confirmed_parameters"],
                "natural_case_added": confirmation["natural_case_added"],
            },
        },
        "open_links": {
            "nontriton_candidate_dispositions": live["counts"],
            "nontriton_candidates_pending_live_full_coordinate": live["counts"].get(
                "PENDING_LIVE_FULL_COORDINATE_FOLLOWUP", 0
            ),
            "nontriton_t1_positive_pending_case_gates": live["counts"].get(
                "T1_POSITIVE_PENDING_COMPLETE_CASE_GATES", 0
            ),
            "static_source_mapping_cells": len(static["cells"]),
            "static_source_mapping_all_mapped": all(
                row["mapped_invocations"] == row["runtime_invocations"] and row["unresolved_invocations"] == 0
                for row in static["cells"]
            ),
            "real_replay_runs_planned": schedule["planned_runs"],
            "real_replay_runs_complete": sum(row["status"] == "COMPLETE" for row in replay["cells"]),
            "real_replay_status": replay["numeric_replay"],
            "priority_weight_carrier_chains": chains["chain_count"],
            "priority_weight_carrier_replay_pending": all(
                row["evolving_carrier_replay"] == "PENDING_GPU_REMEASUREMENT" for row in chains["rows"]
            ),
        },
        "interpretation": {
            "supported_now": (
                f"Local finite implementation residuals are common. {cases['counts']['strict_flash_style_cases']} "
                "strict Flash-style cases are closed, including the layer-23 q_proj S_bwd "
                "semantic-region attribution. The frozen 83-candidate non-Triton denominator "
                "now has no pending live full-coordinate follow-ups. All six DeepSeek candidates "
                "failed the corrected direction gate; one Qwen seq64 T1-positive mechanism still "
                "needs complete case gates."
            ),
            "not_supported_now": "The 28 priority chains are not Flash-style coherent carrier cases; this does not establish a universal absence outside the current Qwen3-1.7B scope.",
            "next_decision": (
                "Close the remaining Qwen seq64 mechanism, complete the softmax saved-state "
                "repair, then rebuild the invalid typed Triton reference before adding models."
            ),
        },
        "natural_bias_case_added": cases["counts"]["strict_flash_style_cases"] > 0,
        "property_claim": False,
        "boundary": "The diagnostic separates measured cancellation evidence from carrier-level attribution; it never promotes a local residual to a Flash-style case.",
    }
    output["result_sha256"] = hashlib.sha256(
        json.dumps(output, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = FINAL / "missing_case_diagnosis.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(path),
        "changed_units": output["measured_links"]["local_difference"]["changed_closed_fbv_units"],
        "persistent_intervention_arms": output["measured_links"]["local_to_carrier_intervention"]["persistent_direction_arms"],
        "replay_status": output["open_links"]["real_replay_status"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
