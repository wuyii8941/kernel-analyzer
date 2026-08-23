#!/usr/bin/env python3
"""Audit the frozen sample-completion plan without running a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/property/sample_completion_v1"
OUT = BASE / "completion_audit.json"


def load(name: str) -> dict[str, Any]:
    return json.loads((BASE / name).read_text(encoding="utf-8"))


def check(name: str, passed: bool, evidence: str, detail: str) -> dict[str, Any]:
    return {
        "item": name,
        "status": "COMPLETE" if passed else "PENDING",
        "evidence": evidence,
        "detail": detail,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    protocol = load("protocol.json")
    roster = load("roster.json")
    execution = load("execution_manifest.json")
    preflight = load("preflight_report.json")
    heldout = load("heldout_roster.json")
    stage = load("stage_manifest.json")
    coverage = json.loads(
        (ROOT / "results/coverage/coverage_table_v1.json").read_text(encoding="utf-8")
    )
    uniform = coverage["uniform_sample_completion"]
    rows = [
        check(
            "protocol_frozen",
            protocol.get("status") == "FROZEN_BEFORE_NEW_MEASUREMENT",
            "results/property/sample_completion_v1/protocol.json",
            protocol.get("status"),
        ),
        check(
            "frozen_roster_has_24_units",
            len(roster.get("base_cases", [])) + len(roster.get("search_units", [])) >= 20,
            "results/property/sample_completion_v1/roster.json",
            f"base={len(roster.get('base_cases', []))}, search={len(roster.get('search_units', []))}",
        ),
        check(
            "coverage_denominator_audited",
            coverage.get("status", "").startswith("COMPLETE_FOR_DECLARED_COVERAGE"),
            "results/coverage/coverage_table_v1.json",
            f"models={coverage.get('model_count_with_actual_artifacts')}, systematic={coverage.get('systematic_census_model_count')}",
        ),
        check(
            "candidate_commands_and_bindings_ready",
            execution.get("status") == "READY_FOR_GPU_BUT_NOT_EXECUTED",
            "results/property/sample_completion_v1/execution_manifest.json",
            f"units={execution.get('total_search_units_with_exact_command')}",
        ),
        check(
            "all_required_files_and_model_paths_ready",
            preflight.get("status") in {"READY_TO_START", "BLOCKED_PRECAMPAIGN"}
            and all(row.get("ready_without_gpu") for row in preflight.get("groups", [])),
            "results/property/sample_completion_v1/preflight_report.json",
            preflight.get("gpu", {}).get("detail", "unknown"),
        ),
        check(
            "heldout_split_frozen_before_measurement",
            heldout.get("status") == "FROZEN_BEFORE_MEASUREMENT"
            and len(heldout.get("rows", [])) >= 8,
            "results/property/sample_completion_v1/heldout_roster.json",
            f"units={len(heldout.get('rows', []))}, labels_frozen={heldout.get('scientific_labels_frozen')}",
        ),
        check(
            "uniform_cases_20",
            int(uniform.get("current_uniform_cases", 0)) >= int(uniform.get("required_cases", 20)),
            "results/coverage/coverage_table_v1.json",
            f"{uniform.get('current_uniform_cases', 0)}/{uniform.get('required_cases', 20)}",
        ),
        check(
            "valid_controls_15",
            int(uniform.get("current_uniform_controls", 0)) >= int(uniform.get("required_controls", 15)),
            "results/coverage/coverage_table_v1.json",
            f"{uniform.get('current_uniform_controls', 0)}/{uniform.get('required_controls', 15)}",
        ),
        check(
            "32_step_labels_exist",
            int(stage.get("counts", {}).get("uniform_32_complete", 0)) >= 20,
            "results/property/sample_completion_v1/stage_manifest.json",
            f"{stage.get('counts', {}).get('uniform_32_complete', 0)}/20",
        ),
        check(
            "heldout_validation_complete",
            False,
            "results/property/sample_completion_v1/heldout_roster.json",
            "labels are intentionally null until the frozen 32-step run",
        ),
        check(
            "oracle_baselines_and_cost_complete",
            False,
            "not yet generated",
            "requires uniform 16/32-step measurements",
        ),
    ]
    pending = [row["item"] for row in rows if row["status"] != "COMPLETE"]
    result = {
        "schema": "kernel-analyzer-sample-completion-audit-v1",
        "status": "INCOMPLETE_GPU_CAMPAIGN_NOT_RUN" if pending else "COMPLETE",
        "items": rows,
        "pending_items": pending,
        "scientific_results_written": False,
        "claim_boundary": "This audit distinguishes preparation from measurements; it cannot promote legacy artifacts to uniform cases.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "pending": pending}, sort_keys=True))


if __name__ == "__main__":
    main()
