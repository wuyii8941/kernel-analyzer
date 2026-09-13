#!/usr/bin/env python3
"""Recompute the bounded mainline-completion decision from saved evidence.

This audit checks execution accounting and already-declared evidence.  It does
not verify runtime identity from first principles, infer root causes from
coverage measurements, or broaden any statistical or training claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    "campaign_manifest": ROOT / "results/property/numerical_coverage_v1/triton_signature_campaigns_v4/manifest.json",
    "campaign": ROOT / "results/property/numerical_coverage_v1/triton_signature_campaign_audit_v1.json",
    "catalog": ROOT / "results/property/numerical_coverage_v1/observed_kernel_catalog_with_signature_measurements_summary_v3.json",
    "deduplication": ROOT / "results/property/numerical_coverage_v1/triton_signature_deduplication_v1.json",
    "bounded_q": ROOT / "results/property/numerical_coverage_v1/bounded_population_equivalence_validation_v3.json",
    "exceedance": ROOT / "results/property/numerical_coverage_v1/population_statewise_exceedance_validation_v1.json",
    "mechanism": ROOT / "results/property/numerical_coverage_v1/adamw8bit_error_compensation_probe_v1/verification.json",
    "training": ROOT / "results/property/numerical_coverage_v1/adamw8bit_error_compensation_training_v1/verification.json",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(paths: dict[str, Path]) -> dict[str, Any]:
    values = {name: load(path) for name, path in paths.items()}
    campaign_manifest = values["campaign_manifest"]
    campaign = values["campaign"]
    catalog = values["catalog"]
    dedup = values["deduplication"]
    bounded_q = values["bounded_q"]
    exceedance = values["exceedance"]
    mechanism = values["mechanism"]
    training = values["training"]

    counts = campaign.get("final_status_counts", {})
    accounted = sum(int(value) for value in counts.values())
    training_decision = training.get("recomputed", {}).get("decision")
    checks = {
        "frozen_campaign_is_outcome_blind_and_triton_only": (
            campaign_manifest.get("selection_uses_numerical_outcomes") is False
            and campaign.get("selection_uses_numerical_outcomes") is False
            and len(campaign_manifest.get("campaigns", [])) == 64
            and all(
                row.get("implementation_kind") == "TRITON"
                for row in campaign_manifest.get("campaigns", [])
            )
        ),
        "all_frozen_tasks_have_terminal_status": (
            campaign.get("all_frozen_tasks_accounted") is True
            and accounted == campaign.get("frozen_task_count") == 64
            and counts.get("VALID") == 31
        ),
        "coverage_results_are_not_called_root_causes": (
            campaign.get("valid_measurements_are_not_root_causes") is True
            and dedup.get("independent_root_cause_count") is None
        ),
        "observed_catalog_is_complete_for_supplied_inventory": (
            catalog.get("all_positions_classified_including_explicit_unresolved") is True
            and catalog.get("canonical_position_count") == 146104
            and catalog.get("operator_family_count") == 19
            and catalog.get("distinct_position_support_status_counts", {}).get(
                "VALID_MEASUREMENT_COMPLETED"
            )
            == 551
        ),
        "new_valid_results_are_deduplicated": (
            dedup.get("valid_position_count") == 31
            and dedup.get("structural_signature_count") == 31
            and dedup.get("exact_candidate_computation_count") == 30
            and dedup.get("broad_catalogue_family_count") == 4
        ),
        "bounded_population_q_method_validated": bounded_q.get("status") == "PASS",
        "statewise_exceedance_method_validated": exceedance.get("status") == "PASS",
        "adamw8bit_mechanism_modification_verified": mechanism.get("status") == "VERIFIED",
        "adamw8bit_training_improvement_verified": (
            training.get("status") == "VERIFIED"
            and training.get("stream_count") == 8
            and training_decision == "MATERIAL_IMPROVEMENT"
        ),
    }
    complete = all(checks.values())
    return {
        "schema": "kernel-analyzer-scoped-mainline-completion-audit-v1",
        "status": "COMPLETE_SCOPED_MAINLINE" if complete else "INCOMPLETE",
        "checks": checks,
        "observed_counts": {
            "canonical_positions": catalog.get("canonical_position_count"),
            "directory_families": catalog.get("operator_family_count"),
            "valid_measurement_positions": catalog.get(
                "distinct_position_support_status_counts", {}
            ).get("VALID_MEASUREMENT_COMPLETED"),
            "frozen_signature_tasks": campaign.get("frozen_task_count"),
            "valid_signature_measurements": counts.get("VALID"),
            "signature_timeouts": counts.get("EXECUTION_TIMEOUT_NOT_MEASURED"),
            "signature_execution_failures": counts.get("EXECUTION_FAILED_NOT_MEASURED"),
            "signature_runtime_path_mismatches": counts.get(
                "RUNTIME_PATH_MISMATCH_NOT_MEASURED"
            ),
            "deduplicated_compiled_computations": dedup.get(
                "exact_candidate_computation_count"
            ),
            "adamw8bit_training_streams": training.get("stream_count"),
        },
        "claim_scope": {
            "included": [
                "automated inventory, execution accounting, measurement and recomputation",
                "real Triton implementation measurements",
                "fixed-suite analysis and conditional population procedures",
                "one verified mechanism-to-modification-to-training chain",
            ],
            "not_claimed": [
                "dynamic analysis of every observed kernel position",
                "unconditional finite-sample population-mean Q equivalence",
                "cross-model or cross-checkpoint training improvement",
                "natural training collapse",
                "one independent mechanism per valid signature measurement",
            ],
        },
        "input_sha256": {name: sha256(path) for name, path in paths.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name, default in DEFAULTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=default)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {name: getattr(args, name) for name in DEFAULTS}
    result = build(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if result["status"] != "COMPLETE_SCOPED_MAINLINE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
