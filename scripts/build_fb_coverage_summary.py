#!/usr/bin/env python3
"""Summarize census, F+B proof-unit, and candidate-region denominators."""

from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "results/coverage"


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    contract = json.loads((COVERAGE / "coverage_contract.json").read_text())
    with gzip.open(COVERAGE / "fb_proof_unit_ledger.json.gz", "rt") as handle:
        ledger: dict[str, Any] = json.load(handle)
    by_model = {}
    for model_key, model in contract["models"].items():
        accounting_units = [
            row for row in ledger["units"] if row["model"] == model_key
        ]
        units = [
            row for row in accounting_units
            if row["denominator_role"] == "PRIMARY_FB_PROOF"
        ]
        candidate_summary = {}
        for candidate in model["candidate_configurations"]:
            cells = [row["candidate_cells"][candidate] for row in units]
            candidate_summary[candidate] = {
                "total_fb_units": len(units),
                "exactly_bound_fb_units": sum(
                    cell["mapping_status"] == "EXACT_ALL_MEMBERS" for cell in cells
                ),
                "numerically_tested_fb_units": sum(
                    cell["measurement_status"] == "COMPLETE_ALL_MEMBER_ENDPOINTS"
                    for cell in cells
                ),
                "correctness_certified_fb_units": sum(
                    cell["correctness_status"] == "EQUIVALENT" for cell in cells
                ),
                "directional_bias_confirmed_fb_units": sum(
                    cell["directional_bias_status"] == "PASS" for cell in cells
                ),
                "abstained_or_unresolved_fb_units": sum(
                    cell["correctness_status"] == "UNRESOLVED" for cell in cells
                ),
            }
        by_model[model_key] = {
            "scope": model["scope"],
            "execution_census_invocations": ledger["model_audits"][model_key]["source_invocations"],
            "closed_accounting_components": len(accounting_units),
            "auxiliary_backward_accounting_units": sum(
                row["denominator_role"] == "AUXILIARY_BACKWARD_ACCOUNTING"
                for row in accounting_units
            ),
            "primary_fb_proof_units": len(units),
            "origin_bound_fb_units": sum(
                row["gates"].get("FB_ORIGIN_BOUND", False) for row in units
            ),
            "formula_registered_fb_units": sum(
                row["gates"].get("FORMULA_REGISTERED", False) for row in units
            ),
            "analytic_fb_proof_units": sum(
                row["gates"].get("FB_ANALYTICALLY_PROVED", False) for row in units
            ),
            "unit_kind_counts": dict(sorted(Counter(row["unit_kind"] for row in units).items())),
            "candidate_configurations": candidate_summary,
            "shape_coverage": ledger["shape_coverage_matrix"][model_key],
            "fb_denominator_cells": ledger["fb_denominator_cells"][model_key],
        }
    qwen_generated = None
    qwen_generated_source = None
    qwen_path = COVERAGE / "qwen_generated_inventory.json.gz"
    if qwen_path.exists():
        with gzip.open(qwen_path, "rt") as handle:
            generated = json.load(handle)
        qwen_generated = generated["compute_dataflow"]["denominator"]
        qwen_generated_source = {
            "path": str(qwen_path.relative_to(ROOT)),
            "result_sha256": generated.get("result_sha256"),
        }
    candidate_region_cells = {}
    for model_key, model in contract["models"].items():
        candidate_region_cells[model_key] = {}
        for candidate in model["candidate_configurations"]:
            candidate_region_cells[model_key][candidate] = {}
            for sequence in contract["execution_strata"]["sequence_lengths"]:
                cell_key = f"batch1_seq{sequence}"
                captured = (
                    model_key == "qwen3_1p7b"
                    and candidate == "bf16_inductor_full_step"
                    and sequence == 64
                    and qwen_generated is not None
                )
                candidate_region_cells[model_key][candidate][cell_key] = {
                    "status": (
                        "CAPTURED_EXECUTION_DERIVED"
                        if captured else "PENDING_EXECUTION_DERIVED_WITNESS"
                    ),
                    "source": qwen_generated_source if captured else None,
                    "runtime_region_denominator": qwen_generated if captured else None,
                }
    active_candidate_cells = sum(
        len(model["candidate_configurations"])
        * len(contract["execution_strata"]["sequence_lengths"])
        for model in contract["models"].values()
        if model["scope"] == "FULL_STEP"
    )
    captured_candidate_cells = sum(
        cell["status"] == "CAPTURED_EXECUTION_DERIVED"
        for model in candidate_region_cells.values()
        for candidate in model.values()
        for cell in candidate.values()
    )
    payload = {
        "schema": "kernel-analyzer-fb-coverage-summary-v2",
        "status": "PARTIAL_FAIL_CLOSED",
        "contract_sha256": contract["contract_sha256"],
        "fb_ledger_sha256": ledger["result_sha256"],
        "global_denominators": ledger["denominators"],
        "models": by_model,
        "candidate_runtime_region_denominator": {
            "qwen3_1p7b_bf16_inductor_full_step": qwen_generated,
            "cells": candidate_region_cells,
            "declared_active_model_candidate_shape_cells": active_candidate_cells,
            "captured_active_model_candidate_shape_cells": captured_candidate_cells,
            "pending_active_model_candidate_shape_cells": (
                active_candidate_cells - captured_candidate_cells
            ),
            "never_substitutes_for_fb_proof_units": True,
        },
        "reporting_rule": (
            "Always report confirmed / numerically tested / total F+B proof units. "
            "Unresolved, invalid, and abstained units remain in total."
        ),
    }
    payload["result_sha256"] = digest(payload)
    output = COVERAGE / "fb_coverage_summary.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output.relative_to(ROOT)), "sha256": payload["result_sha256"]}))


if __name__ == "__main__":
    main()
