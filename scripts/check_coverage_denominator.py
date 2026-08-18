#!/usr/bin/env python3
"""Fail-closed verifier for the frozen coverage denominator package."""

from __future__ import annotations

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


def without(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = dict(value)
    result.pop(key)
    return result


def load_gzip(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def verify() -> dict[str, int]:
    contract = json.loads((COVERAGE / "coverage_contract.json").read_text())
    ledger = load_gzip(COVERAGE / "fb_proof_unit_ledger.json.gz")
    summary = json.loads((COVERAGE / "fb_coverage_summary.json").read_text())

    if digest(without(contract, "contract_sha256")) != contract["contract_sha256"]:
        raise ValueError("coverage contract self-digest differs")
    if digest(without(ledger, "result_sha256")) != ledger["result_sha256"]:
        raise ValueError("F+B ledger self-digest differs")
    if digest(without(summary, "result_sha256")) != summary["result_sha256"]:
        raise ValueError("coverage summary self-digest differs")
    if ledger["contract_sha256"] != contract["contract_sha256"]:
        raise ValueError("ledger was not built from the current contract")
    if summary["contract_sha256"] != contract["contract_sha256"]:
        raise ValueError("summary was not built from the current contract")
    if summary["fb_ledger_sha256"] != ledger["result_sha256"]:
        raise ValueError("summary was not built from the current F+B ledger")

    source_rows = set()
    for model_key, model in contract["models"].items():
        source = load_gzip(ROOT / model["ledger"])
        binding = ledger["source_ledgers"][model_key]
        if binding["result_sha256"] != source["result_sha256"]:
            raise ValueError(f"{model_key}: source ledger digest differs")
        for row in source["rows"]:
            row_id = row["row_id"]
            if row_id in source_rows:
                raise ValueError(f"duplicate source row id: {row_id}")
            source_rows.add(row_id)

    owned_rows = []
    primary = auxiliary = 0
    for unit in ledger["units"]:
        members = unit["members"]
        owned_rows.extend(members["all_invocation_rows"]["ids"])
        forward_count = members["forward_invocation_rows"]["count"]
        if unit["denominator_role"] == "PRIMARY_FB_PROOF":
            primary += 1
            if forward_count == 0:
                raise ValueError(f"primary unit lacks forward: {unit['unit_id']}")
        elif unit["denominator_role"] == "AUXILIARY_BACKWARD_ACCOUNTING":
            auxiliary += 1
            if forward_count != 0:
                raise ValueError(f"auxiliary unit contains forward: {unit['unit_id']}")
        else:
            raise ValueError(f"unknown denominator role: {unit['denominator_role']}")

    if len(owned_rows) != len(set(owned_rows)):
        raise ValueError("an invocation has multiple accounting owners")
    if set(owned_rows) != source_rows:
        raise ValueError("owned invocation set differs from source census")
    denominators = ledger["denominators"]
    if len(source_rows) != denominators["execution_census_invocations"]:
        raise ValueError("execution census count differs")
    if primary != denominators["primary_fb_proof_units"]:
        raise ValueError("primary F+B count differs")
    if auxiliary != denominators["auxiliary_backward_accounting_units"]:
        raise ValueError("auxiliary count differs")

    for model_key, cells in ledger["fb_denominator_cells"].items():
        for shape, cell in cells.items():
            captured = cell["status"] == "CAPTURED_EXECUTION_DERIVED"
            counts = (
                cell["execution_census_invocations"],
                cell["primary_fb_proof_units"],
                cell["auxiliary_backward_accounting_units"],
            )
            if captured and any(value is None for value in counts):
                raise ValueError(f"{model_key}/{shape}: captured cell lacks counts")
            if not captured and any(value is not None for value in counts):
                raise ValueError(f"{model_key}/{shape}: pending cell inherited counts")

    legacy_counts = {
        "execution_census_invocations": len(source_rows),
        "primary_fb_proof_units": primary,
        "auxiliary_backward_accounting_units": auxiliary,
    }

    # The legacy contract above remains a retained seq64/paused-Granite
    # accounting package.  The active scientific authority is the independent
    # four-model x three-shape ledger and its layered fail-closed status.
    multishape = load_gzip(COVERAGE / "fb_multishape_ledger.json.gz")
    multishape_summary = json.loads(
        (COVERAGE / "fb_multishape_ledger.summary.json").read_text()
    )
    status = json.loads(
        (COVERAGE / "four_model_full_operator_status.json").read_text()
    )
    audit = json.loads(
        (COVERAGE / "four_model_full_operator_audit.json").read_text()
    )
    abi = json.loads((COVERAGE / "triton_reference_abi_audit.json").read_text())
    compact_path = COVERAGE / "invalid_triton_raw_manifest.json"
    compact = json.loads(compact_path.read_text()) if compact_path.exists() else None
    for label, value in (
        ("multishape", multishape),
        ("four-model status", status),
        ("four-model audit", audit),
        ("Triton ABI audit", abi),
    ):
        if digest(without(value, "result_sha256")) != value["result_sha256"]:
            raise ValueError(f"{label} self-digest differs")
    if compact is not None:
        if digest(without(compact, "result_sha256")) != compact["result_sha256"]:
            raise ValueError("invalid Triton raw manifest self-digest differs")
        if compact["status"] == "COMPACTED_INVALID_RAW_REMOVED":
            if len(compact["files"]) != 12:
                raise ValueError("compacted Triton manifest does not account for 12 cells")
            for row in compact["files"]:
                if (ROOT / row["path"]).exists():
                    raise ValueError("compacted invalid raw screen unexpectedly exists")
                if not (ROOT / row["retained_oracle"]).is_file():
                    raise ValueError("compaction removed a required compact Oracle")
    if multishape_summary["result_sha256"] != multishape["result_sha256"]:
        raise ValueError("multishape summary is stale")
    counts = status["counts"]
    if counts["declared_cells"] != 12 or counts["fb_origin_bound"] != 12:
        raise ValueError("active 12-cell execution/origin denominator is incomplete")
    if counts["fully_closed_cells"] != 0 or status["status"] != "PARTIAL_FAIL_CLOSED":
        raise ValueError("invalid or unresolved scientific gates were promoted")
    if abi["status"] != "INVALID_REFERENCE_ABI":
        raise ValueError("Triton ABI invalidation unexpectedly disappeared")
    if counts["triton_execution_censuses_closed"] != 12:
        raise ValueError("Triton execution census was lost during invalidation")
    if counts["triton_precision_oracles_closed"] != 0:
        raise ValueError("invalid Triton numerical references were promoted")
    for cell in status["cells"]:
        gates = cell["gates"]
        if not gates["execution_census"] or not gates["fb_origin_bound"]:
            raise ValueError("captured cell lost execution/origin accounting")
        if gates["triton_numeric_reference_valid"]:
            raise ValueError("invalid Triton reference passed a cell gate")
        if cell["status"] != "PENDING_FAIL_CLOSED":
            raise ValueError("incomplete cell did not fail closed")
    if audit["status"] != "COMPLETE_EXECUTION_AUDIT_PARTIAL_SCIENTIFIC_GATES":
        raise ValueError("four-model audit obscures partial scientific gates")

    return {
        **{"legacy_" + key: value for key, value in legacy_counts.items()},
        "active_execution_census_invocations": multishape["denominators"][
            "execution_census_invocations"
        ],
        "active_primary_fb_accounting_units": multishape["denominators"][
            "primary_fb_proof_units"
        ],
        "active_analytic_fb_proof_units": multishape["denominators"][
            "analytic_fb_proof_units"
        ],
        "active_fully_closed_cells": counts["fully_closed_cells"],
    }


def main() -> None:
    print(json.dumps({"status": "VALID", **verify()}, sort_keys=True))


if __name__ == "__main__":
    main()
