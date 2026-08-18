#!/usr/bin/env python3
"""Build the active four-model, per-shape F+B denominator without inheritance."""

from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any

from build_fb_proof_unit_ledger import build_components, digest, proof_unit


ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "results/coverage"
OUTPUT = COVERAGE / "fb_multishape_ledger.json.gz"

MODEL_CELLS: dict[str, dict[str, str | None]] = {
    "qwen3_1p7b": {
        "batch1_seq64": "results/coverage/qwen_invocation_ledger.json.gz",
        "batch1_seq128": "results/coverage/qwen_seq128_invocation_ledger.json.gz",
        "batch1_seq256": "results/coverage/qwen_seq256_invocation_ledger.json.gz",
    },
    "mamba_130m": {
        "batch1_seq64": "results/coverage/mamba_invocation_ledger.json.gz",
        "batch1_seq128": "results/coverage/mamba_seq128_invocation_ledger.json.gz",
        "batch1_seq256": "results/coverage/mamba_seq256_invocation_ledger.json.gz",
    },
    "phi4_mini_3p8b": {
        "batch1_seq64": "results/coverage/phi4_seq64_invocation_ledger.json.gz",
        "batch1_seq128": "results/coverage/phi4_seq128_invocation_ledger.json.gz",
        "batch1_seq256": "results/coverage/phi4_seq256_invocation_ledger.json.gz",
    },
    "deepseek_r1_0528_qwen3_8b": {
        "batch1_seq64": "results/coverage/deepseek8b_seq64_invocation_ledger.json.gz",
        "batch1_seq128": "results/coverage/deepseek8b_seq128_invocation_ledger.json.gz",
        "batch1_seq256": "results/coverage/deepseek8b_seq256_invocation_ledger.json.gz",
    },
}

WITNESS_CELLS: dict[str, dict[str, str]] = {
    "qwen3_1p7b": {
        "batch1_seq64": "results/coverage/concrete_fb_witnesses/qwen_seq64.json.gz",
        "batch1_seq128": "results/coverage/concrete_fb_witnesses/qwen_seq128.json.gz",
        "batch1_seq256": "results/coverage/concrete_fb_witnesses/qwen_seq256.json.gz",
    },
    "mamba_130m": {
        "batch1_seq64": "results/coverage/concrete_fb_witnesses/mamba_seq64.json.gz",
        "batch1_seq128": "results/coverage/concrete_fb_witnesses/mamba_seq128.json.gz",
        "batch1_seq256": "results/coverage/concrete_fb_witnesses/mamba_seq256.json.gz",
    },
    "phi4_mini_3p8b": {
        "batch1_seq64": "results/coverage/concrete_fb_witnesses/phi4_seq64.json.gz",
        "batch1_seq128": "results/coverage/concrete_fb_witnesses/phi4_seq128.json.gz",
        "batch1_seq256": "results/coverage/concrete_fb_witnesses/phi4_seq256.json.gz",
    },
    "deepseek_r1_0528_qwen3_8b": {
        "batch1_seq64": "results/coverage/concrete_fb_witnesses/deepseek8b_seq64.json.gz",
        "batch1_seq128": "results/coverage/concrete_fb_witnesses/deepseek8b_seq128.json.gz",
        "batch1_seq256": "results/coverage/concrete_fb_witnesses/deepseek8b_seq256.json.gz",
    },
}


def load(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    contract = json.loads((COVERAGE / "coverage_contract.json").read_text())
    units: list[dict[str, Any]] = []
    cells: dict[str, dict[str, Any]] = {}
    sources: dict[str, dict[str, Any]] = {}
    owners: dict[str, str] = {}
    for model_key, shape_paths in MODEL_CELLS.items():
        model = contract["models"][model_key]
        cells[model_key] = {}
        sources[model_key] = {}
        for shape, relative in shape_paths.items():
            if relative is None:
                cells[model_key][shape] = {
                    "status": "PENDING_EXECUTION_DERIVED_WITNESS",
                    "source_ledger": None,
                    "execution_census_invocations": None,
                    "closed_accounting_components": None,
                    "primary_fb_proof_units": None,
                    "auxiliary_backward_accounting_units": None,
                }
                continue
            path = ROOT / relative
            ledger = load(path)
            components, audit = build_components(model_key, ledger)
            if audit["dangling_origin_links"] or not audit["all_rows_in_exactly_one_component"]:
                raise RuntimeError(f"{model_key}/{shape}: F+B partition is not closed")
            witness_relative = WITNESS_CELLS[model_key][shape]
            witness_payload = load(ROOT / witness_relative)
            if witness_payload["status"] != "COMPLETE_CONCRETE_FB_WITNESSES":
                raise RuntimeError(f"{model_key}/{shape}: concrete F+B witnesses incomplete")
            if witness_payload["bindings"]["ledger_result_sha256"] != ledger["result_sha256"]:
                raise RuntimeError(f"{model_key}/{shape}: witness ledger binding mismatch")
            witness_by_members = {
                row["member_row_ids_sha256"]: row
                for row in witness_payload["witnesses"]
            }
            cell_units = [
                proof_unit(
                    model_key, model, members, ledger["result_sha256"], shape,
                    witness_by_members.get(digest(sorted(
                        row["row_id"] for row in members
                    ))),
                )
                for members in components
            ]
            primary_components = sum(any(
                row["invocation"]["phase"] == "FORWARD" for row in members
            ) for members in components)
            if len(witness_by_members) != primary_components:
                raise RuntimeError(f"{model_key}/{shape}: witness denominator mismatch")
            # Origin accounting and analytic proof are separate gates.  Older
            # architecture ledgers bind actual forward/backward origins and
            # register overload formulas, but do not carry a concrete
            # saved-tensor/cotangent/backward-program proof for every row.
            if not all(unit["gates"]["FB_ORIGIN_BOUND"] for unit in cell_units):
                raise RuntimeError(f"{model_key}/{shape}: F+B origin is unresolved")
            for unit, members in zip(cell_units, components):
                for member in members:
                    qualified = f"{model_key}::{shape}::{member['row_id']}"
                    if qualified in owners:
                        raise RuntimeError(f"qualified invocation has two owners: {qualified}")
                    owners[qualified] = unit["unit_id"]
            primary = sum(
                unit["denominator_role"] == "PRIMARY_FB_PROOF" for unit in cell_units
            )
            auxiliary = len(cell_units) - primary
            analytic_primary = sum(
                unit["denominator_role"] == "PRIMARY_FB_PROOF"
                and unit["gates"]["FB_ANALYTICALLY_PROVED"]
                for unit in cell_units
            )
            cells[model_key][shape] = {
                "status": (
                    "CAPTURED_EXECUTION_DERIVED_ANALYTIC_FB_PROVED"
                    if analytic_primary == primary else
                    "CAPTURED_EXECUTION_DERIVED_ORIGIN_BOUND_FORMULA_ONLY"
                ),
                "source_ledger": relative,
                "execution_census_invocations": audit["source_invocations"],
                "closed_accounting_components": len(cell_units),
                "primary_fb_proof_units": primary,
                "analytic_fb_proof_units": analytic_primary,
                "origin_bound_fb_units": primary,
                "auxiliary_backward_accounting_units": auxiliary,
            }
            sources[model_key][shape] = {
                "path": relative,
                "result_sha256": ledger["result_sha256"],
                "concrete_witness_path": witness_relative,
                "concrete_witness_result_sha256": witness_payload["result_sha256"],
            }
            units.extend(cell_units)

    captured = sum(
        cell["status"].startswith("CAPTURED_")
        for model in cells.values() for cell in model.values()
    )
    total_cells = sum(len(model) for model in cells.values())
    payload = {
        "schema": "kernel-analyzer-active-four-model-multishape-fb-ledger-v1",
        "status": (
            "COMPLETE_EXECUTION_AND_ORIGIN_ACCOUNTING_ONLY"
            if captured == total_cells else "PARTIAL_FAIL_CLOSED"
        ),
        "contract_sha256": contract["contract_sha256"],
        "model_cells": MODEL_CELLS,
        "source_ledgers": sources,
        "denominator_cells": cells,
        "denominators": {
            "active_models": len(MODEL_CELLS),
            "declared_model_shape_cells": total_cells,
            "captured_origin_bound_cells": captured,
            "pending_cells": total_cells - captured,
            "execution_census_invocations": sum(
                cell["execution_census_invocations"] or 0
                for model in cells.values() for cell in model.values()
            ),
            "closed_accounting_components": len(units),
            "primary_fb_proof_units": sum(
                unit["denominator_role"] == "PRIMARY_FB_PROOF" for unit in units
            ),
            "analytic_fb_proof_units": sum(
                unit["denominator_role"] == "PRIMARY_FB_PROOF"
                and unit["gates"]["FB_ANALYTICALLY_PROVED"]
                for unit in units
            ),
            "auxiliary_backward_accounting_units": sum(
                unit["denominator_role"] == "AUXILIARY_BACKWARD_ACCOUNTING"
                for unit in units
            ),
        },
        "audits": {
            "all_captured_cells_origin_bound": all(
                unit["gates"]["FB_ORIGIN_BOUND"] for unit in units
            ),
            "all_captured_cells_analytically_proved": all(
                unit["gates"]["FB_ANALYTICALLY_PROVED"] for unit in units
                if unit["denominator_role"] == "PRIMARY_FB_PROOF"
            ),
            "all_qualified_invocations_owned_once": len(owners) == sum(
                cell["execution_census_invocations"] or 0
                for model in cells.values() for cell in model.values()
            ),
            "qualified_owner_map_sha256": digest(owners),
            "unit_kind_counts": dict(sorted(Counter(unit["unit_kind"] for unit in units).items())),
            "shape_evidence_inherited": False,
            "operator_family_deduplication_used": False,
        },
        "units": sorted(units, key=lambda row: row["unit_id"]),
        "claim_boundary": (
            "This ledger proves execution census partitioning and actual forward/backward origin "
            "accounting for captured cells. ANALYTICALLY_PROVED additionally requires an explicit "
            "concrete saved-tensor, cotangent and backward-program derivation; a registered overload "
            "formula or seq_nr link alone never passes that gate. Candidate mapping and numerical "
            "correctness remain separate."
        ),
    }
    payload = json.loads(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    payload["result_sha256"] = digest(payload)
    temporary_output = OUTPUT.with_name(f".{OUTPUT.name}.tmp")
    temporary_output.unlink(missing_ok=True)
    with temporary_output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    with gzip.open(temporary_output, "rt", encoding="utf-8") as handle:
        written = json.load(handle)
    if written.get("result_sha256") != payload["result_sha256"]:
        temporary_output.unlink(missing_ok=True)
        raise RuntimeError("multishape ledger failed post-write digest validation")
    temporary_output.replace(OUTPUT)
    summary = {
        "schema": "kernel-analyzer-multishape-fb-ledger-summary-v1",
        "status": payload["status"], "result_sha256": payload["result_sha256"],
        "denominator_cells": payload["denominator_cells"],
        "denominators": payload["denominators"],
        "source_ledger": str(OUTPUT.relative_to(ROOT)),
        "source_ledger_bytes": OUTPUT.stat().st_size,
    }
    summary_output = OUTPUT.with_name("fb_multishape_ledger.summary.json")
    summary_output.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT.relative_to(ROOT)), **payload["denominators"], "result_sha256": payload["result_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
