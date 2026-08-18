#!/usr/bin/env python3
"""Audit the final four-model, three-shape full-invocation closure."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "results/coverage"
CELLS = {
    ("qwen3_1p7b", 64): ("qwen", "results/coverage/qwen_seq64_input_bank.json"),
    ("qwen3_1p7b", 128): ("qwen", "results/coverage/qwen_seq128_input_bank.json"),
    ("qwen3_1p7b", 256): ("qwen", "results/coverage/qwen_seq256_input_bank.json"),
    ("mamba_130m", 64): ("mamba", "results/mamba_scan/input_bank.json"),
    ("mamba_130m", 128): ("mamba", "results/coverage/mamba_seq128_input_bank.json"),
    ("mamba_130m", 256): ("mamba", "results/coverage/mamba_seq256_input_bank.json"),
    ("phi4_mini_3p8b", 64): ("phi4", "results/coverage/phi4_seq64_input_bank.json"),
    ("phi4_mini_3p8b", 128): ("phi4", "results/coverage/phi4_seq128_input_bank.json"),
    ("phi4_mini_3p8b", 256): ("phi4", "results/coverage/phi4_seq256_input_bank.json"),
    ("deepseek_r1_0528_qwen3_8b", 64): ("deepseek8b", "results/coverage/deepseek8b_seq64_input_bank.json"),
    ("deepseek_r1_0528_qwen3_8b", 128): ("deepseek8b", "results/coverage/deepseek8b_seq128_input_bank.json"),
    ("deepseek_r1_0528_qwen3_8b", 256): ("deepseek8b", "results/coverage/deepseek8b_seq256_input_bank.json"),
}


def load(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def verify_result_hash(data: dict[str, Any], label: str) -> None:
    expected = data.get("result_sha256")
    body = dict(data)
    body.pop("result_sha256", None)
    if expected != canonical_hash(body):
        raise RuntimeError(f"invalid result hash: {label}")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    status_path = COVERAGE / "four_model_full_operator_status.json"
    status = load(status_path)
    compact_path = COVERAGE / "invalid_triton_raw_manifest.json"
    compact = load(compact_path) if compact_path.exists() else None
    compact_rows = {row["path"]: row for row in (compact or {}).get("files", [])}
    verify_result_hash(status, "four-model status")
    if status["counts"]["declared_cells"] != 12:
        raise RuntimeError("four-model status changed the declared 12-cell matrix")
    if status["counts"]["fb_origin_bound"] != 12:
        raise RuntimeError("four-model F+B origin accounting is incomplete")
    if status["scope"].get("deduplication_across_models_shapes_or_families") is not False:
        raise RuntimeError("coverage status permits forbidden deduplication")
    indexed = {
        (row["model"], int(row["shape"][len("batch1_seq"):])): row
        for row in status["cells"]
    }
    if set(indexed) != set(CELLS):
        raise RuntimeError("declared cell cross product changed")

    rows = []
    housekeeping_gaps = []
    totals = {
        "eager_invocations": 0,
        "primary_fb_proof_units": 0,
        "candidate_compute_invocations_per_cross_product": 0,
        "triton_runtime_invocations": 0,
        "nontriton_runtime_invocations": 0,
        "invalid_triton_reference_rows": 0,
    }
    for key, (prefix, bank_name) in CELLS.items():
        model, shape = key
        row = indexed[key]
        gates = row["gates"]
        release = COVERAGE / "runtime_releases" / f"{prefix}_seq{shape}_r1"
        bank = ROOT / bank_name
        capture = load(release / "capture.json")
        verify_result_hash(capture, f"{key} capture")
        if capture["status"] != "COMPLETE_EXACT_EXECUTED_FORWARD_BACKWARD_SOURCE_CAPTURE":
            raise RuntimeError(f"incomplete capture: {key}")
        if capture.get("same_process_measurement_required") is not True:
            raise RuntimeError(f"same-process contract absent: {key}")
        if set(capture["phase_module_counts"]) != {"FORWARD", "BACKWARD"}:
            raise RuntimeError(f"capture lacks F+B wrappers: {key}")
        if capture["input"]["input_bank_sha256"] != file_hash(bank):
            raise RuntimeError(f"input bank drift: {key}")
        for module in capture["modules"]:
            source = Path(module["captured_source"])
            if not source.exists() or file_hash(source) != module["sha256"]:
                raise RuntimeError(f"captured wrapper drift: {key} {source}")

        inventory_summary = load(release / "inventory.summary.json")
        campaign = load(release / "campaign.json.gz")
        verify_result_hash(campaign, f"{key} campaign")
        triton = load(release / "triton_oracle.json.gz")
        nontriton = load(release / "nontriton_oracle.json.gz")
        verify_result_hash(triton, f"{key} Triton Oracle")
        verify_result_hash(nontriton, f"{key} non-Triton Oracle")
        inventory_sha = row["candidate_inventory"]["result_sha256"]
        if inventory_summary["result_sha256"] != inventory_sha:
            raise RuntimeError(f"inventory summary drift: {key}")
        if campaign["result_sha256"] != triton["campaign_sha256"]:
            raise RuntimeError(f"campaign/Oracle binding failed: {key}")
        if nontriton["inventory_sha256"] != inventory_sha:
            raise RuntimeError(f"inventory/Oracle binding failed: {key}")
        if triton["input_bank_sha256"] != capture["input"]["input_bank_sha256"]:
            raise RuntimeError(f"Triton input binding failed: {key}")
        if nontriton["input_bank_sha256"] != capture["input"]["input_bank_sha256"]:
            raise RuntimeError(f"non-Triton input binding failed: {key}")

        tri_den = triton["denominator"]
        non_den = nontriton["denominator"]
        for label, denominator in (("triton", tri_den), ("nontriton", non_den)):
            per_state = denominator["actual_runtime_invocations_per_state"]
            if denominator["states"] != 32 or denominator["repeats_per_state"] != 2:
                raise RuntimeError(f"wrong held-out population: {key} {label}")
            if len(per_state["by_state"]) != 32 or per_state["min"] != per_state["max"]:
                raise RuntimeError(f"runtime denominator drift: {key} {label}")
            if denominator["actual_runtime_invocations_total"] != 32 * per_state["min"]:
                raise RuntimeError(f"runtime denominator total mismatch: {key} {label}")
        candidate = row["candidate_inventory"]["denominator"]["compute_invocations"]
        accounted = (
            tri_den["actual_runtime_invocations_per_state"]["min"]
            + non_den["actual_runtime_invocations_per_state"]["min"]
            + non_den["static_generated_calls_not_executed_in_at_least_one_state"]
        )
        if candidate != accounted:
            raise RuntimeError(f"candidate denominator is not losslessly partitioned: {key}")
        # The runtime release's non-Triton oracle is the retained, typed
        # precision reference.  ``nontriton_screen.json.gz`` was an older
        # intermediate screen and is intentionally not part of the release
        # contract.  Triton raw screens, on the other hand, may be absent only
        # when their invalid ABI is recorded in the compaction manifest.
        for final_screen in ("triton_screen.json.gz",):
            path = release / final_screen
            if path.exists() and path.stat().st_size > 0:
                continue
            relative = str(path.relative_to(ROOT))
            compact_row = compact_rows.get(relative)
            compact_valid = (
                final_screen == "triton_screen.json.gz"
                and compact is not None
                and compact.get("status") == "COMPACTED_INVALID_RAW_REMOVED"
                and compact_row is not None
                and compact_row.get("state") == "DELETED_INVALID_ABI_REGENERABLE"
                and compact_row.get("retained_oracle")
                == str((release / "triton_oracle.json.gz").relative_to(ROOT))
            )
            if not compact_valid:
                raise RuntimeError(
                    f"missing final screen without compact invalidation record: {key} {final_screen}"
                )
        leftovers = [
            path.name for path in release.iterdir()
            if path.name.startswith(".") or "checkpoint" in path.name
        ]
        if leftovers:
            # A completed release can still have an old resumable checkpoint.
            # Keep this visible as a cleanup gap, but do not turn it into a
            # false scientific coverage failure: the checkpoint is not an
            # execution denominator and the finalized oracle is hash-bound.
            housekeeping_gaps.append({
                "model": model,
                "shape": f"batch1_seq{shape}",
                "artifacts": leftovers,
                "disposition": "STALE_CHECKPOINT_RETAINED_PENDING_CLEANUP",
            })

        math = row["canonical_eager_fb_math"]
        totals["eager_invocations"] += math["execution_census_invocations"]
        totals["primary_fb_proof_units"] += math["primary_fb_proof_units"]
        totals["candidate_compute_invocations_per_cross_product"] += candidate
        totals["triton_runtime_invocations"] += tri_den["actual_runtime_invocations_total"]
        totals["nontriton_runtime_invocations"] += non_den["actual_runtime_invocations_total"]
        abi = row["triton_precision_oracle"].get("abi_audit") or {}
        totals["invalid_triton_reference_rows"] += int(
            abi.get("invalid_reference_abi_campaign_rows", 0)
        )
        rows.append({
            "model": model,
            "shape": f"batch1_seq{shape}",
            "eager_invocations": math["execution_census_invocations"],
            "primary_fb_proof_units": math["primary_fb_proof_units"],
            "candidate_compute_invocations": candidate,
            "triton_invocations_per_state": tri_den["actual_runtime_invocations_per_state"]["min"],
            "nontriton_invocations_per_state": non_den["actual_runtime_invocations_per_state"]["min"],
            "static_nonexecution": non_den["static_generated_calls_not_executed_in_at_least_one_state"],
            "gates": gates,
            "status": row["status"],
        })

    audit = {
        "schema": "kernel-analyzer-four-model-full-coverage-audit-v1",
        "status": (
            "COMPLETE"
            if status["counts"]["fully_closed_cells"] == 12 and not housekeeping_gaps
            else (
                "COMPLETE_EXECUTION_AUDIT_WITH_HOUSEKEEPING_GAPS"
                if status["counts"]["fully_closed_cells"] == 12
                else "COMPLETE_EXECUTION_AUDIT_PARTIAL_SCIENTIFIC_GATES"
            )
        ),
        "requirements": {
            "models": 4,
            "shapes_per_model": 3,
            "cells": 12,
            "states_per_cell": 32,
            "repeats_per_state": 2,
            "deduplication_across_models_shapes_or_families": False,
            "gates": [
                "eager_execution_and_forward_backward_origin",
                "canonical_eager_forward_backward_proof",
                "actual_default_aot_forward_backward_proof",
                "exact_candidate_inventory",
                "candidate_to_forward_backward_binding",
                "valid_typed_triton_reference",
                "valid_nontriton_precision_reference",
                "same_dtype_optimization_reference",
            ],
        },
        "totals": totals,
        "cells": rows,
        "housekeeping_gaps": housekeeping_gaps,
        "claim_boundary": (
            "This audit proves the declared four-model execution census, wrapper identity and "
            "runtime call accounting. It reports analytic-proof, candidate-binding and numerical-"
            "reference gaps without converting them into equivalence. Numerical screen positives "
            "still require a complete causal carrier and accumulation proof before promotion."
        ),
    }
    audit["result_sha256"] = canonical_hash(audit)
    output = COVERAGE / "four_model_full_operator_audit.json"
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(audit, sort_keys=True, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps({"output": str(output.relative_to(ROOT)), **totals}, sort_keys=True))


if __name__ == "__main__":
    main()
