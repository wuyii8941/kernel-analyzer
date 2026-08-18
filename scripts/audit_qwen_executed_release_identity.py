#!/usr/bin/env python3
"""Prove an executed Qwen release is the exact program bound by the F+B registry."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = ROOT / "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/full_step_inventory"


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def exact_flow_checks(old_row: dict[str, Any], new_row: dict[str, Any]) -> dict[str, bool]:
    """Return kind-specific exact executable-boundary identity checks."""
    checks = {
        "phase": old_row["phase"] == new_row["phase"],
        "kind": old_row["kind"] == new_row["kind"],
    }
    if old_row["kind"] == "TRITON":
        checks["embedded_program_sha256"] = (
            old_row["boundary_witness"].get("embedded_program_sha256")
            == new_row["boundary_witness"].get("embedded_program_sha256")
            and old_row["boundary_witness"].get("embedded_program_sha256") is not None
        )
    elif old_row["kind"] == "DIRECT_ATEN":
        for field in ("symbol", "call_expression", "source_line_sha256"):
            checks[field] = old_row.get(field) == new_row.get(field)
        for field in ("boundary_source", "mutated_target", "accumulate_expression"):
            checks[field] = (
                old_row["boundary_witness"].get(field)
                == new_row["boundary_witness"].get(field)
            )
    else:
        for field in ("symbol", "call_expression", "source_line_sha256"):
            checks[field] = old_row.get(field) == new_row.get(field)
        checks["boundary_source"] = (
            old_row["boundary_witness"].get("boundary_source")
            == new_row["boundary_witness"].get("boundary_source")
        )
        for field in (
            "input_tensor_variables",
            "output_tensor_variables",
            "input_storage_root_variables",
            "output_storage_root_variables",
        ):
            checks[field] = old_row.get(field) == new_row.get(field)
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.inventory, "rt", encoding="utf-8") as handle:
        current = json.load(handle)
    old_regions_path = OLD_ROOT / "inductor_generated_region_inventory.json"
    old_dataflow_path = OLD_ROOT / "generated_compute_dataflow_audit_v1.json"
    registry_path = OLD_ROOT / "joint_forward_backward_candidate_registry_v2.json"
    old_regions = json.loads(old_regions_path.read_text())["inventory"]["regions"]
    old_dataflow = json.loads(old_dataflow_path.read_text())["rows"]
    new_regions = current["generated_regions"]["inventory"]["regions"]
    new_dataflow = current["compute_dataflow"]["rows"]
    old_by_id = {row["region_id"]: row for row in old_regions}
    new_by_id = {row["region_id"]: row for row in new_regions}
    old_flow = {row["region_id"]: row for row in old_dataflow}
    new_flow = {row["region_id"]: row for row in new_dataflow}
    fields = ("phase", "kind", "symbol", "original_aten", "source_nodes")
    if old_by_id.keys() != new_by_id.keys() or len(old_by_id) != 1446:
        raise RuntimeError("generated region invocation denominator changed")
    region_rows = []
    for region_id in sorted(old_by_id):
        checks = {
            field: old_by_id[region_id].get(field) == new_by_id[region_id].get(field)
            for field in fields
        }
        if not all(checks.values()):
            raise RuntimeError(f"generated region identity changed: {region_id}")
        row = {"region_id": region_id, "checks": checks}
        row["row_sha256"] = digest(row)
        region_rows.append(row)
    if old_flow.keys() != new_flow.keys() or len(old_flow) != 1447:
        raise RuntimeError("expanded compute boundary denominator changed")
    flow_rows = []
    for region_id in sorted(old_flow):
        checks = exact_flow_checks(old_flow[region_id], new_flow[region_id])
        if not all(checks.values()):
            raise RuntimeError(f"compute boundary program changed: {region_id}")
        row = {"region_id": region_id, "checks": checks}
        row["row_sha256"] = digest(row)
        flow_rows.append(row)
    payload = {
        "schema": "kernel-analyzer-qwen-executed-release-identity-v1",
        "status": "COMPLETE_EXACT_PROGRAM_IDENTITY_TO_FB_REGISTRY",
        "current_inventory": str(args.inventory.resolve().relative_to(ROOT)),
        "current_inventory_sha256": current["result_sha256"],
        "registry": str(registry_path.relative_to(ROOT)),
        "registry_file_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        "denominator": {
            "generated_regions_exact": len(region_rows),
            "expanded_compute_boundaries_exact": len(flow_rows),
            "triton": 686, "external": 760, "direct_aten": 1,
        },
        "gates": {
            "all_invocation_descriptors_exact": True,
            "all_generated_embedded_program_hashes_exact": True,
            "all_external_callsites_and_tensor_abi_exact": True,
            "all_direct_aten_callsites_and_mutation_semantics_exact": True,
            "name_shape_or_ordinal_similarity_used": False,
            "fb_registry_transfer_allowed": True,
        },
        "region_rows": region_rows,
        "compute_rows": flow_rows,
        "claim_boundary": (
            "The executed release is the exact generated program population already bound "
            "to the frozen F+B registry. This transfers identity, not numerical verdicts."
        ),
    }
    payload["result_sha256"] = digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    print(json.dumps({"output": str(args.output), **payload["denominator"], "result_sha256": payload["result_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
