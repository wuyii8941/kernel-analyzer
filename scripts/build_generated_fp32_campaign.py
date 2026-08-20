#!/usr/bin/env python3
"""Freeze an all-Triton FP32 replay plan from an executed schedule."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def pointer_abi(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    bindings = row["boundary_witness"]["formal_to_actual_pointer_binding"]
    inputs = sorted(
        name for name, value in bindings.items()
        if value["loaded"] or name.startswith("in_out_ptr")
    )
    outputs = sorted(name for name, value in bindings.items() if value["stored"])
    return inputs, outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.inventory, "rt", encoding="utf-8") as handle:
        inventory = json.load(handle)
    if inventory["status"] != "COMPLETE_GENERATED_SCHEDULE_AND_POINTER_DATAFLOW":
        raise RuntimeError("executed generated inventory is incomplete")
    regions = [
        row for row in inventory["generated_regions"]["inventory"]["regions"]
        if row["kind"] == "TRITON"
    ]
    flow = {
        row["region_id"]: row for row in inventory["compute_dataflow"]["rows"]
        if row["kind"] == "TRITON"
    }
    if len(regions) != len(flow):
        raise RuntimeError("Triton inventory and pointer-dataflow denominators differ")
    rows = []
    for region in regions:
        dataflow = flow[region["region_id"]]
        inputs, outputs = pointer_abi(dataflow)
        if not outputs:
            raise RuntimeError(f"Triton region has no stored endpoint: {region['region_id']}")
        row = {
            "region_id": region["region_id"],
            "phase": region["phase"],
            "symbol": region["symbol"],
            "source_path": dataflow["source_path"],
            "source_line": int(dataflow["source_line"]),
            "source_line_sha256": dataflow["source_line_sha256"],
            "original_aten": region["original_aten"],
            "source_nodes": region["source_nodes"],
            "input_names": inputs,
            "output_names": outputs,
            "embedded_program_sha256": dataflow["boundary_witness"][
                "embedded_program_sha256"
            ],
            "reference_kind": "SAME_GENERATED_PROGRAM_FP32_STORAGE_REPLAY",
            "reference_role": "PRECISION_ONLY_GENERATED_PROGRAM_COUNTERFACTUAL",
            "candidate_values_used_to_select_reference": False,
            "status": "EXACT_STATIC_PROGRAM_AND_POINTER_ABI_REPLAY_PLAN",
        }
        row["row_sha256"] = digest(row)
        rows.append(row)
    payload = {
        "schema": "kernel-analyzer-generated-fp32-replay-campaign-v1",
        "status": "COMPLETE_ALL_TRITON_FP32_REPLAY_PLAN",
        "architecture": inventory.get("architecture", "qwen"),
        "sequence_length": inventory["generated_regions"]["state"]["length"],
        "denominator": {
            "triton_invocations": len(rows),
            "exact_static_replay_plans": len(rows),
            "unresolved": 0,
        },
        "bindings": {
            "executed_inventory": str(args.inventory.resolve().relative_to(ROOT)),
            "executed_inventory_sha256": inventory["result_sha256"],
        },
        "gates": {
            "all_actual_triton_invocations_retained": True,
            "all_pointer_abis_execution_derived": True,
            "all_generated_program_hashes_bound": True,
            "candidate_values_used_to_select_reference": False,
            "operator_name_shape_or_ordinal_pairing_used": False,
            "semantic_equivalence_inferred": False,
        },
        "rows": rows,
        "claim_boundary": (
            "The reference reruns the identical generated Triton program with floating storages "
            "promoted to FP32 while preserving pointer aliasing, views, grid and scalar arguments. "
            "It detects precision-induced error in that program; it does not prove the program's "
            "semantic equivalence to eager ATen."
        ),
    }
    payload["result_sha256"] = digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    with gzip.open(temporary, "rt", encoding="utf-8") as handle:
        if json.load(handle)["result_sha256"] != payload["result_sha256"]:
            raise RuntimeError("campaign post-write validation failed")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), **payload["denominator"]}, sort_keys=True))


if __name__ == "__main__":
    main()
