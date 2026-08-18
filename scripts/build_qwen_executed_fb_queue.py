#!/usr/bin/env python3
"""Bind current T1-positive Qwen regions to exact frozen F+B proof units."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/full_step_inventory/joint_forward_backward_candidate_registry_v2.json"
POSITIVE = "DIRECTIONAL_BIAS_SCREEN_POSITIVE"
EXACT_VJP = "EXACT_ACTUAL_BACKWARD_PROGRAM"


def read(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    oracle = read(args.oracle)
    identity = read(args.identity)
    registry_file = read(REGISTRY)
    registry = registry_file["registry"]
    if oracle["status"] != "COMPLETE_TRITON_DENOMINATOR_HELDOUT_SCREEN":
        raise RuntimeError("T1 oracle is not complete")
    if identity["status"] == "COMPLETE_EXACT_PROGRAM_IDENTITY_TO_FB_REGISTRY":
        if identity["registry_file_sha256"] != hashlib.sha256(REGISTRY.read_bytes()).hexdigest():
            raise RuntimeError("identity artifact does not bind the loaded F+B registry")
        if not identity["gates"]["fb_registry_transfer_allowed"]:
            raise RuntimeError("F+B registry transfer is not allowed")
        source_regions_by_current = None
        binding_mode = "EXACT_EXECUTED_RELEASE_IDENTITY"
    elif identity["status"] == "COMPLETE_ALL_EXECUTED_REGIONS_BOUND_TO_FROZEN_FB_REGISTRY":
        required = (
            "all_current_generated_regions_retained",
            "all_repeated_semantic_keys_have_uniform_fb_binding",
            "all_schedule_refinements_preserve_exact_aot_source_nodes",
        )
        if not all(identity["gates"][gate] for gate in required):
            raise RuntimeError("shape-region bridge is not complete")
        if identity["gates"]["operator_name_shape_or_ordinal_similarity_used"]:
            raise RuntimeError("shape-region bridge used a forbidden similarity match")
        if identity["bindings"]["registry_sha256"] != registry["registry_sha256"]:
            raise RuntimeError("shape-region bridge does not bind the loaded F+B registry")
        source_regions_by_current = {
            row["current_region_id"]: row["source_seq64_region_ids"]
            for row in identity["rows"]
        }
        if len(source_regions_by_current) != len(identity["rows"]):
            raise RuntimeError("shape-region bridge has duplicate current region IDs")
        binding_mode = "EXACT_SHAPE_REGION_REFINEMENT_TO_SEQ64_FB_REGISTRY"
    else:
        raise RuntimeError("executed-region F+B binding is not complete")

    candidate_by_id = {row["region_id"]: row for row in registry["candidate_regions"]}
    unit_by_id = {row["unit_id"]: row for row in registry["forward_vjp_units"]}
    positive_endpoints: dict[str, list[str]] = defaultdict(list)
    for row in oracle["rows"]:
        if row["verdict"] == POSITIVE:
            positive_endpoints[row["region_id"]].append(row["endpoint"])

    rows = []
    selected_units: set[str] = set()
    for region_id in sorted(positive_endpoints):
        source_region_ids = (
            [region_id] if source_regions_by_current is None
            else source_regions_by_current.get(region_id)
        )
        if not source_region_ids:
            raise RuntimeError(f"positive region absent from exact F+B bridge: {region_id}")
        candidates = [candidate_by_id.get(source_id) for source_id in source_region_ids]
        if any(candidate is None for candidate in candidates):
            raise RuntimeError(f"bridged seq64 region absent from registry: {region_id}")
        if len({candidate["phase"] for candidate in candidates}) != 1:
            raise RuntimeError(f"shape refinement crosses phases: {region_id}")
        exact_unit_ids = sorted(
            record["record_id"]
            for candidate in candidates
            for record in candidate["exact_proof_records"]
            if record["record_kind"] == "FORWARD_VJP_UNIT"
        )
        exact_unit_ids = sorted(set(exact_unit_ids))
        exact_units = [unit_by_id[unit_id] for unit_id in exact_unit_ids]
        closed_units = [
            unit
            for unit in exact_units
            if unit["vjp_status"] == EXACT_VJP and unit["actual_backward_node_ids"]
        ]
        selected_units.update(unit["unit_id"] for unit in closed_units)
        status = "ELIGIBLE_EXACT_FB_FOR_T2" if closed_units else "UNRESOLVED_NO_ACTUAL_BACKWARD"
        row = {
            "region_id": region_id,
            "source_seq64_region_ids": sorted(source_region_ids),
            "phase": candidates[0]["phase"],
            "kind": sorted({candidate["kind"] for candidate in candidates}),
            "positive_endpoints": sorted(positive_endpoints[region_id]),
            "candidate_region_binding_status": sorted(
                {candidate["status"] for candidate in candidates}
            ),
            "eligible_exact_fb_units": [
                {
                    "unit_id": unit["unit_id"],
                    "forward_node_ids": unit["forward_node_ids"],
                    "actual_backward_node_ids": unit["actual_backward_node_ids"],
                    "mathematical_derivation_sha256": unit["mathematical_derivation_sha256"],
                    "structural_proof_unit_id": unit["structural_proof_unit_id"],
                }
                for unit in closed_units
            ],
            "excluded_exact_records": [
                {
                    "unit_id": unit["unit_id"],
                    "vjp_status": unit["vjp_status"],
                    "actual_backward_node_ids": unit["actual_backward_node_ids"],
                }
                for unit in exact_units
                if unit not in closed_units
            ],
            "status": status,
        }
        row["row_sha256"] = digest(row)
        rows.append(row)

    eligible = sum(row["status"] == "ELIGIBLE_EXACT_FB_FOR_T2" for row in rows)
    unresolved = len(rows) - eligible
    payload = {
        "schema": "kernel-analyzer-qwen-executed-t1-fb-queue-v1",
        "status": "COMPLETE_FAIL_CLOSED_T1_TO_FB_BINDING",
        "bindings": {
            "oracle": str(args.oracle.resolve().relative_to(ROOT)),
            "oracle_sha256": oracle["result_sha256"],
            "executed_release_identity": str(args.identity.resolve().relative_to(ROOT)),
            "executed_release_identity_sha256": identity["result_sha256"],
            "binding_mode": binding_mode,
            "fb_registry": str(REGISTRY.relative_to(ROOT)),
            "fb_registry_sha256": registry["registry_sha256"],
        },
        "denominator": {
            "t1_positive_endpoints": sum(len(value) for value in positive_endpoints.values()),
            "t1_positive_regions": len(rows),
            "regions_eligible_for_t2": eligible,
            "regions_unresolved_before_t2": unresolved,
            "unique_exact_fb_units_eligible_for_t2": len(selected_units),
        },
        "gates": {
            "all_t1_positive_regions_retained": len(rows) == len(positive_endpoints),
            "only_exact_candidate_region_records_used": True,
            "only_actual_backward_program_units_eligible": True,
            "analytic_zero_or_unreached_vjp_not_promoted": True,
            "candidate_correctness_inferred_from_identity": False,
            "t2_or_later_verdict_granted": False,
        },
        "rows": rows,
        "claim_boundary": (
            "This queue binds current T1-positive executed regions to independently derived "
            "forward maps and their exact actual backward programs. It grants no causal, "
            "carrier-coherence, accumulation, or correctness verdict."
        ),
    }
    payload["result_sha256"] = digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    print(json.dumps({"output": str(args.output), **payload["denominator"], "result_sha256": payload["result_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
