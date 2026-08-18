#!/usr/bin/env python3
"""Map shape-specific executed Qwen regions to the frozen seq64 F+B registry."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = ROOT / "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/full_step_inventory"
OLD_INVENTORY = OLD_ROOT / "inductor_generated_region_inventory.json"
OLD_FLOW = OLD_ROOT / "generated_compute_dataflow_audit_v1.json"
REGISTRY = OLD_ROOT / "joint_forward_backward_candidate_registry_v2.json"


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["phase"], row["kind"], tuple(row["original_aten"]),
        tuple(row["source_nodes"]),
    )


def proof_fingerprint(row: dict[str, Any]) -> str:
    return digest({
        field: row[field]
        for field in (
            "exact_proof_records", "provenance_only_proof_records",
            "component_cut_ids", "partial_cut_ids", "status",
        )
    })


def is_subsequence(values: list[str], whole: list[str]) -> bool:
    iterator = iter(whole)
    return all(any(candidate == value for candidate in iterator) for value in values)


def alpha_normalize_buffers(expression: str) -> str:
    names: dict[str, str] = {}
    def replace(match: re.Match[str]) -> str:
        value = match.group(0)
        names.setdefault(value, f"buf#{len(names)}")
        return names[value]
    return re.sub(r"\bbuf\d+\b", replace, expression)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--shape-isomorphism", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.inventory, "rt", encoding="utf-8") as handle:
        current = json.load(handle)
    isomorphism = json.loads(args.shape_isomorphism.read_text())
    if isomorphism["status"] != "COMPLETE_EXACT_SHAPE_PARAMETRIC_PROGRAM_ISOMORPHISM":
        raise RuntimeError("shape-program isomorphism is not complete")
    old_regions = json.loads(OLD_INVENTORY.read_text())["inventory"]["regions"]
    old_flow = json.loads(OLD_FLOW.read_text())["rows"]
    registry_file = json.loads(REGISTRY.read_text())
    registry = registry_file["registry"]
    proof_by_region = {row["region_id"]: row for row in registry["candidate_regions"]}
    old_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in old_regions:
        old_by_key[key(row)].append(row)
    for semantic_key, rows in old_by_key.items():
        fingerprints = {proof_fingerprint(proof_by_region[row["region_id"]]) for row in rows}
        if len(fingerprints) != 1:
            raise RuntimeError(f"repeated seq64 semantic key has nonuniform proof binding: {semantic_key}")

    current_regions = current["generated_regions"]["inventory"]["regions"]
    rows = []
    exact_key = split_refinement = loss_refinement = 0
    old_dual = next(
        row for row in old_regions
        if row["region_id"] == "backward:924"
    )
    old_nll = next(
        row for row in old_regions
        if row["region_id"] == "forward:1445"
    )
    for current_row in current_regions:
        sources = old_by_key.get(key(current_row), [])
        method = "EXACT_PHASE_KIND_ORDERED_ATEN_AND_SOURCE_NODE_KEY"
        if sources:
            exact_key += 1
        elif (
            current_row["phase"] == "BACKWARD"
            and set(current_row["source_nodes"]) < set(old_dual["source_nodes"])
            and set(current_row["source_nodes"]) in (
                set(old_dual["source_nodes"][:10]), set(old_dual["source_nodes"][10:])
            )
            and is_subsequence(current_row["original_aten"], old_dual["original_aten"])
        ):
            sources = [old_dual]
            method = "EXACT_SEQ256_PARTITION_OF_SEQ64_DUAL_OUTPUT_SOURCE_NODE_REGION"
            split_refinement += 1
        elif (
            current_row["phase"] == "FORWARD"
            and current_row["source_nodes"] == old_nll["source_nodes"]
            and current_row["original_aten"]
            == [value for value in old_nll["original_aten"] if value != "prims.prepare_softmax_online"]
        ):
            sources = [old_nll]
            method = "EXACT_SEQ256_LOSS_SCHEDULE_REFINEMENT_SAME_AOT_SOURCE_NODES"
            loss_refinement += 1
        else:
            raise RuntimeError(f"unmapped current generated region: {current_row['region_id']}")
        proof_rows = [proof_by_region[source["region_id"]] for source in sources]
        fingerprints = {proof_fingerprint(row) for row in proof_rows}
        if len(fingerprints) != 1:
            raise RuntimeError(f"nonuniform source proof binding: {current_row['region_id']}")
        row = {
            "current_region_id": current_row["region_id"],
            "phase": current_row["phase"],
            "kind": current_row["kind"],
            "source_seq64_region_ids": sorted(source["region_id"] for source in sources),
            "proof_binding_fingerprint": next(iter(fingerprints)),
            "method": method,
            "candidate_correctness_granted": False,
        }
        row["row_sha256"] = digest(row)
        rows.append(row)

    old_direct = next(row for row in old_flow if row["kind"] == "DIRECT_ATEN")
    current_direct = [
        row for row in current["compute_dataflow"]["rows"]
        if row["kind"] == "DIRECT_ATEN"
    ]
    if len(current_direct) != 1:
        raise RuntimeError("current direct ATen denominator changed")
    direct = current_direct[0]
    direct_checks = {
        field: old_direct.get(field) == direct.get(field)
        for field in ("phase", "kind", "symbol")
    }
    direct_checks["alpha_equivalent_call_ast"] = alpha_normalize_buffers(
        old_direct["call_expression"]
    ) == alpha_normalize_buffers(direct["call_expression"])
    for field in ("boundary_source", "accumulate_expression"):
        direct_checks[field] = (
            old_direct["boundary_witness"].get(field)
            == direct["boundary_witness"].get(field)
        )
    direct_checks["mutated_target_is_first_call_argument"] = all(
        row["call_expression"].startswith(
            f"aten.index_put_({row['boundary_witness']['mutated_target']},"
        ) for row in (old_direct, direct)
    )
    if not all(direct_checks.values()):
        raise RuntimeError("direct ATen call identity changed across shapes")

    payload = {
        "schema": "kernel-analyzer-qwen-shape-candidate-region-fb-bridge-v1",
        "status": "COMPLETE_ALL_EXECUTED_REGIONS_BOUND_TO_FROZEN_FB_REGISTRY",
        "bindings": {
            "current_inventory": str(args.inventory.resolve().relative_to(ROOT)),
            "current_inventory_sha256": current["result_sha256"],
            "shape_isomorphism_sha256": isomorphism["result_sha256"],
            "registry_sha256": registry["registry_sha256"],
        },
        "denominator": {
            "current_generated_regions": len(rows),
            "exact_semantic_key_regions": exact_key,
            "seq256_split_refinement_regions": split_refinement,
            "seq256_loss_refinement_regions": loss_refinement,
            "direct_aten_regions": 1,
            "all_compute_boundaries": len(rows) + 1,
        },
        "gates": {
            "all_current_generated_regions_retained": len(rows) == len(current_regions),
            "all_repeated_semantic_keys_have_uniform_fb_binding": True,
            "all_schedule_refinements_preserve_exact_aot_source_nodes": True,
            "direct_aten_call_and_mutation_identity_exact": True,
            "operator_name_shape_or_ordinal_similarity_used": False,
            "candidate_values_used": False,
            "candidate_correctness_inferred": False,
        },
        "direct_aten": {
            "current_region_id": direct["region_id"],
            "source_seq64_region_id": old_direct["region_id"],
            "checks": direct_checks,
        },
        "rows": rows,
        "claim_boundary": (
            "Every executed shape-specific compute region is structurally bound to the frozen "
            "F+B registry. This transfers program identity and proof ownership, not numerical "
            "correctness or a bias verdict."
        ),
    }
    payload["result_sha256"] = digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    print(json.dumps({"output": str(args.output), **payload["denominator"], "result_sha256": payload["result_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
