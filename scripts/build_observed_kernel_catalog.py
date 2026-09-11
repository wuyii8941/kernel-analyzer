#!/usr/bin/env python3
"""Build an exhaustive, status-preserving catalogue of observed kernels.

The input inventory is the denominator.  Saved release task packages provide
exact AOT endpoint identities where available.  Classification never creates
a reference adapter or upgrades a measurement status.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import re

from kernel_analyzer.operator_taxonomy import FAMILY_LABELS, classify_observed_position


VALID_MEASUREMENTS = {"VERIFIED", "RECORDED_MEASUREMENT_CHECKED"}


def _read(path: Path) -> dict:
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as stream:
        return json.load(stream)


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _support_status(row: dict, available_references: list[dict]) -> tuple[str, str | None]:
    runtime = row.get("runtime_measurement_status", "NOT_ASSESSED_BY_THIS_INVENTORY")
    if runtime in VALID_MEASUREMENTS:
        return "VALID_MEASUREMENT_COMPLETED", None
    if available_references and row.get("carrier"):
        return "READY_FOR_MEASUREMENT", "VALID_MEASUREMENT_NOT_COMPLETED"
    if available_references:
        return "REFERENCE_AVAILABLE", "TRAINING_PARAMETER_BINDING_REQUIRED"
    eligibility = row.get("eligibility")
    if eligibility == "OTHER_IMPLEMENTATION_REQUIRES_REFERENCE_AUDIT":
        return "IDENTIFIED", "ORDINARY_IMPLEMENTATION_REFERENCE_AUDIT_REQUIRED"
    if eligibility == "AMBIGUOUS_SOURCE_DEFINITION":
        return "IDENTIFIED", "AMBIGUOUS_SOURCE_DEFINITION"
    return "IDENTIFIED", "REFERENCE_ADAPTER_REQUIRED"


def _signature(symbol: str) -> str:
    return re.sub(r"_\d+$", "_{N}", symbol)


def build(inventory: dict, *, root: Path) -> tuple[dict, dict]:
    records = inventory["records"]
    releases = sorted({Path(row["release"]) for row in records})
    task_maps: dict[str, dict[str, dict]] = {}
    release_cut_ids: dict[str, set[str]] = {}
    source_hashes = {}
    for release in releases:
        task_path = release / "same_dtype_tasks.json.gz"
        if not task_path.exists():
            raise ValueError(f"Missing saved task package: {task_path}")
        payload = _read(task_path)
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise ValueError(f"Malformed saved task package: {task_path}")
        task_maps[str(release)] = {row["task_id"]: row for row in rows}
        release_cut_ids[str(release)] = {
            str(row["task_id"]).removeprefix("same-dtype:")
            for row in payload.get("reference_cut_tasks", [])
        }
        if len(task_maps[str(release)]) != len(rows):
            raise ValueError(f"Duplicate task ID in {task_path}")
        source_hashes[str(task_path)] = _digest(task_path)

    releases_by_digest = defaultdict(list)
    for task_path, task_digest in source_hashes.items():
        releases_by_digest[task_digest].append(str(Path(task_path).parent))
    canonical_release_by_digest = {}
    for task_digest, candidates in releases_by_digest.items():
        canonical_release_by_digest[task_digest] = min(
            candidates,
            key=lambda value: (
                "/results/coverage/runtime_releases/" not in value,
                value,
            ),
        )

    positions = []
    seen = set()
    kernels: dict[tuple, dict] = {}
    for inventory_row in records:
        identity = (inventory_row["release"], inventory_row["task_id"])
        if identity in seen:
            raise ValueError(f"Duplicate inventory position: {identity}")
        seen.add(identity)
        task = task_maps[inventory_row["release"]].get(inventory_row["task_id"])
        if task is None:
            raise ValueError(f"Inventory position absent from saved task package: {identity}")
        if task.get("symbol") != inventory_row.get("symbol") or task.get("implementation_kind") != inventory_row.get("implementation_kind"):
            raise ValueError(f"Saved implementation identity mismatch: {identity}")
        classification = classify_observed_position(
            reference_families=[row["family"] for row in inventory_row.get("reference_candidates", [])],
            exact_semantic_endpoint_id=task.get("exact_semantic_endpoint_id"),
            exact_aot_endpoint_id=task.get("exact_aot_endpoint_id"),
            symbol=inventory_row["symbol"],
        )
        available_references = list(inventory_row.get("reference_candidates", []))
        reference_discovery = "AUDITED_REFERENCE_INVENTORY" if available_references else None
        symbol = str(inventory_row.get("symbol", "")).removeprefix("extern_kernels.")
        if not available_references and inventory_row.get("carrier"):
            if (inventory_row.get("implementation_kind") == "EXTERN"
                    and symbol in {"mm", "bmm", "addmm"}):
                available_references = [{
                    "family": "LINEAR_FP32_RECOMPUTE",
                    "reference_method": "EXTERNAL_FP32_RECOMPUTE",
                    "reference_scope": "COMMON_OPERAND_EXTERNAL_RECOMPUTE",
                }]
                reference_discovery = "EXISTING_GENERIC_EXTERNAL_REFERENCE_CAPABILITY"
            elif (task.get("status") == "EXACT_CANDIDATE_BUFFER_TO_AOT_SEMANTIC_ENDPOINT"
                  and str(task.get("exact_aot_endpoint_id")) in release_cut_ids[inventory_row["release"]]):
                available_references = [{
                    "family": "CLOSED_AOT_REGION_REPLAY",
                    "reference_method": "AOT_REPLAY",
                    "reference_scope": "REFERENCE_GRAPH_ENDPOINT_SUBSTITUTION",
                    "single_kernel_source_attribution": "NOT_ESTABLISHED",
                }]
                reference_discovery = "EXISTING_GENERIC_AOT_REPLAY_CAPABILITY"
        support, reason = _support_status(inventory_row, available_references)
        kernel_key = (
            inventory_row["release"], task.get("candidate_region_id"),
            inventory_row["phase"], inventory_row["symbol"],
        )
        kernel_id = hashlib.sha256("\0".join(str(value) for value in kernel_key).encode()).hexdigest()[:24]
        task_path = str(Path(inventory_row["release"]) / "same_dtype_tasks.json.gz")
        task_digest = source_hashes[task_path]
        canonical_release = canonical_release_by_digest[task_digest]
        position = {
            "release": inventory_row["release"],
            "task_id": inventory_row["task_id"],
            "kernel_id": kernel_id,
            "release_task_package_sha256": task_digest,
            "canonical_release": canonical_release,
            "is_canonical_release_package": inventory_row["release"] == canonical_release,
            "candidate_region_id": task.get("candidate_region_id"),
            "phase": inventory_row["phase"],
            "implementation_kind": inventory_row["implementation_kind"],
            "symbol": inventory_row["symbol"],
            "symbol_signature": _signature(inventory_row["symbol"]),
            "formal_pointer": inventory_row.get("formal_pointer"),
            "exact_semantic_endpoint_id": task.get("exact_semantic_endpoint_id"),
            "task_binding_status": task.get("status"),
            "operator_family": classification["family"],
            "operator_family_label": FAMILY_LABELS[classification["family"]],
            "classification_basis": classification["classification_basis"],
            "classification_confidence": classification["classification_confidence"],
            "candidate_families": classification["candidate_families"],
            "unknown_reference_labels": classification["unknown_reference_labels"],
            "support_status": support,
            "unsupported_reason": reason,
            "runtime_measurement_status": inventory_row.get("runtime_measurement_status"),
            "reference_candidates": inventory_row.get("reference_candidates", []),
            "available_reference_bindings": available_references,
            "reference_discovery": reference_discovery,
            "carrier": inventory_row.get("carrier"),
        }
        positions.append(position)
        aggregate = kernels.setdefault(kernel_key, {
            "kernel_id": kernel_id,
            "release": inventory_row["release"],
            "candidate_region_id": task.get("candidate_region_id"),
            "phase": inventory_row["phase"],
            "implementation_kind": inventory_row["implementation_kind"],
            "symbol": inventory_row["symbol"],
            "symbol_signature": _signature(inventory_row["symbol"]),
            "position_count": 0,
            "operator_families": set(),
            "support_status_counts": Counter(),
            "classification_confidence_counts": Counter(),
        })
        aggregate["position_count"] += 1
        aggregate["operator_families"].add(classification["family"])
        aggregate["support_status_counts"][support] += 1
        aggregate["classification_confidence_counts"][classification["classification_confidence"]] += 1

    kernel_rows = []
    for aggregate in kernels.values():
        row = dict(aggregate)
        row["operator_families"] = sorted(row["operator_families"])
        row["support_status_counts"] = dict(row["support_status_counts"])
        row["classification_confidence_counts"] = dict(row["classification_confidence_counts"])
        row["kernel_family_status"] = (
            "SINGLE_FAMILY" if len(row["operator_families"]) == 1 else "MULTI_OUTPUT_OR_FUSED_FAMILIES"
        )
        kernel_rows.append(row)
    kernel_rows.sort(key=lambda row: (row["release"], row["phase"], str(row["candidate_region_id"]), row["symbol"]))

    family_positions = Counter(row["operator_family"] for row in positions)
    family_kernels = Counter(family for row in kernel_rows for family in row["operator_families"])
    support_positions = Counter(row["support_status"] for row in positions)
    confidence_positions = Counter(row["classification_confidence"] for row in positions)
    implementation_positions = Counter(row["implementation_kind"] for row in positions)
    family_support = defaultdict(Counter)
    family_impl = defaultdict(Counter)
    for row in positions:
        family_support[row["operator_family"]][row["support_status"]] += 1
        family_impl[row["operator_family"]][row["implementation_kind"]] += 1
    support_rank = {
        "IDENTIFIED": 0, "REFERENCE_AVAILABLE": 1,
        "READY_FOR_MEASUREMENT": 2, "VALID_MEASUREMENT_COMPLETED": 3,
    }
    confidence_rank = {
        "UNRESOLVED": 0, "STRUCTURAL_SUGGESTION": 1,
        "EXACT_ENDPOINT": 2, "AUDITED": 3,
    }
    distinct_positions = {}
    for row in positions:
        identity = (row["release_task_package_sha256"], row["task_id"])
        existing = distinct_positions.get(identity)
        key = (support_rank[row["support_status"]], confidence_rank[row["classification_confidence"]])
        existing_key = (-1, -1) if existing is None else (
            support_rank[existing["support_status"]],
            confidence_rank[existing["classification_confidence"]],
        )
        if key > existing_key:
            distinct_positions[identity] = row
    distinct_rows = list(distinct_positions.values())
    distinct_family_positions = Counter(row["operator_family"] for row in distinct_rows)
    distinct_support = Counter(row["support_status"] for row in distinct_rows)
    distinct_family_support = defaultdict(Counter)
    distinct_family_impl = defaultdict(Counter)
    for row in distinct_rows:
        distinct_family_support[row["operator_family"]][row["support_status"]] += 1
        distinct_family_impl[row["operator_family"]][row["implementation_kind"]] += 1
    summary = {
        "schema": "observed-kernel-catalog-summary-v1",
        "scope": "ALL_POSITIONS_IN_SUPPLIED_INVENTORY_AND_THEIR_SAVED_RELEASE_KERNEL_CALLS",
        "classification_is_reference_or_bias_proof": False,
        "position_count": len(positions),
        "kernel_invocation_count": len(kernel_rows),
        "release_count": len(releases),
        "distinct_release_task_package_count": len(releases_by_digest),
        "duplicate_release_directory_count": len(releases) - len(releases_by_digest),
        "canonical_position_count": sum(row["is_canonical_release_package"] for row in positions),
        "distinct_release_qualified_position_count": len(distinct_rows),
        "operator_family_count": len(FAMILY_LABELS),
        "all_positions_classified_including_explicit_unresolved": len(positions) == sum(family_positions.values()),
        "position_family_counts": dict(family_positions),
        "distinct_position_family_counts": dict(distinct_family_positions),
        "kernel_family_counts": dict(family_kernels),
        "position_support_status_counts": dict(support_positions),
        "distinct_position_support_status_counts": dict(distinct_support),
        "position_classification_confidence_counts": dict(confidence_positions),
        "position_implementation_kind_counts": dict(implementation_positions),
        "family_support_status_counts": {key: dict(value) for key, value in family_support.items()},
        "distinct_family_support_status_counts": {
            key: dict(value) for key, value in distinct_family_support.items()
        },
        "family_implementation_kind_counts": {key: dict(value) for key, value in family_impl.items()},
        "distinct_family_implementation_kind_counts": {
            key: dict(value) for key, value in distinct_family_impl.items()
        },
        "input_inventory_sha256": hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "release_task_package_sha256": source_hashes,
    }
    catalogue = {
        "schema": "observed-kernel-catalog-v1",
        "summary": summary,
        "positions": positions,
        "kernel_invocations": kernel_rows,
    }
    return catalogue, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--measurement-audit", type=Path)
    parser.add_argument("--measurement-merge-manifest", type=Path)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.catalog, args.summary):
        if output.exists() or not output.resolve().is_relative_to(Path("/data1/tzh")):
            parser.error("Choose new outputs under /data1/tzh")
    if bool(args.measurement_audit) != bool(args.measurement_merge_manifest):
        parser.error("--measurement-audit and --measurement-merge-manifest are required together")
    inventory = _read(args.inventory)
    input_files = {str(args.inventory.resolve()): _digest(args.inventory)}
    if args.measurement_audit:
        from scripts.merge_family_execution_measurements import merge
        audit = _read(args.measurement_audit)
        manifest = _read(args.measurement_merge_manifest)
        if manifest.get("schema") != "family-measurement-inventory-merge-v1":
            raise ValueError("Unexpected measurement merge manifest")
        inventory = merge(
            inventory, audit, manifest["mappings"], verify_artifacts=True,
        )
        input_files[str(args.measurement_audit.resolve())] = _digest(args.measurement_audit)
        input_files[str(args.measurement_merge_manifest.resolve())] = _digest(args.measurement_merge_manifest)
    catalogue, summary = build(inventory, root=Path(__file__).resolve().parents[1])
    taxonomy_source = Path(__file__).resolve().parents[1] / "src/kernel_analyzer/operator_taxonomy.py"
    summary["input_file_sha256"] = input_files
    summary["source_sha256"] = {
        str(Path(__file__).resolve()): _digest(Path(__file__).resolve()),
        str(taxonomy_source.resolve()): _digest(taxonomy_source.resolve()),
    }
    args.catalog.parent.mkdir(parents=True, exist_ok=True)
    # Freeze gzip metadata as well as JSON ordering so downstream manifests can
    # hash this catalogue reproducibly.
    with args.catalog.open("xb") as raw_stream:
        with gzip.GzipFile(filename="", fileobj=raw_stream, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as stream:
                json.dump(catalogue, stream, separators=(",", ":"), allow_nan=False)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("x") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({key: summary[key] for key in (
        "position_count", "kernel_invocation_count", "release_count",
        "operator_family_count", "position_support_status_counts",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
