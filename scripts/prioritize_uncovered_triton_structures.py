#!/usr/bin/env python3
"""Rank uncovered saved Triton structures without reading numerical results.

The output is a review queue, not a semantic operator-family assignment.  A
shared structural fingerprint can suggest where one reference adapter may be
reusable, but a researcher must still check the mathematical computation.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re


IGNORED_CALLS = {
    "tl.arange", "tl.broadcast_to", "tl.full", "tl.load", "tl.program_id",
    "tl.store",
}

EXISTING_FAMILY_NAME_HINTS = (
    ("CROSS_ENTROPY", ("nll_loss",)),
    ("SOFTMAX", ("softmax",)),
    ("SILU_GATING", ("silu",)),
    ("SOFTPLUS", ("softplus",)),
    ("GELU", ("gelu",)),
    ("EMBEDDING", ("embedding",)),
    ("CONVOLUTION", ("convolution", "conv1d")),
)


def stable_calls(calls):
    """Keep language/library calls while dropping expression-specific calls."""
    kept = []
    for name, count in calls.items():
        if name in IGNORED_CALLS:
            continue
        # ast.unparse also yields expression-specific names such as
        # ``tl.load(...).to``.  Those encode dimensions and offsets and would
        # split one reusable structure into hundreds of apparent groups.
        if (name.startswith(("tl.", "libdevice.", "tl_math.", "triton_helpers."))
                and re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+", name)):
            kept.append((name, count))
    return tuple(sorted(kept))


def fingerprint(operation):
    if operation.get("status") != "BODY_INSPECTED":
        raise ValueError("Only inspected bodies can be prioritized")
    return (
        operation["structural_role"],
        stable_calls(operation["calls"]),
        tuple(sorted(operation["arithmetic"].items())),
        operation["loop_count"],
        operation["has_atomic"],
        operation["has_reduction"],
    )


def priority_features(operation):
    calls = stable_calls(operation["calls"])
    return {
        "has_atomic": operation["has_atomic"],
        "has_reduction": operation["has_reduction"],
        "has_nonlinear_math": any(
            name.startswith(("libdevice.", "tl_math.")) for name, _ in calls
        ),
        "language_calls": [{"name": name, "count": count} for name, count in calls],
        "arithmetic": operation["arithmetic"],
        "loop_count": operation["loop_count"],
        "structural_role": operation["structural_role"],
    }


def existing_family_hints(rows, operation):
    joined = " ".join(row["symbol"].lower() for row in rows)
    hints = {family for family, tokens in EXISTING_FAMILY_NAME_HINTS
             if any(token in joined for token in tokens)}
    calls = {name for name, _ in stable_calls(operation["calls"])}
    if any(name.endswith(".rsqrt") for name in calls):
        hints.add("NORMALIZATION")
    if (any(name.endswith(".sin") for name in calls)
            and any(name.endswith(".cos") for name in calls)):
        hints.add("ROTARY")
    return sorted(hints)


def prioritize(records, type_records=None):
    type_map = None
    if type_records is not None:
        type_map = {}
        for row in type_records:
            key = (row["source"], row["symbol"])
            if key in type_map:
                raise ValueError("Duplicate type-audit identity")
            type_map[key] = row["type_review"]["status"]
        expected = {(row["source"], row["symbol"]) for row in records}
        if set(type_map) != expected:
            raise ValueError("Type audit and operation inventory identities differ")
    groups = defaultdict(list)
    covered = defaultdict(set)
    for row in records:
        operation = row["operation_inventory"]
        if operation.get("status") != "BODY_INSPECTED":
            continue
        key = fingerprint(operation)
        if row.get("status") != "REFERENCE_ADAPTER_REQUIRED":
            covered[key].update(row.get("matched_reference_families", []))
            continue
        groups[key].append(row)
    output = []
    for rows in groups.values():
        rows = sorted(rows, key=lambda row: (row["source"], row["symbol"]))
        operation = rows[0]["operation_inventory"]
        features = priority_features(operation)
        family_hints = existing_family_hints(rows, operation)
        type_statuses = sorted({
            type_map[(row["source"], row["symbol"])] if type_map is not None
            else "TYPE_AUDIT_NOT_SUPPLIED"
            for row in rows
        })
        has_floating_pointers = "FLOATING_POINTERS_REQUIRES_SEMANTIC_REVIEW" in type_statuses
        # This ordering uses only saved source structure.  It intentionally does
        # not predict bias or replace semantic review.
        risk_feature_count = sum(
            bool(features[name])
            for name in ("has_atomic", "has_reduction", "has_nonlinear_math")
        )
        output.append({
            "definition_count": len(rows),
            "source_file_count": len({row["source"] for row in rows}),
            "risk_feature_count": risk_feature_count,
            "features": features,
            "representative_symbol": rows[0]["symbol"],
            "representative_source": rows[0]["source"],
            "members": [
                {"source": row["source"], "symbol": row["symbol"]} for row in rows
            ],
            "structurally_seen_with_registered_reference": fingerprint(operation) in covered,
            "structurally_matching_reference_families": sorted(covered.get(fingerprint(operation), set())),
            "likely_existing_family_hints": family_hints,
            "pointer_type_statuses": type_statuses,
            "has_floating_pointers": has_floating_pointers,
            "semantic_family_status": "REQUIRES_HUMAN_REVIEW",
        })
    return sorted(
        output,
        key=lambda row: (
            not row["has_floating_pointers"],
            row["structurally_seen_with_registered_reference"],
            bool(row["likely_existing_family_hints"]),
            -row["risk_feature_count"], -row["definition_count"],
            row["representative_symbol"], row["representative_source"],
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--type-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("Choose a new output under /data1/tzh")
    raw = args.inventory.read_bytes()
    inventory = json.loads(raw)
    type_raw = args.type_audit.read_bytes() if args.type_audit else None
    type_document = json.loads(type_raw) if type_raw is not None else None
    rows = prioritize(
        inventory["records"],
        None if type_document is None else type_document["records"],
    )
    result = {
        "schema": "uncovered-triton-structural-priority-v1",
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "type_audit_sha256": (
            hashlib.sha256(type_raw).hexdigest() if type_raw is not None else None
        ),
        "selection_uses_numerical_results": False,
        "scope": "SAVED_TRITON_DEFINITIONS_WITHOUT_A_REGISTERED_REFERENCE",
        "warning": (
            "Structural clusters are review candidates, not verified mathematical "
            "operator families or independent bias mechanisms."
        ),
        "uncovered_definition_count": sum(row["definition_count"] for row in rows),
        "structural_cluster_count": len(rows),
        "clusters": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({
        "uncovered_definitions": result["uncovered_definition_count"],
        "structural_clusters": result["structural_cluster_count"],
    }))


if __name__ == "__main__":
    main()
