#!/usr/bin/env python3
"""Prioritize unmeasured, reference-bound Triton work across operator families.

The queue uses no numerical outcomes.  It first covers distinct families, then
distinct structural signatures, and only then additional positions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


def _read(path: Path) -> dict:
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as stream:
        return json.load(stream)


def build(catalogue: dict) -> dict:
    grouped = {}
    for row in catalogue["positions"]:
        identity = (row.get("release_task_package_sha256", row["release"]), row["task_id"])
        grouped.setdefault(identity, []).append(row)
    rows = []
    for copies in grouped.values():
        if any(row["support_status"] == "VALID_MEASUREMENT_COMPLETED" for row in copies):
            continue
        ready = [row for row in copies if row["support_status"] == "READY_FOR_MEASUREMENT"]
        if not ready:
            continue
        chosen = max(ready, key=lambda row: (
            row["classification_confidence"] == "AUDITED",
            row.get("is_canonical_release_package", False),
            row["release"],
        )).copy()
        evidence_release = chosen["release"]
        chosen["release"] = chosen.get("canonical_release", chosen["release"])
        chosen["reference_binding_evidence_release"] = evidence_release
        rows.append(chosen)
    # One task can only occur once in the catalogue; preserve that invariant.
    identities = [(row["release"], row["task_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate release-qualified task in executable queue")
    rows.sort(key=lambda row: (
        row["implementation_kind"] != "TRITON",
        row["operator_family"] in {"FUSED_MIXED", "UNRESOLVED_LOW_LEVEL"},
        row["operator_family"], row["symbol_signature"], row["release"], row["task_id"],
    ))
    selected = []
    selected_identities = set()
    seen_families = set()
    seen_signatures = set()
    for wave in ("NEW_FAMILY_FIRST", "NEW_SIGNATURE_SECOND", "REMAINING_POSITIONS"):
        for row in rows:
            identity = (row["release"], row["task_id"])
            if identity in selected_identities:
                continue
            signature = (row["operator_family"], row["implementation_kind"], row["phase"], row["symbol_signature"])
            if wave == "NEW_FAMILY_FIRST" and row["operator_family"] in seen_families:
                continue
            if wave == "NEW_SIGNATURE_SECOND" and signature in seen_signatures:
                continue
            selected.append({
                "priority_index": len(selected),
                "wave": wave,
                "operator_family": row["operator_family"],
                "implementation_kind": row["implementation_kind"],
                "release": row["release"],
                "task_id": row["task_id"],
                "symbol": row["symbol"],
                "symbol_signature": row["symbol_signature"],
                "carrier": row["carrier"],
                "reference_candidates": row["available_reference_bindings"],
                "reference_discovery": row["reference_discovery"],
                "reference_binding_evidence_release": row["reference_binding_evidence_release"],
                "classification_confidence": row["classification_confidence"],
                "selection_uses_numerical_outcomes": False,
            })
            selected_identities.add(identity)
            seen_families.add(row["operator_family"])
            seen_signatures.add(signature)
    if len(selected) != len(rows):
        raise AssertionError("Queue construction dropped executable positions")
    return {
        "schema": "family-first-execution-queue-v1",
        "scope": "ALL_REFERENCE_AND_TRAINING_BOUND_UNMEASURED_POSITIONS_IN_CATALOG",
        "duplicate_release_packages_excluded": True,
        "selection_uses_numerical_outcomes": False,
        "priority_policy": [
            "TRITON_BEFORE_OTHER_IMPLEMENTATIONS",
            "ONE_POSITION_PER_OPERATOR_FAMILY",
            "ONE_POSITION_PER_FAMILY_IMPLEMENTATION_PHASE_SYMBOL_SIGNATURE",
            "REMAINING_RELEASE_QUALIFIED_POSITIONS",
        ],
        "position_count": len(selected),
        "wave_counts": dict(Counter(row["wave"] for row in selected)),
        "family_counts": dict(Counter(row["operator_family"] for row in selected)),
        "implementation_kind_counts": dict(Counter(row["implementation_kind"] for row in selected)),
        "rows": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("Choose a new output under /data1/tzh")
    raw = args.catalog.read_bytes()
    result = build(_read(args.catalog))
    result["catalog_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({key: result[key] for key in (
        "position_count", "wave_counts", "family_counts", "implementation_kind_counts",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
