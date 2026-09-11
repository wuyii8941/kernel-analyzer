#!/usr/bin/env python3
"""Verify catalogue, summary, and family-first queue without GPU execution."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


def read(path: Path) -> dict:
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as stream:
        return json.load(stream)


def verify(catalogue: dict, summary: dict, queue: dict, *, catalog_sha256: str) -> dict:
    if catalogue.get("schema") != "observed-kernel-catalog-v1":
        raise ValueError("Unexpected catalogue schema")
    if summary.get("schema") != "observed-kernel-catalog-summary-v1":
        raise ValueError("Unexpected summary schema")
    if catalogue.get("summary") != summary:
        raise ValueError("Standalone summary differs from catalogue summary")
    positions = catalogue["positions"]
    if len(positions) != summary["position_count"]:
        raise ValueError("Position count changed")
    identities = [(row["release"], row["task_id"]) for row in positions]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate release-qualified position")
    distinct = {(row["release_task_package_sha256"], row["task_id"]) for row in positions}
    if len(distinct) != summary["distinct_release_qualified_position_count"]:
        raise ValueError("Distinct task-package position count changed")
    if Counter(row["support_status"] for row in positions) != Counter(summary["position_support_status_counts"]):
        raise ValueError("Support counts changed")
    if queue.get("catalog_sha256") != catalog_sha256:
        raise ValueError("Queue points to another catalogue")
    queued = [(row["release"], row["task_id"]) for row in queue["rows"]]
    if len(queued) != len(set(queued)) or len(queued) != queue["position_count"]:
        raise ValueError("Queue identities are duplicated or incomplete")
    if queue.get("selection_uses_numerical_outcomes") is not False:
        raise ValueError("Queue outcome-independence declaration absent")
    return {
        "schema": "observed-kernel-catalog-verification-v1",
        "status": "VERIFIED",
        "position_count": len(positions),
        "distinct_position_count": len(distinct),
        "queued_position_count": len(queued),
        "valid_measurement_count": summary["distinct_position_support_status_counts"].get(
            "VALID_MEASUREMENT_COMPLETED", 0),
        "all_positions_have_a_family_label": all(row.get("operator_family") for row in positions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--recomputed-catalog", type=Path)
    parser.add_argument("--recomputed-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("Choose a new output under /data1/tzh")
    catalog_bytes = args.catalog.read_bytes()
    result = verify(
        read(args.catalog), read(args.summary), read(args.queue),
        catalog_sha256=hashlib.sha256(catalog_bytes).hexdigest(),
    )
    result["input_sha256"] = {
        str(args.catalog): hashlib.sha256(catalog_bytes).hexdigest(),
        str(args.summary): hashlib.sha256(args.summary.read_bytes()).hexdigest(),
        str(args.queue): hashlib.sha256(args.queue.read_bytes()).hexdigest(),
    }
    if bool(args.recomputed_catalog) != bool(args.recomputed_summary):
        parser.error("both recomputed inputs are required together")
    if args.recomputed_catalog:
        result["deterministic_recomputation"] = {
            "catalog_byte_identical": args.recomputed_catalog.read_bytes() == catalog_bytes,
            "summary_byte_identical": args.recomputed_summary.read_bytes() == args.summary.read_bytes(),
            "recomputed_catalog_sha256": hashlib.sha256(
                args.recomputed_catalog.read_bytes()
            ).hexdigest(),
            "recomputed_summary_sha256": hashlib.sha256(
                args.recomputed_summary.read_bytes()
            ).hexdigest(),
        }
        if not all(result["deterministic_recomputation"][key] for key in (
            "catalog_byte_identical", "summary_byte_identical"
        )):
            raise ValueError("catalogue recomputation is not byte-identical")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
