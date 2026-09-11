#!/usr/bin/env python3
"""Build a deterministic, Triton-first coverage frontier.

This is a planning artifact, not a numerical result.  It groups the observed
execution inventory by the saved implementation identity and operator family,
then records the next actionable representative without using any numerical
outcome.  A family is not considered measured merely because it appears in the
inventory or has a source label.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


IMPLEMENTATION_PRIORITY = {"TRITON": 0, "DIRECT_ATEN": 1, "DIRECT_TORCH_OP": 2,
                           "DIRECT_TENSOR_METHOD": 3, "EXTERN": 4}
STATUS_PRIORITY = {
    "VALID_MEASUREMENT_COMPLETED": 0,
    "READY_FOR_MEASUREMENT": 1,
    "REFERENCE_AVAILABLE": 2,
    "IDENTIFIED": 3,
}


def read_json(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    """Deduplicate release copies without merging distinct execution tasks."""
    release = row.get("release_task_package_sha256") or row.get("canonical_release") or row["release"]
    return (str(release), str(row["task_id"]), str(row.get("formal_pointer", "")),
            str(row.get("phase", "")))


def deduplicate(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = canonical_key(row)
        previous = selected.get(key)
        if previous is None:
            selected[key] = row
            continue
        # Prefer the canonical package and the stronger saved evidence.  This
        # is a provenance choice, never a numerical-outcome choice.
        previous_rank = (
            not bool(previous.get("is_canonical_release_package", False)),
            STATUS_PRIORITY.get(previous.get("support_status"), 99),
            previous.get("release", ""),
        )
        current_rank = (
            not bool(row.get("is_canonical_release_package", False)),
            STATUS_PRIORITY.get(row.get("support_status"), 99),
            row.get("release", ""),
        )
        if current_rank < previous_rank:
            selected[key] = row
    return list(selected.values())


def action_for(rows: list[dict[str, Any]]) -> str:
    statuses = Counter(row.get("support_status", "UNDECLARED") for row in rows)
    if statuses["READY_FOR_MEASUREMENT"]:
        return "RUN_DECLARED_MEASUREMENT"
    if statuses["REFERENCE_AVAILABLE"]:
        return "BIND_TRAINING_PARAMETER_OR_EXECUTION_REFERENCE"
    if statuses["IDENTIFIED"]:
        return "ADD_REFERENCE_OR_PARAMETER_MAPPING"
    if statuses["VALID_MEASUREMENT_COMPLETED"]:
        return "NO_NEW_POSITION_REQUIRED"
    return "REVIEW_SUPPORT_STATUS"


def representative(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [row for row in rows if row.get("support_status") != "VALID_MEASUREMENT_COMPLETED"]
    if not eligible:
        return None
    eligible.sort(key=lambda row: (
        STATUS_PRIORITY.get(row.get("support_status"), 99),
        IMPLEMENTATION_PRIORITY.get(row.get("implementation_kind"), 99),
        row.get("classification_confidence") != "AUDITED",
        row.get("release", ""), row.get("task_id", ""),
    ))
    row = eligible[0]
    return {
        "release": row.get("release"),
        "canonical_release": row.get("canonical_release", row.get("release")),
        "task_id": row.get("task_id"),
        "phase": row.get("phase"),
        "symbol": row.get("symbol"),
        "symbol_signature": row.get("symbol_signature"),
        "formal_pointer": row.get("formal_pointer"),
        "carrier": row.get("carrier"),
        "support_status": row.get("support_status"),
        "reference_candidates": row.get("available_reference_bindings", []),
        "classification_confidence": row.get("classification_confidence"),
        "selection_rule": "FIRST_ACTIONABLE_ROW_BY_STATUS_AND_IMPLEMENTATION; NO_NUMERICAL_OUTCOME",
    }


def build(catalogue: dict[str, Any], *, input_sha256: str) -> dict[str, Any]:
    rows = deduplicate(catalogue.get("positions", []))
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("operator_family", "UNRESOLVED_LOW_LEVEL")),
                str(row.get("implementation_kind", "UNDECLARED")))].append(row)

    family_rows = []
    for (family, implementation_kind), family_positions in sorted(groups.items()):
        statuses = Counter(row.get("support_status", "UNDECLARED") for row in family_positions)
        family_rows.append({
            "operator_family": family,
            "implementation_kind": implementation_kind,
            "position_count": len(family_positions),
            "support_status_counts": dict(sorted(statuses.items())),
            "valid_measurement_count": statuses.get("VALID_MEASUREMENT_COMPLETED", 0),
            "ready_for_measurement_count": statuses.get("READY_FOR_MEASUREMENT", 0),
            "identified_only_count": statuses.get("IDENTIFIED", 0),
            "reference_available_count": statuses.get("REFERENCE_AVAILABLE", 0),
            "next_action": action_for(family_positions),
            "next_representative": representative(family_positions),
        })

    triton = [row for row in family_rows if row["implementation_kind"] == "TRITON"]
    return {
        "schema": "triton-coverage-frontier-v1",
        "scope": "OBSERVED_EXECUTION_INVENTORY; FAMILY_LEVEL_PLANNING_NOT_NUMERICAL_RESULT",
        "selection_uses_numerical_outcomes": False,
        "deduplication": "release_task_package_hash + task_id + formal_pointer + phase",
        "position_count_after_deduplication": len(rows),
        "operator_family_count": len({row["operator_family"] for row in family_rows}),
        "triton_family_count": len({row["operator_family"] for row in triton}),
        "implementation_kind_counts": dict(sorted(Counter(
            row.get("implementation_kind", "UNDECLARED") for row in rows).items())),
        "triton_position_count": sum(
            row["position_count"] for row in triton),
        "triton_valid_measurement_count": sum(
            row["valid_measurement_count"] for row in triton),
        "triton_ready_for_measurement_count": sum(
            row["ready_for_measurement_count"] for row in triton),
        "catalogue_sha256": input_sha256,
        "families": family_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("output must remain under /data1/tzh")
    if args.output.exists():
        parser.error("refusing to overwrite an existing frontier")
    payload = build(read_json(args.catalog), input_sha256=digest(args.catalog))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: payload[key] for key in (
        "position_count_after_deduplication", "operator_family_count",
        "triton_family_count", "triton_position_count",
        "triton_valid_measurement_count", "triton_ready_for_measurement_count",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
