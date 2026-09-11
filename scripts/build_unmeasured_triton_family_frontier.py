#!/usr/bin/env python3
"""Build a family-level frontier without selecting on numerical outcomes.

The observed-kernel catalogue contains positions, while the research plan is
organized around operator/problem families.  This report joins the catalogue
with the family report and the already recorded case roles, then emits at
most one static candidate per family that has no prior measurement evidence.
It deliberately does not call a family negative merely because a run is
missing or failed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _valid_count(row: dict[str, Any]) -> int:
    counts = row.get("support_stage_counts", {})
    return sum(int(value) for key, value in counts.items() if "VALID" in key)


def _family_evidence(row: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not row:
        return []
    evidence: list[dict[str, Any]] = []
    valid = _valid_count(row)
    if valid:
        evidence.append({"kind": "CATALOG_VALID_MEASUREMENT", "count": valid})
    roles = int(row.get("historical_role_records", 0) or 0)
    if roles:
        evidence.append({"kind": "HISTORICAL_MAINLINE_ROLE", "count": roles})
    for key in ("additional_historical_artifacts", "additional_measurement_evidence"):
        for item in row.get(key, []) or []:
            evidence.append({
                "kind": key.upper(),
                "path": item.get("path") or item.get("artifact_path"),
                "measurement_status": item.get("measurement_status") or item.get("historical_status"),
            })
    return evidence


def _catalog_stats(positions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for position in positions:
        grouped[str(position["operator_family"])].append(position)
    result = {}
    for family, rows in sorted(grouped.items()):
        support = Counter(str(row.get("support_status", "UNDECLARED")) for row in rows)
        kinds = Counter(str(row.get("implementation_kind", "UNDECLARED")) for row in rows)
        ready = [row for row in rows if row.get("support_status") == "READY_FOR_MEASUREMENT"]
        result[family] = {
            "catalogue_position_count": len(rows),
            "implementation_kind_counts": dict(sorted(kinds.items())),
            "support_status_counts": dict(sorted(support.items())),
            "ready_position_count": len(ready),
            "identified_position_count": sum(
                value for key, value in support.items() if key in {"IDENTIFIED", "REFERENCE_AVAILABLE"}
            ),
            "triton_position_count": sum(row.get("implementation_kind") == "TRITON" for row in rows),
        }
    return result


def _queue_candidates(queue: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Choose one queue row per family using only static queue priority."""
    candidates: dict[str, dict[str, Any]] = {}
    wave_rank = {"NEW_FAMILY_FIRST": 0, "NEW_SIGNATURE_SECOND": 1, "REMAINING_POSITIONS": 2}
    for row in queue.get("rows", []):
        family = str(row["operator_family"])
        if family in candidates:
            continue
        candidates[family] = {
            "wave": row.get("wave"),
            "wave_rank": wave_rank.get(row.get("wave"), 99),
            "release": row.get("release"),
            "task_id": row.get("task_id"),
            "symbol": row.get("symbol"),
            "symbol_signature": row.get("symbol_signature"),
            "phase": row.get("phase"),
            "implementation_kind": row.get("implementation_kind"),
            "carrier": row.get("carrier"),
            "reference_candidates": row.get("reference_candidates", []),
            "reference_discovery": row.get("reference_discovery"),
            "classification_confidence": row.get("classification_confidence"),
            "selection_uses_numerical_outcomes": False,
        }
    return candidates


def _prior_campaign_status(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        payload = _read(path)
        for row in payload.get("rows", []):
            family = row.get("operator_family")
            if family:
                execution_status = row.get("execution_status") or row.get("status")
                result[str(family)].append({
                    "summary": str(path),
                    "execution_status": execution_status,
                    "measurement_status": row.get("measurement_status"),
                    "failure_reason": row.get("failure_reason"),
                    "case_id": row.get("case_id"),
                })
    return result


def build(
    catalogue: dict[str, Any],
    queue: dict[str, Any],
    family_report: dict[str, Any] | None = None,
    roles: dict[str, Any] | None = None,
    prior_campaigns: list[Path] | None = None,
    *,
    max_targets: int | None = None,
) -> dict[str, Any]:
    positions = catalogue.get("positions", [])
    catalogue_stats = _catalog_stats(positions)
    report_rows = {str(row.get("family_id")): row for row in (family_report or {}).get("families", [])}
    queue_rows = _queue_candidates(queue)
    prior = _prior_campaign_status(prior_campaigns or [])
    all_families = set(catalogue_stats) | set(report_rows)
    roles_by_family: dict[str, int] = Counter()
    for record in (roles or {}).get("records", []):
        # Roles may not have a catalogue family; preserve them only as an audit
        # count.  Explicit family-role mappings in the family report remain the
        # authoritative family-level evidence.
        family = record.get("operator_family")
        if family:
            roles_by_family[str(family)] += 1

    rows: list[dict[str, Any]] = []
    new_targets: list[dict[str, Any]] = []
    covered: list[str] = []
    blocked: list[dict[str, Any]] = []
    for family in sorted(all_families):
        stats = catalogue_stats.get(family, {})
        report = report_rows.get(family)
        evidence = _family_evidence(report)
        if roles_by_family.get(family):
            evidence.append({"kind": "EXPLICIT_ROLE_FAMILY", "count": roles_by_family[family]})
        candidate = queue_rows.get(family)
        if evidence:
            status = "ALREADY_COVERED_DO_NOT_REPEAT"
            action = "DO_NOT_REPEAT_FAMILY_ONLY_FOR_MORE_POSITIONS"
            covered.append(family)
        elif candidate:
            attempts = prior.get(family, [])
            valid_attempts = [item for item in attempts if item.get("execution_status") in {"MEASUREMENT_VALID", "VALID"}]
            failed_attempts = [item for item in attempts if item.get("execution_status") == "EXECUTION_FAILED"]
            if valid_attempts:
                evidence.append({
                    "kind": "AUTOMATED_FAMILY_CAMPAIGN_VALID",
                    "count": len(valid_attempts),
                    "summaries": [item["summary"] for item in valid_attempts],
                })
                status = "ALREADY_COVERED_DO_NOT_REPEAT"
                action = "DO_NOT_REPEAT_FAMILY_ONLY_FOR_MORE_POSITIONS"
                covered.append(family)
            elif failed_attempts:
                status = "MEASUREMENT_BLOCKED_AFTER_PRIOR_ATTEMPT"
                action = "FIX_REFERENCE_BINDING_NOT_RERUN"
                blocked.append({
                    "operator_family": family,
                    "reason": status,
                    "prior_attempts": failed_attempts,
                })
            else:
                status = "UNMEASURED_FAMILY_CANDIDATE"
                action = "PREPARE_OR_RUN_ONE_FAMILY_CAMPAIGN"
                new_targets.append({"operator_family": family, **candidate})
        elif stats.get("ready_position_count", 0):
            status = "READY_BUT_NOT_IN_EXECUTION_QUEUE"
            action = "REBUILD_QUEUE_OR_BIND_REFERENCE"
            blocked.append({"operator_family": family, "reason": status})
        elif stats:
            status = "NO_REFERENCE_BOUND_EXECUTABLE_POSITION"
            action = "BUILD_REFERENCE_ADAPTER_BEFORE_MEASUREMENT"
            blocked.append({"operator_family": family, "reason": status})
        else:
            status = "EXTERNAL_EVIDENCE_WITHOUT_CATALOGUE_POSITION"
            action = "KEEP_AS_EXTERNAL_EVIDENCE"
        rows.append({
            "operator_family": family,
            "status": status,
            "next_action": action,
            "catalogue": stats,
            "historical_or_measurement_evidence": evidence,
            "candidate": candidate,
            "prior_campaign_status": prior.get(family, []),
            "selection_uses_numerical_outcomes": False,
        })

    if max_targets is not None:
        if max_targets < 0:
            raise ValueError("max_targets must be non-negative")
        new_targets = new_targets[:max_targets]
    return {
        "schema": "unmeasured-triton-family-frontier-v1",
        "scope": "FAMILY_LEVEL_COVERAGE_AND_NEXT_ACTION_PLANNING",
        "selection_uses_numerical_outcomes": False,
        "position_count_is_not_family_count": True,
        "negative_results_are_not_inferred_from_missing_runs": True,
        "catalogue_family_count": len(catalogue_stats),
        "family_count_in_report_or_catalogue": len(all_families),
        "already_covered_family_count": len(covered),
        "unmeasured_family_count": len(new_targets),
        "blocked_or_unbound_family_count": len(blocked),
        "already_covered_families": covered,
        "new_family_targets": new_targets,
        "blocked_or_unbound_families": blocked,
        "families": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--family-report", type=Path)
    parser.add_argument("--roles", type=Path)
    parser.add_argument(
        "--prior-summary", type=Path, action="append", default=[],
        help="Prior family campaign summary; failed families are blocked, not rerun.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-targets", type=int)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("Choose a new output under /data1/tzh")
    catalogue = _read(args.catalog)
    queue = _read(args.queue)
    family_report = _read(args.family_report) if args.family_report else None
    roles = _read(args.roles) if args.roles else None
    result = build(
        catalogue, queue, family_report, roles,
        [Path(path) for path in args.prior_summary], max_targets=args.max_targets,
    )
    result["input_sha256"] = {
        "catalog": hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
        "queue": hashlib.sha256(args.queue.read_bytes()).hexdigest(),
    }
    if args.family_report:
        result["input_sha256"]["family_report"] = hashlib.sha256(args.family_report.read_bytes()).hexdigest()
    if args.roles:
        result["input_sha256"]["roles"] = hashlib.sha256(args.roles.read_bytes()).hexdigest()
    result["input_sha256"]["prior_summaries"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in args.prior_summary
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in (
        "family_count_in_report_or_catalogue", "already_covered_family_count",
        "unmeasured_family_count", "blocked_or_unbound_family_count",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
