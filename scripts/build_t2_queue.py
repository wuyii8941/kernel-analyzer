#!/usr/bin/env python3
"""Build a deduplicated executable T1-to-T2 region intervention queue.

The queue is region based: one fused candidate region may account for several
closed F+B proof units, and a proof unit may touch several candidate regions.
No invocation is duplicated merely because the mapping is many-to-one.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return json.load(handle)


def build_queue(t1: dict[str, Any], campaign: dict[str, Any], limit: int | None,
                observations: dict[str, Any] | None = None,
                unit_limit: int | None = None,
                proof_units: dict[str, Any] | None = None) -> dict[str, Any]:
    campaign_by_region = {str(row["region_id"]): row for row in campaign["rows"]}
    proof_kind = {
        str(row["unit_id"]): str(row.get("unit_kind"))
        for row in (proof_units or {}).get("proof_units", ())
    }
    metric_by_region = {
        str(row["region_id"]): float(row.get("max_abs_max", 0.0))
        for row in (observations or {}).get("rows", ())
    }
    regions: dict[str, dict[str, Any]] = {}
    passing_units = 0
    for unit_id, tier in t1["rows"].items():
        if tier.get("status") != "PASS":
            continue
        passing_units += 1
        evidence = tier.get("evidence", {})
        for region_id in evidence.get("candidate_region_ids", ()):
            row = regions.setdefault(str(region_id), {
                "region_id": str(region_id),
                "proof_unit_ids": [],
                "t1_max_abs": 0.0,
            })
            row["proof_unit_ids"].append(str(unit_id))
            # Unit-level max_abs can combine several fused regions.  Never
            # attribute that aggregate to an individual region when the
            # region-level observation is available.
            row["t1_max_abs"] = metric_by_region.get(
                str(region_id),
                max(row["t1_max_abs"], float(evidence.get("max_abs", 0.0))),
            )

    for row in regions.values():
        row["proof_unit_ids"] = sorted(set(row["proof_unit_ids"]))
        source = campaign_by_region.get(row["region_id"])
        row["executable"] = bool(source and source.get("reference_entrypoint"))
        row["unresolved_reason"] = None if row["executable"] else (
            "NO_EXACT_EXECUTABLE_REFERENCE_ADAPTER"
            if source else "REGION_ABSENT_FROM_CAMPAIGN"
        )
        if source:
            row.update({
                "symbol": source.get("symbol"),
                "phase": source.get("phase"),
                "reference_symbol": source.get("reference_symbol"),
                "endpoints": list(source.get("output_names", ())),
            })

    ordered = sorted(
        regions.values(),
        key=lambda row: (
            not row["executable"], -row["t1_max_abs"], row["region_id"]
        ),
    )
    executable = [row for row in ordered if row["executable"]]
    selected = executable
    selected_unit_ids: list[str] = []
    if unit_limit is not None:
        unit_regions: dict[str, set[str]] = {}
        for row in executable:
            for unit_id in row["proof_unit_ids"]:
                if proof_kind and proof_kind.get(unit_id) != "FORWARD_ACTUAL_BACKWARD_UNIT":
                    continue
                unit_regions.setdefault(unit_id, set()).add(row["region_id"])
        # Prefer cheap complete closures, then stronger local differences.
        ranked_units = sorted(
            unit_regions,
            key=lambda unit_id: (
                len(unit_regions[unit_id]),
                -max(regions[value]["t1_max_abs"] for value in unit_regions[unit_id]),
                unit_id,
            ),
        )
        selected_unit_ids = ranked_units[:unit_limit]
        selected_region_ids = set().union(*(unit_regions[value] for value in selected_unit_ids))
        selected = [row for row in executable if row["region_id"] in selected_region_ids]
    elif limit is not None:
        # A small causal pilot should exercise distinct generated programs
        # before repeating the same symbol at another layer/invocation.
        selected, repeated = [], []
        seen_symbols: set[str] = set()
        for row in executable:
            symbol = str(row.get("symbol"))
            if symbol in seen_symbols:
                repeated.append(row)
            else:
                selected.append(row)
                seen_symbols.add(symbol)
            if len(selected) == limit:
                break
        if len(selected) < limit:
            selected.extend(repeated[:limit - len(selected)])
    selected_ids = {row["region_id"] for row in selected}
    for row in ordered:
        row["selected_for_batch"] = row["region_id"] in selected_ids

    payload: dict[str, Any] = {
        "schema": "kernel-analyzer-t2-region-queue-v1",
        "selection_role": "CAUSAL_FOLLOWUP_AFTER_T1",
        "candidate_values_used_for_mathematical_derivation": False,
        "candidate_values_used_for_t2_priority": True,
        "t1_passing_proof_units": passing_units,
        "unique_t1_regions": len(ordered),
        "executable_regions": sum(bool(row["executable"]) for row in ordered),
        "selected_regions": len(selected),
        "selected_proof_unit_ids": selected_unit_ids,
        "proof_unit_eligibility": (
            "FORWARD_ACTUAL_BACKWARD_UNIT_ONLY" if proof_kind else "NOT_PROVIDED"
        ),
        "selected_arms": [
            {"region_id": row["region_id"], "endpoints": row.get("endpoints", [])}
            for row in selected
        ],
        "rows": ordered,
        "gates": {
            "proof_units_not_treated_as_independent_regions": True,
            "many_to_one_mapping_retained": True,
            "only_exact_executable_adapters_selected": all(row["executable"] for row in selected),
            "t3_confirmation_independence_not_claimed": True,
        },
    }
    payload["result_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--observations", type=Path, default=None)
    parser.add_argument("--proof-units", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--unit-limit", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.limit is not None and args.unit_limit is not None:
        raise ValueError("choose only one of --limit and --unit-limit")
    if any(value is not None and value < 1 for value in (args.limit, args.unit_limit)):
        raise ValueError("limits must be positive")
    payload = build_queue(
        _read(args.t1), _read(args.campaign), args.limit,
        _read(args.observations) if args.observations is not None else None,
        args.unit_limit,
        _read(args.proof_units) if args.proof_units is not None else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "passing_units": payload["t1_passing_proof_units"],
        "unique_regions": payload["unique_t1_regions"],
        "executable_regions": payload["executable_regions"],
        "selected_regions": payload["selected_regions"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
