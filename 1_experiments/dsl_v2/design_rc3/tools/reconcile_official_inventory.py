#!/usr/bin/env python3
"""Read-only reconciliation of normalized OFFICIAL inventories.

This does not enumerate MLIR itself, validate proofs, run kernels, or certify
that the input inventories are complete. Producers must retain source/command
receipts. Nonzero exit means incomplete evidence or an unexplained discrepancy.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any

KINDS = ("source", "registered", "observed")


def _read(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected object")
    return raw


def reconcile(inventories: dict[str, dict[str, Any]], rules: list[dict[str, Any]],
              explanations: list[dict[str, str]] | None = None) -> dict[str, Any]:
    errors: list[str] = []
    missing = sorted(set(KINDS) - inventories.keys())
    if missing:
        return {"status": "INCOMPLETE", "errors": ["missing inventories: " + ",".join(missing)],
                "is_semantic_certificate": False}
    sets: dict[str, dict[str, dict[str, Any]]] = {}
    profile_ids = set()
    for kind in KINDS:
        inv = inventories[kind]
        if inv.get("kind") != kind:
            errors.append(f"{kind}: wrong kind field")
        for field in ("profile_id", "source_commit", "producer", "receipt"):
            if not inv.get(field):
                errors.append(f"{kind}: missing {field}")
        profile_ids.add((inv.get("profile_id"), inv.get("source_commit")))
        if kind != "observed" and inv.get("complete_for_profile") is not True:
            errors.append(f"{kind}: producer has not attested profile enumeration")
        records = inv.get("records")
        if not isinstance(records, list) or not records:
            errors.append(f"{kind}: no records")
            sets[kind] = {}
            continue
        seen: dict[str, dict[str, Any]] = {}
        for record in records:
            if not isinstance(record, dict) or not record.get("id") or not record.get("contract_hash"):
                errors.append(f"{kind}: malformed record")
                continue
            if record["id"] in seen:
                errors.append(f"{kind}: duplicate {record['id']}")
            seen[record["id"]] = record
        sets[kind] = seen
    if len(profile_ids) != 1:
        errors.append("inventory profile/commit mismatch")
    declared = {}
    for e in explanations or []:
        if not all(e.get(k) for k in ("difference", "id", "reason", "evidence")):
            errors.append("difference explanation lacks reason/evidence")
        else:
            declared[(e["difference"], e["id"])] = e
    diffs = {
        "source_only": sorted(sets["source"].keys() - sets["registered"].keys()),
        "registered_not_source": sorted(sets["registered"].keys() - sets["source"].keys()),
        "observed_not_registered": sorted(sets["observed"].keys() - sets["registered"].keys()),
    }
    for dtype, names in diffs.items():
        for name in names:
            if (dtype, name) not in declared:
                errors.append(f"unexplained {dtype}: {name}")
    for a, b in (("source", "registered"), ("registered", "observed")):
        for name in sets[a].keys() & sets[b].keys():
            if sets[a][name]["contract_hash"] != sets[b][name]["contract_hash"]:
                errors.append(f"contract mismatch {a}/{b}: {name}")
    routes = {}
    for rule in rules:
        if not isinstance(rule, dict) or not rule.get("id") or not rule.get("route"):
            errors.append("malformed route record")
            continue
        if rule["id"] in routes:
            errors.append(f"duplicate route: {rule['id']}")
        routes[rule["id"]] = rule
    all_known = set().union(*(set(v) for v in sets.values()))
    for name in sorted(all_known):
        if name not in routes:
            errors.append(f"missing target route: {name}")
    # Planned routes are allowed for target bookkeeping, but never count as implemented.
    implementations = sum(routes.get(n, {}).get("implementation_status") == "validated-scope"
                          for n in all_known)
    return {"status": "READY_FOR_MANUAL_REVIEW" if not errors else "INCOMPLETE",
            "errors": errors, "differences": diffs,
            "counts": {k: len(v) for k, v in sets.items()},
            "target_records": len(all_known), "validated_route_claims": implementations,
            "is_semantic_certificate": False,
            "warning": "Receipts, extractor completeness, rule correctness and device validity require independent review."}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for k in KINDS:
        p.add_argument(f"--{k}", type=Path, required=True)
    p.add_argument("--routes", type=Path, required=True)
    p.add_argument("--explanations", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    try:
        invs = {k: _read(getattr(a, k)) for k in KINDS}
        routes = _read(a.routes)["records"]
        exps = _read(a.explanations)["records"] if a.explanations else []
        result = reconcile(invs, routes, exps)
    except (OSError, ValueError, TypeError, KeyError) as e:
        result = {"status": "INCOMPLETE", "errors": [str(e)], "is_semantic_certificate": False}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if a.output:
        # Explicit new output only: never overwrite evidence by accident.
        with a.output.open("x", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)
    return 0 if result["status"] == "READY_FOR_MANUAL_REVIEW" else 2


if __name__ == "__main__":
    raise SystemExit(main())
