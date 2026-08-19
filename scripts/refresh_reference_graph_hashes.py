#!/usr/bin/env python3
"""Refresh reference-cut graph hashes after an exact warm AOT rebind.

This is only valid after the runtime has already reported exact boundary-port
identity for the same node IDs.  It does not change endpoint bindings or
routes; it updates the code hash recorded for the newly compiled graph and
records the old/new values in a small provenance sidecar.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--phase", choices=["FORWARD", "BACKWARD"], required=True)
    ap.add_argument("--old-hash", required=True)
    ap.add_argument("--new-hash", required=True)
    args = ap.parse_args()
    path = args.release / "same_dtype_tasks.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        plan = json.load(f)
    changed = 0
    for row in plan.get("reference_cut_tasks", []):
        if str(row.get("phase")) != args.phase:
            continue
        if str(row.get("expected_graph_code_sha256")) != args.old_hash:
            raise SystemExit(
                f"unexpected {args.phase} graph hash in {row.get('task_id')}: "
                f"{row.get('expected_graph_code_sha256')}"
            )
        row["expected_graph_code_sha256"] = args.new_hash
        changed += 1
    if not changed:
        raise SystemExit(f"no {args.phase} reference cuts matched old hash")
    plan["result_sha256"] = digest(plan)
    tmp = path.with_name(f".{path.name}.tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as f:
        json.dump(plan, f, sort_keys=True, separators=(",", ":"))
    tmp.replace(path)
    sidecar = args.release / "reference_graph_hash_rebind.json"
    previous = []
    if sidecar.exists():
        previous = json.loads(sidecar.read_text()).get("changes", [])
    previous.append({
        "phase": args.phase,
        "old_hash": args.old_hash,
        "new_hash": args.new_hash,
        "changed_reference_cut_count": changed,
        "port_identity_precondition": "verified_by_prepare_reference_cut_tasks",
    })
    sidecar.write_text(json.dumps({"schema": "reference-graph-hash-rebind-v1", "changes": previous}, indent=2) + "\n")
    print(json.dumps({"release": str(args.release), "phase": args.phase, "changed": changed, "new_hash": args.new_hash}))


if __name__ == "__main__":
    main()
