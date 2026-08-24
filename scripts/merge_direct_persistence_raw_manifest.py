#!/usr/bin/env python3
"""Merge an exact formation manifest with a later update-only replay.

The large tensors stay in the external cache.  Only the two compact manifests
are combined, and the script refuses to merge different target or carrier
identities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formation", type=Path, required=True)
    parser.add_argument("--updates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    formation = json.loads(args.formation.read_text())
    updates = json.loads(args.updates.read_text())
    for key in ("target_region", "target_endpoint", "carrier"):
        if formation.get(key) != updates.get(key):
            raise SystemExit(f"identity mismatch for {key}: {formation.get(key)!r} != {updates.get(key)!r}")
    if not updates.get("trajectory_update_records"):
        raise SystemExit("update manifest has no trajectory records")
    merged = dict(formation)
    merged["schema"] = "kernel-analyzer-gemma4-v4-merged-raw-capture-v1"
    merged["status"] = "COMPLETE_FORMATION_AND_UPDATE_PAIRS"
    merged["case_id"] = formation.get("case_id", updates.get("case_id"))
    merged["trajectory_update_records"] = updates["trajectory_update_records"]
    merged["update_only_source_manifest"] = str(args.updates)
    merged["claim_boundary"] = (
        "Formation endpoint/gradient pairs come from the exact frozen formation replay; "
        "same-state effective-update pairs come from a separate exact update-only replay "
        "with the same model, input bank, target, carrier, optimizer and runtime seed."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": merged["status"],
        "formation_states": len(merged.get("formation_gradient_records", [])),
        "update_steps": len(merged["trajectory_update_records"]),
        "output": str(args.output),
    }))


if __name__ == "__main__":
    main()
