#!/usr/bin/env python3
"""Manifest and remove one explicitly abandoned RUNNING live campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_TEMP = Path("/data1/tzh/cache/kernel_analyzer_contrasts")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.output.read_text())
    if payload.get("status") != "RUNNING":
        raise RuntimeError("only an incomplete RUNNING artifact may be abandoned")
    paths = {Path(row["path"]).resolve() for state in payload.get("states", {}).values()
             for row in state.get("vectors", {}).values() if row.get("path")}
    if not paths or any(ALLOWED_TEMP.resolve() not in path.parents for path in paths):
        raise RuntimeError("partial vectors are outside the guarded temporary root")
    campaign_dirs = {next(parent for parent in path.parents if parent.parent == ALLOWED_TEMP.resolve())
                     for path in paths}
    manifest = {
        "schema": "kernel-analyzer-abandoned-live-partial-v1",
        "status": "COMPLETE_REGENERABLE_PARTIAL_REMOVED",
        "reason": "float64 per-state spool violated bounded-storage policy",
        "output": str(args.output), "campaign_id": payload.get("campaign_id"),
        "completed_state_ids": list(payload.get("states", {})),
        "vector_count": len(paths),
        "bytes_removed": sum(path.stat().st_size for path in paths if path.exists()),
        "vector_sha256": sorted(row["sha256"] for state in payload.get("states", {}).values()
                                for row in state.get("vectors", {}).values()),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    for directory in campaign_dirs:
        shutil.rmtree(directory)
    args.output.unlink()
    print(json.dumps({"bytes_removed": manifest["bytes_removed"],
                      "completed_states_removed": len(manifest["completed_state_ids"])}))


if __name__ == "__main__":
    main()
