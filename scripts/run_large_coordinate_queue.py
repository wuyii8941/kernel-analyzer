#!/usr/bin/env python3
"""Exhaust every >4096-coordinate candidate in one frozen model/shape cell.

The queue is deterministic, resumable, and never removes a candidate from the
denominator.  It groups adjacent candidates only while the requested complete
coordinate spool stays below the configured byte/coordinate budget.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/data1/tzh/miniconda3/envs/pt_nightly/bin/python")

sys.path.insert(0, str(ROOT / "scripts"))
from run_large_coordinate_case import frozen_large_candidates  # noqa: E402


def completed_task_ids(cell: str) -> set[str]:
    root = ROOT / "results/coverage/cases/full_coordinate/large" / cell
    observed: set[str] = set()
    if not root.exists():
        return observed
    for path in sorted(root.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("status") == "PARTIAL_FAIL_CLOSED":
            continue
        observed.update(str(row["task_id"]) for row in payload.get("rows", []))
    return observed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-prefix", required=True,
                        choices=("qwen", "phi4", "mamba", "deepseek8b"))
    parser.add_argument("--sequence-length", required=True, type=int,
                        choices=(64, 128, 256))
    parser.add_argument("--device-index", required=True, type=int)
    parser.add_argument("--max-batch-coordinates", type=int, default=100_000_000)
    args = parser.parse_args()

    cell = f"{args.model_prefix}_seq{args.sequence_length}_r1"
    release = ROOT / "results/coverage/runtime_releases" / cell
    selected = frozen_large_candidates(release, 4096)
    if not selected:
        print(json.dumps({"event": "QUEUE_COMPLETE", "cell": cell,
                          "large_candidates": 0}), flush=True)
        return

    while True:
        done = completed_task_ids(cell)
        missing = [index for index, row in enumerate(selected) if row[0] not in done]
        if not missing:
            print(json.dumps({"event": "QUEUE_COMPLETE", "cell": cell,
                              "large_candidates": len(selected)}), flush=True)
            return
        start = missing[0]
        count = 0
        coordinates = 0
        for index in range(start, len(selected)):
            task_id, size = selected[index]
            if task_id in done or (count and coordinates + size > args.max_batch_coordinates):
                break
            if size > args.max_batch_coordinates:
                # A single tensor must still be audited; the budget controls
                # batching, not eligibility or the denominator.
                if count:
                    break
            coordinates += size
            count += 1
        command = [
            str(PYTHON), str(ROOT / "scripts/run_large_coordinate_case.py"),
            "--model-prefix", args.model_prefix,
            "--sequence-length", str(args.sequence_length),
            "--candidate-index", str(start), "--candidate-count", str(count),
            "--max-batch-coordinates", str(max(coordinates, args.max_batch_coordinates)),
            "--device-index", str(args.device_index),
        ]
        print(json.dumps({"event": "BATCH_START", "cell": cell, "start": start,
                          "count": count, "coordinates": coordinates,
                          "remaining_before_batch": len(missing)}), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
