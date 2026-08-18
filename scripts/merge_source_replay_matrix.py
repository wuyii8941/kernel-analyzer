#!/usr/bin/env python3
"""Validate and merge the six-cell source-replay cross product.

The GPU worker writes one JSON file per natural checkpoint.  This wrapper
keeps the cross-product boundary explicit: missing files remain pending, and
only a complete cell is passed to ``merge_dtype_semantic.py``.  It never
assigns a correctness or bias verdict by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="directory containing per-checkpoint worker JSON files")
    parser.add_argument("--schedule", type=Path,
                        default=FINAL / "source_replay_schedule.json")
    parser.add_argument("--final-dir", type=Path, default=FINAL)
    parser.add_argument("--run-mergers", action="store_true",
                        help="merge complete cells with merge_dtype_semantic.py")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schedule = json.loads(args.schedule.read_text())
    rows = schedule["rows"]
    cells: dict[tuple[str, bool, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["dtype"]), bool(row["tf32"]), int(row["seq_len"]))
        cells.setdefault(key, []).append(row)

    output_cells = []
    for (dtype, tf32, seq_len), cell_rows in sorted(cells.items(), key=lambda item: item[0]):
        cell_rows.sort(key=lambda row: int(row["step"]))
        mapping = args.final_dir / str(cell_rows[0]["mapping_file"])
        files = [args.output_dir / str(row["output_file"]) for row in cell_rows]
        missing = [str(path) for path in files if not path.exists()]
        merged_name = ("tf32" if tf32 else "fp32") + f"_seq{seq_len}_merged.json"
        merged = args.final_dir / merged_name
        status = "PENDING_GPU_REMEASUREMENT" if missing else "READY_TO_MERGE"
        if not missing and args.run_mergers:
            command = [
                sys.executable,
                str(ROOT / "scripts" / "merge_dtype_semantic.py"),
                "--seq-len", str(seq_len),
                "--dtype-mapping", str(mapping),
                "--inputs", *[str(path) for path in files],
                "--output", str(merged),
            ]
            completed = subprocess.run(command, cwd=ROOT, check=False,
                                       capture_output=True, text=True)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"merge failed for {dtype}/tf32={tf32}/seq{seq_len}: "
                    f"{completed.stderr.strip()}"
                )
            status = "COMPLETE"
        output_cells.append({
            "dtype": dtype,
            "tf32": tf32,
            "seq_len": seq_len,
            "checkpoint_steps": [int(row["step"]) for row in cell_rows],
            "mapping_file": str(mapping),
            "mapping_sha256": digest(mapping),
            "expected_invocations": int(cell_rows[0]["expected_invocations"]),
            "repeat_count": int(cell_rows[0]["repeat_count"]),
            "worker_files": [str(path) for path in files],
            "missing_worker_files": missing,
            "merged_file": str(merged),
            "status": status,
            "numeric_verdicts": "NOT_ASSIGNED",
            "natural_bias_case_added": False,
        })

    complete = all(cell["status"] == "COMPLETE" for cell in output_cells)
    output = {
        "schema": "kernel-analyzer-source-replay-matrix-v1",
        "subject": "Qwen3-1.7B candidate-blind FP32/TF32 source-mapped replay",
        "schedule": str(args.schedule),
        "schedule_sha256": digest(args.schedule),
        "output_dir": str(args.output_dir),
        "candidate_values_used_to_select_or_classify": False,
        "cell_count": len(output_cells),
        "planned_runs": len(rows),
        "complete_cells": sum(cell["status"] == "COMPLETE" for cell in output_cells),
        "pending_cells": sum(cell["status"] != "COMPLETE" for cell in output_cells),
        "numeric_replay": "COMPLETE" if complete else "PENDING_GPU_REMEASUREMENT",
        "natural_bias_case_added": False,
        "property_claim": False,
        "cells": output_cells,
        "boundary": "A complete merge only certifies worker coverage and preserves endpoint metrics; a natural bias case requires the separate cross-checkpoint carrier and live-weight tests.",
    }
    output["result_sha256"] = hashlib.sha256(
        json.dumps(output, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    destination = args.final_dir / "source_replay_matrix.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(destination),
        "complete_cells": output["complete_cells"],
        "pending_cells": output["pending_cells"],
        "numeric_replay": output["numeric_replay"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
