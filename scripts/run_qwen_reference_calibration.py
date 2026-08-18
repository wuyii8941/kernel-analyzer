#!/usr/bin/env python3
"""Run one shard of the frozen 48-state Qwen reference calibration.

Each task is a single-process full loss forward/backward observation.  Raw JSON
is gzip-compressed immediately after validation so interrupted campaigns remain
restartable without retaining bulky transient files.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/state_design.json"
PROTOCOL = ROOT / "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/full_step_inventory/all_op_reference_protocol_v2.json"
WORKER = ROOT / "archive/round1_code/scripts/capture_qwen3_full_step_numerical_sketch.py"
FORKCERT = ROOT / "archive/round1_code/src"
MODEL = Path("/data1/tzh/models/Qwen/Qwen3-1.7B")
OUTPUT = ROOT / "results/calibration"


def artifact_name(state_id: str, dtype: str, repeat: int) -> str:
    short = "bf16" if dtype == "bfloat16" else "fp32"
    return f"{state_id}.{short}.r{repeat}.json"


def valid_gzip(path: Path, state_id: str, dtype: str, repeat: int) -> bool:
    if not path.exists():
        return False
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        return (
            payload["status"].startswith("VALID_COMPLETE_STEP_")
            and payload["state"]["sequence_id"] == state_id
            and payload["state"]["repeat_id"] == str(repeat)
            and payload["environment"]["dtype"] == dtype
            and payload["coverage_certificate"]["denominator"]["dispatch_invocations"]
            == (9269 if dtype == "bfloat16" else 8701)
        )
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index")

    design = json.loads(DESIGN.read_text())
    states = [row for row in design["records"] if row["split"] == "calibration"]
    tasks = [
        (state["sequence_id"], dtype, repeat)
        for state in states
        for dtype, repeat in (("bfloat16", 0), ("bfloat16", 1), ("float32", 0))
    ]
    tasks = [
        task for index, task in enumerate(tasks)
        if index % args.shard_count == args.shard_index
    ]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(FORKCERT)
    completed = 0
    skipped = 0
    for task_index, (state_id, dtype, repeat) in enumerate(tasks, 1):
        raw = OUTPUT / artifact_name(state_id, dtype, repeat)
        compressed = raw.with_suffix(raw.suffix + ".gz")
        if valid_gzip(compressed, state_id, dtype, repeat):
            skipped += 1
            print(json.dumps({
                "event": "SKIP_VALID",
                "shard": args.shard_index,
                "task": task_index,
                "tasks": len(tasks),
                "artifact": compressed.name,
            }), flush=True)
            continue
        if compressed.exists():
            raise RuntimeError(f"invalid compressed artifact requires audit: {compressed}")
        command = [
            "/data1/tzh/miniconda3/envs/pt_nightly/bin/python",
            str(WORKER),
            "--model", str(MODEL),
            "--state-design", str(DESIGN),
            "--protocol", str(PROTOCOL),
            "--sequence-id", state_id,
            "--repeat-id", str(repeat),
            "--device", args.device,
            "--dtype", dtype,
            "--output", str(raw),
        ]
        print(json.dumps({
            "event": "START",
            "shard": args.shard_index,
            "task": task_index,
            "tasks": len(tasks),
            "state": state_id,
            "dtype": dtype,
            "repeat": repeat,
        }), flush=True)
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
        payload = json.loads(raw.read_text())
        if not payload["status"].startswith("VALID_COMPLETE_STEP_"):
            raise RuntimeError(f"invalid worker artifact: {raw}")
        with raw.open("rb") as source, gzip.open(compressed, "wb", compresslevel=9) as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        raw.unlink()
        if not valid_gzip(compressed, state_id, dtype, repeat):
            raise RuntimeError(f"compressed artifact validation failed: {compressed}")
        completed += 1
        print(json.dumps({
            "event": "COMPLETE",
            "shard": args.shard_index,
            "task": task_index,
            "tasks": len(tasks),
            "artifact": compressed.name,
            "bytes": compressed.stat().st_size,
        }), flush=True)
    print(json.dumps({
        "event": "SHARD_COMPLETE",
        "shard": args.shard_index,
        "completed": completed,
        "skipped": skipped,
        "tasks": len(tasks),
    }), flush=True)


if __name__ == "__main__":
    main()
