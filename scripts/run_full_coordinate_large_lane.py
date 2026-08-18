#!/usr/bin/env python3
"""Run resumable large-candidate queues sequentially on one GPU lane."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/data1/tzh/miniconda3/envs/pt_nightly/bin/python")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument("--cell", action="append", required=True,
                        help="model-prefix:sequence-length, in execution order")
    parser.add_argument("--wait-path", action="append", default=[])
    parser.add_argument("--max-batch-coordinates", type=int, default=100_000_000)
    args = parser.parse_args()

    waits = [Path(value) if Path(value).is_absolute() else ROOT / value
             for value in args.wait_path]
    for path in waits:
        while not path.exists():
            time.sleep(30)
    for value in args.cell:
        model, raw_length = value.split(":", 1)
        length = int(raw_length)
        small = ROOT / "results/coverage/cases/full_coordinate" / f"{model}_seq{length}_small.json.gz"
        while not small.exists():
            time.sleep(30)
        subprocess.run([
            str(PYTHON), str(ROOT / "scripts/run_large_coordinate_queue.py"),
            "--model-prefix", model, "--sequence-length", str(length),
            "--device-index", str(args.device_index),
            "--max-batch-coordinates", str(args.max_batch_coordinates),
        ], cwd=ROOT, check=True)
        subprocess.run([
            str(PYTHON), str(ROOT / "scripts/run_t2_causal_queue.py"),
            "--model-prefix", model, "--sequence-length", str(length),
            "--device-index", str(args.device_index),
        ], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
