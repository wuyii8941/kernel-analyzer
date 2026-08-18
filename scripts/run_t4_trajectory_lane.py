#!/usr/bin/env python3
"""Wait for T3 cells and run their exhaustive paired T4 queues."""

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
    parser.add_argument("--cell", action="append", required=True)
    args = parser.parse_args()
    for cell in args.cell:
        model, raw_length = cell.split(":", 1)
        marker = ROOT / "results/coverage/cases/carrier" / f"{model}_seq{int(raw_length)}_r1" / "queue_complete.json"
        while not marker.exists():
            time.sleep(30)
        time.sleep(5)
        subprocess.run([
            str(PYTHON), str(ROOT / "scripts/run_t4_trajectory_queue.py"),
            "--model-prefix", model, "--sequence-length", str(int(raw_length)),
            "--device-index", str(args.device_index),
        ], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
