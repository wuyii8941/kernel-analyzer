#!/usr/bin/env python3
"""Run every requested paired development job, retaining logs and exit codes."""
import argparse
import os
import subprocess
import sys
from scripts.run_liger_language_followup import OUT
from scripts.run_training_numerical_v2 import ROOT, save_new


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", nargs="+", type=int, choices=range(4), required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    failed = False
    for pair in args.pairs:
        for condition in ("candidate", "reference"):
            command = [sys.executable, "scripts/run_liger_language_followup.py", "train",
                       "--pair", str(pair), "--condition", condition, "--device", args.device]
            record = {"command": command, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
            destination = OUT / f"pair{pair}"
            save_new(destination / f"{condition}_execution_plan.json", record)
            with (destination / f"{condition}.log").open("x") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            save_new(destination / f"{condition}_execution_result.json", {**record, "returncode": result.returncode})
            failed |= bool(result.returncode)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
