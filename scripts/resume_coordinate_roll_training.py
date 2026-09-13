#!/usr/bin/env python3
"""Resume frozen coordinate-roll training while retaining per-stream failures."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

from scripts import run_structured_residual_training as frozen


ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("all outputs must stay inside kernel-analyzer")

    protocol = frozen.checked(output)
    recovery = output / "coordinate_roll_recovery.json"
    if not recovery.exists():
        save_new(recovery, {
            "schema": "structured-residual-training-operational-recovery-v1",
            "purpose": "Continue later streams after retaining a nonfinite earlier stream",
            "scientific_condition_changed": False,
            "condition": "COORDINATE_ROLL",
            "frozen_protocol_sha256": sha(output / "protocol.json"),
            "frozen_runner_sha256": sha(Path(frozen.__file__).resolve()),
            "recovery_runner_sha256": sha(Path(__file__).resolve()),
        })

    for stream in range(protocol["stream_count"]):
        run_path = output / "runs" / f"stream_{stream:02d}_COORDINATE_ROLL.json"
        failure_path = output / "failures" / f"stream_{stream:02d}_COORDINATE_ROLL.json"
        if run_path.exists() or failure_path.exists():
            continue
        print(json.dumps({"event": "START", "stream": stream,
                          "condition": "COORDINATE_ROLL"}), flush=True)
        try:
            result = frozen.run_condition(protocol, stream, "COORDINATE_ROLL", args.device)
            save_new(run_path, result)
        except Exception as error:  # retain the endpoint, then continue the frozen condition
            match = re.search(r"step (\d+)", str(error))
            save_new(failure_path, {
                "schema": "structured-residual-training-failure-v1",
                "status": "NONFINITE_TRAINING_FAILURE" if "nonfinite loss" in str(error)
                else "EXECUTION_FAILURE",
                "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "stream": stream,
                "condition": "COORDINATE_ROLL",
                "failure_step": int(match.group(1)) if match else None,
                "error_type": type(error).__name__,
                "error": str(error),
                "protocol_sha256": sha(output / "protocol.json"),
                "frozen_runner_sha256": sha(Path(frozen.__file__).resolve()),
                "recovery_runner_sha256": sha(Path(__file__).resolve()),
                "interpretation_boundary": [
                    "The frozen model, data, optimizer, and coordinate-roll condition are unchanged.",
                    "This runner only retains a failure and proceeds to the next frozen stream.",
                    "A failed trajectory is never replaced or counted as a completed trajectory.",
                ],
            })
            print(json.dumps({"event": "FAILURE_RETAINED", "stream": stream,
                              "error": str(error)}), flush=True)


if __name__ == "__main__":
    main()
