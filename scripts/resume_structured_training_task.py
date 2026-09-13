#!/usr/bin/env python3
"""Execute one missing frozen structured-training task and retain failures."""
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
    parser.add_argument("--stream", type=int, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("all outputs must stay inside kernel-analyzer")
    protocol = frozen.checked(output)
    if not 0 <= args.stream < int(protocol["stream_count"]):
        parser.error("stream is outside the frozen task set")
    if args.condition not in protocol["new_conditions"]:
        parser.error("condition is outside the frozen task set")
    run_path = output / "runs" / f"stream_{args.stream:02d}_{args.condition}.json"
    failure_path = output / "failures" / (
        f"stream_{args.stream:02d}_{args.condition}_captured.json"
    )
    if run_path.exists() or failure_path.exists():
        parser.error("this frozen task already has a terminal captured record")

    metadata = output / (
        f"task_recovery_stream_{args.stream:02d}_{args.condition}.json"
    )
    save_new(metadata, {
        "schema": "structured-residual-training-task-recovery-v1",
        "purpose": "Capture a terminal record for one interrupted frozen task",
        "scientific_condition_changed": False,
        "stream": args.stream,
        "condition": args.condition,
        "frozen_protocol_sha256": sha(output / "protocol.json"),
        "frozen_runner_sha256": sha(Path(frozen.__file__).resolve()),
        "recovery_runner_sha256": sha(Path(__file__).resolve()),
        "interrupted_attempt_increases_sample_size": False,
    })
    print(json.dumps({"event": "FROZEN_TASK_RETRY", "stream": args.stream,
                      "condition": args.condition}), flush=True)
    try:
        save_new(run_path, frozen.run_condition(
            protocol, args.stream, args.condition, args.device,
        ))
    except Exception as error:
        match = re.search(r"step (\d+)", str(error))
        status = (
            "NONFINITE_TRAINING_FAILURE" if "nonfinite loss" in str(error)
            else "EXECUTION_FAILURE"
        )
        save_new(failure_path, {
            "schema": "structured-residual-training-failure-v1",
            "status": status,
            "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "stream": args.stream,
            "condition": args.condition,
            "failure_step": int(match.group(1)) if match else None,
            "error_type": type(error).__name__,
            "error": str(error),
            "protocol_sha256": sha(output / "protocol.json"),
            "frozen_runner_sha256": sha(Path(frozen.__file__).resolve()),
            "recovery_runner_sha256": sha(Path(__file__).resolve()),
            "interpretation_boundary": [
                "The scientific condition and frozen training task are unchanged.",
                "The interrupted attempt and this retry are one statistical task.",
                "A numerical failure has no finite endpoint and is not imputed.",
            ],
        })
        print(json.dumps({"event": status, "stream": args.stream,
                          "condition": args.condition, "error": str(error)}), flush=True)


if __name__ == "__main__":
    main()
