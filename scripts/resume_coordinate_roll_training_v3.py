#!/usr/bin/env python3
"""Run missing frozen coordinate-roll streams after environment-only failures."""
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


def load(path: Path):
    return json.loads(path.read_text())


def save_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(); output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("all outputs must stay inside kernel-analyzer")
    protocol = frozen.checked(output)
    metadata = output / "coordinate_roll_recovery_v3.json"
    if not metadata.exists():
        save_new(metadata, {
            "schema": "structured-residual-training-operational-recovery-v3",
            "purpose": "Run missing streams with the repository src path available",
            "scientific_condition_changed": False,
            "condition": "COORDINATE_ROLL",
            "frozen_protocol_sha256": sha(output / "protocol.json"),
            "frozen_runner_sha256": sha(Path(frozen.__file__).resolve()),
            "recovery_runner_sha256": sha(Path(__file__).resolve()),
            "prior_environment_failures_retained": True,
        })
    for stream in range(protocol["stream_count"]):
        run_path = output / "runs" / f"stream_{stream:02d}_COORDINATE_ROLL.json"
        original_failure = output / "failures" / f"stream_{stream:02d}_COORDINATE_ROLL.json"
        if run_path.exists():
            continue
        if original_failure.exists() and load(original_failure)["status"] == "NONFINITE_TRAINING_FAILURE":
            continue
        print(json.dumps({"event": "SCIENTIFIC_RETRY", "stream": stream,
                          "condition": "COORDINATE_ROLL"}), flush=True)
        try:
            save_new(run_path, frozen.run_condition(
                protocol, stream, "COORDINATE_ROLL", args.device,
            ))
        except Exception as error:
            match = re.search(r"step (\d+)", str(error))
            outcome = output / "failures" / f"stream_{stream:02d}_COORDINATE_ROLL_scientific.json"
            save_new(outcome, {
                "schema": "structured-residual-training-failure-v1",
                "status": "NONFINITE_TRAINING_FAILURE" if "nonfinite loss" in str(error)
                else "EXECUTION_FAILURE",
                "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "stream": stream, "condition": "COORDINATE_ROLL",
                "failure_step": int(match.group(1)) if match else None,
                "error_type": type(error).__name__, "error": str(error),
                "protocol_sha256": sha(output / "protocol.json"),
                "frozen_runner_sha256": sha(Path(frozen.__file__).resolve()),
                "recovery_runner_sha256": sha(Path(__file__).resolve()),
                "interpretation_boundary": [
                    "Earlier environment-only failure records remain preserved.",
                    "This record comes from executing the unchanged frozen condition with CUDA available.",
                    "A failed trajectory is not counted as a completed trajectory.",
                ],
            })
            print(json.dumps({"event": "SCIENTIFIC_FAILURE_RETAINED", "stream": stream,
                              "error": str(error)}), flush=True)


if __name__ == "__main__":
    main()
