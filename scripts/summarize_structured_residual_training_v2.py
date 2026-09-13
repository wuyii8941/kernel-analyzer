#!/usr/bin/env python3
"""Create a failure-aware summary without changing the frozen training runner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kernel_analyzer.training_outcome_summary import summarize_failure_aware_training
from scripts import run_structured_residual_training as frozen


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); root = args.root.resolve(); output = args.output.resolve()
    if not root.is_relative_to(ROOT) or not output.is_relative_to(ROOT):
        parser.error("inputs and outputs must stay inside kernel-analyzer")
    if output.exists():
        parser.error("choose a new output; prior summaries are immutable")
    protocol = frozen.checked(root)
    new_runs = [load(path) for path in sorted((root / "runs").glob("*.json"))]
    failures = [load(path) for path in sorted((root / "failures").glob("*.json"))]
    historical = {
        (stream, condition): load(
            frozen.HISTORICAL / "runs" / f"stream_{stream:02d}_{condition}.json"
        )
        for stream in range(int(protocol["stream_count"]))
        for condition in protocol["historical_conditions"]
    }
    result = summarize_failure_aware_training(protocol, new_runs, failures, historical)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "status": result["status"],
        "condition_outcomes": result["condition_outcomes"],
        "decisions": {name: row["decision"] for name, row in result["contrasts"].items()},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
