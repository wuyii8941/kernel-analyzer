#!/usr/bin/env python3
"""Render per-model/shape dynamic-reach plans for all bound semantic cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/property/bias_formation/hotspot_search/multishape_backward_carriers.json"
OUTPUT = ROOT / "results/property/bias_formation/hotspot_search/multishape_reach_plans"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    source = json.loads(args.source.read_text()); args.output_dir.mkdir(parents=True, exist_ok=True)
    for model in ("qwen", "phi4", "deepseek8b", "mamba"):
        for sequence_length in (64, 128, 256):
            cells = [row for row in source["cells"] if row["model"] == model
                     and row["sequence_length"] == sequence_length
                     and row["nearest_carrier"] is not None
                     and row["capture_boundary"] == "EXACT_AOT_ENDPOINT"]
            payload = {
                "schema": "kernel-analyzer-multishape-backward-capture-plan-v1",
                "model": model, "sequence_length": sequence_length,
                "selection_uses_candidate_values_or_historical_verdict": False,
                "cases": [{
                    "case_id": row["cell_id"],
                    "task_id": row["representative"]["task_id"],
                    "carrier": row["nearest_carrier"]["name"],
                    "family": row["family"], "depth_stratum": row["depth_stratum"],
                    "member_count": row["member_count"],
                } for row in cells],
            }
            target = args.output_dir / f"{model}_seq{sequence_length}.json"
            target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            print(json.dumps({"output": str(target), "cases": len(cells)}))


if __name__ == "__main__":
    main()
