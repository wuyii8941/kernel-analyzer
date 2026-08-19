#!/usr/bin/env python3
"""Finalize a completed short screening spool without rerunning F+B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.bias_formation_v21 import (  # noqa: E402
    FormationPolicy,
    summarize_streamed_state_vector_files,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--case-plan", type=Path, required=True)
    parser.add_argument("--spool-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--states", type=int, default=4)
    args = parser.parse_args()
    cases = json.loads(args.case_plan.read_text())["cases"]
    policy = FormationPolicy(min_states=args.states, bootstrap_samples=2000)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial = args.output_dir / ".screening_gram.partial.json"
    output_cases = json.loads(partial.read_text())["cases"] if partial.exists() else []
    completed = {row["case_id"] for row in output_cases}
    for case in cases:
        case_id = str(case["case_id"])
        if case_id in completed:
            continue
        layers = {}
        for layer_name, directory in (
            ("LOCAL_ENDPOINT", "local"),
            ("PARAMETER_GRADIENT", "gradient"),
            ("EFFECTIVE_UPDATE", "update"),
        ):
            files = sorted((args.spool_dir / case_id / directory / "calibration").glob("*.f32"))
            if len(files) != args.states:
                raise RuntimeError(f"{case_id}:{layer_name} has {len(files)} states")
            observations = [{
                "state_id": path.stem,
                "path": str(path), "storage_dtype": "float32",
                "coordinate_count": path.stat().st_size // 4,
            } for path in files]
            layers[layer_name] = summarize_streamed_state_vector_files(
                observations, layer=layer_name, partition="screening", policy=policy
            ).as_dict()
        output_cases.append({
            "case_id": case_id, "task_id": str(case["task_id"]),
            "carrier": str(case["carrier"]), "layers": layers,
        })
        partial.write_text(json.dumps({"cases": output_cases}, sort_keys=True) + "\n")
        # Each cell is complete once its three small Gram matrices exist.
        # Reclaim the large transient vectors immediately rather than keeping
        # the whole model's spool until the final JSON write.
        shutil.rmtree(args.spool_dir / case_id)
    payload = {
        "schema": "kernel-analyzer-backward-equivalence-screening-gram-v1",
        "status": "SCREENING_ONLY_NO_SCIENTIFIC_VERDICT",
        "architecture": args.architecture, "state_count": args.states,
        "selection_rule": (
            "Shortlist by gradient cross-state geometry without requiring local bias; "
            "all promoted cases require independent 16+16 confirmation."
        ),
        "cases": output_cases,
    }
    target = args.output_dir / "screening_gram.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    partial.unlink(missing_ok=True)
    shutil.rmtree(args.spool_dir)
    print(json.dumps({"output": str(target), "cases": len(output_cases)}))


if __name__ == "__main__":
    main()
