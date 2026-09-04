#!/usr/bin/env python3
"""Freeze a metadata-only cross-model Training Bias Profile benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "results/property/tcmp_allop_v1/semantic_family_heldout_pool_v1.json"
PRIOR_PROTOCOLS = (
    ROOT / "results/property/training_bias_profile_v2/prospective_batch_1_protocol.json",
    ROOT / "results/property/training_bias_profile_v2/prospective_batch_2_protocol.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=16)
    args = parser.parse_args()
    pool = json.loads(POOL.read_text())
    prior = set()
    for path in PRIOR_PROTOCOLS:
        protocol = json.loads(path.read_text())
        for row in protocol["cases"]:
            prior.add((row["model"], int(row["sequence_length"]), row["task_id"]))

    eligible = [
        row for row in pool["rows"]
        if row.get("pool_status") == "PRE_MEASUREMENT_CANDIDATE"
        and row.get("exact_endpoint_executable")
        and row.get("has_exact_downstream_closure")
        and (row["model"], int(row["sequence_length"]), row["task_id"]) not in prior
    ]
    # First pass maximizes model/family breadth.  The second pass fills any
    # remaining slots by the already frozen pool rank, without reading values.
    chosen = []
    used_model_family = set()
    for row in sorted(eligible, key=lambda item: int(item["frozen_pool_rank"])):
        key = (row["model"], row["family"])
        if key in used_model_family:
            continue
        chosen.append(row)
        used_model_family.add(key)
        if len(chosen) == args.cases:
            break
    chosen_keys = {
        (row["model"], int(row["sequence_length"]), row["task_id"])
        for row in chosen
    }
    if len(chosen) < args.cases:
        for row in sorted(eligible, key=lambda item: int(item["frozen_pool_rank"])):
            key = (row["model"], int(row["sequence_length"]), row["task_id"])
            if key in chosen_keys:
                continue
            chosen.append(row)
            chosen_keys.add(key)
            if len(chosen) >= args.cases:
                break
    if len(chosen) != args.cases:
        raise RuntimeError(f"requested {args.cases} cases but only selected {len(chosen)}")

    cases = [{
        "benchmark_index": index,
        "model": row["model"],
        "sequence_length": int(row["sequence_length"]),
        "task_id": row["task_id"],
        "family": row["family"],
        "implementation_kind": row["implementation_kind"],
        "region_symbol": row["region_symbol"],
        "frozen_pool_rank": int(row["frozen_pool_rank"]),
        "cell_id": row["cell_id"],
    } for index, row in enumerate(chosen)]
    payload = {
        "schema": "kernel-analyzer-generalization-benchmark-v1",
        "status": "FROZEN_BEFORE_ANY_BENCHMARK_MEASUREMENT",
        "selection_source": str(POOL.relative_to(ROOT)),
        "selection_source_sha256": sha256(POOL),
        "excluded_prior_protocols": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in PRIOR_PROTOCOLS
        ],
        "selection_rule": (
            "Exclude prior development and prospective cases; then scan the existing frozen "
            "pool rank, first taking one case per model and training family, then filling by "
            "pool rank. No numerical result or historical label is read."
        ),
        "primary_stage": "ADAMW_UPDATE",
        "primary_effects": ["additive", "repair_aligned", "residual_direction"],
        "primary_comparison_count": len(cases) * 3,
        "multiplicity": "one Holm family across every case and all three update effects",
        "unresolved_rule": "retain in the denominator with p=1; do not replace after results are known",
        "cases": cases,
        "claim_boundary": (
            "This benchmark tests new frozen training positions across the four existing core "
            "models. It does not by itself establish a new-model result."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "cases": len(cases), "output": str(args.output)}))


if __name__ == "__main__":
    main()
