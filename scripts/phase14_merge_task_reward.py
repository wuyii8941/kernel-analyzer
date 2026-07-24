#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(fraction * (len(ordered) - 1))))
    return ordered[index]


def paired_bootstrap_ci(differences: list[float], repeats: int = 5000, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    estimates = []
    for _ in range(repeats):
        sample = [differences[rng.randrange(len(differences))] for _ in differences]
        estimates.append(sum(sample) / len(sample))
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def merge_replicates(paths: list[str]) -> dict[str, Any]:
    runs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    if len({run["arm"] for run in runs}) != 1:
        raise ValueError("replicate arm names differ")
    hashes = {run["evaluation"]["completion_ids_sha256"] for run in runs}
    if len(hashes) != 1:
        raise ValueError(f"independent task-evaluation replicates differ for {runs[0]['arm']}")
    return {
        "arm": runs[0]["arm"],
        "independent_runs": len(runs),
        "independent_outputs_exact": True,
        "evaluation": runs[0]["evaluation"],
        "rows": runs[0]["rows"],
        "model": runs[0]["model"],
        "environment": runs[0]["environment"],
    }


def paired_comparison(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_rows = {int(row["dataset_index"]): row for row in left["rows"]}
    right_rows = {int(row["dataset_index"]): row for row in right["rows"]}
    if left_rows.keys() != right_rows.keys():
        raise ValueError("held-out task prompt sets differ")
    differences = [right_rows[key]["reward"] - left_rows[key]["reward"] for key in sorted(left_rows)]
    return {
        "left": left["arm"],
        "right": right["arm"],
        "prompts": len(differences),
        "completion_token_forks": sum(
            left_rows[key]["completion_token_ids"] != right_rows[key]["completion_token_ids"]
            for key in left_rows
        ),
        "reward_differences": sum(value != 0.0 for value in differences),
        "exact_outcome_forks": sum(
            bool(left_rows[key]["exact"]) != bool(right_rows[key]["exact"]) for key in left_rows
        ),
        "mean_reward_difference_right_minus_left": sum(differences) / len(differences),
        "paired_bootstrap_95ci": paired_bootstrap_ci(differences),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge independent held-out task-reward evaluation arms.")
    parser.add_argument("--initial", action="append", required=True)
    parser.add_argument("--clean", action="append", required=True)
    parser.add_argument("--mutation", action="append", required=True)
    parser.add_argument("--out", default="results/phase14_task_reward.json")
    parser.add_argument("--report", default="reports/phase14_task_reward.md")
    args = parser.parse_args()
    initial = merge_replicates(args.initial)
    clean = merge_replicates(args.clean)
    mutation = merge_replicates(args.mutation)
    comparisons = [
        paired_comparison(initial, clean),
        paired_comparison(initial, mutation),
        paired_comparison(clean, mutation),
    ]
    payload = {
        "schema_version": "forkcert.task_reward.v1",
        "arms": [
            {key: value for key, value in arm.items() if key != "rows"}
            for arm in [initial, clean, mutation]
        ],
        "comparisons": comparisons,
        "native_bf16_hardware": False,
        "bf16_status": "not_run_t4_sm75_has_no_native_bf16",
        "claim_scope": (
            "Held-out greedy generation on the synthetic arithmetic reward used in Phase 0. Reward divergence is "
            "task-level evidence for this finite task; reward equality does not prove trajectory equivalence or harmlessness."
        ),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Phase 14 Held-out Task Reward",
        "",
        "## Objective",
        "",
        "Determine whether the clean/mutation parameter divergence observed after matched updates changes held-out generated answers or the actual Phase-0 numeric reward.",
        "",
        "## Controls",
        "",
        "- Arithmetic indices 64-127 are disjoint from the Phase-0 fallback training prompts 0-63.",
        "- Greedy generation, fixed tokenizer, eager SDPA-MATH FP16 evaluation and two independent processes per arm.",
        "- Mutation is absent during evaluation; only its saved training-state consequence remains.",
        "",
        "## Arm Summary",
        "",
        "| Arm | Prompts | Exact answers | Mean reward | Independent outputs exact |",
        "|---|---:|---:|---:|---|",
    ]
    for arm in payload["arms"]:
        evaluation = arm["evaluation"]
        lines.append(
            f"| {arm['arm']} | {evaluation['count']} | {evaluation['exact_count']} | "
            f"{evaluation['mean_numeric_reward']:.6g} | {arm['independent_outputs_exact']} |"
        )
    lines.extend(
        [
            "",
            "## Paired Comparisons",
            "",
            "| Left -> right | Completion forks | Reward differences | Exact-outcome forks | Mean reward difference | 95% paired bootstrap CI |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| {row['left']} -> {row['right']} | {row['completion_token_forks']} | "
            f"{row['reward_differences']} | {row['exact_outcome_forks']} | "
            f"{row['mean_reward_difference_right_minus_left']:.6g} | {row['paired_bootstrap_95ci']} |"
        )
    lines.extend(
        [
            "",
            "## BF16 Status",
            "",
            "Native BF16 was not run: all visible GPUs are Tesla T4 (SM 7.5), which do not provide native BF16 execution. This is an external-validity gap requiring Ampere or newer hardware.",
            "",
            "## Interpretation Boundary",
            "",
            payload["claim_scope"],
            "",
            "## Artifacts",
            "",
            f"- `{args.out}`",
            "- `results/phase14_task_reward/`",
            "- `scripts/phase14_task_reward_eval_once.py`",
            "- `scripts/phase14_merge_task_reward.py`",
            "",
        ]
    )
    Path(args.report).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": args.out, "report": args.report, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
