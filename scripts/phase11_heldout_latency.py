#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.phase8_matched_step import state_distance


def branch_fork_count(clean_row: dict[str, Any], mutation_row: dict[str, Any]) -> int:
    return sum(
        left is not None and right is not None and left != right
        for left, right in zip(clean_row["clip_active"], mutation_row["clip_active"], strict=True)
    )


def summarize_case(
    clean: dict[str, Any], mutation: dict[str, Any], root: Path, initial_forks: int
) -> dict[str, Any]:
    if clean["replay"] != mutation["replay"]:
        raise ValueError(f"replay metadata differs for {mutation['mutation']}")
    if clean["steps"] != mutation["steps"]:
        raise ValueError("trajectory length mismatch")
    trajectory = []
    first_delayed = None
    for clean_row, mutation_row in zip(clean["trajectory"], mutation["trajectory"], strict=True):
        if clean_row["step"] != mutation_row["step"]:
            raise ValueError("trajectory step mismatch")
        forks = branch_fork_count(clean_row, mutation_row)
        if forks and first_delayed is None:
            first_delayed = int(clean_row["step"])
        deltas = [
            abs(left - right)
            for left, right in zip(clean_row["logps"], mutation_row["logps"], strict=True)
        ]
        trajectory.append(
            {
                "step": int(clean_row["step"]),
                "clipping_branch_forks": forks,
                "max_abs_logp_delta": max(deltas),
                "mean_abs_logp_delta": sum(deltas) / len(deltas),
                "loss_delta": mutation_row["loss"] - clean_row["loss"],
                "gradient_norm_delta": (
                    mutation_row["full_gradient_norm"] - clean_row["full_gradient_norm"]
                ),
            }
        )
    distances = []
    for step in clean["checkpoint_steps"]:
        distances.append(
            {
                "step": step,
                **state_distance(
                    root / "clean_reference" / f"step_{step:02d}",
                    root / f"mutation_{mutation['mutation']}" / f"step_{step:02d}",
                ),
            }
        )
    return {
        "mutation": mutation["mutation"],
        "initial_catalog_clipping_forks": initial_forks,
        "latency_definition": "0 means the held-out frozen state already forks; otherwise first matched update index",
        "clipping_fork_latency": 0 if initial_forks else first_delayed,
        "first_matched_update_fork_step": first_delayed,
        "total_matched_update_branch_forks": sum(row["clipping_branch_forks"] for row in trajectory),
        "continuous_update_divergence_before_first_observed_fork": (
            initial_forks == 0
            and trajectory[0]["clipping_branch_forks"] == 0
            and distances[0]["l2"] > 0
        ),
        "trajectory": trajectory,
        "parameter_distances": distances,
    }


def render_report(payload: dict[str, Any]) -> str:
    replay = payload["heldout_replay"]
    lines = [
        "# Phase 11 Held-out Fork Latency",
        "",
        "## Objective",
        "",
        "Test whether the step-2/3 delayed clipping forks from Phase 10 reproduce after changing both checkpoint and replay batch.",
        "",
        "## Controls",
        "",
        f"- Held-out optimizer step: `{replay['optimizer_step']}`",
        f"- Held-out rollout batch: `{replay['rollout_batch']}`",
        f"- Replay batch SHA256: `{replay['batch_sha256']}`",
        "- Clean and mutation arms use identical token IDs, old logprobs, advantages, optimizer, seed and initial weights.",
        "- Mutation eligibility is determined on the held-out frozen state before trajectory execution.",
        "- No step-5 legal-path envelope is reused at this checkpoint.",
        "",
        "## Held-out Initial Gate",
        "",
        "| Mutation | Discovery-state forks | Held-out-state forks | Immediate latency |",
        "|---|---:|---:|---:|",
    ]
    for row in payload["heldout_initial_gate"]:
        lines.append(
            f"| {row['mutation']} | {row['discovery_initial_forks']} | "
            f"{row['heldout_initial_forks']} | {0 if row['heldout_initial_forks'] else 'not immediate'} |"
        )
    lines.extend([
        "",
        "## Results",
        "",
        "| Mutation | Initial held-out forks | Fork latency | Total trajectory forks | Step-1 parameter L2 | Continuous divergence before fork |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for case in payload["cases"]:
        latency = case["clipping_fork_latency"]
        step1 = next(row for row in case["parameter_distances"] if row["step"] == 1)
        lines.append(
            f"| {case['mutation']} | {case['initial_catalog_clipping_forks']} | "
            f"{latency if latency is not None else '> horizon'} | "
            f"{case['total_matched_update_branch_forks']} | {step1['l2']:.6g} | "
            f"{case['continuous_update_divergence_before_first_observed_fork']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            payload["claim_scope"],
            "",
            "Parameter divergence is a state-level effect, not evidence of reward or task-quality harm.",
            "",
            "## Artifacts",
            "",
            "- `results/phase11_heldout_latency.json`",
            "- `results/phase11_heldout_mutations/summary.json`",
            "- `results/phase11_heldout_twins/`",
            "- `scripts/phase11_heldout_latency.py`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge held-out zero-fork mutation trajectories.")
    parser.add_argument("--clean", required=True)
    parser.add_argument("--mutation", action="append", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--discovery-summary", default="results/phase9_mutations_gated/summary.json")
    parser.add_argument("--out", default="results/phase11_heldout_latency.json")
    parser.add_argument("--report", default="reports/phase11_heldout_latency.md")
    args = parser.parse_args()
    clean = json.loads(Path(args.clean).read_text(encoding="utf-8"))
    mutations = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.mutation]
    catalog = json.loads(Path(args.catalog).read_text(encoding="utf-8"))
    discovery = json.loads(Path(args.discovery_summary).read_text(encoding="utf-8"))
    initial = {row["name"]: int(row["branch_forks"]) for row in catalog["mutations"]}
    discovery_forks = {row["name"]: int(row["branch_forks"]) for row in discovery["mutations"]}
    cases = [summarize_case(clean, mutation, Path(args.root), initial[mutation["mutation"]]) for mutation in mutations]
    payload = {
        "schema_version": "forkcert.heldout_fork_latency.v1",
        "discovery_replay": {
            "optimizer_step": discovery["optimizer_step"],
            "rollout_batch": discovery["rollout_batch"],
        },
        "heldout_replay": clean["replay"],
        "checkpoint_and_batch_held_out": (
            clean["replay"]["optimizer_step"] != discovery["optimizer_step"]
            and clean["replay"]["rollout_batch"] != discovery["rollout_batch"]
        ),
        "discovery_initial_forks": {case["mutation"]: discovery_forks[case["mutation"]] for case in cases},
        "heldout_initial_gate": [
            {
                "mutation": name,
                "discovery_initial_forks": discovery_forks[name],
                "heldout_initial_forks": forks,
            }
            for name, forks in initial.items()
        ],
        "cases": cases,
        "claim_scope": (
            "A held-out checkpoint/batch replication of mutation-to-clipping-fork latency under a frozen repeated-batch "
            "matched-update protocol. It tests checkpoint/batch dependence, but not fresh-batch training or task harm."
        ),
    }
    if not payload["checkpoint_and_batch_held_out"]:
        raise ValueError("held-out run reuses the discovery checkpoint or rollout batch")
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.report).write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({"cases": len(cases), "out": args.out, "report": args.report}, indent=2))


if __name__ == "__main__":
    main()
