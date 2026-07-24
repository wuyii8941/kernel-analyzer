#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.phase8_matched_step import state_distance


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge clean and zero-initial-fork mutation trajectory arms.")
    parser.add_argument("--clean", required=True)
    parser.add_argument("--mutation", action="append", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--legal-envelope", default="results/trajectory_step5_fusion/merged.json")
    parser.add_argument("--clean-control-root", default="results/trajectory_step5_fusion/A_reference")
    parser.add_argument("--out", default="results/phase10_zero_fork_mutation_trajectories.json")
    parser.add_argument("--report", default="reports/phase10_zero_fork_mutation_trajectories.md")
    args = parser.parse_args()
    clean = json.loads(Path(args.clean).read_text(encoding="utf-8"))
    mutations = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.mutation]
    legal = json.loads(Path(args.legal_envelope).read_text(encoding="utf-8"))
    legal_by_step = {int(row["step"]): row["A_B"] for row in legal["distances"]}
    root = Path(args.root)
    clean_controls = []
    for step in clean["checkpoint_steps"]:
        distance = state_distance(
            root / "clean_reference" / f"step_{step:02d}",
            Path(args.clean_control_root) / f"step_{step:02d}",
        )
        clean_controls.append({"step": step, **distance})
    if any(row["l2"] != 0.0 for row in clean_controls):
        raise ValueError(f"independent clean trajectory control is nonzero: {clean_controls}")
    cases = []
    for mutation in mutations:
        if mutation["steps"] != clean["steps"]:
            raise ValueError("trajectory length mismatch")
        rows = []
        first_branch_fork = None
        for clean_row, mutation_row in zip(clean["trajectory"], mutation["trajectory"], strict=True):
            if clean_row["step"] != mutation_row["step"]:
                raise ValueError("trajectory step mismatch")
            branch_forks = sum(
                left is not None and right is not None and left != right
                for left, right in zip(clean_row["clip_active"], mutation_row["clip_active"], strict=True)
            )
            if branch_forks and first_branch_fork is None:
                first_branch_fork = clean_row["step"]
            deltas = [
                abs(left - right)
                for left, right in zip(clean_row["logps"], mutation_row["logps"], strict=True)
            ]
            rows.append(
                {
                    "step": clean_row["step"],
                    "clipping_branch_forks": branch_forks,
                    "max_abs_logp_delta": max(deltas),
                    "mean_abs_logp_delta": sum(deltas) / len(deltas),
                    "loss_delta": mutation_row["loss"] - clean_row["loss"],
                    "gradient_norm_delta": mutation_row["full_gradient_norm"] - clean_row["full_gradient_norm"],
                }
            )
        distances = []
        mutation_name = mutation["mutation"]
        for step in clean["checkpoint_steps"]:
            clean_dir = root / "clean_reference" / f"step_{step:02d}"
            mutation_dir = root / f"mutation_{mutation_name}" / f"step_{step:02d}"
            distance = state_distance(clean_dir, mutation_dir)
            envelope = legal_by_step.get(step)
            distances.append(
                {
                    "step": step,
                    "clean_mutation": distance,
                    "legal_eager_compile": envelope,
                    "ratio_to_legal_envelope": (
                        distance["l2"] / envelope["l2"] if envelope and envelope["l2"] else None
                    ),
                }
            )
        cases.append(
            {
                "mutation": mutation_name,
                "first_clipping_branch_fork_step": first_branch_fork,
                "total_clipping_branch_forks_over_steps": sum(row["clipping_branch_forks"] for row in rows),
                "step1_has_continuous_update_divergence_before_clipping_fork": (
                    rows[0]["clipping_branch_forks"] == 0 and distances[0]["clean_mutation"]["l2"] > 0
                ),
                "trajectory": rows,
                "distances": distances,
            }
        )
    payload = {
        "schema_version": "forkcert.zero_fork_mutation_trajectories.v1",
        "cases": cases,
        "independent_clean_checkpoint_control": clean_controls,
        "claim_scope": (
            "A zero clipping-fork count is not training equivalence. Distances show state divergence, not task harm; "
            "the legal eager/compile trajectory is an empirical comparison envelope, not a certified bound."
        ),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Phase 10 Zero-Initial-Clipping-Fork Mutation Trajectories",
        "",
        "## Objective",
        "",
        "Determine whether mutations with zero clipping forks at the frozen initial state can still change PPO gradients and parameter updates through the continuous within-branch channel.",
        "",
        "## Results",
        "",
        "| Mutation | First clipping-fork step | Total branch forks | Step-1 parameter L2 | Step-1 ratio to legal pair | Continuous divergence before fork |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for case in cases:
        first = case["first_clipping_branch_fork_step"]
        distance = case["distances"][0]
        lines.append(
            f"| {case['mutation']} | {first if first is not None else 'none'} | "
            f"{case['total_clipping_branch_forks_over_steps']} | {distance['clean_mutation']['l2']:.6g} | "
            f"{distance['ratio_to_legal_envelope']:.6g} | "
            f"{case['step1_has_continuous_update_divergence_before_clipping_fork']} |"
        )
    lines.extend(
        [
            "",
            "## Parameter Distance Growth",
            "",
            "| Mutation | Step 1 vs legal | Step 5 vs legal | Step 20 vs legal |",
            "|---|---:|---:|---:|",
        ]
    )
    for case in cases:
        ratios = {row["step"]: row["ratio_to_legal_envelope"] for row in case["distances"]}
        lines.append(
            f"| {case['mutation']} | {ratios[1]:.6g} | {ratios[5]:.6g} | {ratios[20]:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Independent Clean Control",
            "",
            "The clean trajectory was independently rerun. Its saved full-model checkpoints are bitwise identical to the earlier clean A trajectory at steps 1, 5, and 20 (`L2=0` for all three).",
            "",
            "## Interpretation Boundary",
            "",
            payload["claim_scope"],
            "",
            "A mutation may be called equivalent only if no input can distinguish it from the original program. These finite traces can establish neither program equivalence nor downstream task harm.",
            "",
            "## Artifacts",
            "",
            f"- `{args.out}`",
            "- `scripts/phase10_mutation_trajectory_arm.py`",
            "- `scripts/phase10_merge_mutation_trajectories.py`",
            "",
        ]
    )
    Path(args.report).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"cases": len(cases), "out": args.out, "report": args.report}, indent=2))


if __name__ == "__main__":
    main()
