#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable


def mean_difference(values: list[float], fork_mask: list[bool]) -> float:
    fork = [value for value, is_fork in zip(values, fork_mask, strict=True) if is_fork]
    nonfork = [value for value, is_fork in zip(values, fork_mask, strict=True) if not is_fork]
    return mean(fork) - mean(nonfork)


def exact_permutation_pvalue(values: list[float], fork_mask: list[bool]) -> float:
    """Two-sided exact randomization test with the observed group sizes fixed."""
    fork_count = sum(fork_mask)
    observed = abs(mean_difference(values, fork_mask))
    extreme = 0
    total = 0
    for fork_indices in itertools.combinations(range(len(values)), fork_count):
        selected = set(fork_indices)
        candidate = [index in selected for index in range(len(values))]
        if abs(mean_difference(values, candidate)) >= observed - 1e-15:
            extreme += 1
        total += 1
    return extreme / total


def bootstrap_mean_difference(
    values: list[float], fork_mask: list[bool], draws: int = 10_000, seed: int = 0
) -> list[float]:
    fork = [value for value, is_fork in zip(values, fork_mask, strict=True) if is_fork]
    nonfork = [value for value, is_fork in zip(values, fork_mask, strict=True) if not is_fork]
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        fork_sample = [rng.choice(fork) for _ in fork]
        nonfork_sample = [rng.choice(nonfork) for _ in nonfork]
        samples.append(mean(fork_sample) - mean(nonfork_sample))
    samples.sort()
    return [samples[int(0.025 * (draws - 1))], samples[int(0.975 * (draws - 1))]]


def summarize_metric(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(row[key]) for row in rows]
    mask = [bool(row["is_fork_step"]) for row in rows]
    fork = [value for value, is_fork in zip(values, mask, strict=True) if is_fork]
    nonfork = [value for value, is_fork in zip(values, mask, strict=True) if not is_fork]
    fork_mean = mean(fork)
    nonfork_mean = mean(nonfork)
    return {
        "fork_n": len(fork),
        "nonfork_n": len(nonfork),
        "fork_mean": fork_mean,
        "nonfork_mean": nonfork_mean,
        "fork_median": median(fork),
        "nonfork_median": median(nonfork),
        "mean_difference": fork_mean - nonfork_mean,
        "fork_over_nonfork_ratio": fork_mean / nonfork_mean if nonfork_mean else None,
        "bootstrap_mean_difference_95pct": bootstrap_mean_difference(values, mask),
        "exact_permutation_pvalue_two_sided": exact_permutation_pvalue(values, mask),
    }


def build_analysis(merged: dict[str, Any]) -> dict[str, Any]:
    arms = {arm["arm"]: arm for arm in merged["arms"]}
    a_rows = {int(row["step"]): row for row in arms["A_reference"]["trajectory"]}
    b_rows = {int(row["step"]): row for row in arms["B_alternative"]["trajectory"]}
    if set(a_rows) != set(b_rows):
        raise ValueError("A/B trajectory steps differ")

    rows = []
    for step in sorted(a_rows):
        a = a_rows[step]
        b = b_rows[step]
        a_norm = float(a["full_gradient_norm"])
        b_norm = float(b["full_gradient_norm"])
        average_norm = (a_norm + b_norm) / 2.0
        norm_gap = abs(a_norm - b_norm)
        rows.append(
            {
                "step": step,
                "is_fork_step": bool(a["target_clip_active"]) != bool(b["target_clip_active"]),
                "target_clip_active_a": bool(a["target_clip_active"]),
                "target_clip_active_b": bool(b["target_clip_active"]),
                "target_gradient_semantic_fork": math.isclose(float(a["target_loss_gradient"]), 0.0, abs_tol=1e-15)
                != math.isclose(float(b["target_loss_gradient"]), 0.0, abs_tol=1e-15),
                "average_gradient_norm": average_norm,
                "absolute_gradient_norm_gap": norm_gap,
                "normalized_gradient_norm_gap": norm_gap / average_norm if average_norm else None,
                "absolute_loss_gap": abs(float(a["loss"]) - float(b["loss"])),
                "absolute_target_logp_gap": abs(float(a["target_logp"]) - float(b["target_logp"])),
            }
        )

    metrics = {
        key: summarize_metric(rows, key)
        for key in [
            "average_gradient_norm",
            "absolute_gradient_norm_gap",
            "normalized_gradient_norm_gap",
            "absolute_loss_gap",
            "absolute_target_logp_gap",
        ]
    }
    return {
        "schema_version": "forkcert.matched_step_counterfactual.v1",
        "fork_id": merged["fork_id"],
        "protocol": "20-step A/B trajectory; exact permutation test over the seven observed fork-step labels",
        "fork_definition": "A.target_clip_active != B.target_clip_active for the frozen target token",
        "rows": rows,
        "summary": {
            "steps": len(rows),
            "fork_steps": [row["step"] for row in rows if row["is_fork_step"]],
            "fork_step_count": sum(row["is_fork_step"] for row in rows),
            "target_gradient_semantic_fork_count": sum(row["target_gradient_semantic_fork"] for row in rows),
            "metrics": metrics,
            "checkpoint_distance_anchors": merged["distances"],
        },
        "interpretation_limits": [
            "Only steps 1, 5, and 20 have parameter checkpoints, so per-step parameter-distance increments are unavailable.",
            "The normalized gradient-norm gap is a confound diagnostic and not a substitute for a parameter-distance jump.",
            "The 20 steps are a single trajectory and are serially dependent; permutation and bootstrap results are descriptive sensitivity analyses, not population-level inference.",
        ],
    }


def render_report(analysis: dict[str, Any]) -> str:
    summary = analysis["summary"]
    metrics = summary["metrics"]

    def metric_row(label: str, key: str) -> str:
        item = metrics[key]
        ratio = item["fork_over_nonfork_ratio"]
        ratio_text = "n/a" if ratio is None else f"{ratio:.3f}"
        ci = item["bootstrap_mean_difference_95pct"]
        return (
            f"| {label} | {item['fork_mean']:.8g} | {item['nonfork_mean']:.8g} | "
            f"{ratio_text} | {item['mean_difference']:.8g} | [{ci[0]:.3g}, {ci[1]:.3g}] | "
            f"{item['exact_permutation_pvalue_two_sided']:.4g} |"
        )

    avg = metrics["average_gradient_norm"]
    normalized = metrics["normalized_gradient_norm_gap"]
    lines = [
        "# Phase 8 Matched-Step Counterfactual Confound Analysis",
        "",
        "## Objective",
        "",
        "Test whether the previously observed association between clipping-fork steps and faster parameter divergence can be explained trivially by larger full-batch gradient scale.",
        "",
        "## Controls And Scope",
        "",
        "- Same frozen step-5 state, batch, old logprobs, advantages, optimizer, learning rate and target token for A eager and B default compile.",
        "- A fork step is defined before analysis as `A.target_clip_active != B.target_clip_active`.",
        "- There are 20 matched steps and 7 target-token fork steps: " + ", ".join(map(str, summary["fork_steps"])) + ".",
        "- Parameter checkpoints exist only at steps 1, 5 and 20. Therefore this analysis does **not** reconstruct a 20-step parameter-distance increment series.",
        "",
        "## Result",
        "",
        "| Metric | Fork-step mean | Non-fork mean | Ratio | Mean difference | Bootstrap 95% CI | Exact permutation p |",
        "|---|---:|---:|---:|---:|---:|---:|",
        metric_row("Average full-gradient norm", "average_gradient_norm"),
        metric_row("Absolute gradient-norm gap", "absolute_gradient_norm_gap"),
        metric_row("Normalized gradient-norm gap", "normalized_gradient_norm_gap"),
        metric_row("Absolute loss gap", "absolute_loss_gap"),
        metric_row("Absolute target-logp gap", "absolute_target_logp_gap"),
        "",
        f"Fork steps have an average full-gradient norm ratio of `{avg['fork_over_nonfork_ratio']:.3f}` relative to non-fork steps. The normalized A/B gradient-norm gap ratio is `{normalized['fork_over_nonfork_ratio']:.3f}`.",
        "",
        f"All `{summary['target_gradient_semantic_fork_count']}` branch-fork rows also disagree on whether the target token has zero versus non-zero loss gradient.",
        "",
        "## Parameter-Distance Anchors",
        "",
        "| Step | A-B L2 | A-C L2 | A-C / A-B |",
        "|---:|---:|---:|---:|",
    ]
    for item in summary["checkpoint_distance_anchors"]:
        lines.append(
            f"| {item['step']} | {item['A_B']['l2']:.8g} | {item['A_C']['l2']:.8g} | "
            f"{item['recovery_ratio_A_C_over_A_B']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This analysis separates two questions. The average full-gradient norm checks the simple scale confound; the normalized A/B norm gap checks whether fork steps coincide with a larger path-dependent gradient disturbance after dividing out batch gradient scale.",
            "",
            "It cannot validate the earlier `6.49x` parameter-divergence jump ratio because that result comes from a different 100-step twin trajectory and the current 20-step run lacks intermediate parameter checkpoints. A strict normalized jump-ratio claim requires rerunning and saving every step, or recording gradient vectors sufficient to predict each update distance.",
            "",
            "The exact permutation and bootstrap numbers are descriptive only: the 20 observations are serially dependent and come from one replay trajectory. They must not be presented as cross-prompt or cross-checkpoint significance.",
            "",
            "## Artifacts",
            "",
            "- Source trajectories: `results/trajectory_step5_fusion/A_reference.json`, `B_alternative.json`",
            "- Checkpoint anchors: `results/trajectory_step5_fusion/merged.json`",
            "- Structured analysis: `results/phase8_matched_step_counterfactual.json`",
            "- Analysis script: `scripts/phase8_matched_step_counterfactual.py`",
            "",
            "## Decision",
            "",
            "**REVISE.** Use this as a gradient-scale confound audit, not as proof of a normalized parameter-jump effect. The single-step A/B/C intervention remains the direct causal result; the long-horizon 6.49x timing result remains coupling evidence pending a fully instrumented rerun.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze gradient-scale confounding in the matched A/B trajectory.")
    parser.add_argument("--merged", default="results/trajectory_step5_fusion/merged.json")
    parser.add_argument("--out", default="results/phase8_matched_step_counterfactual.json")
    parser.add_argument("--report", default="reports/phase8_matched_step_counterfactual.md")
    args = parser.parse_args()
    merged = json.loads(Path(args.merged).read_text(encoding="utf-8"))
    analysis = build_analysis(merged)
    Path(args.out).write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.report).write_text(render_report(analysis), encoding="utf-8")
    print(json.dumps(analysis["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
