#!/usr/bin/env python
"""Exploratory, claim-limited analysis of the complete ForkCert online scan.

This script treats eager--compiled differences as implementation-relative
discrepancies.  It does not label either path as mathematical truth and does
not treat clipping disagreement as a correctness violation.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def q(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability)) if len(values) else math.nan


def summary(values: np.ndarray, prefix: str = "") -> dict:
    if not len(values):
        return {f"{prefix}n": 0}
    return {
        f"{prefix}n": int(len(values)),
        f"{prefix}mean": float(values.mean()),
        f"{prefix}std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        f"{prefix}p01": q(values, 0.01),
        f"{prefix}p50": q(values, 0.50),
        f"{prefix}p99": q(values, 0.99),
        f"{prefix}min": float(values.min()),
        f"{prefix}max": float(values.max()),
    }


def weighted_decomposition(values: np.ndarray, groups: list[str | int]) -> dict:
    """Empirical total-variance identity using population (ddof=0) moments."""
    grouped: dict[str | int, list[float]] = defaultdict(list)
    for value, group in zip(values, groups, strict=True):
        grouped[group].append(float(value))
    total_mean = float(values.mean())
    total_var = float(values.var(ddof=0))
    between = 0.0
    within = 0.0
    n = len(values)
    group_means = []
    for group_values in grouped.values():
        array = np.asarray(group_values, dtype=np.float64)
        weight = len(array) / n
        group_mean = float(array.mean())
        group_means.append(group_mean)
        between += weight * (group_mean - total_mean) ** 2
        within += weight * float(array.var(ddof=0))
    return {
        "groups": len(grouped),
        "total_variance": total_var,
        "between_group_variance": between,
        "mean_within_group_variance": within,
        "identity_residual": total_var - between - within,
        "between_fraction": between / total_var if total_var else 0.0,
        "unweighted_group_mean_std": float(np.std(group_means, ddof=1)) if len(group_means) > 1 else 0.0,
    }


def moving_block_bootstrap(
    rollout_values: dict[int, tuple[float, int]], *, block_length: int, draws: int, seed: int
) -> dict:
    """Circular moving-block bootstrap over sequential rollout clusters."""
    rollout_ids = sorted(rollout_values)
    sums = np.asarray([rollout_values[item][0] for item in rollout_ids], dtype=np.float64)
    counts = np.asarray([rollout_values[item][1] for item in rollout_ids], dtype=np.float64)
    n_rollouts = len(rollout_ids)
    blocks_needed = math.ceil(n_rollouts / block_length)
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        selected: list[int] = []
        for start in rng.integers(0, n_rollouts, size=blocks_needed):
            selected.extend((int(start) + offset) % n_rollouts for offset in range(block_length))
        chosen = np.asarray(selected[:n_rollouts], dtype=np.int64)
        estimates[draw] = sums[chosen].sum() / counts[chosen].sum()
    return {
        "block_length_rollouts": block_length,
        "draws": draws,
        "ci95_low": q(estimates, 0.025),
        "ci95_high": q(estimates, 0.975),
    }


def aggregate_for_bootstrap(rows: list[dict], field: str) -> dict[int, tuple[float, int]]:
    aggregates: dict[int, tuple[float, int]] = {}
    for row in rows:
        rollout = int(row["rollout_batch"])
        total, count = aggregates.get(rollout, (0.0, 0))
        aggregates[rollout] = (total + float(row[field]), count + 1)
    return aggregates


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "(none)"
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                cells.append(f"{value:.8g}")
            else:
                cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, rule, *body])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/phase4_online_full_logprobs.jsonl")
    parser.add_argument("--out-json", default="theory_oracle/exploration_existing.json")
    parser.add_argument("--report", default="theory_oracle/EXPLORATION_EXISTING.md")
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--bootstrap-block", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--max-rollout-exclusive",
        type=int,
        default=None,
        help="Optional claim-scope filter used when later rollouts lack a validated compiled-path canary.",
    )
    args = parser.parse_args()

    source = Path(args.input)
    rows = load_jsonl(source)
    if args.max_rollout_exclusive is not None:
        rows = [row for row in rows if int(row["rollout_batch"]) < args.max_rollout_exclusive]
    applicable = [row for row in rows if int(row.get("advantage_sign", 0)) != 0]
    upper = math.log1p(args.eps)
    lower = math.log1p(-args.eps)

    for row in applicable:
        sign = int(row["advantage_sign"])
        ref_ratio = float(row["logp_ref"]) - float(row["old_logp"])
        alt_ratio = float(row["logp_alt"]) - float(row["old_logp"])
        signed_delta = float(row["logp_alt"]) - float(row["logp_ref"])
        if sign > 0:
            ref_event_margin = ref_ratio - upper
            alt_event_margin = alt_ratio - upper
        else:
            ref_event_margin = lower - ref_ratio
            alt_event_margin = lower - alt_ratio
        row["signed_delta"] = signed_delta
        row["abs_delta"] = abs(signed_delta)
        row["ref_event_margin"] = ref_event_margin
        row["boundary_distance"] = abs(ref_event_margin)
        row["event_oriented_shift"] = alt_event_margin - ref_event_margin
        row["ref_event"] = float(ref_event_margin > 0.0)
        row["alt_event"] = float(alt_event_margin > 0.0)
        row["up_flip"] = float(ref_event_margin <= 0.0 < alt_event_margin)
        row["down_flip"] = float(alt_event_margin <= 0.0 < ref_event_margin)
        row["disagreement"] = row["up_flip"] + row["down_flip"]
        row["directional_event"] = row["up_flip"] - row["down_flip"]

    deltas = np.asarray([row["signed_delta"] for row in applicable], dtype=np.float64)
    abs_deltas = np.abs(deltas)
    oriented = np.asarray([row["event_oriented_shift"] for row in applicable], dtype=np.float64)
    distances = np.asarray([row["boundary_distance"] for row in applicable], dtype=np.float64)
    disagreements = np.asarray([row["disagreement"] for row in applicable], dtype=np.float64)
    directional = np.asarray([row["directional_event"] for row in applicable], dtype=np.float64)

    self_ref_max = max(float(row.get("delta_self_ref", 0.0)) for row in rows)
    self_alt_max = max(float(row.get("delta_self_alt", 0.0)) for row in rows)
    correlation = stats.spearmanr(np.log10(np.maximum(abs_deltas, 1e-30)), np.log10(np.maximum(distances, 1e-30)))

    numeric = {
        **summary(deltas, "signed_"),
        **summary(abs_deltas, "absolute_"),
        **summary(oriented, "event_oriented_"),
        "fraction_signed_positive": float(np.mean(deltas > 0.0)),
        "fraction_signed_negative": float(np.mean(deltas < 0.0)),
        "fraction_signed_zero": float(np.mean(deltas == 0.0)),
        "self_ref_max": self_ref_max,
        "self_alt_max": self_alt_max,
    }
    semantic = {
        "applicable_tokens": len(applicable),
        "reference_event_rate": float(np.mean([row["ref_event"] for row in applicable])),
        "compiled_event_rate": float(np.mean([row["alt_event"] for row in applicable])),
        "up_flip_count": int(sum(row["up_flip"] for row in applicable)),
        "down_flip_count": int(sum(row["down_flip"] for row in applicable)),
        "semantic_disagreement": float(disagreements.mean()),
        "directional_semantic_shift": float(directional.mean()),
        "compiled_minus_reference_event_rate": float(
            np.mean([row["alt_event"] - row["ref_event"] for row in applicable])
        ),
    }
    hierarchy = {
        "by_case": weighted_decomposition(deltas, [str(row["case_id"]) for row in applicable]),
        "by_rollout": weighted_decomposition(deltas, [int(row["rollout_batch"]) for row in applicable]),
    }

    bootstrap = {
        "signed_mean": moving_block_bootstrap(
            aggregate_for_bootstrap(applicable, "signed_delta"),
            block_length=args.bootstrap_block,
            draws=args.bootstrap_draws,
            seed=args.seed,
        ),
        "event_oriented_mean": moving_block_bootstrap(
            aggregate_for_bootstrap(applicable, "event_oriented_shift"),
            block_length=args.bootstrap_block,
            draws=args.bootstrap_draws,
            seed=args.seed + 1,
        ),
        "semantic_disagreement": moving_block_bootstrap(
            aggregate_for_bootstrap(applicable, "disagreement"),
            block_length=args.bootstrap_block,
            draws=args.bootstrap_draws,
            seed=args.seed + 2,
        ),
        "directional_semantic_shift": moving_block_bootstrap(
            aggregate_for_bootstrap(applicable, "directional_event"),
            block_length=args.bootstrap_block,
            draws=args.bootstrap_draws,
            seed=args.seed + 3,
        ),
    }

    boundary_edges = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, math.inf]
    boundary_rows = []
    for low, high in zip(boundary_edges[:-1], boundary_edges[1:], strict=True):
        selected = [row for row in applicable if low <= row["boundary_distance"] < high]
        if not selected:
            continue
        boundary_rows.append(
            {
                "distance_bin": f"[{low:g},{high:g})",
                "tokens": len(selected),
                "signed_mean": float(np.mean([row["signed_delta"] for row in selected])),
                "abs_delta_mean": float(np.mean([row["abs_delta"] for row in selected])),
                "oriented_mean": float(np.mean([row["event_oriented_shift"] for row in selected])),
                "up_flips": int(sum(row["up_flip"] for row in selected)),
                "down_flips": int(sum(row["down_flip"] for row in selected)),
                "disagreement_rate": float(np.mean([row["disagreement"] for row in selected])),
            }
        )

    observed_rollouts = sorted({int(row["rollout_batch"]) for row in rows})
    rollout_start = min(observed_rollouts)
    rollout_stop = max(observed_rollouts) + 1
    cut1 = rollout_start + math.ceil((rollout_stop - rollout_start) / 3)
    cut2 = rollout_start + math.ceil(2 * (rollout_stop - rollout_start) / 3)
    stage_rows = []
    for name, low, high in [
        ("early", rollout_start, cut1),
        ("middle", cut1, cut2),
        ("late", cut2, rollout_stop),
    ]:
        selected = [row for row in applicable if low <= int(row["rollout_batch"]) < high]
        stage_rows.append(
            {
                "stage": name,
                "rollouts": len({int(row["rollout_batch"]) for row in selected}),
                "tokens": len(selected),
                "signed_mean": float(np.mean([row["signed_delta"] for row in selected])),
                "signed_std": float(np.std([row["signed_delta"] for row in selected], ddof=1)),
                "oriented_mean": float(np.mean([row["event_oriented_shift"] for row in selected])),
                "up_flips": int(sum(row["up_flip"] for row in selected)),
                "down_flips": int(sum(row["down_flip"] for row in selected)),
            }
        )

    payload = {
        "claim_scope": {
            "comparison": "compiled minus eager; implementation-relative, not truth-relative",
            "state_population": (
                "one canonical Qwen3-0.6B GRPO trajectory"
                + (
                    f", rollout batches [0,{args.max_rollout_exclusive}) only"
                    if args.max_rollout_exclusive is not None
                    else ", all recorded rollout batches"
                )
            ),
            "observable": "selected-token log probability and PPO/GRPO clipping event",
            "limitations": [
                "one trajectory and one T4 FP16 configuration",
                "token/case/rollout heterogeneity is not within-state runtime variance",
                "no gradient, update, optimizer-state, or next-state endpoint in this dataset",
                "semantic disagreement is impact evidence, not correctness evidence",
                "the source data do not contain a per-row generated-code or compiled-path canary",
            ],
        },
        "coverage": {
            "rows": len(rows),
            "applicable_rows": len(applicable),
            "cases": len({str(row["case_id"]) for row in rows}),
            "rollouts": len({int(row["rollout_batch"]) for row in rows}),
            "optimizer_steps": len({int(row["optimizer_step"]) for row in rows}),
        },
        "numeric": numeric,
        "semantic": semantic,
        "hierarchical_descriptive_decomposition": hierarchy,
        "moving_block_bootstrap": bootstrap,
        "boundary_conditioning": boundary_rows,
        "training_stage": stage_rows,
        "associations": {
            "spearman_log_abs_delta_vs_log_boundary_distance": float(correlation.statistic),
            "pvalue_descriptive_only": float(correlation.pvalue),
        },
    }

    Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = "\n".join(
        [
            "# Existing-Data Oracle Exploration",
            "",
            "## Scope",
            "",
            (
                "This is an exploratory reanalysis of the online scan"
                + (
                    f", restricted to rollout batches [0,{args.max_rollout_exclusive})"
                    if args.max_rollout_exclusive is not None
                    else ""
                )
                + ". Compiled-minus-eager is an implementation-relative discrepancy, not a truth-relative error. Results are conditional on one canonical trajectory and the recorded T4 FP16 configuration."
            ),
            "",
            "## Coverage",
            "",
            markdown_table([payload["coverage"]], list(payload["coverage"].keys())),
            "",
            "## Numerical Endpoints",
            "",
            markdown_table([numeric], list(numeric.keys())),
            "",
            "## Semantic Endpoints",
            "",
            markdown_table([semantic], list(semantic.keys())),
            "",
            "## Training-Stage Conditioning",
            "",
            markdown_table(stage_rows, list(stage_rows[0].keys())),
            "",
            "## Boundary Conditioning",
            "",
            markdown_table(boundary_rows, list(boundary_rows[0].keys())),
            "",
            "## Descriptive Variance Decomposition",
            "",
            "The between-case and between-rollout terms describe heterogeneity across sampled units. They are not runtime variance components.",
            "",
            markdown_table(
                [
                    {"level": "case", **hierarchy["by_case"]},
                    {"level": "rollout", **hierarchy["by_rollout"]},
                ],
                ["level", *hierarchy["by_case"].keys()],
            ),
            "",
            "## Block-Bootstrap Intervals",
            "",
            markdown_table(
                [{"estimand": key, **value} for key, value in bootstrap.items()],
                ["estimand", "block_length_rollouts", "draws", "ci95_low", "ci95_high"],
            ),
            "",
            "## Interpretation Limits",
            "",
            "- Attached self-pair maxima describe the observed deterministic floor; they do not estimate a general runtime-noise law.",
            "- The scan has no gradient/update/next-state endpoints, so it cannot select a complete impact Oracle by itself.",
            "- Five clipping disagreements are too sparse for a stable universal event-rate claim.",
            "- All inferential numbers remain conditional on this single serial trajectory.",
            "",
        ]
    )
    Path(args.report).write_text(report, encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
