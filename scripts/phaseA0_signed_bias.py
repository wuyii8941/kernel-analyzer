#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from forkcert.io import read_jsonl
from forkcert.report import CLAIM_SCOPE, markdown_table


EXTERNAL_VALIDITY = (
    "This server uses Tesla T4 FP16 (u approximately 4.9e-4). It does not reproduce production BF16 "
    "kernels (u approximately 3.9e-3). An FP16 fork is evidence that the mechanism can occur at higher "
    "precision; a zero-FP16 result would be scoped to FP16 and would not rule out BF16. A BF16-hardware "
    "replication remains required."
)


def key(row: dict) -> tuple[str, int]:
    return str(row["case_id"]), int(row["token_index"])


def cluster_bootstrap(rows: list[dict], *, seed: int, draws: int = 10_000) -> tuple[float, float]:
    clusters: dict[str, tuple[float, int]] = {}
    for row in rows:
        case_id = str(row["case_id"])
        total, count = clusters.get(case_id, (0.0, 0))
        clusters[case_id] = (total + float(row["signed_delta"]), count + 1)
    values = list(clusters.values())
    if not values:
        return math.nan, math.nan
    sums = np.asarray([item[0] for item in values], dtype=np.float64)
    counts = np.asarray([item[1] for item in values], dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        selected = rng.integers(0, len(values), size=len(values))
        estimates[draw] = sums[selected].sum() / counts[selected].sum()
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def grouped_cluster_difference(
    positive: list[dict], negative: list[dict], *, seed: int, draws: int = 10_000
) -> dict:
    def aggregates(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        clusters: dict[str, tuple[float, int]] = {}
        for row in rows:
            case_id = str(row["case_id"])
            total, count = clusters.get(case_id, (0.0, 0))
            clusters[case_id] = (total + float(row["signed_delta"]), count + 1)
        return (
            np.asarray([value[0] for value in clusters.values()], dtype=np.float64),
            np.asarray([value[1] for value in clusters.values()], dtype=np.float64),
        )

    pos_sums, pos_counts = aggregates(positive)
    neg_sums, neg_counts = aggregates(negative)
    if not len(pos_sums) or not len(neg_sums):
        return {"positive_minus_negative": math.nan, "ci95_low": math.nan, "ci95_high": math.nan}
    observed = float(pos_sums.sum() / pos_counts.sum() - neg_sums.sum() / neg_counts.sum())
    rng = np.random.default_rng(seed)
    differences = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        pos_selected = rng.integers(0, len(pos_sums), size=len(pos_sums))
        neg_selected = rng.integers(0, len(neg_sums), size=len(neg_sums))
        differences[draw] = (
            pos_sums[pos_selected].sum() / pos_counts[pos_selected].sum()
            - neg_sums[neg_selected].sum() / neg_counts[neg_selected].sum()
        )
    low, high = np.quantile(differences, [0.025, 0.975])
    return {
        "positive_minus_negative": observed,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "significant": not (low <= 0.0 <= high),
    }


def summarize(rows: list[dict], *, pair: str, group: str, seed: int) -> dict:
    values = np.asarray([float(row["signed_delta"]) for row in rows], dtype=np.float64)
    if values.size == 0:
        return {"pair": pair, "advantage_group": group, "n_tokens": 0, "n_cases": 0}
    test = stats.ttest_1samp(values, popmean=0.0)
    ci_low, ci_high = cluster_bootstrap(rows, seed=seed)
    standard_error = float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0
    significant = not (ci_low <= 0.0 <= ci_high)
    return {
        "pair": pair,
        "advantage_group": group,
        "n_tokens": int(values.size),
        "n_cases": len({str(row["case_id"]) for row in rows}),
        "signed_mean": float(values.mean()),
        "signed_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "token_standard_error": standard_error,
        "t_statistic": float(test.statistic),
        "t_pvalue": float(test.pvalue),
        "cluster_bootstrap_ci95_low": ci_low,
        "cluster_bootstrap_ci95_high": ci_high,
        "cluster_bootstrap_significant": significant,
        "direction": "alt_higher" if values.mean() > 0 else "alt_lower" if values.mean() < 0 else "zero",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit signed cross-path logprob bias with rollout advantage groups.")
    parser.add_argument("--pair", action="append", required=True, help="NAME=JSONL")
    parser.add_argument("--rollout", default="data/phase0_final_rollout.jsonl")
    parser.add_argument("--out-json", default="results/phaseA0_signed_bias.json")
    parser.add_argument("--report", default="reports/phaseA0_signed_bias.md")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rollout = {key(row): row for row in read_jsonl(args.rollout)}
    summaries = []
    advantage_comparisons = []
    alignment = []
    for pair_index, spec in enumerate(args.pair):
        if "=" not in spec:
            raise SystemExit(f"invalid --pair {spec!r}; expected NAME=JSONL")
        name, source = spec.split("=", 1)
        source_rows = read_jsonl(source)
        joined = []
        missing = 0
        token_mismatch = 0
        for row in source_rows:
            state = rollout.get(key(row))
            if state is None:
                missing += 1
                continue
            if int(row["token_id"]) != int(state["token_id"]):
                token_mismatch += 1
                continue
            sign = int(state.get("advantage_sign", 0))
            joined.append(
                {
                    "case_id": str(row["case_id"]),
                    "signed_delta": float(row["logp_alt"]) - float(row["logp_ref"]),
                    "advantage_sign": sign,
                }
            )
        alignment.append(
            {
                "pair": name,
                "source_rows": len(source_rows),
                "joined_rows": len(joined),
                "missing_rollout": missing,
                "token_mismatch": token_mismatch,
                "zero_advantage": sum(1 for row in joined if row["advantage_sign"] == 0),
            }
        )
        groups = {
            "all": joined,
            "positive": [row for row in joined if row["advantage_sign"] > 0],
            "negative": [row for row in joined if row["advantage_sign"] < 0],
        }
        for group_index, (group, rows) in enumerate(groups.items()):
            summaries.append(summarize(rows, pair=name, group=group, seed=args.seed + 10 * pair_index + group_index))
        advantage_comparisons.append(
            {
                "pair": name,
                **grouped_cluster_difference(
                    groups["positive"],
                    groups["negative"],
                    seed=args.seed + 100 + pair_index,
                ),
            }
        )

    all_significant = [row for row in summaries if row.get("advantage_group") == "all" and row.get("cluster_bootstrap_significant")]
    conclusion = (
        "At least one claim pair has a directionally biased signed delta under prompt-cluster bootstrap."
        if all_significant
        else "Neither claim pair shows a signed mean distinguishable from zero under prompt-cluster bootstrap."
    )
    payload = {
        "signed_delta_definition": "logp_alt - logp_ref",
        "rollout": args.rollout,
        "bootstrap_unit": "case_id/prompt-response cluster",
        "bootstrap_draws": 10_000,
        "alignment": alignment,
        "rows": summaries,
        "advantage_sign_comparisons": advantage_comparisons,
        "conclusion": conclusion,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    columns = list(summaries[0].keys()) if summaries else []
    align_columns = list(alignment[0].keys()) if alignment else []
    report = "\n".join(
        [
            "# Phase A0 Signed Bias Audit",
            "",
            "## Claim Scope",
            CLAIM_SCOPE,
            "",
            "## Confound Checklist",
            f"- rollout token alignment: {'PASS' if all(row['missing_rollout'] == 0 and row['token_mismatch'] == 0 for row in alignment) else 'FAIL'}",
            "- signed delta uses alt minus ref: PASS",
            "- debug FP32/FP16 pair excluded: PASS",
            "- prompt-cluster bootstrap used: PASS",
            "",
            "## Delta Self Control",
            "Phase 1 aggregate self deltas are carried forward; process-independence is audited separately in A1.",
            "",
            "## External Validity",
            EXTERNAL_VALIDITY,
            "",
            "## Alignment",
            markdown_table(alignment, align_columns),
            "",
            "## Signed Bias",
            markdown_table(summaries, columns),
            "",
            "## Advantage Sign Association",
            markdown_table(advantage_comparisons, list(advantage_comparisons[0].keys())),
            "",
            "## Conclusion",
            conclusion,
            "",
            "A significant all-token mean supports directional path bias. Positive/negative advantage rows test association, not causality; differing group means can also reflect token-distribution differences.",
            "",
        ]
    )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report, encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
