#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from forkcert.io import read_jsonl


def exact_mcnemar_pvalue(fork_only_correct: int, delta_only_correct: int) -> float:
    discordant = fork_only_correct + delta_only_correct
    if discordant == 0:
        return 1.0
    lower = min(fork_only_correct, delta_only_correct)
    tail = sum(math.comb(discordant, value) for value in range(lower + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def exact_sign_flip_pvalue(differences: list[int]) -> float:
    observed = abs(sum(differences))
    extreme = 0
    for mask in range(1 << len(differences)):
        value = sum((1 if mask & (1 << index) else -1) * difference for index, difference in enumerate(differences))
        extreme += abs(value) >= observed
    return extreme / (1 << len(differences))


def percentile_interval(values: list[float]) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def metric(label: np.ndarray, selected: np.ndarray) -> tuple[float, float]:
    true_positives = int(label[selected].sum())
    alarms = int(selected.sum())
    positives = int(label.sum())
    return true_positives / alarms if alarms else float("nan"), true_positives / positives if positives else float("nan")


def clustered_bootstrap(
    rows: list[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    labels = np.asarray([row["label"] for row in rows], dtype=np.int8)
    deltas = np.asarray([row["delta"] for row in rows], dtype=np.float64)
    forks = np.asarray([row["fork"] for row in rows], dtype=bool)
    legal_clusters: dict[str, list[int]] = defaultdict(list)
    mutation_clusters: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        target = mutation_clusters if row["label"] else legal_clusters
        target[row["case_id"]].append(index)
    legal_ids = sorted(legal_clusters)
    mutation_ids = sorted(mutation_clusters)
    legal_arrays = [np.asarray(legal_clusters[key], dtype=np.int64) for key in legal_ids]
    mutation_arrays = [np.asarray(mutation_clusters[key], dtype=np.int64) for key in mutation_ids]
    rng = np.random.default_rng(seed)
    precision_differences = []
    recall_differences = []
    for _ in range(replicates):
        sampled_legal = rng.integers(0, len(legal_arrays), size=len(legal_arrays))
        sampled_mutation = rng.integers(0, len(mutation_arrays), size=len(mutation_arrays))
        indices = np.concatenate(
            [*(legal_arrays[index] for index in sampled_legal), *(mutation_arrays[index] for index in sampled_mutation)]
        )
        sample_labels = labels[indices]
        sample_deltas = deltas[indices]
        sample_forks = forks[indices]
        budget = int(sample_forks.sum())
        if budget <= 0:
            continue
        selected_delta = np.zeros(indices.size, dtype=bool)
        chosen = np.argpartition(sample_deltas, -budget)[-budget:]
        selected_delta[chosen] = True
        fork_precision, fork_recall = metric(sample_labels, sample_forks)
        delta_precision, delta_recall = metric(sample_labels, selected_delta)
        precision_differences.append(delta_precision - fork_precision)
        recall_differences.append(delta_recall - fork_recall)
    return {
        "replicates_requested": replicates,
        "replicates_valid": len(precision_differences),
        "seed": seed,
        "cluster_contract": (
            "Resample legal case_id clusters and the four aligned mutation case_id clusters independently; "
            "all 15 mutations for a sampled mutation case remain together. The mutation catalog is conditioned on, not resampled."
        ),
        "legal_clusters": len(legal_ids),
        "mutation_case_clusters": len(mutation_ids),
        "delta_minus_fork_precision_ci95": percentile_interval(precision_differences),
        "delta_minus_fork_recall_ci95": percentile_interval(recall_differences),
        "probability_delta_precision_greater": float(np.mean(np.asarray(precision_differences) > 0)),
        "probability_delta_recall_greater": float(np.mean(np.asarray(recall_differences) > 0)),
    }


def analyze(legal_path: str, injected_path: str, *, bootstrap_replicates: int, seed: int) -> dict[str, Any]:
    legal = [
        {
            "id": ("legal", str(row["case_id"]), int(row["token_index"])),
            "case_id": str(row["case_id"]),
            "kind": "legal_eager_compile",
            "label": 0,
            "delta": float(row["logprob_delta"]),
            "fork": bool(row["actual_fork"]),
        }
        for row in read_jsonl(legal_path)
        if int(row.get("advantage_sign", 0)) != 0
    ]
    injected = [
        {
            "id": (str(row["bug"]), str(row["case_id"]), int(row["token_index"])),
            "case_id": str(row["case_id"]),
            "kind": str(row["bug"]),
            "label": 1,
            "delta": float(row["logprob_delta"]),
            "fork": bool(row["actual_clip_branch_fork"]),
        }
        for row in read_jsonl(injected_path)
    ]
    rows = legal + injected
    budget = sum(row["fork"] for row in rows)
    ranked_indices = sorted(range(len(rows)), key=lambda index: (-rows[index]["delta"], rows[index]["id"]))
    delta_selected = set(ranked_indices[:budget])
    cutoff = rows[ranked_indices[budget - 1]]["delta"]
    fork_only_correct = 0
    delta_only_correct = 0
    both_correct = 0
    both_wrong = 0
    discordance = Counter()
    true_positives_by_kind: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for index, row in enumerate(rows):
        fork_prediction = bool(row["fork"])
        delta_prediction = index in delta_selected
        label = bool(row["label"])
        fork_correct = fork_prediction == label
        delta_correct = delta_prediction == label
        if fork_correct and delta_correct:
            both_correct += 1
        elif fork_correct:
            fork_only_correct += 1
        elif delta_correct:
            delta_only_correct += 1
        else:
            both_wrong += 1
        discordance[(fork_prediction, delta_prediction, label)] += 1
        if label:
            true_positives_by_kind[row["kind"]][0] += int(fork_prediction)
            true_positives_by_kind[row["kind"]][1] += int(delta_prediction)
    per_kind = {
        kind: {"fork_true_positives": values[0], "delta_true_positives": values[1], "difference_fork_minus_delta": values[0] - values[1]}
        for kind, values in sorted(true_positives_by_kind.items())
    }
    fork_tp = sum(values[0] for values in true_positives_by_kind.values())
    delta_tp = sum(values[1] for values in true_positives_by_kind.values())
    fork_kinds = sum(values[0] > 0 for values in true_positives_by_kind.values())
    delta_kinds = sum(values[1] > 0 for values in true_positives_by_kind.values())
    fork_kind_counts = sorted((values[0] for values in true_positives_by_kind.values()), reverse=True)
    delta_kind_counts = sorted((values[1] for values in true_positives_by_kind.values()), reverse=True)
    return {
        "schema_version": "forkcert.testing_utility_significance.v1",
        "scope": "Artificial-mutation identification at the fork signal's matched alarm budget.",
        "rows": {"legal": len(legal), "injected": len(injected), "total": len(rows)},
        "matched_budget": budget,
        "delta_cutoff": cutoff,
        "delta_cutoff_ties": {
            "strictly_above": sum(row["delta"] > cutoff for row in rows),
            "equal": sum(row["delta"] == cutoff for row in rows),
        },
        "paired_correctness": {
            "both_correct": both_correct,
            "fork_only_correct": fork_only_correct,
            "delta_only_correct": delta_only_correct,
            "both_wrong": both_wrong,
            "exact_mcnemar_pvalue": exact_mcnemar_pvalue(fork_only_correct, delta_only_correct),
        },
        "prediction_discordance": {str(key): value for key, value in sorted(discordance.items(), key=str)},
        "mutation_true_positives": {"fork": fork_tp, "delta": delta_tp},
        "mutation_kind_coverage": {
            "total_kinds": len(true_positives_by_kind),
            "fork_kinds_with_true_positive": fork_kinds,
            "delta_kinds_with_true_positive": delta_kinds,
            "fork_top_kind_share": fork_kind_counts[0] / fork_tp if fork_tp else None,
            "delta_top_kind_share": delta_kind_counts[0] / delta_tp if delta_tp else None,
        },
        "per_mutation_kind": per_kind,
        "mutation_kind_sign_flip": {
            "observed_total_difference_fork_minus_delta": fork_tp - delta_tp,
            "exact_pvalue": exact_sign_flip_pvalue([item["difference_fork_minus_delta"] for item in per_kind.values()]),
        },
        "cluster_bootstrap": clustered_bootstrap(rows, replicates=bootstrap_replicates, seed=seed),
        "interpretation_contract": (
            "Token-level mutation identification and mutation-family coverage are different endpoints. "
            "Neither endpoint establishes downstream training harm; that requires an independent trajectory experiment."
        ),
    }


def render_report(payload: dict[str, Any]) -> str:
    paired = payload["paired_correctness"]
    coverage = payload["mutation_kind_coverage"]
    bootstrap = payload["cluster_bootstrap"]
    lines = [
        "# Phase 9 Testing Utility Significance Audit",
        "",
        "## Objective",
        "",
        "Test whether the matched-budget difference between clipping-fork alarms and absolute-delta ranking is statistically supported, and measure how each method allocates alarms across mutation families.",
        "",
        "## Paired Result",
        "",
        f"At `{payload['matched_budget']}` alarms, the paired correctness table has `{paired['fork_only_correct']}` fork-only correct rows and `{paired['delta_only_correct']}` delta-only correct rows. The exact McNemar p-value is `{paired['exact_mcnemar_pvalue']:.6f}`.",
        "",
        f"The cluster-bootstrap 95% interval for delta-minus-fork precision is `{bootstrap['delta_minus_fork_precision_ci95']}`; for recall it is `{bootstrap['delta_minus_fork_recall_ci95']}`. The bootstrap conditions on the 15-mutation catalog and resamples aligned case clusters.",
        "",
        "The small aggregate advantage previously assigned to delta ranking is not statistically distinguishable from zero under either paired token analysis or clustered resampling.",
        "",
        "## Mutation-Family Coverage",
        "",
        f"Fork alarms identify at least one row in `{coverage['fork_kinds_with_true_positive']}/{coverage['total_kinds']}` mutation kinds; delta top-budget alarms identify `{coverage['delta_kinds_with_true_positive']}/{coverage['total_kinds']}`.",
        "",
        f"The largest mutation family accounts for `{coverage['fork_top_kind_share']:.3%}` of fork true positives and `{coverage['delta_top_kind_share']:.3%}` of delta-ranking true positives. In this catalog, every delta top-budget true positive comes from the catastrophic `rmsnorm_no_upcast` mutation.",
        "",
        "This diversity endpoint favors fork alarms, but the 15 operators are artificial and not independent historical bugs. It supports a test-budget allocation claim, not a universal detection-superiority claim.",
        "",
        "## Ground-Truth Limitation",
        "",
        "The current labels ask whether a row came from an artificial mutation. They do not say whether that mutation changes short-horizon training behavior. Calling zero-clipping-fork mutations training-equivalent would therefore be premature. A matched trajectory comparison must independently measure gradient/update/trajectory effects and include other discrete events, especially sampling forks.",
        "",
        "## Artifacts",
        "",
        "- `results/phase9_testing_utility_significance.json`",
        "- `scripts/phase9_testing_utility_significance.py`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired and clustered significance audit for RQ5 testing utility.")
    parser.add_argument("--legal", default="results/phase4_certificates.jsonl")
    parser.add_argument("--injected", default="results/phase9_mutations_gated/all_mutation_rows.jsonl")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/phase9_testing_utility_significance.json")
    parser.add_argument("--report", default="reports/phase9_testing_utility_significance.md")
    args = parser.parse_args()
    payload = analyze(args.legal, args.injected, bootstrap_replicates=args.bootstrap_replicates, seed=args.seed)
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.report).write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "per_mutation_kind"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
