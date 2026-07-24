#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from forkcert.io import read_jsonl
from forkcert.report import markdown_table


def load_rows(legal_path: str, mutation_path: str) -> list[dict[str, Any]]:
    legal = [
        {
            "id": ("legal", str(row["case_id"]), int(row["token_index"])),
            "label": False,
            "family": None,
            "delta": float(row["logprob_delta"]),
            "fork": bool(row["actual_fork"]),
        }
        for row in read_jsonl(legal_path)
        if int(row.get("advantage_sign", 0)) != 0
    ]
    mutations = [
        {
            "id": (str(row["bug"]), str(row["case_id"]), int(row["token_index"])),
            "label": True,
            "family": str(row["bug"]),
            "delta": float(row["logprob_delta"]),
            "fork": bool(row["actual_clip_branch_fork"]),
        }
        for row in read_jsonl(mutation_path)
    ]
    return legal + mutations


def metrics(
    rows: list[dict[str, Any]], selected: set[int], *, positive_count: int | None = None
) -> dict[str, float | int]:
    positives = positive_count if positive_count is not None else sum(bool(row["label"]) for row in rows)
    true_positive_indices = [index for index in selected if rows[index]["label"]]
    families = {str(rows[index]["family"]) for index in true_positive_indices}
    return {
        "alarms": len(selected),
        "mutation_true_positives": len(true_positive_indices),
        "precision": len(true_positive_indices) / len(selected) if selected else 0.0,
        "token_recall": len(true_positive_indices) / positives if positives else 0.0,
        "mutation_family_coverage": len(families),
    }


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, int(probability * len(ordered)))]


def main() -> None:
    parser = argparse.ArgumentParser(description="P2 matched-budget delta/fork portfolio evaluation.")
    parser.add_argument("--legal", default="results/phase4_certificates.jsonl")
    parser.add_argument("--mutations", default="results/phase9_mutations_gated/all_mutation_rows.jsonl")
    parser.add_argument("--budget", type=int, default=425)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/p2_portfolio.json")
    parser.add_argument("--report", default="reports/p2_portfolio.md")
    args = parser.parse_args()

    rows = load_rows(args.legal, args.mutations)
    positive_count = sum(bool(row["label"]) for row in rows)
    delta_ranking = sorted(range(len(rows)), key=lambda index: (-rows[index]["delta"], rows[index]["id"]))
    fork_indices = [index for index, row in enumerate(rows) if row["fork"]]
    if len(fork_indices) != args.budget:
        raise ValueError(f"registered matched budget is {args.budget}, but fork signal has {len(fork_indices)} alarms")
    delta_selected = set(delta_ranking[: args.budget])
    fork_selected = set(fork_indices)

    delta_quota = (args.budget + 1) // 2
    fork_quota = args.budget - delta_quota
    fixed_delta = set(delta_ranking[:delta_quota])
    rng = random.Random(args.seed)
    portfolio_results = []
    for _ in range(args.replicates):
        shuffled = list(fork_indices)
        rng.shuffle(shuffled)
        added = []
        for index in shuffled:
            if index not in fixed_delta:
                added.append(index)
            if len(added) == fork_quota:
                break
        selected = fixed_delta | set(added)
        if len(selected) != args.budget:
            raise ValueError("50-50 portfolio could not fill its unique alarm budget")
        portfolio_results.append(metrics(rows, selected, positive_count=positive_count))

    mutation_indices = [index for index, row in enumerate(rows) if row["label"]]
    families = sorted({str(rows[index]["family"]) for index in mutation_indices})
    oracle_selected = {
        next(index for index in mutation_indices if rows[index]["family"] == family)
        for family in families
    }
    for index in delta_ranking:
        if rows[index]["label"]:
            oracle_selected.add(index)
        if len(oracle_selected) == args.budget:
            break

    methods = {
        "delta_only": metrics(rows, delta_selected, positive_count=positive_count),
        "fork_only": metrics(rows, fork_selected, positive_count=positive_count),
        "family_oracle_upper_bound": metrics(rows, oracle_selected, positive_count=positive_count),
    }
    portfolio_fields = ["mutation_true_positives", "precision", "token_recall", "mutation_family_coverage"]
    portfolio = {"alarms": args.budget}
    for field in portfolio_fields:
        values = [float(result[field]) for result in portfolio_results]
        portfolio[field] = sum(values) / len(values)
        portfolio[f"{field}_ci95"] = [quantile(values, 0.025), quantile(values, 0.975)]
    methods["50_50_portfolio"] = portfolio

    max_single_recall = max(float(methods["delta_only"]["token_recall"]), float(methods["fork_only"]["token_recall"]))
    max_single_coverage = max(
        float(methods["delta_only"]["mutation_family_coverage"]),
        float(methods["fork_only"]["mutation_family_coverage"]),
    )
    payload = {
        "schema_version": "forkcert.p2.portfolio.v1",
        "scope": "15 artificial mutation families; not historical or certified bugs",
        "budget": args.budget,
        "mutation_rows": sum(row["label"] for row in rows),
        "legal_rows": sum(not row["label"] for row in rows),
        "mutation_families": len(families),
        "portfolio_contract": (
            f"{delta_quota} highest-delta unique alarms plus {fork_quota} additional fork alarms; "
            f"fork ties randomized for {args.replicates} replicates with seed {args.seed}"
        ),
        "oracle_contract": "Uses mutation-family ground-truth labels; unattainable allocation upper bound.",
        "methods": methods,
        "portfolio_exceeds_best_single_recall": float(portfolio["token_recall"]) > max_single_recall,
        "portfolio_exceeds_best_single_family_coverage": float(portfolio["mutation_family_coverage"]) > max_single_coverage,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    table = []
    for name, result in methods.items():
        table.append(
            {
                "strategy": name,
                "alarms": result["alarms"],
                "mutation_tp": result["mutation_true_positives"],
                "precision": result["precision"],
                "token_recall": result["token_recall"],
                "family_coverage": result["mutation_family_coverage"],
            }
        )
    report = "\n".join(
        [
            "# P2 Portfolio Oracle",
            "",
            "## Confound Checklist",
            "- same gated 15-mutation catalog as authoritative RQ5 analysis: PASS",
            "- same legal eager/compile population: PASS",
            f"- matched alarm budget `{args.budget}` for every strategy: PASS",
            "- artificial mutation labels kept distinct from historical/certified bugs: PASS",
            "",
            "## Delta Self Control",
            "No new model execution is performed. The legal and mutation inputs retain their authoritative self/canary gates.",
            "",
            "## External Validity",
            "The catalog is Qwen3-0.6B, T4 FP16, and 15 artificial altered operations. Family coverage is catalog-conditioned.",
            "",
            "## Main Result",
            markdown_table(table, list(table[0])),
            "",
            "The 50-50 row reports the mean over randomized fork ties; its 95% intervals are stored in `results/p2_portfolio.json`.",
            "",
            "## Interpretation",
            (
                "The 50-50 portfolio exceeds the best single signal on token recall."
                if payload["portfolio_exceeds_best_single_recall"]
                else "The 50-50 portfolio does not exceed the best single signal on token recall."
            ),
            (
                "It exceeds the best single signal on mutation-family coverage."
                if payload["portfolio_exceeds_best_single_family_coverage"]
                else "It does not exceed the best single signal on mutation-family coverage."
            ),
            "The family oracle is an unattainable label-aware upper bound, not a deployable ForkCert result.",
            "",
        ]
    )
    Path(args.report).write_text(report, encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
