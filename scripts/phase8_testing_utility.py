#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from forkcert.io import read_jsonl


def confusion(rows, predicate):
    tp = sum(row["label"] and predicate(row) for row in rows)
    fp = sum((not row["label"]) and predicate(row) for row in rows)
    fn = sum(row["label"] and not predicate(row) for row in rows)
    return {
        "alarms": tp + fp,
        "true_anomalies": tp,
        "false_alarms": fp,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare tolerance, delta ranking and fork signal for empirical anomaly triage.")
    parser.add_argument("--legal", default="results/phase4_certificates.jsonl")
    parser.add_argument("--injected", default="results/phase5_empirical_bug_rows.jsonl")
    parser.add_argument("--out", default="results/phase8_testing_utility.json")
    parser.add_argument("--report")
    args = parser.parse_args()
    legal = [
        {"label": 0, "delta": float(row["logprob_delta"]), "fork": bool(row["actual_fork"]), "kind": "legal_eager_compile"}
        for row in read_jsonl(args.legal) if int(row.get("advantage_sign", 0)) != 0
    ]
    injected = [
        {"label": 1, "delta": float(row["logprob_delta"]), "fork": bool(row["actual_clip_branch_fork"]), "kind": row["bug"]}
        for row in read_jsonl(args.injected)
    ]
    rows = legal + injected
    methods = {"fork_signal": confusion(rows, lambda row: row["fork"])}
    for threshold in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        methods[f"abs_delta_gt_{threshold:g}"] = confusion(rows, lambda row, threshold=threshold: row["delta"] > threshold)
    ranked = sorted(rows, key=lambda row: row["delta"], reverse=True)
    budgets = {}
    fork_alarms = int(methods["fork_signal"]["alarms"])
    for budget in sorted({100, 500, 1000, 2000, 5000, fork_alarms}):
        selected = ranked[:budget]
        found = sum(row["label"] for row in selected)
        budgets[str(budget)] = {"anomalies_found": found, "discovery_rate": found / budget, "recall": found / len(injected)}
    by_injection = {}
    for kind in sorted({row["kind"] for row in injected}):
        subset = [row for row in injected if row["kind"] == kind]
        by_injection[kind] = {"rows": len(subset), "forks": sum(row["fork"] for row in subset), "fork_recall": sum(row["fork"] for row in subset) / len(subset)}
    mutation_kinds = sorted({row["kind"] for row in injected})
    matched = budgets[str(fork_alarms)]
    payload = {
        "schema_version": "forkcert.testing_utility.v1",
        "legal_rows": len(legal),
        "injected_rows": len(injected),
        "mutation_kinds": len(mutation_kinds),
        "methods": methods,
        "delta_ranking_budgets": budgets,
        "matched_alarm_budget": {
            "alarms": fork_alarms,
            "fork_signal_precision": methods["fork_signal"]["precision"],
            "fork_signal_recall": methods["fork_signal"]["recall"],
            "delta_ranking_precision": matched["discovery_rate"],
            "delta_ranking_recall": matched["recall"],
            "delta_ranking_dominates_on_this_catalog": (
                matched["discovery_rate"] >= methods["fork_signal"]["precision"]
                and matched["recall"] >= methods["fork_signal"]["recall"]
            ),
        },
        "fork_recall_by_injection": by_injection,
        "label_contract": "Only artificial mutation rows are positive. Legal eager/compile rows, including audited natural semantic forks, are negative for this mutation-classification task.",
        "claim_scope": f"Empirical triage of {len(mutation_kinds)} artificial altered operations; injected rows are not certified or historical real bugs.",
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.report:
        fork = methods["fork_signal"]
        lines = [
            "# Phase 9 Testing Utility",
            "",
            "## Objective",
            "",
            "Compare clipping-fork alarms with fixed absolute-delta thresholds and delta ranking for identifying artificial execution mutations.",
            "",
            "## Label Contract",
            "",
            payload["label_contract"],
            "This is intentionally different from semantic-risk detection, where an audited natural fork is itself a positive event.",
            "",
            "## Result",
            "",
            f"The evaluation contains `{len(legal)}` legal-path rows and `{len(injected)}` rows from `{len(mutation_kinds)}` artificial mutations.",
            "",
            "| Method | Alarms | Precision | Recall |",
            "|---|---:|---:|---:|",
            f"| Fork signal | {fork['alarms']} | {fork['precision']:.4f} | {fork['recall']:.4f} |",
        ]
        for name, item in methods.items():
            if not name.startswith("abs_delta"):
                continue
            lines.append(f"| `{name}` | {item['alarms']} | {item['precision']:.4f} | {item['recall']:.4f} |")
        lines.extend(
            [
                "",
                "## Matched Alarm Budget",
                "",
                f"At `{fork_alarms}` alarms, delta ranking has precision `{matched['discovery_rate']:.4f}` and recall `{matched['recall']:.4f}`. Fork signal has precision `{fork['precision']:.4f}` and recall `{fork['recall']:.4f}`.",
                "",
                "On this catalog, delta ranking dominates fork signal for artificial-mutation classification at the matched alarm budget. The preregistered expectation that fork would have higher precision is not supported.",
                "",
                "## Interpretation",
                "",
                "Fork signal remains a tolerance-free indicator that a training decision changed, but that property does not make it a universal classifier of implementation mutations. It deliberately ignores mutations that alter numerical amplitude without crossing the frozen clipping boundary.",
                "",
                "The five audited eager/compile natural forks count as false alarms under the mutation label contract, even though they are true semantic-risk events. Therefore this result evaluates testing utility for mutation identification, not the validity of the natural-fork existence claim.",
                "",
                "## Artifacts",
                "",
                f"- `{args.out}`",
                f"- `{args.injected}`",
                f"- `{args.legal}`",
                "",
            ]
        )
        Path(args.report).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
