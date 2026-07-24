#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from forkcert.detector import clip_boundary
from forkcert.io import read_jsonl
from forkcert.report import CLAIM_SCOPE, markdown_table
from forkcert.stats import percentile


def main() -> None:
    parser = argparse.ArgumentParser(description="Conditional stable/unknown coverage from a log-softmax bound.")
    parser.add_argument("--bound", required=True)
    parser.add_argument("--margin-jsonl", required=True)
    parser.add_argument("--policy-iteration", type=int, default=2)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--out", default="results/phase2_logsoftmax_coverage.json")
    parser.add_argument("--report", default="reports/phase2_logsoftmax_coverage.md")
    args = parser.parse_args()

    bound = json.loads(Path(args.bound).read_text(encoding="utf-8"))
    value = float(bound["deterministic_conditional_bound"])
    rows = [
        row
        for row in read_jsonl(args.margin_jsonl)
        if int(row.get("policy_iteration", -1)) == args.policy_iteration
        and int(row.get("advantage_sign", 0)) != 0
    ]
    margins = [
        abs(
            (float(row["new_logp"]) - float(row["old_logp"]))
            - clip_boundary(int(row["advantage_sign"]), args.eps)
        )
        for row in rows
    ]
    stable = sum(margin > value for margin in margins)
    result = {
        "schema_version": "forkcert.phase2.logsoftmax_coverage.v1",
        "status": "completed",
        "bound_kind": bound["certificate_kind"],
        "analytic_legal": bool(bound["analytic_legal"]),
        "conditional_bound": value,
        "policy_iteration": args.policy_iteration,
        "applicable_decisions": len(margins),
        "conditional_stable_count": stable,
        "conditional_unknown_count": len(margins) - stable,
        "conditional_stable_rate": stable / len(margins) if margins else None,
        "margin_min": min(margins) if margins else None,
        "margin_p1": percentile(margins, 1) if margins else None,
        "margin_p5": percentile(margins, 5) if margins else None,
        "classification_permission": "conditional stable/unknown only; no fragile or bug labels",
        "same_state_delta_measured": False,
        "reason_not_fragile": (
            "Rows inside the envelope lack same-state L4 alternate logprobs and the bound is not analytic_legal."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    keys = [
        "bound_kind",
        "conditional_bound",
        "applicable_decisions",
        "conditional_stable_count",
        "conditional_unknown_count",
        "conditional_stable_rate",
        "analytic_legal",
    ]
    report = "\n".join(
        [
            "# Phase 2 Log-Softmax Conditional Coverage",
            "",
            "## Claim Scope",
            CLAIM_SCOPE,
            "",
            "## Confound Checklist",
            "- real iteration-2 old/new logprobs and advantages: PASS",
            "- clipping sign branches applied separately: PASS",
            "- same-state L4 alternate logprobs: FAIL / not recorded",
            "- analytic legal bound: FAIL",
            "- fragile or bug labels emitted: PASS / prohibited",
            "",
            "## Delta Self Control",
            "The L4 isolation paths are bitwise self-consistent. This coverage calculation consumes margins only and does not invent an alternate-path delta.",
            "",
            "## Summary",
            markdown_table([{key: result[key] for key in keys}], keys),
            "",
            "## Interpretation",
            "`margin > B_conditional` is reported as conditional stable coverage under the documented CUDA assumptions. Remaining rows are unknown, not fragile. The result cannot support bug classification.",
            "",
            "## External Validity",
            "The envelope and margins are T4 FP16 results. Native BF16 requires new kernels, deltas, and bounds.",
            "",
        ]
    )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report, encoding="utf-8")
    print(json.dumps({key: result[key] for key in keys}, indent=2))


if __name__ == "__main__":
    main()
