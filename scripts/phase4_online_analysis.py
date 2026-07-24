#!/usr/bin/env python
from __future__ import annotations

import argparse
import bisect
import json
import math
from collections import defaultdict
from pathlib import Path

from forkcert.detector import clip_boundary
from forkcert.io import read_jsonl
from forkcert.report import CLAIM_SCOPE, markdown_table
from forkcert.stats import mean, percentile


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return numerator / (dx * dy) if dx and dy else None


def predicted_rate(margins: list[float], deltas: list[float]) -> float:
    ordered = sorted(margins)
    return sum(bisect.bisect_left(ordered, delta) / len(ordered) for delta in deltas) / len(deltas)


def delta_stats(rows: list[dict]) -> dict:
    values = [float(row["logprob_delta"]) for row in rows]
    return {
        "tokens": len(values),
        "delta_mean": mean(values) if values else None,
        "delta_p50": percentile(values, 50) if values else None,
        "delta_p99": percentile(values, 99) if values else None,
        "delta_max": max(values) if values else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze an online state-aligned Phase 4 scan.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--expected-tokens", type=int, default=51200)
    parser.add_argument("--out-json", default="results/phase4_online_analysis.json")
    parser.add_argument("--report", default="reports/phase4_online_analysis.md")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    applicable = [row for row in rows if int(row.get("advantage_sign", 0)) != 0]
    margins = []
    signed_deltas = []
    actual = []
    possible = []
    by_rollout: dict[int, list[dict]] = defaultdict(list)
    for row in applicable:
        sign = int(row["advantage_sign"])
        boundary = clip_boundary(sign, args.eps)
        ref_ratio_log = float(row["logp_ref"]) - float(row["old_logp"])
        alt_ratio_log = float(row["logp_alt"]) - float(row["old_logp"])
        margin = abs(ref_ratio_log - boundary)
        margins.append(margin)
        signed_deltas.append(float(row["logp_alt"]) - float(row["logp_ref"]))
        possible.append(float(row["logprob_delta"]) >= margin)
        clip_ref = ref_ratio_log > boundary if sign > 0 else ref_ratio_log < boundary
        clip_alt = alt_ratio_log > boundary if sign > 0 else alt_ratio_log < boundary
        actual.append(clip_ref != clip_alt)
        by_rollout[int(row["rollout_batch"])].append({**row, "clip_margin": margin, "actual_fork": clip_ref != clip_alt})

    deltas = [float(row["logprob_delta"]) for row in applicable]
    self_ref = [float(row["delta_self_ref"]) for row in applicable]
    self_alt = [float(row["delta_self_alt"]) for row in applicable]
    p50_cross = percentile(deltas, 50) if deltas else 0.0
    p99_self_ref = percentile(self_ref, 99) if self_ref else 0.0
    p99_self_alt = percentile(self_alt, 99) if self_alt else 0.0
    independent = predicted_rate(margins, deltas) if margins and deltas else 0.0
    near = [row for row, margin in zip(applicable, margins, strict=True) if margin < 1e-2]
    far = [row for row, margin in zip(applicable, margins, strict=True) if margin >= 1e-2]
    positive_bias = [delta for delta, row in zip(signed_deltas, applicable, strict=True) if int(row["advantage_sign"]) > 0]
    negative_bias = [delta for delta, row in zip(signed_deltas, applicable, strict=True) if int(row["advantage_sign"]) < 0]
    exact_zero_self = max(self_ref, default=0.0) == 0.0 and max(self_alt, default=0.0) == 0.0
    scalar_self_gate = p50_cross > 0 and p99_self_ref < 0.1 * p50_cross and p99_self_alt < 0.1 * p50_cross
    degenerate_zero_gate = p50_cross == 0 and exact_zero_self and any(delta > 0 for delta in deltas)
    total_rollouts = len({int(row["rollout_batch"]) for row in rows})
    compute_dtypes = sorted(
        {
            str(row.get("training_compute_dtype") or ("bf16" if "-bf16-" in str(row.get("path_ref")) else "fp16"))
            for row in rows
        }
    )
    summary = {
        "tokens": len(applicable),
        "total_rows": len(rows),
        "expected_tokens": args.expected_tokens,
        "coverage_complete": len(rows) == args.expected_tokens,
        "rollouts": total_rollouts,
        "training_compute_dtypes": compute_dtypes,
        "rollouts_with_applicable_decisions": len(by_rollout),
        "zero_advantage_rows": len(rows) - len(applicable),
        "self_ref_p99": p99_self_ref,
        "self_alt_p99": p99_self_alt,
        "cross_p50": p50_cross,
        "self_gate": scalar_self_gate or degenerate_zero_gate,
        "self_gate_rule": "original_strict_ratio" if scalar_self_gate else "exact_zero_self_with_nonzero_cross" if degenerate_zero_gate else "failed",
        "original_strict_ratio_gate_degenerate": p50_cross == 0,
        "near_boundary_tokens_margin_lt_1e_2": len(near),
        "fork_possible_count": sum(possible),
        "actual_fork_count": sum(actual),
        "actual_fork_rate": sum(actual) / len(actual) if actual else 0.0,
        "independent_convolution_predicted_rate": independent,
        "independent_convolution_predicted_count": independent * len(applicable),
        "observed_minus_predicted_count": sum(actual) - independent * len(applicable),
        "pearson_log_margin_vs_log_delta": pearson(
            [math.log10(max(value, 1e-30)) for value in margins],
            [math.log10(max(value, 1e-30)) for value in deltas],
        ),
        "signed_delta_mean_all": mean(signed_deltas) if signed_deltas else None,
        "signed_delta_mean_positive_advantage": mean(positive_bias) if positive_bias else None,
        "signed_delta_mean_negative_advantage": mean(negative_bias) if negative_bias else None,
    }
    conditional = [
        {"group": "margin_lt_1e-2", **delta_stats(near)},
        {"group": "margin_ge_1e-2", **delta_stats(far)},
    ]
    rollout_rows = []
    for rollout, group in sorted(by_rollout.items()):
        rollout_rows.append(
            {
                "rollout_batch": rollout,
                "optimizer_step": min(int(row["optimizer_step"]) for row in group),
                "tokens": len(group),
                "near_boundary": sum(float(row["clip_margin"]) < 1e-2 for row in group),
                "fork_possible": sum(float(row["logprob_delta"]) >= float(row["clip_margin"]) for row in group),
                "actual_forks": sum(bool(row["actual_fork"]) for row in group),
            }
        )
    payload = {"summary": summary, "conditional_delta": conditional, "by_rollout": rollout_rows}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = "\n".join(
        [
            "# Phase 4 Online State-Aligned Analysis",
            "",
            "## Claim Scope",
            CLAIM_SCOPE,
            "",
            "## Confound Checklist",
            "- every row marked online_state_aligned: " + ("PASS" if all(row.get("online_state_aligned") is True for row in rows) else "FAIL"),
            "- every row is policy_iteration=2 pre-minibatch: " + ("PASS" if all(int(row.get("policy_iteration", -1)) == 2 and row.get("state") == "pre_minibatch" for row in rows) else "FAIL"),
            "- attention backend locked to MATH: " + ("PASS" if all(row.get("attention_backend_locked") == "MATH" for row in rows) else "FAIL"),
            "- online self separation: " + ("PASS" if summary["self_gate"] else "FAIL"),
            "- expected token coverage: " + ("PASS" if summary["coverage_complete"] else "INCOMPLETE"),
            "",
            "## Delta Self Control",
            f"ref p99={p99_self_ref:.6g}, alt p99={p99_self_alt:.6g}, cross p50={p50_cross:.6g}. "
            "When cross p50 is zero, the original strict-ratio scalar gate is degenerate; pass requires both self maxima to be exactly zero and at least one nonzero cross delta.",
            "",
            "## External Validity",
            "This online scan reports compute dtype(s) " + str(compute_dtypes) + ". Hardware identity and capability must be read from the aligned training metadata sidecar; conclusions remain conditional on that environment.",
            "",
            "## Summary",
            markdown_table([summary], list(summary.keys())),
            "",
            "## Conditional Delta",
            markdown_table(conditional, list(conditional[0].keys())),
            "",
            "## By Rollout",
            markdown_table(rollout_rows, list(rollout_rows[0].keys()) if rollout_rows else []),
            "",
        ]
    )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
