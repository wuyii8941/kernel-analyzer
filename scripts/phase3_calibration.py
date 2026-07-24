#!/usr/bin/env python
from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

from forkcert.detector import clip_boundary, detect_clipping_fork
from forkcert.io import read_jsonl, write_jsonl
from forkcert.report import markdown_table, write_phase_report
from forkcert.stats import percentile


def construct_old_logp(logp_ref: float, sign: int, eps: float, margin: float) -> float:
    return logp_ref - clip_boundary(sign, eps) + margin


def empirical_predicted_fork_rate(margins: list[float], deltas: list[float]) -> float:
    if not margins or not deltas:
        return 0.0
    ordered = sorted(margins)
    return sum(bisect.bisect_left(ordered, delta) / len(ordered) for delta in deltas) / len(deltas)


def load_margin_groups(path: str, eps: float) -> tuple[list[float], list[float], int]:
    rows = read_jsonl(path)
    parsed = []
    max_iteration = max((int(row.get("policy_iteration", row.get("minibatch", -1))) for row in rows), default=-1)
    for row in rows:
        sign = int(row.get("advantage_sign", 0))
        if sign == 0:
            continue
        new_logp = row.get("new_logp", row.get("logp_ref"))
        if new_logp is None:
            raise ValueError("margin row requires new_logp or logp_ref")
        margin = abs((float(new_logp) - float(row["old_logp"])) - clip_boundary(sign, eps))
        iteration = int(row.get("policy_iteration", row.get("minibatch", -1)))
        parsed.append((margin, iteration))
    return [item[0] for item in parsed], [item[0] for item in parsed if item[1] == max_iteration], max_iteration


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 controlled boundary construction and fork-rate calibration.")
    parser.add_argument("--logprob-jsonl", required=True, help="Phase 1 token logprob JSONL.")
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--bound", type=float, default=None)
    parser.add_argument("--margin-jsonl", required=True, help="Canonical Phase 0 per-token GRPO margin dump.")
    parser.add_argument("--out-model-json", default="results/phase3_calibration.json")
    parser.add_argument("--out-jsonl", default="results/phase3_controlled_certificates.jsonl")
    parser.add_argument("--report", default="reports/phase3.md")
    args = parser.parse_args()

    rows = read_jsonl(args.logprob_jsonl)
    certs = []
    possible = 0
    actual = 0
    for i, row in enumerate(rows):
        logp_ref = float(row["logp_ref"])
        logp_alt = float(row["logp_alt"])
        delta = abs(logp_alt - logp_ref)
        if delta == 0:
            continue
        sign = 1 if i % 2 == 0 else -1
        for multiplier in [-0.75, -0.25, 0.25, 0.75]:
            margin = multiplier * delta
            old_logp = construct_old_logp(logp_ref, sign, args.eps, margin)
            cert = detect_clipping_fork(
                case_id=str(row.get("case_id", f"row_{i}")),
                token_index=int(row.get("token_index", i)),
                token_id=row.get("token_id"),
                token_text=row.get("token_text"),
                path_ref=row.get("path_ref", row.get("metadata", {}).get("path_ref", "path_ref")),
                path_alt=row.get("path_alt", row.get("metadata", {}).get("path_alt", "path_alt")),
                logp_ref=logp_ref,
                logp_alt=logp_alt,
                old_logp=old_logp,
                advantage_sign_value=sign,
                eps=args.eps,
                delta_self_ref=row.get("delta_self_ref"),
                delta_self_alt=row.get("delta_self_alt"),
                delta_bound_legal=args.bound,
                metadata={
                    "phase": "phase3_controlled",
                    "source": args.logprob_jsonl,
                    "phase1_metadata": row.get("metadata", {}),
                    "tokenization": {
                        key: row.get(key)
                        for key in [
                            "prompt_token_hash",
                            "response_token_hash",
                            "full_token_hash",
                            "prompt_token_count",
                            "response_token_count",
                            "full_token_count",
                        ]
                        if key in row
                    },
                },
            )
            certs.append(cert.to_json_dict())
            possible += int(cert.fork_possible)
            actual += int(cert.actual_fork)

    write_jsonl(args.out_jsonl, certs)
    deltas = [
        float(r["logprob_delta"])
        for r in rows
        if int(r.get("advantage_sign", 1)) != 0
        and float(r.get("logprob_delta", abs(float(r["logp_alt"]) - float(r["logp_ref"])))) >= 0
    ]
    all_margins, late_margins, late_iteration = load_margin_groups(args.margin_jsonl, args.eps)
    model = {
        "model_kind": "empirical_independent_margin_delta_convolution",
        "margin_source": args.margin_jsonl,
        "delta_source": args.logprob_jsonl,
        "eps": args.eps,
        "margin_count": len(all_margins),
        "late_margin_count": len(late_margins),
        "delta_count": len(deltas),
        "late_policy_iteration": late_iteration,
        "predicted_fork_rate_overall": empirical_predicted_fork_rate(all_margins, deltas),
        "predicted_fork_rate_late": empirical_predicted_fork_rate(late_margins, deltas),
        "independence_assumption": True,
    }
    model_out = Path(args.out_model_json)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model_out.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_rows = [
        {
            "controlled_cases": len(certs),
            "fork_possible_rate": possible / len(certs) if certs else 0,
            "actual_fork_rate": actual / len(certs) if certs else 0,
            "delta_p50": percentile(deltas, 50) if deltas else 0,
            "delta_p99": percentile(deltas, 99) if deltas else 0,
            "predicted_fork_rate_overall": model["predicted_fork_rate_overall"],
            "predicted_fork_rate_late": model["predicted_fork_rate_late"],
        }
    ]
    fork_cases = [c for c in certs if c["actual_fork"]]
    min_case = min(fork_cases, key=lambda c: c["logprob_delta"]) if fork_cases else None
    min_case_text = json.dumps(min_case, indent=2, sort_keys=True) if min_case else "_No actual fork found._"
    write_phase_report(
        args.report,
        title="Phase 3 Controlled Calibration",
        confound_checklist={
            "fixed_response_tokens": True,
            "controlled_old_logp_only": True,
            "not_used_as_final_claim": True,
            "same_token_comparison": True,
        },
        delta_self_summary="Uses Phase 1 delta_self fields if present; this script does not establish self consistency.",
        summary="Controlled boundary construction generated certificates for detector calibration.",
        sections={
            "Calibration": markdown_table(summary_rows, list(summary_rows[0].keys())),
            "Minimum Actual Fork Case": f"```json\n{min_case_text}\n```",
        },
    )
    print(json.dumps(summary_rows[0], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
