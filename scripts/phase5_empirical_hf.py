#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from forkcert.config import load_config
from forkcert.detector import clip_active
from forkcert.io import read_jsonl, write_jsonl
from forkcert.logprob_runner import cleanup_memory, configure_determinism, load_hf_path
from forkcert.report import CLAIM_SCOPE, markdown_table
from forkcert.stats import mean, percentile
from phase5_hf_bug_injection import BUGS, execute_bug, path_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Executed HF bug injections with empirical-only anomaly reporting.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--rollout-jsonl", required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--out-jsonl", default="results/phase5_empirical_bug_rows.jsonl")
    parser.add_argument("--report", default="reports/phase5_empirical.md")
    args = parser.parse_args()

    cfg = load_config(args.config)
    config = path_config(cfg)
    configure_determinism(seed=int(cfg.get("seed", 0)))
    samples = read_jsonl(args.samples)[: args.max_samples]
    rollout = {(str(row["case_id"]), int(row["token_index"])): row for row in read_jsonl(args.rollout_jsonl)}
    tokenizer, model = load_hf_path(config)
    output = []
    try:
        for sample in samples:
            ref_first = execute_bug(tokenizer, model, config, sample, "none")
            ref_second = execute_bug(tokenizer, model, config, sample, "none")
            if any(a["logp_bug"] != b["logp_bug"] for a, b in zip(ref_first, ref_second, strict=True)):
                raise ValueError(f"empirical Phase 5 reference self mismatch: {sample['case_id']}")
            for bug in BUGS:
                altered = execute_bug(tokenizer, model, config, sample, bug["name"])
                for ref, alt in zip(ref_first, altered, strict=True):
                    state = rollout.get((str(ref["case_id"]), int(ref["token_index"])))
                    if state is None or int(state["token_id"]) != int(ref["token_id"]):
                        raise ValueError(f"missing or mismatched rollout state: {ref['case_id']}:{ref['token_index']}")
                    sign = int(state.get("advantage_sign", 0))
                    ref_logp = float(ref["logp_bug"])
                    alt_logp = float(alt["logp_bug"])
                    branch_fork = False
                    if sign != 0:
                        branch_fork = clip_active(ref_logp, float(state["old_logp"]), sign, args.eps) != clip_active(
                            alt_logp, float(state["old_logp"]), sign, args.eps
                        )
                    output.append(
                        {
                            "case_id": ref["case_id"],
                            "token_index": ref["token_index"],
                            "token_id": ref["token_id"],
                            "bug": bug["name"],
                            "injection_kind": bug["injection_kind"],
                            "logp_ref_recomputed": ref_logp,
                            "logp_bug": alt_logp,
                            "logprob_delta": abs(alt_logp - ref_logp),
                            "delta_self_ref": 0.0,
                            "advantage_sign": sign,
                            "actual_clip_branch_fork": branch_fork,
                            "region": "unknown",
                            "delta_bound_legal": None,
                            "certified_bug": False,
                            "empirical_anomaly_only": True,
                        }
                    )
    finally:
        del model, tokenizer
        cleanup_memory()
    write_jsonl(args.out_jsonl, output)
    by_bug = []
    for bug in BUGS:
        rows = [row for row in output if row["bug"] == bug["name"]]
        deltas = [row["logprob_delta"] for row in rows]
        by_bug.append(
            {
                "bug": bug["name"],
                "rows": len(rows),
                "delta_mean": mean(deltas),
                "delta_p50": percentile(deltas, 50),
                "delta_p99": percentile(deltas, 99),
                "delta_max": max(deltas),
                "clip_branch_forks": sum(row["actual_clip_branch_fork"] for row in rows),
                "certified_bug_count": 0,
            }
        )
    report = "\n".join(
        [
            "# Phase 5 Executed Bugs: Empirical-Only",
            "",
            "## Claim Scope",
            CLAIM_SCOPE,
            "",
            "## Confound Checklist",
            "- three altered model operations executed: PASS",
            "- reference path self exact: PASS",
            "- same single-sample shape for reference and altered path: PASS",
            "- token and rollout alignment: PASS",
            "- analytic legal bound: FAIL; certified bug classification disabled",
            "",
            "## Delta Self Control",
            "Every recomputed reference token matched exactly across two runs.",
            "",
            "## External Validity",
            "Executed on the step-5 T4 FP16 snapshot. Results are empirical anomaly sensitivity only.",
            "",
            "## Results",
            markdown_table(by_bug, list(by_bug[0].keys())),
            "",
            "All rows retain region `unknown`; no empirical threshold is promoted to a legal bug boundary.",
            "",
        ]
    )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report, encoding="utf-8")
    print(json.dumps(by_bug, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
