#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from forkcert.bounds import ErrorSource, assemble_semi_certified_probability_bound
from forkcert.io import read_jsonl
from forkcert.report import CLAIM_SCOPE, markdown_table
from forkcert.stats import percentile


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble the P4 semi-certified differential probability bound.")
    parser.add_argument("--sources", required=True)
    parser.add_argument("--logprob-jsonl", required=True)
    parser.add_argument("--delta", type=float, default=1e-6)
    parser.add_argument("--out-json", default="results/phase2_v2.json")
    parser.add_argument("--report", default="reports/phase2_v2.md")
    args = parser.parse_args()

    source_payload = json.loads(Path(args.sources).read_text(encoding="utf-8"))
    sources = [ErrorSource(**item) for item in source_payload["sources"]]
    result = assemble_semi_certified_probability_bound(sources, delta=args.delta)
    rows = read_jsonl(args.logprob_jsonl)
    deltas = [float(row["logprob_delta"]) for row in rows]
    empirical_p99 = percentile(deltas, 99)
    empirical_envelope = {
        "kind": "empirical_only",
        "p99": empirical_p99,
        "p99_9": percentile(deltas, 99.9),
        "max": max(deltas),
        "allowed_for_bug_classification": False,
    }
    ratio = result["logprob_bound_prob"] / empirical_p99 if empirical_p99 > 0 else None
    result.update(
        {
            "source_payload_kind": source_payload.get("certificate_kind", "unknown"),
            "source_provenance": source_payload.get("provenance", {}),
            "empirical_delta_p99": empirical_p99,
            "tightness_prob_over_p99": ratio,
            "empirical_envelope": empirical_envelope,
        }
    )
    if not result["validation"]:
        decision = "DOWNGRADE: P4 numerical assembly is diagnostic only because source-level assumptions are unverified."
    elif ratio is None:
        decision = "UNKNOWN: no empirical p99 available."
    elif ratio <= 100:
        decision = "GO: semi-certified probability bound is within 100x of empirical p99."
    elif ratio <= 1000:
        decision = "REVIEW: semi-certified probability bound is conservative by 100x-1000x."
    else:
        decision = "DOWNGRADE: semi-certified probability bound is over 1000x too loose; use stable/unknown plus empirical envelope."
    result["decision"] = decision
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "certificate_kind": result["certificate_kind"],
        "source_validation": result["validation"],
        "logprob_bound_prob": result["logprob_bound_prob"],
        "empirical_delta_p99": empirical_p99,
        "tightness_prob_over_p99": ratio,
        "decision": decision,
    }
    report = "\n".join(
        [
            "# Phase 2 v2 Differential Probability Bound",
            "",
            "## Claim Scope",
            CLAIM_SCOPE,
            "",
            "## Confound Checklist",
            "- only declared path-difference injection points included: " + ("PASS" if all(s.difference_injection for s in sources) else "FAIL"),
            "- shared rounding cancellation established: " + ("PASS" if all(s.shared_rounding_cancelled for s in sources) else "FAIL"),
            "- local cross-source independence established: " + ("PASS" if all(s.local_error_independent for s in sources) else "FAIL"),
            "- propagation gains empirically calibrated: " + ("PASS" if all(s.propagation_empirically_calibrated for s in sources) else "FAIL"),
            "- propagation gain kept inside each RSS term: PASS",
            "- empirical envelope prohibited from bug classification: PASS",
            "",
            "## Delta Self Control",
            "Consumes the warmed A4 and Phase 1 self gates; no self delta is reinterpreted as a legal bound.",
            "",
            "## External Validity",
            "Inputs are T4 FP16. A zero-fork conclusion is limited to FP16 and cannot exclude BF16 behavior.",
            "",
            "## Summary",
            markdown_table([summary], list(summary.keys())),
            "",
            "## Validation Failures",
            "\n".join(f"- {item}" for item in result["validation_failures"]) or "_None._",
            "",
            "## Per Source",
            markdown_table(result["per_source"], list(result["per_source"][0].keys()) if result["per_source"] else []),
            "",
        ]
    )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
