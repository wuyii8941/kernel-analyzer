#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from forkcert.io import read_jsonl
from forkcert.stats import percentile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an explicitly empirical cross-path envelope after Phase 2 downgrade.")
    parser.add_argument("--logprob-jsonl", required=True)
    parser.add_argument("--analytic-attempt", required=True)
    parser.add_argument("--out", default="results/phase2_bounds.empirical.json")
    args = parser.parse_args()

    source = Path(args.logprob_jsonl)
    rows = read_jsonl(source)
    deltas = [float(row["logprob_delta"]) for row in rows]
    if not deltas:
        raise SystemExit("cannot build empirical envelope from an empty logprob file")
    analytic = json.loads(Path(args.analytic_attempt).read_text(encoding="utf-8"))
    payload = {
        "certificate_kind": "empirical_cross_path_envelope",
        "decision": "EMPIRICAL_ONLY: legal analytic classifier unavailable; do not label fragile or bug.",
        "classification_mode": "empirically_stable_or_unknown",
        "generalization_certified": False,
        "in_sample_envelope": True,
        "logprob_bound_empirical": max(deltas),
        "empirical_delta_p50": percentile(deltas, 50),
        "empirical_delta_p95": percentile(deltas, 95),
        "empirical_delta_p99": percentile(deltas, 99),
        "empirical_delta_max": max(deltas),
        "sample_count": len({str(row["case_id"]) for row in rows}),
        "token_count": len(rows),
        "source": str(source),
        "source_sha256": sha256(source),
        "analytic_attempt": args.analytic_attempt,
        "analytic_decision": analytic.get("decision"),
        "analytic_tightness_worst_over_p99": analytic.get("tightness_worst_over_p99"),
        "limitations": [
            "The envelope is the maximum observed legal-pair delta on this dataset, not a rounding-error proof.",
            "empirically_stable means stable against the observed envelope only.",
            "delta above the envelope is an empirical anomaly, not proof of an implementation bug.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
