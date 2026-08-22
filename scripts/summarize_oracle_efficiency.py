#!/usr/bin/env python3
"""Report the measured triage numbers without inventing runtime savings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.comparison.read_text())
    op = source["operating_point"]
    cohort = source["cohort"]
    oracle = source["comparisons"]["prefix16_local_persistence_oracle"]
    payload = {
        "schema": "kernel-analyzer-oracle-efficiency-v1",
        "status": "COMPLETE_RETROSPECTIVE_RUNTIME_UNRECORDED",
        "source": str(args.comparison),
        "source_sha256": sha256(args.comparison),
        "cohort": {
            "rows": int(cohort["rows"]),
            "positives": int(cohort["positives"]),
            "negatives": int(cohort["negatives"]),
            "selection": cohort["selection"],
        },
        "screening": {
            "rule": op["threshold_rule"],
            "flagged": int(op["flagged"]),
            "flag_rate": float(op["flag_rate"]),
            "recall": float(op["recall"]),
            "miss_rate": float(op["miss_rate"]),
            "false_positive_rate": float(op["false_positive_rate"]),
            "precision": float(op["precision"]),
            "retrospective_auroc": float(oracle["auroc"]),
            "retrospective_bootstrap_95": oracle["stratified_bootstrap_95"],
            "avoided_full_runs_if_used_as_triage": 1.0 - float(op["flag_rate"]),
        },
        "runtime_savings": {
            "status": "UNRESOLVED_NOT_RECORDED",
            "reason": "The frozen 12-row comparison contains labels and scores but no complete per-row wall-clock/GPU-hour ledger.",
        },
        "claim_boundary": "These are retrospective triage statistics with one positive. The avoided-run fraction is not a GPU-hour saving claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "flag_rate": payload["screening"]["flag_rate"], "auroc": payload["screening"]["retrospective_auroc"]}))


if __name__ == "__main__":
    main()
