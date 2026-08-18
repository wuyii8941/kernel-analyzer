#!/usr/bin/env python3
"""Project the authoritative fail-closed ledger into the compact summary."""

from __future__ import annotations

import gzip
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    summary_path = ROOT / "results/final/summary.json"
    with gzip.open(ROOT / "results/coverage/qwen_invocation_ledger.json.gz", "rt") as handle:
        ledger = json.load(handle)
    gaps = json.loads((ROOT / "results/coverage/qwen_gap_audit.json").read_text())
    architecture_overview = json.loads((ROOT / "results/coverage/summary.json").read_text())
    summary = json.loads(summary_path.read_text())
    summary["claims"]["qwen_invocation_coverage_complete"] = False
    summary["claims"]["declared_implementation_cells_complete"] = True
    summary["coverage"]["fail_closed_invocation_ledger"] = {
        "status": ledger["status"],
        **ledger["summary"],
        "gates": ledger["gates"],
    }
    summary["coverage"]["architecture_invocation_ledgers"] = architecture_overview
    summary["implementation_difference"]["legacy_excluded_nonclosed_units"] = 122
    summary["implementation_difference"]["explicit_empty_or_elided_vjp_units_reclassified"] = gaps[
        "changed_nonclosed_reclassification"
    ]["reclassified_complete"]
    summary["implementation_difference"]["remaining_unresolved_changed_nonclosed_units"] = gaps[
        "changed_nonclosed_reclassification"
    ]["remaining_unresolved"]
    summary["next"] = (
        "complete held-out Qwen per-invocation numerical measurements and verdicts, then build "
        "the same exact eager/AOT/candidate bridges and local measurements for every Mamba and "
        "MoE invocation; keep property induction paused"
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(summary_path), "coverage_status": ledger["status"]}))


if __name__ == "__main__":
    main()
