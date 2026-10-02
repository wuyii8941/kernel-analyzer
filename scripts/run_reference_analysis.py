#!/usr/bin/env python3
"""CLI of the unified entry (``kernel_analyzer.reference_eval.analysis``).

    # composed reference for some units (can run in parallel on disjoint unit sets)
    python scripts/run_reference_analysis.py --declaration D.json --stage reference --out DIR --units unit000,unit001
    # statistics over all units of DIR
    python scripts/run_reference_analysis.py --declaration D.json --stage statistics --out DIR --report R.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.reference_eval.analysis import load_declaration, reference_stage, statistics_stage  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--stage", choices=("reference", "statistics"), required=True)
    parser.add_argument("--out", type=Path, required=True, help="directory of per-unit reference arrays")
    parser.add_argument("--capture-root", type=Path, help="override the declaration's capture_root")
    parser.add_argument("--units", help="comma-separated unit directory names (reference stage)")
    parser.add_argument("--delete-captures", action="store_true")
    parser.add_argument("--report", type=Path, help="statistics report path")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    decl = load_declaration(args.declaration)
    if args.capture_root:
        decl["capture_root"] = str(args.capture_root)
    if args.stage == "reference":
        units = args.units.split(",") if args.units else None
        for summary in reference_stage(decl, args.out, units, args.delete_captures):
            print(json.dumps({"unit": summary["unit"], **{v: {k: summary[v][k] for k in (
                "classes", "residual_positive", "residual_negative", "seconds")} for v in decl["variants"]}}),
                flush=True)
    else:
        report = statistics_stage(decl, args.out, args.device)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        for r in report["results"]:
            print(json.dumps({k: r.get(k) for k in ("comparison", "rule", "mean_projection", "t_interval",
                                                      "positive", "negative", "final_verdict")}))


if __name__ == "__main__":
    main()
