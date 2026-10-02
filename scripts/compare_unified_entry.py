#!/usr/bin/env python3
"""Check that the unified entry reproduces the case-specific Liger step-4/5 result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, default=ROOT / "results/reference_eval/liger_fp32_order_auto_reference.json")
    parser.add_argument("--unified", type=Path, default=ROOT / "results/reference_eval/liger_fp32_order_unified_entry.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results/reference_eval/liger_fp32_order_unified_check.json")
    args = parser.parse_args()
    old = json.loads(args.original.read_text())
    new = json.loads(args.unified.read_text())
    rows = []
    # The case-specific script named the exact-reference comparisons "<variant>_vs_KR".
    old_by = {(r["comparison"].replace("_vs_KR", "_vs_K_R"), r["rule"]): r for r in old["results"]}
    for r in new["results"]:
        o = old_by.get((r["comparison"], r["rule"]))
        if o is None:
            rows.append({"comparison": r["comparison"], "rule": r["rule"], "status": "MISSING_IN_ORIGINAL"})
            continue
        row = {"comparison": r["comparison"], "rule": r["rule"],
               "mean_projection": [o["mean_projection"], r["mean_projection"]],
               "interval": [o["t_interval_95"], r["t_interval"]],
               "positive_negative": [[o["positive"], o["negative"]], [r["positive"], r["negative"]]],
               "final_verdict": [o["final_verdict"], r["final_verdict"]]}
        rel = abs(o["mean_projection"] - r["mean_projection"]) / max(abs(o["mean_projection"]), 1e-300)
        row["mean_relative_difference"] = rel
        row["identical"] = (rel == 0 and o["positive"] == r["positive"] and o["negative"] == r["negative"]
                            and o["final_verdict"] == r["final_verdict"])
        rows.append(row)
    step4_old = old["step4_automatic_vs_manual_reference"]["variants"]
    step4 = {}
    for v, s in new["reference"].items():
        o = step4_old[v]
        step4[v] = {
            "classes": [o["classes_total"], s["classes_total"]],
            "residual_positive_negative_zero": [
                [o["residual_positive_total"], o["residual_negative_total"], o["residual_contains_zero_total"]],
                [s["residual_positive_total"], s["residual_negative_total"], s["residual_contains_zero_total"]]],
            "reference_width_max": [o["reference_width_max"], s["reference_width_max"]],
        }
        step4[v]["identical"] = (step4[v]["residual_positive_negative_zero"][0] == step4[v]["residual_positive_negative_zero"][1]
                                 and o["classes_total"]["complete_composed"] == s["classes_total"]["complete_composed"])
    report = {"results": rows, "reference": step4,
              "all_identical": all(r.get("identical") for r in rows) and all(v["identical"] for v in step4.values())}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for r in rows:
        print(r["comparison"], r["rule"], "identical" if r.get("identical") else r)
    for v, s in step4.items():
        print(v, "reference identical" if s["identical"] else s)
    print("ALL IDENTICAL" if report["all_identical"] else "DIFFERENCES FOUND")


if __name__ == "__main__":
    main()
