#!/usr/bin/env python3
"""H-G1 (protocol.md): row-by-row comparison of a new regression_v11 run with reg20261009T2319.

The reference summary is read from git (commit 3532066, the audit baseline), so it stays available after the new run
replaces it in the working tree.  Compared per program: expected / observed / met, launches per input, the e_num and
e_sem summaries, the execution status and every v3.1 comparison field.  Fields the new run adds are not differences.

    python compare_regression.py --new 1_experiments/dsl_v2/regression_v11/<run>/summary.json --out result.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD = "1_experiments/dsl_v2/regression_v11/reg20261009T2319/summary.json"
KEYS = ("expected", "observed", "met", "launches_per_input", "e_num", "e_sem", "execution", "v31_comparison")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--old-commit", default="3532066")
    a = ap.parse_args()
    old = json.loads(subprocess.run(["git", "-C", str(ROOT), "show", f"{a.old_commit}:{OLD}"], capture_output=True,
                                    text=True, check=True).stdout)
    new = json.loads(Path(a.new).read_text())
    o = {r["program"]: r for r in old["rows"]}
    n = {r["program"]: r for r in new["rows"]}
    diffs = []
    for pid in sorted(set(o) | set(n)):
        if pid not in o or pid not in n:
            diffs.append({"program": pid, "field": "presence", "old": pid in o, "new": pid in n})
            continue
        for k in KEYS:
            if o[pid].get(k) != n[pid].get(k):
                diffs.append({"program": pid, "field": k, "old": o[pid].get(k), "new": n[pid].get(k)})
    res = {"old": OLD + f" @ {a.old_commit}", "new": a.new, "programs": len(n),
           "expectations_met": {"old": old.get("expectations_met"), "new": new.get("expectations_met")},
           "safety_violations": {"old": old.get("safety_violations"), "new": new.get("safety_violations")},
           "differences": diffs, "state": "成立" if not diffs and new.get("safety_violations") == [] else "不成立"}
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print(res["state"], len(diffs), "differences")


if __name__ == "__main__":
    main()
