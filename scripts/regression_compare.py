#!/usr/bin/env python3
"""Compare two directories of reports case by case (kernel_analyzer.check.compare_reports): verdicts, reference
classes and special-value counts must match; relative sizes within 1e-6; e_num differences of outputs written with
float atomics are listed as tolerated (K depends on the atomic order).

    python scripts/regression_compare.py results/regression/pre_refactor results/regression/post_refactor
"""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kernel_analyzer.check import compare_reports  # noqa: E402

old_dir, new_dir = Path(sys.argv[1]), Path(sys.argv[2])
old = {Path(f).stem: f for f in glob.glob(str(old_dir / "*.json"))}
new = {Path(f).stem: f for f in glob.glob(str(new_dir / "*.json"))}
changed = 0
for case in sorted(old):
    if case not in new:
        print(f"{case:34s} missing in {new_dir}")
        changed += 1
        continue
    diffs = [d for d in compare_reports(json.load(open(old[case])), json.load(open(new[case]))) if d[0] != "/mode"]
    real = [d for d in diffs if "[tolerated" not in d[0]]
    print(f"{case:34s} {'identical' if not diffs else ('tolerated only' if not real else 'CHANGED')}"
          + ("" if not diffs else "  " + "; ".join(f"{p}: {a} -> {b}" for p, a, b in diffs[:4])))
    changed += bool(real)
print(f"cases {len(old)}, changed {changed}")
sys.exit(1 if changed else 0)
