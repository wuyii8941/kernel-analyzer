#!/usr/bin/env python3
"""Triage of an OpInfo screening directory: e_sem bins, candidates / small / special-value mismatches, errors.

    python scripts/opinfo_triage.py results/tool_spec/opinfo_onesample
"""
import glob
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
d = Path(sys.argv[1])
rel = d.resolve().relative_to(ROOT / "results" / "tool_spec")
tmp = ROOT / ".cache/tmp" / f"triage_{str(rel).replace('/', '_')}.md"
subprocess.run([sys.executable, str(ROOT / "scripts/tool_spec_summary.py"), "--groups", str(rel), "--out", str(tmp)],
               check=True, capture_output=True)
rows = json.load(open(tmp.with_suffix(".json")))
print("cases done:", len(glob.glob(str(d / "*.json"))), " outputs:", len(rows), dict(Counter(r["sem"] for r in rows)))
for r in rows:
    sm = tuple(r["special_mismatch"])
    if r["sem"] in ("candidate", "small", "mixed (external re-entry)", "unresolved") or sm != (0, 0):
        print(f'  {r["case"]:52s} {r["output"]:5s} {r["sem"]:26s} rel={r["sem_rel"]} complete={r["complete"]:.2f} '
              f'special(K_R/K vs f)={sm}')
errs = Counter()
examples = {}
for f in glob.glob(str(d / "*.json")):
    j = json.load(open(f))
    if "error" in j:
        e = j["error"].splitlines()[0][:160]
        errs[e] += 1
        examples.setdefault(e, j["case"])
print("errors:")
for e, n in errs.most_common():
    print(f"  {n:3d}  {e}   (e.g. {examples[e]})")
