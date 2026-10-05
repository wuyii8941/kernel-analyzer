#!/usr/bin/env python3
"""Cases whose direct-differential verdict changes between static (dynamic=False) and dynamic (dynamic=True) compiles.

    python scripts/baseline_dynamic_diff.py results/baseline/opinfo_onesample results/baseline/dyn_onesample
"""
import glob
import json
import sys
from pathlib import Path

static_dir, dyn_dir = Path(sys.argv[1]), Path(sys.argv[2])
n = 0
for f in sorted(glob.glob(str(dyn_dir / "*.json"))):
    name = Path(f).stem
    d = json.load(open(f))
    sf = static_dir / f"{name}.json"
    if not sf.exists():
        continue
    s = json.load(open(sf))
    n += 1
    if "error" in d or "error" in s:
        if ("error" in d) != ("error" in s):
            print(f"{name:42s} static: {s.get('error', 'ok')[:60]!s:62s} dynamic: {d.get('error', 'ok')[:90]}")
        continue
    for o, e in d["outputs"].items():
        es = s["outputs"].get(o)
        if es is None:
            print(f"{name:42s} {o}: output only in dynamic")
            continue
        if e["ci_found"] != es["ci_found"] or e["hp_special_mismatch"] != es["hp_special_mismatch"] or e["shape_mismatch"] != es["shape_mismatch"]:
            print(f"{name:42s} {o:5s} static ci={es['ci_found']} hp={es['hp_rel_rms_max']} sp={es['hp_special_mismatch']} | "
                  f"dynamic ci={e['ci_found']} hp={e['hp_rel_rms_max']} sp={e['hp_special_mismatch']} shape_mismatch={e['shape_mismatch']} "
                  f"{e['ci_example'].get('why', '')[:60]}")
print("compared", n)
