#!/usr/bin/env python3
"""B012 deep case: the tool's e_sem / e_num / total on the param output, as fractions of the specification's update
|f - theta_old| (docs/radam_prediction_20261006.md states the predictions in those units).

    python scripts/radam_deep_check.py results/tool_spec/final/radam_deep [results/tool_spec/final/radam_deep_fixed]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import tool_spec_cases_inductor2 as m  # noqa: E402
from tool_spec_summary import sem_bin  # noqa: E402

cases = {c.name: c for c in m.CASES}


def update_ratio(case, seeds=range(4)):
    """RMS of the specification's update over RMS of the specification's parameter."""
    r = []
    for s in seeds:
        inp = case.inputs(s)
        lo, hi = case.spec(inp)["param"]
        f = 0.5 * (lo + hi)
        old = inp["p"].double().cpu().numpy()
        r.append(np.sqrt(np.mean((f - old) ** 2)) / np.sqrt(np.mean(f ** 2)))
    return float(np.mean(r))


for d in sys.argv[1:]:
    print(f"## {d}")
    print("| case | e_sem bin | e_sem / update | e_num / update | total / update |")
    print("|---|---|---:|---:|---:|")
    for f in sorted(Path(d).glob("*.json")):
        rep = json.loads(f.read_text())
        if "error" in rep:
            print(f"| {rep['case']} | error: {rep['error'][:60]} | | | |")
            continue
        e = rep["outputs"]["param"]
        ratio = update_ratio(cases[rep["case"]])
        sem = (e["semantic"].get("scale") or {}).get("relative_rms", float("nan"))
        num = (e["numerical"].get("scale") or {}).get("relative_rms", float("nan"))
        tot = (e.get("total") or {}).get("relative_rms", float("nan"))
        b = sem_bin(e["semantic"], False, tot, e.get("semantic_elementwise"))
        print(f"| {rep['case']} | {b} | {sem / ratio:.2g} | {num / ratio:.2g} | {tot / ratio:.2g} |")
