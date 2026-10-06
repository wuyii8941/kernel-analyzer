#!/usr/bin/env python3
"""Equal-budget comparison (evaluation plan section 3, second fairness view; work item H / M3).

For each tier-2 condition of the external corpus (docs/external_eval_protocol_20261006.md) the budget is the tool's
measured wall time on that condition (96 seeds: capture, reference, specification, statistics; after the libtriton
hashing fix).  Each output-level baseline (B1 the authors' tolerance, B2 allclose, B3 TTrace-style threshold) spends
the same wall time on fresh seeds (96, 97, ...): kernel call, the authors' float64 reference, and for B3 the torch
float32 implementation.  Reported: seeds used, and whether the condition is flagged at the default parameter and at
the working point selected in the tier-1 sweep (results/external/gpuemu/summary.json).

    python scripts/equal_budget.py      # -> results/external/gpuemu/equal_budget.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import external_corpus as ec  # noqa: E402
import external_corpus_semantics as sem  # noqa: E402
from external_corpus_summary import DEFAULTS, control_of  # noqa: E402

OUT = ROOT / "results/external/gpuemu/equal_budget.json"


def main():
    summary = json.loads((ec.OUT / "summary.json").read_text())
    wp = summary["rq2_tier1"]["working_points"]
    rows = []
    for e in ec.entries():
        if not e["triton"]:
            continue
        cond = ec.tier2_condition(e["meta"])
        rep = json.loads((ec.OUT / "tier2" / f"{e['name']}__{cond['name']}.json").read_text())
        budget = float(rep["wall_seconds"])
        tol = e["meta"]["tolerances"][ec.DTYPE]
        op = sem.op_of(e["name"])
        stats = {"b1": [], "b2": [], "b3": []}
        t0, seed = time.time(), 96
        ec.call_kernel(e, ec.make_inputs(e["meta"], cond, 0))  # compiled already by the tool run; warm the process
        t0 = time.time()
        while time.time() - t0 < budget:
            inp = ec.make_inputs(e["meta"], cond, seed)
            _, k = ec.call_kernel(e, inp)
            k = np.asarray(k, np.float64).reshape(-1)
            f = ec.reference_f64(e, inp).reshape(-1)
            t32 = sem.trusted32(op, inp)
            fin = np.isfinite(k).all()
            rel = lambda a: float(np.linalg.norm(a - f) / max(np.linalg.norm(f), 1e-300))  # noqa: E731
            stats["b1"].append(float(np.abs(k - f.astype(np.float32)).max()) if fin else float("inf"))
            stats["b2"].append(float((np.abs(k - f) / (1e-5 + 1.3e-6 * np.abs(f))).max()) if fin else float("inf"))
            stats["b3"].append(rel(k) / max(rel(t32), 2.0 ** -24) if fin else float("inf"))
            seed += 1
        n = len(stats["b1"])
        row = {"entry": e["name"], "condition": cond["name"], "budget_seconds": budget, "seeds": n,
               "tool_seeds": 96, "planted_active": e["name"].endswith("buggy") and e["name"] != "softmax_triton_buggy"}
        for key, method in (("b1", "B1"), ("b2", "B2"), ("b3", "B3")):
            thr_default = DEFAULTS[method] * (tol if method == "B1" else 1.0)
            w = wp.get(f"{method} {control_of(e['name'])}")
            row[f"{method}_default_flag"] = bool(max(stats[key]) > thr_default)
            row[f"{method}_swept_flag"] = bool(w is not None and max(stats[key]) > w * (tol if method == "B1" else 1.0))
            row[f"{method}_max_statistic"] = max(stats[key])
        rows.append(row)
        print(f"{e['name']:30s} budget {budget:6.1f}s seeds {n:6d} " +
              " ".join(f"{m}:{int(row[m + '_default_flag'])}/{int(row[m + '_swept_flag'])}" for m in ("B1", "B2", "B3")),
              flush=True)
    OUT.write_text(json.dumps({"rule": "budget = the tool's tier-2 wall time per condition; fresh seeds from 96",
                               "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
