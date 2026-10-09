#!/usr/bin/env python3
"""W0 (rc3 04): a semantic route for every operation of the official source inventories, plus the merged source file and
the explanations, for tools/reconcile_official_inventory.py of the rc3 package.

    python scripts/dsl_v2/w0_routes.py --sources S1.json S2.json --registered R.json --observed O.json --outdir DIR

A route is a plan, not support: operations with a rule of the 3.6.0 regression profile are "implemented-unvalidated";
everything else is "planned" with the rc3 design section and work package that owns it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from kernel_analyzer.reference_eval import ttir_mapping as M  # noqa: E402

PLAN = {
    "ttg": "02 6.6 distributed layouts / shared memory and 02 6.3 async copies (W3 layouts, W4 effects)",
    "ttng": "02 6.9 NVIDIA tensor memory, TMA, mbarrier, warp specialization, cluster/CLC (W3 NVIDIA module, W4 events)",
    "amdg": "02 6.9 AMD buffer / LDS / MFMA / WMMA / TDM / async (W3 AMD module, W4 events)",
    "nvws": "02 6.9 warp-specialization arefs and tokens (W3, W4 events)",
    "nvg": "02 6.9 NVIDIA GPU-level operations (W3 NVIDIA module)",
    "gluon": "02 6.9 Gluon explicit layouts (W3 Gluon importer)",
    "tti": "02 6.10 instrumentation: observable effects kept unless proven non-interfering (W5)",
    "proton": "02 6.10 profiling: observable effects kept unless proven non-interfering (W5)",
    "proton_gpu": "02 6.10 profiling: observable effects kept unless proven non-interfering (W5)",
    "llvm": "lowered IR (00 2): not a reference-language input; used only by lowering verification (emulate) unless an "
            "imported path needs it",
    "nvvm": "lowered IR (00 2): lowering verification only unless an imported path needs it",
    "rocdl": "lowered IR (00 2): lowering verification only unless an imported path needs it",
    "gpu": "host / launch-side dialect: registered by the build; reference role per use (00 2)",
}
SPECIFIC = {
    "tt.atomic_load": "02 6.9 atomic load / store / poll relations with memory order and scope (W4)",
    "tt.atomic_store": "02 6.9 atomic load / store / poll relations with memory order and scope (W4)",
    "tt.atomic_poll": "02 6.9 polling with a progress contract (W4)",
    "tt.approx_divf": "02 6.5 approximate instruction: real division in numerical-difference mode, documented error "
                      "contract in rounding verification (W5)",
    "tt.grid_dependency_wait": "02 6.9 grid dependency events (W4)",
    "tt.grid_dependency_launch_dependents": "02 6.9 grid dependency events (W4)",
    "tt.atomic_cas": "02 6.9 CAS relation: finite interleavings / protocol summary / set target (W4)",
}


def route(op: str) -> dict:
    rule = M.rule_for(op)
    if rule is not None and rule.status == "SUPPORTED":
        return {"id": op, "route": f"rule {rule.internal} (3.6.0 regression profile; contract in "
                                   "1_experiments/dsl_v2/w1_contracts)", "implementation_status": "implemented-unvalidated"}
    if op in SPECIFIC:
        return {"id": op, "route": SPECIFIC[op], "implementation_status": "planned"}
    if rule is not None:
        return {"id": op, "route": f"needs specification: rejected in the 3.6.0 profile ({rule.reason}); rc3 03 2.1 "
                                   "triage", "implementation_status": "planned"}
    d = op.split(".")[0]
    return {"id": op, "route": PLAN.get(d, "02 generic rule from the op's documented semantics (W2)"),
            "implementation_status": "planned"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", type=Path, required=True)
    ap.add_argument("--registered", type=Path, required=True)
    ap.add_argument("--observed", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    srcs = [json.loads(p.read_text()) for p in a.sources]
    merged = dict(srcs[0])
    merged["producer"] = " + ".join(s["producer"] for s in srcs)
    merged["receipt"] = {f"part{i}": s["receipt"] for i, s in enumerate(srcs)}
    merged["complete_for_profile"] = all(s["complete_for_profile"] for s in srcs)
    merged["records"] = sorted({r["id"]: {"id": r["id"], "contract_hash": r["contract_hash"]}
                                for s in srcs for r in s["records"]}.values(), key=lambda r: r["id"])
    (a.outdir / "source.json").write_text(json.dumps(merged, indent=1) + "\n")
    routes = [route(r["id"]) for r in merged["records"]]
    (a.outdir / "routes.json").write_text(json.dumps({"records": routes}, indent=1) + "\n")
    reg = json.loads(a.registered.read_text())
    obs = json.loads(a.observed.read_text())
    src_ids = {r["id"] for r in merged["records"]}
    reg_ids = {r["id"] for r in reg["records"]}
    expl = []
    for r in obs["records"]:
        if r["id"] not in reg_ids:
            expl.append({"difference": "observed_not_registered", "id": r["id"],
                         "reason": "matched by the observation parser but not an operation of the source candidates "
                                   "(attribute, type or symbol text, or an op of a dialect outside the candidate lists)",
                         "evidence": f"observations {r['observations']}; not in the {len(src_ids)} source ids"})
    (a.outdir / "explanations.json").write_text(json.dumps({"records": expl}, indent=1) + "\n")
    print({"source": len(src_ids), "registered": len(reg_ids), "observed": len(obs["records"]),
           "routes_implemented": sum(r["implementation_status"] != "planned" for r in routes),
           "explanations": len(expl)})


if __name__ == "__main__":
    main()
