#!/usr/bin/env python3
"""Analysis of a compiled-workload run (E1 or E3; docs/SUFFICIENCY_PROTOCOL.md sections 3 and 5.3).

Reads ROOT/pipeline_log.jsonl, ROOT/units/*.jsonl, ROOT/candidates.json.gz, ROOT/dynamic_units.json and ROOT/dynamic/*
and writes ROOT/analysis.json with:
  - model-configs attempted and their capture result (failed ones stay in the account);
  - candidate coverage over all captured units;
  - over the dynamic set (all units, or the registered sample): complete reference (development level: drafted GRAPH
    composition) and admitted (strict) counts with Wilson 95% intervals, bias-query availability;
  - the five outcome classes (first blocking stage in pipeline order), with sub-reasons;
  - groupings by model, kernel structure and semantic class.

    python 5_specification/sufficiency/analyze_compiled.py ROOT
"""
from __future__ import annotations

import collections
import gzip
import json
import math
import sys
from pathlib import Path

import yaml

SPEC = Path(__file__).resolve().parents[1]
USABLE, MANUAL, EXTEND, NEWC, EVID = ("usable as is", "manual binding needed", "implementation extension needed",
                                      "new or modified task contract needed", "run or evidence insufficient")
BINDING_REFUSALS = ("candidate origin", "no unique", "provenance", "call parameter mapping", "aliased", "shared storage",
                    "complete task graph", "input type or shape", "output type or shape", "fragment references",
                    "recorded input or pre-state", "storage correspondence", "no graph fragment", "call sites",
                    "template matches", "no provenance")
EVIDENCE_REFUSALS = ("complete pre-state not captured", "complete post-state not captured")
SEMANTIC = {"matmul": ("aten.mm", "aten.bmm", "aten.addmm", "aten.baddbmm", "aten.matmul", "aten.linear"),
            "reduction": ("aten.sum", "aten.mean", "aten.amax", "aten.amin", "aten.prod", "aten.var", "aten.var_mean",
                          "aten.cumsum", "aten.max", "aten.min", "aten.argmax", "aten.argmin"),
            "softmax": ("aten._softmax", "aten._log_softmax", "aten.logsumexp", "prims.prepare_softmax_online",
                        "aten._softmax_backward_data", "aten._log_softmax_backward_data"),
            "normalization": ("aten.native_layer_norm", "aten.native_group_norm", "aten.rsqrt"),
            "index_scatter": ("aten.gather", "aten.index", "aten.index_select", "aten.embedding", "aten.scatter",
                              "aten.scatter_add", "aten.index_put", "aten.embedding_dense_backward", "aten.index_add"),
            "random": ("prims.inductor_random", "prims.inductor_lookup_seed", "aten.native_dropout"),
            "inplace_state": ("aten.index_put_", "aten.copy_", "aten.add_", "aten.mul_")}


def wilson(k, n):
    if n == 0:
        return None
    z, p = 1.959963984540054, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(k / n, 4), round(max(0, c - h), 4), round(min(1, c + h), 4)]


def contract_status():
    status = {}
    for f in (SPEC / "contracts").glob("*.yaml"):
        d = yaml.safe_load(f.read_text())
        st = (d.get("activation") or {}).get("implementation_status")
        for c in d.get("covers") or []:
            if isinstance(c, str):
                base = c.split(".")[0] + "." + c.split(".")[1] if c.count(".") >= 1 else c
                status.setdefault(base, set()).add((d["id"], st))
    return status


def structure(kernel):
    for p in ("triton_poi_", "triton_red_", "triton_per_", "triton_tem_"):
        if (kernel or "").startswith(p):
            return p.strip("_").replace("triton_", "")
    return "other"


def semantic_classes(ops):
    ops = set(ops or [])
    return sorted(k for k, names in SEMANTIC.items() if ops & set(names)) or ["elementwise_layout"]


def classify(cand, verify, dyn):
    """(class, sub-reason) for one unit; first blocking stage in pipeline order."""
    v = (verify or "").lower()
    if v.startswith("error") or v == "budget_exceeded":
        return EVID, f"K-G check: {verify[:80]}"
    st = cand.get("status", "")
    if st.startswith("no fragment") or "kernel file missing" in json.dumps(cand.get("problems", [])):
        return EVID, "no graph fragment / kernel source"
    if "uncovered operations" in st:
        return NEWC, "operations without a contract: " + ",".join(sorted(cand.get("uncovered") or [])[:6])
    if "not parsed" in st:
        return EXTEND, "graph fragment not parsed"
    if cand.get("candidate_status") != "matched":
        return MANUAL, f"candidate {cand.get('candidate_status')}: {st[:80]}"
    if dyn is None:
        return None, "not in the dynamic set"
    if "rc" in dyn:                                  # harness record for a run without a result line
        if dyn["rc"] == "budget_exceeded":
            return EVID, "over the 600 s budget"
        if dyn["rc"] == 2 and "absent from the eligible capture records" in dyn.get("stderr", ""):
            return EXTEND, "tool refuses a reference without bit-identical K-G (verify: " + str(verify) + ")"
        return EVID, f"execution error rc={dyn['rc']}: {dyn.get('stderr', '')[-120:]}"
    s = dyn.get("status", "")
    if s.startswith("error"):
        return EVID, s[:120]
    if any(s.startswith(e) for e in EVIDENCE_REFUSALS):
        return EVID, s[:120]
    if any(s.startswith(b) for b in BINDING_REFUSALS):
        return MANUAL, s[:120]
    if s == "full reference comparison":
        if dyn.get("reference_status") == "complete":
            return EXTEND, "only the composition contract GRAPH is not admitted (complete reference evaluated)"
        return EXTEND, f"reference {dyn.get('reference_status')}"
    return EXTEND, s[:120]


def main():
    root = Path(sys.argv[1])
    log = [json.loads(l) for l in open(root / "pipeline_log.jsonl")] if (root / "pipeline_log.jsonl").exists() else []
    captures = [r for r in log if r["stage"] == "capture"]
    verify = {}
    for f in (root / "units").glob("*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            p = Path(r["unit"])
            verify[f"{p.parent.parent.name}/{p.parent.name}/{p.name}"] = r.get("verify")
    cands = {r["unit"]: r for r in json.loads(gzip.open(root / "candidates.json.gz").read())}
    dyn_units = json.loads((root / "dynamic_units.json").read_text())["units"] if (root / "dynamic_units.json").exists() else []
    dyn = {}
    for f in sorted((root / "dynamic").glob("*")):
        if f.suffix == ".jsonl":
            for l in open(f):
                r = json.loads(l)
                dyn[r["unit"]] = r
        elif f.suffix == ".err":
            r = json.loads(f.read_text())
            dyn.setdefault(r["unit"], r)
    cand_ok = [u for u, c in cands.items() if c.get("candidate_status") == "matched"
               and c.get("status") == "fragment matched, operations covered"]
    out = {"root": str(root), "model_configs": {"attempted": len(captures),
                                                "capture_rc": collections.Counter(str(r["rc"]) for r in captures),
                                                "failed": [f"{r['model']}/{r['config']}: {r['rc']}" for r in captures
                                                           if r["rc"] != 0]},
           "units_captured": len(cands),
           "verify": collections.Counter((v or "none")[:40] for v in verify.values()),
           "candidate_coverage": {"k": len(cand_ok), "n": len(cands), "wilson": wilson(len(cand_ok), len(cands))}}
    classes, reasons = collections.Counter(), collections.Counter()
    groups = {"model": collections.defaultdict(collections.Counter), "structure": collections.defaultdict(collections.Counter),
              "semantic": collections.defaultdict(collections.Counter)}
    complete_dev = 0
    rows = []
    for u in dyn_units:
        c = cands.get(u, {})
        d = dyn.get(u)
        cls, why = classify(c, verify.get(u), d)
        if cls is None:
            cls, why = EVID, "no dynamic record"
        classes[cls] += 1
        reasons[f"{cls} | {why[:90]}"] += 1
        dev = bool(d and d.get("status") == "full reference comparison" and d.get("reference_status") == "complete")
        complete_dev += dev
        model = u.split("/")[0] + "/" + u.split("/")[1]
        for g, keys in (("model", [model]), ("structure", [structure(c.get("kernel"))]),
                        ("semantic", semantic_classes(c.get("graph_ops")))):
            for k in keys:
                groups[g][k][cls] += 1
                groups[g][k]["_n"] += 1
                groups[g][k]["_complete_dev"] += dev
        rows.append({"unit": u, "class": cls, "reason": why, "complete_reference_dev": dev})
    n = len(dyn_units)
    out["dynamic_set"] = {"n": n, "of": len(cands), "seed": 20261019}
    out["complete_reference"] = {"admitted_strict": {"k": 0, "n": n, "wilson": wilson(0, n),
                                                     "reason": "GRAPH (composition) is drafted, not validated"},
                                 "development_drafted_GRAPH": {"k": complete_dev, "n": n, "wilson": wilson(complete_dev, n)}}
    out["bias_query_available"] = {"k": 0, "n": n, "reason": "no automatic route from compiled units to the contract measurement mode"}
    out["classes"] = {k: {"k": v, "wilson": wilson(v, n)} for k, v in classes.most_common()}
    out["reasons"] = dict(reasons.most_common())
    out["groups"] = {g: {k: dict(v) for k, v in sorted(d.items())} for g, d in groups.items()}
    (root / "analysis.json").write_text(json.dumps(out, indent=1, default=list))
    (root / "analysis_units.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps({k: out[k] for k in ("model_configs", "units_captured", "candidate_coverage", "dynamic_set",
                                          "complete_reference", "classes")}, indent=1, default=list))
    return 0


if __name__ == "__main__":
    sys.exit(main())
