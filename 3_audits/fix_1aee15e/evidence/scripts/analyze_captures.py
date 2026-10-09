#!/usr/bin/env python3
"""Compare the follow-up captures (.cache/audit_work/rerun2) with the recorded ones in 1_experiments/dsl_v2/captures:
per launch status transitions for main-capture files, per-target totals and per-launch status pairs for the
cross-level files, per kernel for the tutorials.  Writes rerun2/analysis.json and prints a short summary."""
import collections
import gzip
import json
import sys
from pathlib import Path

R = Path("/data1/tzh/kernel-analyzer")
O = R / ".cache/audit_work/rerun2"
CAP = R / "1_experiments/dsl_v2/captures"
sys.path.insert(0, str(R / "2_tool/scripts/dsl_v2"))
from summarize_capture import status  # noqa: E402


def rows(path):
    p = Path(path)
    text = gzip.decompress(p.read_bytes()).decode() if p.suffix == ".gz" else p.read_text()
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def keyed(rs, select=None, field="test"):
    out, seen = {}, collections.Counter()
    for r in rs:
        t = r.get(field, "").split("::", 1)[-1]
        if select and select not in t:
            continue
        k = (t, r.get("target"), seen[(t, r.get("target"))])
        seen[(t, r.get("target"))] += 1
        out[k] = r
    return out


def transitions(old, new, select=None):
    a, b = keyed(old, select), keyed(new, select)
    tr = collections.Counter(f"{status(a[k])} -> {status(b[k])}" for k in a if k in b)
    return {"matched": len(set(a) & set(b)), "baseline_only": len(set(a) - set(b)), "run_only": len(set(b) - set(a)),
            "changed": {k: v for k, v in tr.items() if k.split(" -> ")[0] != k.split(" -> ")[1]},
            "by_status": dict(collections.Counter(status(r) for r in b.values()))}


def cross(old, new):
    out = {}
    for name, rs in (("recorded", old), ("rerun", new)):
        t = collections.defaultdict(collections.Counter)
        for r in rs:
            g = t[r.get("target")]
            g["launches"] += 1
            g["both_complete"] += int(r.get("ttir_status") == "complete" and r.get("ttgir_status") == "complete")
            g["both_ok_elements"] += int(r.get("both_ok") or 0)
            g["disjoint"] += int(r.get("disjoint") or 0)
            g["ttir_only_ok"] += int(r.get("ttir_only_ok") or 0)
            g["ttgir_only_ok"] += int(r.get("ttgir_only_ok") or 0)
        out[name] = {k: dict(v) for k, v in t.items()}
    a, b = keyed(old), keyed(new)
    pairs = collections.Counter(f"{a[k].get('ttir_status')}/{a[k].get('ttgir_status')} -> "
                                f"{b[k].get('ttir_status')}/{b[k].get('ttgir_status')}" for k in a if k in b)
    out["matched"] = len(set(a) & set(b))
    out["status_changes"] = {k: v for k, v in pairs.items() if k.split(" -> ")[0] != k.split(" -> ")[1]}
    return out


res = {}
broad_new = []
for f in ("test_core", "test_random", "test_standard", "test_libdevice", "test_conversions", "test_tensor_descriptor"):
    broad_new += rows(O / f"broad_{f}.jsonl")
(O / "broad_combined.jsonl").write_text("".join(json.dumps(r) + "\n" for r in broad_new))
res["broad_vs_inc15"] = transitions(rows(R / ".cache/audit_work/inc15_broad.jsonl"), broad_new)
res["broad_vs_previous_auditfix"] = transitions(rows(CAP / "auditfix_broad.jsonl.gz"), broad_new)
res["core_atomic_vs_inc14"] = transitions(rows(CAP / "inc14_test_core_atomic.jsonl.gz"), rows(O / "broad_test_core.jsonl"),
                                          "atomic")
g = rows(O / "gluon_test_core.jsonl")
res["gluon_vs_inc10"] = transitions(rows(CAP / "inc10_gluon_test_core.jsonl.gz"), g)
res["gluon_atomic_vs_inc14"] = transitions(rows(CAP / "inc14_gluon_test_core_atomic.jsonl.gz"), g, "atomic")
res["cross_nvidia_vs_inc11"] = cross(rows(CAP / "inc11_cross_level_combined.jsonl.gz"), rows(O / "cross_nvidia.jsonl"))
res["cross_amd_vs_inc12"] = cross(rows(CAP / "inc12_cross_level_amd.jsonl.gz"), rows(O / "cross_amd.jsonl"))
res["cross_amd_supp_core_vs_inc12"] = cross(rows(CAP / "inc12_cross_level_amd_supplementary_core.jsonl.gz"),
                                            rows(O / "cross_amd_supp_core.jsonl"))
res["cross_amd_supp_td_vs_inc12"] = cross(rows(CAP / "inc12_cross_level_amd_supplementary_tensor_descriptor.jsonl.gz"),
                                          rows(O / "cross_amd_supp_td.jsonl"))
tut = rows(O / "tutorials.jsonl")
res["tutorials"] = {"rerun": [(r["tutorial"], r["kernel"], status(r), [n[:90] for n in r.get("notes", [])]) for r in tut],
                    "inc4": [(r["tutorial"], r["kernel"], status(r)) for r in rows(CAP / "inc4_tutorials.jsonl.gz")],
                    "inc7": [(r["tutorial"], r["kernel"], status(r)) for r in rows(CAP / "inc7_tutorial05.jsonl.gz")]}
(O / "analysis.json").write_text(json.dumps(res, indent=1) + "\n")
for k, v in res.items():
    if k == "tutorials":
        print(k, [(t[1], t[2]) for t in v["rerun"]])
    elif k.startswith("cross"):
        print(k, "matched", v["matched"], "changes", v["status_changes"], "| rerun", v["rerun"], "| recorded", v["recorded"])
    else:
        print(k, {x: v[x] for x in ("matched", "baseline_only", "run_only", "changed")})
