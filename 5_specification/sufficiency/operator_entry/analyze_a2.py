#!/usr/bin/env python3
"""Path A2 analysis (OPERATOR_ENTRY_PROTOCOL_A2.md section 5): classes with Wilson intervals (key- and call-weighted),
by operator and workload, triggers, replays (G12), every detection with its largest relative gap, and the key-by-key
comparison with path A.

    analyze_a2.py --a2 ROOT_A2 --a ROOT_A --out analysis_A2.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
from pathlib import Path


def wilson(k, n):
    if not n:
        return None
    z, p = 1.959963984540054, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(p, 4), round(max(0, c - h), 4), round(min(1, c + h), 4)]


def rows(root):
    out = {}
    for f in glob.glob(f"{root}/results/*.json"):
        if f.endswith(".report.json"):
            continue
        r = json.loads(Path(f).read_text())
        out[Path(f).name] = r
    return out


def max_gap(root, r):
    """Largest semantic.max_gap_relative over the selected profile's report outputs (None when not recorded)."""
    sel = r.get("selected")
    for p in r.get("profiles") or []:
        if f"{p['contract']}/{p['profile']}" == sel and p.get("report"):
            path = Path(root) / "results" / p["report"]
            if not path.exists():
                return None
            rep = json.loads(path.read_text())
            gaps = [((o.get("semantic") or {}).get("max_gap_relative")) for lv in rep.get("levels", [])
                    for o in (lv.get("outputs") or {}).values()]
            gaps = [g for g in gaps if g is not None]
            return max(gaps) if gaps else None
    return None


def missing_reports(root, data):
    """Profiles that ran the measurement (rc 0) but whose report file is absent."""
    bad = []
    for name, r in data.items():
        for p in r.get("profiles") or []:
            if p.get("rc") == 0 and p.get("report") and not (Path(root) / "results" / p["report"]).exists():
                bad.append([name, p["report"]])
            if p.get("rc") == 0 and not p.get("report"):
                bad.append([name, "rc 0 without report"])
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a2", type=Path, required=True)
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    d2, d1 = rows(a.a2), rows(a.a)
    model = {k: r for k, r in d2.items() if not r["workload"].startswith("trigger_") and not r.get("replayed_function")}
    n, ncalls = len(model), sum(r.get("calls", 1) for r in model.values())
    cls = collections.Counter(r["class"] for r in model.values())
    clsw = collections.Counter()
    for r in model.values():
        clsw[r["class"]] += r.get("calls", 1)
    diagnosed = [r for r in model.values() if r["class"].startswith("usable")]
    out = {
        "candidate_keys": n, "candidate_calls": ncalls,
        "classes": {k: {"k": v, "wilson": wilson(v, n)} for k, v in cls.most_common()},
        "classes_call_weighted": {k: {"k": v, "share": round(v / ncalls, 4)} for k, v in clsw.most_common()},
        "diagnosed": {"k": len(diagnosed), "wilson": wilson(len(diagnosed), n),
                      "calls": sum(r.get("calls", 1) for r in diagnosed)},
        "verdicts": dict(collections.Counter(r.get("verdict") for r in diagnosed)),
        "not_diagnosed": [{"key": k, "op": r.get("op"), "class": r["class"], "reason": r["reason"][:240]}
                          for k, r in sorted(model.items()) if not r["class"].startswith("usable")],
        "detections": [],
        "replays": [{"key": k, "function": r.get("replayed_function"), "op": r.get("op"), "class": r["class"],
                     "verdict": r.get("verdict"), "reason": r["reason"][:240]}
                    for k, r in sorted(d2.items()) if r.get("replayed_function")],
        "triggers": [{"key": k, "op": r.get("op"), "class": r["class"], "verdict": r.get("verdict"),
                      "reason": r["reason"][:240]}
                     for k, r in sorted(d2.items()) if r["workload"].startswith("trigger_")],
        "by_op": {}, "by_workload": {},
        "missing_reports": missing_reports(a.a2, d2),
    }
    for k, r in sorted(d2.items()):
        if r.get("verdict") == "detected":
            out["detections"].append({"key": k, "op": r.get("op"), "selected": r.get("selected"),
                                      "reason": r["reason"][:240], "max_gap_relative": max_gap(a.a2, r)})
    byop, byw = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    for r in model.values():
        byop[r.get("op", "?")][r["class"]] += 1
        byw[r["workload"]][r["class"]] += 1
    out["by_op"] = {k: dict(v) for k, v in sorted(byop.items())}
    out["by_workload"] = {k: dict(v) for k, v in sorted(byw.items())}
    # key-by-key against path A: same workload and operator; A2 key numbering comes from new recordings, so the match
    # is by (workload, op, argument structure) through the recordings' keys.jsonl
    def sig(root, r):
        k = next((json.loads(l) for l in open(f"{root}/rec/{r['workload']}/keys.jsonl")
                  if json.loads(l)["key"] == r["key"]), None)
        if k is None:
            return None
        strip = json.dumps([k["op"], k["args"], k["kwargs"]], sort_keys=True).replace(', "device": "cuda"', "")
        return (r["workload"], strip)
    a_by_sig = {}
    for r in d1.values():
        if "key" in r:
            s = sig(a.a, r)
            if s:
                a_by_sig[s] = r
    transitions = collections.Counter()
    for r in model.values():
        s = sig(a.a2, r)
        before = a_by_sig.get(s)
        transitions[(before["class"] if before else "new in A2", r["class"])] += 1
    out["a_to_a2"] = [{"A": k[0], "A2": k[1], "keys": v} for k, v in transitions.most_common()]
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("candidate_keys", "diagnosed", "verdicts", "missing_reports")}, indent=1))


if __name__ == "__main__":
    main()
