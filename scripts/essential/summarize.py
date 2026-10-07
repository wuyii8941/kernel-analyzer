#!/usr/bin/env python3
"""Interim summary of phase 1 (protocol sections 7 and 9): counts with denominators, the shared deviations, the W8
baseline table, reading consistency, undefined-case conventions, P and FR results.

    python scripts/essential/summarize.py      # reads results/essential/phase1/classification_*.json
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/essential/phase1"
DEV_PREFIX = ("deviates", "error_variant")


def is_dev(c):
    return c.startswith(DEV_PREFIX)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=OUT, help="directory with classification_*.json.gz (and for summary.json)")
    d_args = ap.parse_args()
    out_dir = d_args.dir
    summary = {}
    for fam in ("ce", "pool", "index"):
        p = out_dir / f"classification_{fam}.json.gz"
        if not p.exists():
            continue
        import gzip
        with gzip.open(p, "rt") as fh:
            d = json.load(fh)
        fs = {"conditions": d["conditions"], "counts": d["counts"], "shared": [], "candidate_errors": [],
              "eager_deviations": [], "eager_compatible_candidate_deviates": [], "readings": defaultdict(Counter),
              "undefined_conventions": defaultdict(Counter), "properties": defaultdict(lambda: defaultdict(lambda: [0, 0])),
              "fr": defaultdict(Counter), "w8": [], "held": 0}
        for r in d["records"]:
            cond = r["condition"]
            if r["held_doc_version_split"]:
                fs["held"] += 1
            for cand, rec in r["candidates"].items():
                for ph in ("fwd", "bwd"):
                    c = rec["class"][ph]
                    cat, cls = c.get("category", ""), c.get("class", "")
                    if c.get("readings_distinct") and c.get("compatible_readings") is not None:
                        key = "+".join(c["compatible_readings"]) or "none"
                        fs["readings"][f"{cand}/{ph}"][key] += 1
                    if cls.startswith(("spec_main_undefined", "spec_main_not_established")):
                        vals = [tuple(s.get("candidate_values", {}).get("loss", s.get("candidate_values", {}).get("out", []))[:1])
                                for s in c["seeds"] if s.get("candidate_values")]
                        fs["undefined_conventions"][f"{cand}/{ph}"][str(vals[:1])] += 1
                    if r["held_doc_version_split"]:
                        continue
                    item = {"condition": cond["id"], "candidate": cand, "phase": ph, "class": cls, "category": cat,
                            "params": {k: v for k, v in cond.items() if k not in ("id",)}}
                    if cat == "shared_deviation":
                        fs["shared"].append(item)
                    elif cat == "candidate_error":
                        fs["candidate_errors"].append(item)
                    elif cat.startswith("eager_itself") and is_dev(cls):
                        fs["eager_deviations"].append(item)
                    elif cat == "candidate_compatible_eager_deviates":
                        fs["eager_compatible_candidate_deviates"].append(item)
                    if is_dev(cls) or cls.startswith("reading:"):
                        e = (rec.get("E") or {}).get(ph) or {}
                        e64 = (rec.get("E64") or {}).get(ph) or {}
                        pv = sum(v["violated"] for v in rec["P"]["per_property"].values())
                        fr = rec.get("FR") or {}
                        fr_sem = None
                        if fr.get("status") == "ok":
                            fr_sem = any(o.get("semantic_vs_main") for o in fr["outputs"].values() if o["phase"] == ph)
                        fs["w8"].append(dict(item, E_detects=e.get("deviating_elements", 0) > 0 if e else None,
                                             FP64_ref_detects=e64.get("deviating_elements", 0) > 0 if e64 else None,
                                             P_any_violation=pv > 0, FR_semantic=fr_sem))
                for prop, v in rec["P"]["per_property"].items():
                    if not rec["P"]["judged"]:              # bf16: recorded only (protocol section 6)
                        continue
                    acc = fs["properties"][cand][prop]
                    acc[0] += v["checked"]
                    acc[1] += v["violated"]
                fr = rec.get("FR")
                if fr:
                    if fr.get("status") != "ok":
                        fs["fr"][cand]["error"] += 1
                        continue
                    for name, o in fr["outputs"].items():
                        if o.get("not_established"):
                            fs["fr"][cand][f"{name}: not established (tool)"] += 1
                            continue
                        if o["semantic_vs_main"] is None:
                            fs["fr"][cand][f"{name}: no spec value"] += 1
                        elif o["semantic_vs_main"]:
                            fs["fr"][cand][f"{name}: semantic vs main"] += 1
                        else:
                            fs["fr"][cand][f"{name}: no semantic difference vs main"] += 1
                        if o["readings"] and o["readings"].get(list(o["readings"])[0]) and o["readings"][list(o["readings"])[0]]["interface_elements"]:
                            fs["fr"][cand][f"{name}: interface/constant items"] += 1
                    for k in fr["notes"].get("outputs_not_written_by_triton") or []:
                        fs["fr"][cand][f"{k}: not written by Triton"] += 1
        fs["readings"] = {k: dict(v) for k, v in fs["readings"].items()}
        fs["undefined_conventions"] = {k: dict(v) for k, v in fs["undefined_conventions"].items()}
        fs["properties"] = {c: {p: {"checked": a[0], "violated": a[1]} for p, a in v.items()} for c, v in fs["properties"].items()}
        fs["fr"] = {k: dict(v) for k, v in fs["fr"].items()}
        summary[fam] = fs
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    for fam, fs in summary.items():
        print(f"== {fam}: {fs['conditions']} conditions; held (doc split) {fs['held']}")
        for cand, ph in fs["counts"].items():
            for phase, cnt in ph.items():
                print(f"   {cand:26s} {phase} n={sum(cnt.values()):4d} {dict(sorted(cnt.items(), key=lambda kv: -kv[1]))}")
        print("   shared deviations:", len(fs["shared"]), " candidate errors:", len(fs["candidate_errors"]),
              " eager deviations:", len(fs["eager_deviations"]), " candidate ok / eager deviates:", len(fs["eager_compatible_candidate_deviates"]))
        print("   readings:", fs["readings"])
        print("   properties:", {c: {p: f"{v['violated']}/{v['checked']}" for p, v in d.items() if v["violated"]} for c, d in fs["properties"].items()})
        print("   FR:", fs["fr"])


if __name__ == "__main__":
    main()
