"""H4-H6 (protocol.md section 9) from plants.jsonl and fixes.jsonl, with the registered criteria; plus descriptive
quantities that were not registered (marked as such).

    python summarize_rest.py plants.jsonl fixes.jsonl
"""
from __future__ import annotations

import json
import math
import sys


def load(path):
    try:
        return [json.loads(l) for l in open(path) if l.strip()]
    except FileNotFoundError:
        return []


def h4(plants):
    rows = []
    for p in plants:
        if p.get("error") or p["kind"] not in ("semantic", "control"):
            continue
        outs = p["outputs"]
        verdicts = {n: o["C_semantic"]["verdict"] for n, o in outs.items()}
        numeric_same = {n: abs(o["C_numeric_after"][0] - o["C_numeric_before"][0]) <= 3 * max(o["C_numeric_after"][1], 0)
                        for n, o in outs.items()}
        if p["kind"] == "semantic":
            flagged = [n for n, v in verdicts.items() if v == "不一致"]
            ok = bool(flagged) and all(numeric_same[n] for n in flagged)
        else:
            flagged = [n for n, v in verdicts.items() if v == "不一致"]
            ok = not flagged
        rows.append({"id": p["id"], "kind": p["kind"], "flagged_outputs": flagged, "numeric_within_3se": numeric_same,
                     "holds": ok,
                     "semantic_alignment": {n: o["C_semantic"]["alignment"] for n, o in outs.items()},
                     "numeric_before_after": {n: [o["C_numeric_before"][0], o["C_numeric_after"][0]] for n, o in outs.items()},
                     "B_f": {n: o["B_f"][0] for n, o in outs.items()}, "B_G": {n: o["B_G"][0] for n, o in outs.items()}})
    return {"plants": rows, "all_hold": all(r["holds"] for r in rows) if rows else None}


def h5_plants(plants):
    rows = []
    for p in plants:
        if p.get("error") or p["kind"] != "numeric":
            continue
        line = p["planted_line_in_patched_module"]
        for n, o in p["outputs"].items():
            c, a = o.get("C_top_site") or {}, o.get("A_top_site") or {}
            rows.append({"id": p["id"], "output": n, "planted": line, "C_line": c.get("line"), "A_line": a.get("line"),
                         "C_hit": c.get("line") == line, "A_hit": a.get("line") == line})
    return {"outputs": rows, "C_hit_rate": (sum(r["C_hit"] for r in rows), len(rows)),
            "A_hit_rate": (sum(r["A_hit"] for r in rows), len(rows))}


def h5_h6_fixes(fixes):
    rows = []
    for f in fixes:
        if f.get("status") != "fixed" or not (f.get("after") or {}).get("truth"):
            continue
        a, se = f["truth"]
        a2, se2 = f["after"]["truth"]
        pc = f["predicted_after"]["C"][0]
        pa = f["predicted_after"]["A"][0]
        a_before = f["A_before"][0]
        tol = max(3 * se2, 0.1 * abs(a))
        rows.append({"unit": f["unit"], "output": f["output"], "mechanism": f["fix"]["mechanism"],
                     "before": a, "after_measured": a2, "after_se": se2,
                     "C_predicted_after": pc, "A_predicted_after": pa, "A_before": a_before,
                     "localized": abs(a2) <= 0.5 * abs(a),
                     "C_after_agrees": abs(pc - a2) <= tol, "A_after_agrees": abs(pa - a2) <= tol,
                     # not registered: the predicted effect of the fix (before - after)
                     "effect_measured": a - a2, "effect_C": a - pc, "effect_A": a_before - pa})
    n = len(rows)
    return {"fixes": rows, "n": n,
            "H5_localized": (sum(r["localized"] for r in rows), n),
            "H6_C_after_agrees": (sum(r["C_after_agrees"] for r in rows), n),
            "A_after_agrees": (sum(r["A_after_agrees"] for r in rows), n),
            "not_registered_effect_within_10pct": {
                "C": sum(abs(r["effect_C"] - r["effect_measured"]) <= 0.1 * abs(r["effect_measured"]) + 3 * r["after_se"]
                         for r in rows),
                "A": sum(abs(r["effect_A"] - r["effect_measured"]) <= 0.1 * abs(r["effect_measured"]) + 3 * r["after_se"]
                         for r in rows), "n": n},
            "not_fixed": {}}


def main():
    plants, fixes = load(sys.argv[1]), load(sys.argv[2])
    out = {"H4": h4(plants), "H5_plants": h5_plants(plants), "H5_H6_fixes": h5_h6_fixes(fixes)}
    status = {}
    for f in fixes:
        status[f.get("status") or ("error" if f.get("error") else "?")] = status.get(f.get("status") or "error", 0) + 1
    out["H5_H6_fixes"]["not_fixed"] = status
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
