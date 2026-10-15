"""Phase-1 statistics (protocol.md sections 4, 5, 9: H1-H3) from analyze_units' JSONL.

    python summarize.py units.jsonl [--out summary.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
from pathlib import Path

PREMISES = ("functions", "fusion", "order", "tensor_core", "subnormals")
METHODS = ("C-exec", "C-model", "A-exec", "A0") + tuple(f"A-{p}" for p in PREMISES)


def cls(a, se):
    if a is None or se is None or not math.isfinite(a):
        return None
    if se > 0 and a > 3 * se:
        return 1
    if se > 0 and a < -3 * se:
        return -1
    if se == 0 and a != 0:
        return 1 if a > 0 else -1
    return 0


def kernel_type(name: str) -> str:
    for k in ("poi", "red", "per", "tem"):
        if name.startswith(f"triton_{k}"):
            return {"poi": "pointwise", "red": "reduction", "per": "persistent", "tem": "template"}[k]
    return "other"


SINGLES = []


def records(path):
    out, units = [], []
    for line in open(path):
        r = json.loads(line)
        parts = Path(r["unit"].rstrip("/")).parts
        model, config = parts[-3], parts[-2]
        units.append({"unit": r["unit"], "model": model, "config": config, "error": r.get("error"),
                      "verify": r.get("verify"), "kernel": r.get("kernel", ""), "prediction": r.get("prediction_status")})
        if "error" in r:
            continue
        for name, o in r.get("outputs", {}).items():
            if "truth" not in o:
                continue
            a, se = o["truth"]
            if o["n"] < 2:          # deviation D3: one element has no statistics; listed separately
                single = {"unit": r["unit"], "model": model, "config": config, "kernel": r["kernel"], "output": name,
                          "elem": o["elem"], "truth": a, **{m: (o.get(m) or [None])[0] for m in METHODS if m != "A0"}}
                SINGLES.append(single)
                continue
            rec = {"unit": r["unit"], "model": model, "config": config, "kernel": r["kernel"],
                   "ktype": kernel_type(r["kernel"]), "output": name, "elem": o["elem"], "n": o["n"],
                   "bitwise": r["verify"] == "bit_identical", "a": a, "se": se, "truth": cls(a, se),
                   "above_resolution": o.get("resolution") is not None and abs(a) > o["resolution"],
                   "predicted": r.get("prediction_status") == "predicted"}
            for m in METHODS:
                if m == "A0":
                    rec[m] = (0.0, 0.0, 0)
                    continue
                v = o.get(m)
                if not v or v[0] is None:
                    rec[m] = None
                    continue
                p, sp = v
                if m == "C-model" and (sp is None):
                    sp = se
                rec[m] = (p, sp, cls(p, sp))
            out.append(rec)
    return out, units


def agree_size(rec, m):
    p, sp, _ = rec[m]
    return abs(p - rec["a"]) <= 3 * math.sqrt(rec["se"] ** 2 + (sp or 0) ** 2) + 0.1 * abs(rec["a"])


def usable(rec, m):
    if rec[m] is None or rec[m][2] is None:
        return False
    if m == "C-model" and not rec["predicted"]:
        return False
    return True


def rates(recs, m):
    sig = [r for r in recs if r["truth"] not in (None, 0) and usable(r, m)]
    null = [r for r in recs if r["truth"] == 0 and usable(r, m)]
    return {"sig_n": len(sig), "sign_agree": sum(r[m][2] == r["truth"] for r in sig),
            "size_agree": sum(r[m][2] == r["truth"] and agree_size(r, m) for r in sig),
            "null_n": len(null), "null_false_signal": sum(r[m][2] != 0 for r in null)}


def bootstrap_h1(recs, reps=2000, seed=0):
    by_unit = collections.defaultdict(list)
    for r in recs:
        if r["truth"] not in (None, 0) and usable(r, "C-model") and usable(r, "A-exec"):
            by_unit[r["unit"]].append(r)
    units = list(by_unit)
    if not units:
        return None
    rng = random.Random(seed)

    def diff(sample):
        rs = [r for u in sample for r in by_unit[u]]
        c = sum(r["C-model"][2] == r["truth"] for r in rs) / len(rs)
        a = sum(r["A-exec"][2] == r["truth"] for r in rs) / len(rs)
        return c - a
    point = diff(units)
    ds = sorted(diff([rng.choice(units) for _ in units]) for _ in range(reps))
    return {"units": len(units), "outputs": sum(len(v) for v in by_unit.values()), "diff": point,
            "ci95": [ds[int(0.025 * reps)], ds[int(0.975 * reps) - 1]]}


def premise_attribution(recs):
    """A-exec errors and the single premises that reproduce them (same predicted class, or for a size error the
    same side of the truth)."""
    errors, explained, by_premise = 0, 0, collections.Counter()
    for r in recs:
        if not usable(r, "A-exec") or r["truth"] is None:
            continue
        p, sp, c = r["A-exec"]
        class_err = c != r["truth"]
        size_err = (not class_err) and r["truth"] != 0 and not agree_size(r, "A-exec")
        if not (class_err or size_err):
            continue
        errors += 1
        hits = []
        for q in PREMISES:
            m = f"A-{q}"
            if not usable(r, m):
                continue
            pq, sq, cq = r[m]
            if class_err and cq == c:
                hits.append(q)
            elif size_err and cq == r["truth"] and not agree_size(r, m) and (pq - r["a"]) * (p - r["a"]) > 0:
                hits.append(q)
        if hits:
            explained += 1
            by_premise.update(hits)
    return {"errors": errors, "explained_by_one_premise": explained, "by_premise": dict(by_premise)}


def table(recs, key):
    groups = collections.defaultdict(list)
    for r in recs:
        groups[key(r)].append(r)
    return {str(k): {m: rates(v, m) for m in ("C-exec", "C-model", "A-exec", "A0")} | {"outputs": len(v)}
            for k, v in sorted(groups.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("units")
    ap.add_argument("--out")
    a = ap.parse_args()
    recs, units = records(a.units)
    primary = [r for r in recs if r["bitwise"]]
    secondary = [r for r in recs if not r["bitwise"]]
    s = {"units": len(units), "unit_errors": sum(u["error"] is not None for u in units),
         "verify": dict(collections.Counter(u["verify"] for u in units)),
         "outputs": len(recs), "primary_outputs": len(primary), "secondary_outputs": len(secondary),
         "primary": {m: rates(primary, m) for m in METHODS},
         "secondary": {m: rates(secondary, m) for m in METHODS if m != "C-exec"},
         "H1": bootstrap_h1(primary), "H3": premise_attribution(primary),
         "H3_secondary": premise_attribution(secondary),
         "by_config_elem": table(primary, lambda r: (r["config"], r["elem"])),
         "by_ktype": table(primary, lambda r: r["ktype"]),
         "by_model": table(primary, lambda r: r["model"]),
         "above_resolution": table(primary, lambda r: r["above_resolution"]),
         "single_element_outputs": SINGLES}
    text = json.dumps(s, indent=1)
    if a.out:
        Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
