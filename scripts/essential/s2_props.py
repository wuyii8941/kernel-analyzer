#!/usr/bin/env python3
"""Phase-1 supplement S2 (docs/protocol_essential_bugs_phase2_20261007.md section 1): the properties phase 1 did not check,
evaluated on the suite-s2 candidate runs (reference-free; post-hoc supplement, reported apart from phase-1 scores).

    ESSENTIAL_SUITE=s2 python scripts/essential/s2_props.py     # -> results/essential/phase1_supplement/s2_properties.json
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("ESSENTIAL_SUITE", "s2")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import candidates as K  # noqa: E402
import classify as Q  # noqa: E402
import common  # noqa: E402
import run_phase1 as R  # noqa: E402

OUT = common.ROOT / "results/essential/phase1_supplement"


def close(a, b, dtype, exact):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape:
        return False
    d, _ = common.deviation(a, b, b, dtype if dtype in common.TAU else "float32", exact)
    return not d.any()


def props(family, cond, cand, rec):
    dtype = Q.cand_dtype(cand)
    exact = Q.exact_inputs(family, cond)
    b = rec.get("base")
    res = {}
    if not b or b["status"] != "ok":
        return res
    o = b["outputs"]
    if family == "ce":
        t = rec.get("masked_rows_changed")
        if t and t["status"] == "ok":
            g0, g1 = np.asarray(o["grad"]), np.asarray(t["outputs"]["grad"])   # rows compared in main(): per-seed targets
            res["masked_rows_do_not_matter"] = close(t["outputs"]["loss"], o["loss"], dtype, False)
            res["masked_rows_bitwise"] = bool(np.array_equal(np.asarray(t["outputs"]["loss"]), np.asarray(o["loss"]), equal_nan=True))
            res["_grads"] = (g0, g1)
    elif family == "pool":
        t = rec.get("channels_reversed")
        if t and t["status"] == "ok":
            res["channel_equivariance"] = bool(np.array_equal(np.asarray(t["outputs"]["out"])[::-1], np.asarray(o["out"]), equal_nan=True))
        ty, txy = rec.get("lin_y"), rec.get("lin_2x3y")
        if ty and txy and ty["status"] == "ok" and txy["status"] == "ok":
            res["linearity"] = close(txy["outputs"]["out"], 2 * np.asarray(o["out"]) + 3 * np.asarray(ty["outputs"]["out"]), dtype, False)
    else:
        ty, txy = rec.get("lin_y"), rec.get("lin_2x3y")
        if ty and txy and ty["status"] == "ok" and txy["status"] == "ok":
            res["linearity"] = close(txy["outputs"]["out"], 2 * np.asarray(o["out"]) + 3 * np.asarray(ty["outputs"]["out"]), dtype, exact)
    return res


def main():
    out = {}
    for fam in ("ce", "pool", "index"):
        conds = {c["id"]: c for c in R.FAMILIES[fam][0]() if c["op"] != "flce"}
        for cand in K.FAMILY_CANDIDATES[fam] + ["nightly_eager_cuda32", "nightly_inductor_cuda32"]:
            p = R.CACHE / "raw" / fam / f"{cand}.pkl"
            if not p.exists():
                continue
            raw = pickle.loads(p.read_bytes())
            judged = Q.cand_dtype(cand) in common.TAU
            tally, examples = {}, []
            for (cid, seed), rec in raw.items():
                cond = conds[cid]
                res = props(fam, cond, cand, rec)
                if fam == "ce" and "_grads" in res:
                    g0, g1 = res.pop("_grads")
                    tgt = R.FAMILIES["ce"][1](cond, seed)["target"]
                    ign = np.array([x == cond["ignore_index"] for x in tgt])
                    res["other_rows_grad_unchanged"] = close(g1[~ign], g0[~ign], Q.cand_dtype(cand), False) if (~ign).any() else None
                    res["ignored_rows_grad_zero"] = bool(np.all(g1[ign] == 0)) if np.isfinite(g1).all() else None
                for k, v in res.items():
                    if v is None:
                        continue
                    t = tally.setdefault(k, {"checked": 0, "violated": 0})
                    t["checked"] += 1
                    t["violated"] += int(not v)
                    if not v and len(examples) < 5:
                        examples.append({"condition": cid, "seed": seed, "property": k})
            out[f"{fam}/{cand}"] = {"judged": judged, "properties": tally, "violation_examples": examples}
            print(fam, cand, "judged" if judged else "recorded", {k: f"{v['violated']}/{v['checked']}" for k, v in tally.items()})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "s2_properties.json").write_text(json.dumps(out, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
