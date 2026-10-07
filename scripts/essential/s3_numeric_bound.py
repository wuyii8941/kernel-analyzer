#!/usr/bin/env python3
"""Phase-1 supplement S3: are the index/scatter forward deviations on extreme / mixed magnitudes rounding?

τ32 = 2^-12 (1 + |f|) is relative to |f|; a float32 sum whose terms cancel cannot meet it.  For every element where a
candidate's forward output deviates from f by more than τ32, compare |K - f| with the a-priori forward-error bound of
recursive float32 summation in any order: γ_k · Σ|terms| (/ count for mean), γ_k = k u / (1 - k u), u = 2^-24,
k = number of terms + 2 (the alpha product and the final division).  f is the exact sum (math.fsum on float64 terms,
which are exact: inputs are float32 values, alpha = 2.5).  Elements within the bound are numerical (cancellation),
not semantic.  Also checks that each deviating element involves terms of opposite sign.

    ESSENTIAL_SUITE=s3 python scripts/essential/s3_numeric_bound.py   # -> results/essential/phase1_supplement/s3/numeric_bound.json
"""
import json
import math
import os
import pickle
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

os.environ.setdefault("ESSENTIAL_SUITE", "s3")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import conditions_supplement as CS  # noqa: E402
import run_phase1 as R  # noqa: E402

U = 2.0 ** -24
TAU32 = 2.0 ** -12
CANDS = ["eager_cpu32", "eager_cuda32", "inductor_cuda32", "nightly_eager_cuda32", "nightly_inductor_cuda32"]
OUT = R.ROOT / "results/essential/phase1_supplement/s3/numeric_bound.json"


def f32(a):
    return np.asarray(a, dtype=np.float64).astype(np.float32).astype(np.float64)


def groups(cond, inp):
    """position in self -> list of source values (float64 of the float32-rounded inputs)."""
    shape, dim = CS._SHAPES[cond["shape"]]
    src, idx = f32(inp["source"]), np.asarray(inp["index"])
    g = {}
    if cond["op"] in ("index_add", "index_reduce"):
        for j, t in enumerate(idx):
            if dim == 0:
                g.setdefault((int(t),), []).append(src[j])
            else:
                for r in range(src.shape[0]):
                    g.setdefault((r, int(t)), []).append(src[r, j])
    else:
        for pos in np.ndindex(*idx.shape):
            tgt = list(pos)
            tgt[dim] = int(idx[pos])
            g.setdefault(tuple(tgt), []).append(src[pos])
    return g


def exact(cond, selfv, vals):
    red = cond.get("reduce")
    if cond["op"] == "index_add":
        terms = [selfv] + [2.5 * v for v in vals]
        return math.fsum(terms), terms, 1
    terms = ([selfv] if cond.get("include_self", True) else []) + list(vals)
    s = Fraction(0)
    for t in terms:
        s += Fraction(t)
    if red == "mean":
        return float(s / len(terms)), terms, len(terms)
    return float(s), terms, 1


def main():
    conds = {c["id"]: c for c in CS.index_conditions() if c.get("reduce") in (None, "sum", "mean")}
    out = {"rule": __doc__.split("\n\n")[1].replace("\n", " "), "candidates": {}}
    for cand in CANDS:
        p = R.CACHE / "raw" / "index" / f"{cand}.pkl"
        if not p.exists():
            continue
        raw = pickle.loads(p.read_bytes())
        n_dev = n_in = 0
        opp = 0
        worst, examples = 0.0, []
        for (cid, seed), rec in raw.items():
            if cid not in conds or rec.get("base", {}).get("status") != "ok":
                continue
            cond = conds[cid]
            inp = CS.make_index_inputs(cond, seed)
            selfv = f32(inp["self"])
            k = np.asarray(rec["base"]["outputs"]["out"], dtype=np.float64)
            for pos, vals in groups(cond, inp).items():
                f, terms, div = exact(cond, selfv[pos], vals)
                d = abs(k[pos] - f)
                if not d > TAU32 * (1 + abs(f)):
                    continue
                n_dev += 1
                kk = len(terms) + 2
                bound = kk * U / (1 - kk * U) * math.fsum(abs(t) for t in terms) / div
                ok = d <= bound
                n_in += int(ok)
                opp += int(min(terms) < 0 < max(terms))
                worst = max(worst, d / bound if bound else math.inf)
                if len(examples) < 6 or not ok:
                    examples.append({"condition": cid, "seed": seed, "position": list(pos), "K": k[pos], "f": f,
                                     "abs_dev": d, "tau32": TAU32 * (1 + abs(f)), "bound": bound, "within_bound": bool(ok),
                                     "terms": len(terms), "sum_abs_over_abs_f": math.fsum(abs(t) for t in terms) / div / max(abs(f), 1e-300)})
        out["candidates"][cand] = {"elements_beyond_tau32": n_dev, "within_summation_bound": n_in,
                                   "with_terms_of_both_signs": opp, "max_dev_over_bound": worst, "examples": examples[:12]}
        print(cand, f"beyond τ32 {n_dev}, within the summation bound {n_in}, both signs {opp}, max dev/bound {worst:.3g}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
