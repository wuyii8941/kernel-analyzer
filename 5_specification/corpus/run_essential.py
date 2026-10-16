#!/usr/bin/env python3
"""PLAN_v3 V3 for the essential-round conditions: regenerate every condition's inputs with the frozen generator
(pre-reorg-20261009:scripts/essential, extracted to .cache/spec_v3/tag_scripts/essential) and evaluate f through the
single entry kernel_spec.registry.query (draft contracts allowed, labelled); record the outcome of every output.

    python 5_specification/corpus/run_essential.py --family matmul_linear reductions gather_layout

Writes corpus/results/essential_<family>.jsonl (one line per condition and seed) and prints a summary. V3 accepts the
outcomes value, domain_undefined and invalid_input; anything else is a gap.
"""
from __future__ import annotations

import argparse
import collections
import importlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = HERE.parent
REPO = SPEC.parent
TAG = REPO / ".cache/spec_v3/tag_scripts/essential"
sys.path.insert(0, str(SPEC))
sys.path.insert(0, str(TAG))

from kernel_spec import registry  # noqa: E402

ACCEPTED = {"value", "domain_undefined", "invalid_input"}
PLAN = REPO / "1_experiments/essential_bugs_round/results/phase2b/coverage_plan.json"


def ten(a, dtype="f32"):
    a = np.asarray(a, dtype=np.float64)
    return {"t": dtype, "v": a.tolist(), "shape": list(a.shape)}


def ints(a):
    a = np.asarray(a)
    return {"t": "i64", "v": a.astype(int).tolist(), "shape": list(a.shape)}


def q(cid, op, args, kwargs=None, profile="real"):
    r = registry.query(cid, profile, {"op": op, "args": args, "kwargs": kwargs or {}}, allow_draft=True)
    return r.outcome, r.note


def load_basic():
    m = importlib.import_module("p2b_basic")

    def plan(family, keys):
        p = json.loads(PLAN.read_text())["tier1"][family]["conditions"]
        out, seen = [], {}
        for c in p:
            key = tuple(str(c[k]) for k in keys)
            if key in seen:
                continue
            d = dict(c)
            d["id"] = family[:4] + "_" + "_".join(str(c[k]).replace("(", "").replace(")", "").replace(",", "x").replace(" ", "")
                                                   for k in keys)
            seen[key] = d
            out.append(d)
        return out
    m.plan = plan
    return m


# ------------------------------------------------------------------ families
def fam_matmul(c, t):
    a, b, bias, g = t["a"], t["b"], t["bias"], t["g"]
    out = {}
    if bias is not None and np.ndim(a) == 2:
        out["out"] = q("MATMUL", "aten.addmm", [ten(bias), ten(a), ten(b)])
        out["grads"] = q("MATMUL", "vjp.aten.addmm", [ten(g), ten(bias), ten(a), ten(b)])
    else:
        out["out"] = q("MATMUL", "aten.matmul", [ten(a), ten(b)])
        out["grads"] = q("MATMUL", "vjp.aten.matmul", [ten(g), ten(a), ten(b)])
        if bias is not None:
            out["bias_add"] = q("ARITH", "aten.add", [ten(np.matmul(a, b)), ten(bias)])
            out["dbias"] = q("REDUCE", "aten.sum", [ten(g), list(range(np.ndim(g) - 1))])
    return out


def fam_reductions(c, t):
    x, g, op = t["x"], t["g"], c["op"]
    dims = {"single": [-1], "multi": [-2, -1], "all": None, "empty-extent": [-1]}[c["dims"]]
    out = {}
    if op in ("softmax", "log_softmax", "cumsum"):
        if dims is None or len(dims) > 1:          # the generator flattens the reduced axes first
            flat = x.reshape(x.shape[0], -1) if dims is not None else x.reshape(1, -1)
            xx, gg = flat, (g.reshape(flat.shape) if g is not None else None)
        else:
            xx, gg = x, g
        if op == "cumsum":
            out["out"] = q("REDUCE", "aten.cumsum", [ten(xx), -1])
            out["dx"] = q("REDUCE", "vjp.aten.cumsum", [ten(gg), ten(xx), -1])
        else:
            aten = "aten._softmax" if op == "softmax" else "aten._log_softmax"
            out["out"] = q("SOFTMAX", aten, [ten(xx), -1, False])
            if out["out"][0] == "value":
                y = _value_of(q_value("SOFTMAX", aten, [ten(xx), -1, False]))
                bwd = "aten._softmax_backward_data" if op == "softmax" else "aten._log_softmax_backward_data"
                out["dx"] = q("SOFTMAX", bwd, [ten(gg), y, -1, "f32"])
        return out
    kw_dims = dims if dims is not None else list(range(x.ndim))
    if op in ("sum", "mean", "prod", "amax", "amin", "logsumexp"):
        if op == "prod" and dims is not None and len(dims) > 1:
            out["out"] = q("REDUCE", "aten.prod", [ten(x.reshape(x.shape[0], -1)), -1])
            out["dx"] = q("REDUCE", "vjp.aten.prod", [ten(g.reshape(-1)), ten(x.reshape(x.shape[0], -1)), -1])
        elif op == "prod" and dims is None:
            out["out"] = q("REDUCE", "aten.prod", [ten(x.reshape(-1)), 0])
            out["dx"] = q("REDUCE", "vjp.aten.prod", [ten(np.asarray(g).reshape(())), ten(x.reshape(-1)), 0])
        elif op == "prod":
            out["out"] = q("REDUCE", "aten.prod", [ten(x), dims[0]])
            out["dx"] = q("REDUCE", "vjp.aten.prod", [ten(g), ten(x), dims[0]])
        else:
            out["out"] = q("REDUCE", f"aten.{op}", [ten(x), kw_dims])
            out["dx"] = q("REDUCE", f"vjp.aten.{op}", [ten(g), ten(x), kw_dims])
    elif op in ("var", "std"):
        out["out"] = q("REDUCE", f"aten.{op}", [ten(x), kw_dims])
        out["dx"] = q("REDUCE", f"vjp.aten.{op}", [ten(g), ten(x), kw_dims])
    else:
        raise KeyError(op)
    return out


def fam_gather(c, t):
    x, idx, g, op = t["x"], t["idx"], t["g"], c["op"]
    out = {}
    if op == "gather":
        out["out"] = q("INDEX", "aten.gather", [ten(x), 1, ints(idx)])
        out["dx"] = q("INDEX", "vjp.aten.gather", [ten(g), ten(x), 1, ints(idx)])
    elif op == "take_along_dim":
        out["out"] = q("INDEX", "aten.take_along_dim", [ten(x), ints(idx), 1])
        out["dx"] = q("INDEX", "vjp.aten.gather", [ten(g), ten(x), 1, ints(idx)])
    elif op == "index_select":
        sel = q_value("INDEX", "aten.index_select", [ten(x), 1, ints(idx[0])])
        out["out"] = (sel.outcome, sel.note)
        if sel.outcome == "value":
            out["slice"] = q("LAYOUT", "aten.slice", [{"t": "f32", "v": sel.fields["out"]}, 1, 0, 5])
        out["dx"] = q("INDEX", "vjp.aten.index_select", [ten(np.pad(g, ((0, 0), (0, max(0, len(idx[0]) - 5))))[:, :len(idx[0])]),
                                                         ten(x), 1, ints(idx[0])])
    elif op == "cat_split":
        parts = q_value("LAYOUT", "aten.split_with_sizes", [ten(x), [2, 3, 4], 1])
        out["split"] = (parts.outcome, parts.note)
        if parts.outcome == "value":
            out["cat"] = q("LAYOUT", "aten.cat", [[{"t": "f32", "v": p} for p in parts.fields["out"]], 1])
        out["dx"] = q("LAYOUT", "aten.constant_pad_nd", [ten(g), [0, 4]])
    else:                                          # view_alias: columns 1..3 doubled through a view, then [:, :5]
        out["scaled"] = q("ARITH", "aten.mul", [ten(x[:, 1:4]), 2])
        out["slice"] = q("LAYOUT", "aten.slice", [ten(x), 1, 0, 5])
        out["dx"] = q("ARITH", "aten.mul", [ten(g), ten(np.array([1, 2, 2, 2, 1], dtype=float))])
    return out


def q_value(cid, op, args, kwargs=None, profile="real"):
    return registry.query(cid, profile, {"op": op, "args": args, "kwargs": kwargs or {}}, allow_draft=True)


def _value_of(r):
    return {"t": "f32", "v": _floatish(r.fields["out"])}


def _floatish(v):
    """Values handed on to the next query: exact rationals stay exact; enclosures are passed as their midpoint
    rationals only for the backward formula's inputs (documented in the result as 'chained')."""
    from fractions import Fraction
    from kernel_spec.num import IV, ends
    if isinstance(v, list):
        return [_floatish(x) for x in v]
    if isinstance(v, IV):
        a, b = ends(v)
        m = (a + b) / 2
        return f"{m.numerator}/{m.denominator}"
    if isinstance(v, Fraction):
        return f"{v.numerator}/{v.denominator}"
    return v


FAMILIES = {"matmul_linear": ("Matmul", fam_matmul), "reductions": ("Reductions", fam_reductions),
            "gather_layout": ("Gather", fam_gather)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", nargs="+", default=list(FAMILIES))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    basic = load_basic()
    (HERE / "results").mkdir(exist_ok=True)
    gaps = 0
    for fam in a.family:
        cls, fn = FAMILIES[fam]
        F = getattr(basic, cls)()
        conds = F.conditions()
        counts = collections.Counter()
        with open(HERE / f"results/essential_{fam}.jsonl", "w") as fh:
            for c in conds:
                for seed in a.seeds:
                    t = F.inputs(c, seed, "base")
                    res = fn(c, t)
                    outcomes = {k: v[0] for k, v in res.items()}
                    worst = "value" if all(o == "value" for o in outcomes.values()) else \
                        next((o for o in outcomes.values() if o not in ACCEPTED), None) or \
                        ("domain_undefined" if "domain_undefined" in outcomes.values() else "invalid_input")
                    counts[worst] += 1
                    fh.write(json.dumps({"family": fam, "condition": c["id"], "seed": seed, "outcome": worst,
                                         "outputs": outcomes, "notes": {k: v[1][:160] for k, v in res.items() if v[1]}}) + "\n")
        bad = sum(v for k, v in counts.items() if k not in ACCEPTED)
        gaps += bad
        print(f"{fam}: {len(conds)} conditions x {len(a.seeds)} seeds -> {dict(counts)}; gaps {bad}")
    return 1 if gaps else 0


if __name__ == "__main__":
    sys.exit(main())
