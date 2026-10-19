#!/usr/bin/env python3
"""PLAN_v3 V3 for the essential-round conditions: regenerate every condition's inputs with the frozen generator
(pre-reorg-20261009:scripts/essential, extracted to .cache/spec_v3/tag_scripts/essential) and evaluate f through the
single entry kernel_spec.registry.query (draft contracts allowed, labelled); record the outcome of every output.

    python 5_specification/corpus/run_essential.py --out-dir NEW --family matmul_linear reductions gather_layout

Writes essential_<family>.jsonl in a new directory. Query outcomes, draft labels and
unestablished kernel binding remain separate; query completion is not certified coverage.
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
from kernel_spec.num import OUTCOMES  # noqa: E402

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
    return r.outcome, r.note, r.labels


def qgraph(graph):
    """Compose within the reference evaluator; never round an intermediate to feed the next query."""
    result = registry.query("GRAPH", "eager", {"graph": graph}, allow_draft=True)
    return result.outcome, result.note, result.labels


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
            out["bias_add"] = qgraph({"inputs": {"a": ten(a), "b": ten(b), "bias": ten(bias)},
                                      "nodes": [{"name": "mm", "op": "aten.matmul", "args": [{"ref": "a"}, {"ref": "b"}]},
                                                {"name": "y", "op": "aten.add", "args": [{"ref": "mm"}, {"ref": "bias"}]}],
                                      "outputs": ["y"]})
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
            bwd = "aten._softmax_backward_data" if op == "softmax" else "aten._log_softmax_backward_data"
            status = qgraph({"inputs": {"x": ten(xx), "g": ten(gg)},
                             "nodes": [{"name": "y", "op": aten, "args": [{"ref": "x"}, -1, False]},
                                       {"name": "dx", "op": bwd,
                                        "args": [{"ref": "g"}, {"ref": "y"}, -1, "f32"]}],
                             "outputs": ["y", "dx"]})
            out["out"], out["dx"] = status, status
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
        out["out"] = (sel.outcome, sel.note, sel.labels)
        if sel.outcome == "value":
            out["slice"] = q("LAYOUT", "aten.slice", [{"t": "f32", "v": sel.fields["out"]}, 1, 0, 5])
        out["dx"] = q("INDEX", "vjp.aten.index_select", [ten(np.pad(g, ((0, 0), (0, max(0, len(idx[0]) - 5))))[:, :len(idx[0])]),
                                                         ten(x), 1, ints(idx[0])])
    elif op == "cat_split":
        parts = q_value("LAYOUT", "aten.split_with_sizes", [ten(x), [2, 3, 4], 1])
        out["split"] = (parts.outcome, parts.note, parts.labels)
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


FAMILIES = {"matmul_linear": ("Matmul", fam_matmul), "reductions": ("Reductions", fam_reductions),
            "gather_layout": ("Gather", fam_gather)}


def run_family(fam, generator, fn, seeds, path):
    """Preserve every condition, including exceptions, without replacing existing evidence."""
    counts = collections.Counter()
    with Path(path).open("x") as fh:
        for condition in generator.conditions():
            for seed in seeds:
                try:
                    tensors = generator.inputs(condition, seed, "base")
                    results = fn(condition, tensors)
                    if not isinstance(results, dict) or not results:
                        raise ValueError("condition returned no output results")
                    for value in results.values():
                        if (not isinstance(value, tuple) or len(value) != 3 or value[0] not in OUTCOMES
                                or not isinstance(value[1], str) or not isinstance(value[2], dict)):
                            raise ValueError("output result must retain outcome, note and labels")
                except Exception as error:
                    results = {"query": ("implementation_error", f"{type(error).__name__}: {error}", {})}
                outcomes = {name: value[0] for name, value in results.items()}
                worst = next(state for state in ("implementation_error", "unsupported", "not_established",
                                                 "invalid_input", "domain_undefined", "value")
                             if state in outcomes.values())
                counts[worst] += 1
                row = {"family": fam, "condition": condition["id"], "seed": seed, "outcome": worst,
                       "outputs": outcomes, "notes": {k: v[1][:200] for k, v in results.items() if v[1]},
                       "reference_labels": {k: v[2] for k, v in results.items()},
                       "input_evidence": "regenerated", "binding_status": "not_established",
                       "semantic_verdict": "not_established"}
                fh.write(json.dumps(row) + "\n")
                fh.flush()
    if not counts:
        raise ValueError(f"{fam}: no query records were generated")
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", nargs="+", choices=list(FAMILIES), default=list(FAMILIES))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out-dir", type=Path, required=True, help="new directory; existing records are never overwritten")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=False)
    basic = load_basic()
    gaps = 0
    for fam in a.family:
        cls, fn = FAMILIES[fam]
        F = getattr(basic, cls)()
        counts = run_family(fam, F, fn, a.seeds, a.out_dir / f"essential_{fam}.jsonl")
        bad = sum(v for k, v in counts.items() if k not in ACCEPTED)
        gaps += bad
        print(f"{fam}: {sum(counts.values())} queries -> {dict(counts)}; gaps {bad}")
    return 1 if gaps else 0


if __name__ == "__main__":
    sys.exit(main())
