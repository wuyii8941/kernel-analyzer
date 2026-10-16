#!/usr/bin/env python3
"""PLAN_v3 V3 for the composition-study library corpus (FlagGems 5.0.2 battery; Liger calls are bound by their own
contracts): regenerate each call's inputs exactly as 1_experiments/composition_study/corpus.py does (CUDA generator,
seed 1, cast to the call's dtype) and evaluate f through the single entry for the call's result.

    CUDA_VISIBLE_DEVICES=0 python 5_specification/corpus/run_corpus.py

Writes corpus/results/corpus_gems.jsonl and prints a summary. Calls whose contract is not written yet are listed as
pending, not as covered.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SPEC = HERE.parent
sys.path.insert(0, str(SPEC))

from kernel_spec import registry  # noqa: E402

ACCEPTED = {"value", "domain_undefined", "invalid_input"}
DT = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
TAG = {"float32": "f32", "float16": "f16", "bfloat16": "bf16"}


def ten(t, tag):
    a = t.detach().double().cpu().numpy()
    return {"t": tag, "v": a.tolist(), "shape": list(a.shape)}


def ints(t):
    a = t.detach().cpu().numpy()
    return {"t": "i64", "v": a.astype(int).tolist(), "shape": list(a.shape)}


def q(cid, op, args, kwargs=None, profile="real"):
    r = registry.query(cid, profile, {"op": op, "args": args, "kwargs": kwargs or {}}, allow_draft=True)
    return r.outcome, r.note


def calls(dt_name):
    dt, tag = DT[dt_name], TAG[dt_name]

    def x_(shape=(64, 512), pos=False):
        g = torch.Generator(device="cuda").manual_seed(1)
        v = torch.randn(*shape, device="cuda", generator=g)
        return (v.abs() + 0.1 if pos else v).to(dt)

    def X(shape=(64, 512), pos=False):
        return ten(x_(shape, pos), tag)

    def randint(hi, shape):
        return torch.randint(0, hi, shape, device="cuda")

    return {
        "abs": lambda: q("ARITH", "aten.abs", [X()]),
        "exp": lambda: q("MATHFN", "aten.exp", [X()]),
        "log": lambda: q("MATHFN", "aten.log", [X(pos=True)]),
        "sin": lambda: q("MATHFN", "aten.sin", [X()]),
        "cos": lambda: q("MATHFN", "aten.cos", [X()]),
        "tanh": lambda: q("MATHFN", "aten.tanh", [X()]),
        "sigmoid": lambda: q("MATHFN", "aten.sigmoid", [X()]),
        "rsqrt": lambda: q("MATHFN", "aten.rsqrt", [X(pos=True)]),
        "sqrt": lambda: q("MATHFN", "aten.sqrt", [X(pos=True)]),
        "erf": lambda: q("MATHFN", "aten.erf", [X()]),
        "reciprocal": lambda: q("ARITH", "aten.reciprocal", [X(pos=True)]),
        "neg": lambda: q("ARITH", "aten.neg", [X()]),
        "pow": lambda: q("ARITH", "aten.pow", [X(pos=True), 1.7]),
        "div": lambda: q("ARITH", "aten.div", [X(), X(pos=True)]),
        "add": lambda: q("ARITH", "aten.add", [X(), X()]),
        "mul": lambda: q("ARITH", "aten.mul", [X(), X()]),
        "sub": lambda: q("ARITH", "aten.sub", [X(), X()]),
        "where": lambda: q("ARITH", "aten.where", [ten(x_() > 0, "b8"), X(), ten(x_() * 2, tag)]),
        "clamp": lambda: q("ARITH", "aten.clamp", [X()], {"min": -0.5, "max": 0.5}),
        "sum": lambda: q("REDUCE", "aten.sum", [X(), [1]]),
        "mean": lambda: q("REDUCE", "aten.mean", [X(), [1]]),
        "amax": lambda: q("REDUCE", "aten.amax", [X(), [1]]),
        "prod": lambda: q("REDUCE", "aten.prod", [X((64, 16)), 1]),
        "argmax": lambda: q("REDUCE", "aten.argmax", [X(), 1]),
        "var": lambda: q("REDUCE", "aten.var", [X(), [1]]),
        "norm": lambda: q("REDUCE", "aten.norm", [X(), 2, [1]]),
        "softmax": lambda: q("SOFTMAX", "aten._softmax", [X(), 1, False]),
        "log_softmax": lambda: q("SOFTMAX", "aten._log_softmax", [X(), 1, False]),
        "cumsum": lambda: q("REDUCE", "aten.cumsum", [X(), 1]),
        "mm": lambda: q("MATMUL", "aten.mm", [X((64, 128)), X((128, 64))]),
        "bmm": lambda: q("MATMUL", "aten.bmm", [X((4, 32, 64)), X((4, 64, 32))]),
        "addmm": lambda: q("MATMUL", "aten.addmm", [X((64,)), X((64, 128)), X((128, 64))]),
        "gather": lambda: q("INDEX", "aten.gather", [X(), 1, ints(randint(512, (64, 16)))]),
        "index_select": lambda: q("INDEX", "aten.index_select", [X(), 0, ints(randint(64, (32,)))]),
        "embedding": lambda: q("INDEX", "aten.embedding", [X(), ints(randint(64, (32,)))]),
        "masked_fill": lambda: q("ARITH", "aten.masked_fill", [X(), ten(x_() > 1, "b8"), 0]),
        "max_dim": lambda: q("REDUCE", "aten.max", [X(), 1]),
        "min": lambda: q("REDUCE", "aten.amin", [X(), [0, 1]]),
        "isnan": lambda: q("LOGIC", "aten.isnan", [X()], profile="ieee_values"),
        "floor_divide": lambda: q("ARITH", "aten.floor_divide", [X(), X(pos=True)]),
        "linspace": lambda: q("LAYOUT", "aten.linspace", [0, 1, 1000]),
    }


PENDING = {"gelu": "ACT", "silu": "ACT", "relu": "ACT", "layer_norm": "NORM", "rms_norm": "NORM", "group_norm": "NORM",
           "cross_entropy": "CE", "sort": "SORT", "topk": "SORT", "dropout": "RANDOM (conditioned on the kernel's mask)"}


def main():
    corpus = json.loads((SPEC.parent / "1_experiments/composition_study/results/corpus.json").read_text())["kernels"]
    tested = sorted({c for k in corpus for c in (k.get("calls") or [k["call"]]) if c.startswith("gems.")})
    (HERE / "results").mkdir(exist_ok=True)
    counts, rows = collections.Counter(), []
    cache = {}
    for call in tested:
        _, op, dt_name = call.split(".", 2)
        fns = cache.setdefault(dt_name, calls(dt_name))
        if op in fns:
            out, note = fns[op]()
            st = out if out in ACCEPTED else f"gap: {out}"
        else:
            out, note, st = None, f"contract {PENDING.get(op, '?')} pending", "pending"
        counts[st] += 1
        rows.append({"call": call, "outcome": out, "status": st, "note": (note or "")[:200]})
        print(call, st, (note or "")[:100], flush=True)
    with open(HERE / "results/corpus_gems.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("calls", len(rows), dict(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
