#!/usr/bin/env python3
"""PLAN_v3 V3 for the composition-study library corpus (FlagGems 5.0.2 battery; Liger calls are bound by their own
contracts): construct inputs for the recorded call shapes and dtypes, and evaluate f through the single entry.
The generated inputs are not certified identical to captured inputs: some use a local seed-1 generator, while
indices and dropout use the current CUDA random state.

    CUDA_VISIBLE_DEVICES=0 python 5_specification/corpus/run_corpus.py --out NEW.jsonl

Writes a new result file and prints a summary. These regenerated-input queries
do not establish captured-kernel binding or certified workload coverage.
"""
from __future__ import annotations

import collections
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SPEC = HERE.parent
sys.path.insert(0, str(SPEC))

from kernel_spec import registry  # noqa: E402
from kernel_spec.num import OUTCOMES, Result  # noqa: E402

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
    return r


def qraw(cid, profile, inp):
    """Contracts whose input is a record of named fields (NORM, CE) rather than an op with args."""
    r = registry.query(cid, profile, inp, allow_draft=True)
    return r


def lst(t):
    return t.detach().double().cpu().tolist()


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
        "gelu": lambda: q("ACT", "aten.gelu", [X()], profile="torch"),
        "silu": lambda: q("ACT", "aten.silu", [X()], profile="torch"),
        "relu": lambda: q("ACT", "aten.relu", [X()], profile="torch"),
        # F.layer_norm / rms_norm / group_norm without weight and bias, as called; eps is the double torch passes
        # (rms_norm eps=None means finfo(dtype).eps). The calls were forward only; the contract's VJP needs an
        # upstream, ones here, which does not change the forward output.
        "layer_norm": lambda: qraw("NORM", "torch", {"op": "layer_norm", "x": lst(x_()), "normalized_shape": [512],
                                                     "weight": None, "bias": None, "eps": 1e-5,
                                                     "upstream": [[1] * 512] * 64}),
        "rms_norm": lambda: qraw("NORM", "torch", {"op": "rms_norm", "x": lst(x_()), "normalized_shape": [512],
                                                   "weight": [1] * 512, "eps": torch.finfo(dt).eps,
                                                   "upstream": [[1] * 512] * 64}),
        "group_norm": lambda: qraw("NORM", "torch", {"op": "group_norm", "x": lst(x_((4, 32, 128))), "num_groups": 8,
                                                     "weight": None, "bias": None, "eps": 1e-5,
                                                     "upstream": [[[1] * 128] * 32] * 4}),
        # targets: any valid class index gives the same outcome class; drawn as in corpus.py (default CUDA generator)
        "cross_entropy": lambda: qraw("CE", "index_unsmoothed", {
            "logits": lst(x_((64, 100)).float()), "target": randint(100, (64,)).cpu().tolist(), "weights": None,
            "eps": 0, "ignore_index": -100, "reduction": "mean", "upstream": 1}),
        "sort": lambda: q("SORT", "aten.sort", [X((16, 256)), 1]),
        "topk": lambda: q("SORT", "aten.topk", [X((16, 256)), 8, 1]),
        # Training dropout at 0 < p < 1 has no unique value reference. The reviewed RANDOM profile deliberately
        # returns unsupported here; distribution testing must be supplied separately.
        "dropout": lambda: q("RANDOM", "aten.dropout", [X(), 0.1, True],
                             profile="deterministic_forward"),
    }


PENDING = {}


def evaluate_calls(tested, output):
    """Retain every attempted query; process errors fail the run without erasing earlier records."""
    counts = collections.Counter()
    cache = {}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as fh:
        for call in tested:
            try:
                _, op, dt_name = call.split(".", 2)
                if dt_name not in cache:
                    cache[dt_name] = calls(dt_name)
                fn = cache[dt_name].get(op)
                result = fn() if fn else Result("unsupported", note=f"contract {PENDING.get(op, '?')} pending")
                if not isinstance(result, Result) or result.outcome not in OUTCOMES:
                    raise TypeError("query did not return a valid Result")
            except Exception as error:
                result = Result("implementation_error", note=f"{type(error).__name__}: {error}")
            counts[result.outcome] += 1
            row = {"call": call, "outcome": result.outcome, "status": result.outcome,
                   "note": result.note[:200], "reference_labels": result.labels,
                   "input_evidence": "regenerated", "reference_status": result.outcome,
                   "binding_status": "not_established", "semantic_verdict": "not_established"}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(call, result.outcome, result.note[:100], flush=True)
    print("calls", sum(counts.values()), dict(counts))
    return 1 if counts["implementation_error"] else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True, help="new result file; existing records are never overwritten")
    args = parser.parse_args()
    corpus = json.loads((SPEC.parent / "1_experiments/composition_study/results/corpus.json").read_text())["kernels"]
    tested = sorted({c for k in corpus for c in (k.get("calls") or [k["call"]]) if c.startswith("gems.")})
    return evaluate_calls(tested, args.out)


if __name__ == "__main__":
    sys.exit(main())
