#!/usr/bin/env python3
"""Section 3 of the general-round protocol: reference quality (complete rate, width over the output dtype's ulp,
fraction resolving 1/8 ulp, phase timings) for programs outside the tier-1 mode-B runs -- the 20 numerical-stream
programs (bf16 / fp16 Inductor), the 10 G6 compositions (float32 Inductor, forward and backward) and the three
torch.sparse Triton operators of the unified-entry demo -- with 3 units each (quality needs no statistics).
KA_ACCUMULATION=gamma gives the tool-2.3 accumulation for the before / after comparison.

    python scripts/general/reference_quality_probe.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "scripts/essential", "scripts/closure", "examples/general"):
    sys.path.insert(0, str(ROOT / p))
import common  # noqa: E402
from kernel_analyzer.measure import reference_quality  # noqa: E402

MODE = os.environ.get("KA_ACCUMULATION", "exact")
OUT = ROOT / f"results/general/reference_quality_probe_{MODE}.json"


def probe(name, setup, inputs, launch, group):
    dtypes = {}

    def launch2(t):
        outs = launch(t)
        dtypes.update({k: str(v.dtype).replace("torch.", "") for k, v in outs.items()})
        return outs
    t0 = time.time()
    rep, keep = common.fr_run(f"probe/{name}", setup, inputs, launch2, lambda _t: None, seeds=(0, 1, 2))
    outs = {}
    for k, rows in keep.items():
        outs[k] = reference_quality(rows, dtypes.get(k, "float32"), 0.125)
    return {"group": group, "outputs": outs, "not_written_by_triton": rep.get("outputs_not_written_by_triton"),
            "aborted": rep.get("outputs_whose_writing_programs_aborted"), "coverage": rep.get("ttir_coverage_complete"),
            "timing_seconds": rep.get("timing_seconds"), "seconds": round(time.time() - t0, 1)}


def stream_cases():
    import numerical_stream as NS
    for name, dt, make, fn in NS.programs():
        st = {}

        def setup(fn=fn, make=make, dt=dt, st=st):
            torch._dynamo.reset()
            st["fn"] = torch.compile(fn, dynamic=False)
            st["fn"](*[a.to("cuda", dt) for a in make(0)])
            torch.cuda.synchronize()

        def inputs(seed, make=make, dt=dt):
            return {"args": [a.to("cuda", dt) for a in make(2000 + seed)], "_seed": seed}

        def launch(t, st=st):
            return {"out": st["fn"](*t["args"])}
        yield name, setup, inputs, launch, "numerical stream (bf16 / fp16 Inductor)"


def g6_cases():
    import p2b_g6_g7 as G
    for name, make in G.G6.items():
        P64, inputs0, fn = make()
        st = {}

        def to_dev(d):
            return {k: (v.to("cuda", torch.float32).requires_grad_(True) if v.is_floating_point() else v.to("cuda"))
                    for k, v in d.items()}

        def call(p, x, fn=fn, name=name):
            out = fn(p, x)
            outs = {"out": out}
            grads_of = [k for k, v in list(p.items()) + list(x.items()) if isinstance(v, torch.Tensor) and v.requires_grad]
            if out.requires_grad and name != "loss_then_adamw_step":
                gs = torch.autograd.grad(out, [({**p, **x})[k] for k in grads_of], torch.ones_like(out) * 0.5,
                                         allow_unused=True)
                outs.update({f"d_{k}": g for k, g in zip(grads_of, gs) if g is not None})
            return outs

        def setup(st=st, P64=P64, call=call, inputs0=inputs0):
            torch._dynamo.reset()
            st["p"] = to_dev(P64)
            st["fn"] = torch.compile(call, dynamic=False)
            st["fn"](st["p"], to_dev(inputs0(0)))
            torch.cuda.synchronize()

        def inputs(seed, inputs0=inputs0):
            d = to_dev(inputs0(seed))
            d["_seed"] = seed
            return d

        def launch(t, st=st):
            return st["fn"](st["p"], {k: v for k, v in t.items() if k != "_seed"})
        yield name, setup, inputs, launch, "G6 compositions (float32 Inductor, fwd + bwd)"


def bsr_cases():
    import bsr_ops as B
    decls = {"bsr_softmax": (B.bsr_softmax, lambda s: {"x": B.block_sparse(s, (64, 64), "float32"), "block": 16}),
             "bsr_dense_mm": (B.bsr_dense_mm, lambda s: {"x": B.block_sparse(s, (64, 64), "float32"), "block": 16,
                                                          "d": torch.randn(64, 32, generator=torch.Generator().manual_seed(s)).cuda()}),
             "sampled_addmm": (B.sampled_addmm, lambda s: {"m": B.block_sparse(s, (64, 64), "float32"), "block": 16,
                                                            "a": torch.randn(64, 48, generator=torch.Generator().manual_seed(s)).cuda(),
                                                            "b": torch.randn(48, 64, generator=torch.Generator().manual_seed(100 + s)).cuda()})}
    for name, (call, mk) in decls.items():
        def setup(call=call, mk=mk):
            call(mk(0))
            torch.cuda.synchronize()

        def inputs(seed, mk=mk):
            d = mk(seed)
            d["_seed"] = seed
            return d

        def launch(t, call=call):
            return call({k: v for k, v in t.items() if k != "_seed"})
        yield name, setup, inputs, launch, "new operators (torch.sparse Triton kernels)"


def main():
    res = {}
    for gen in (stream_cases, g6_cases, bsr_cases):
        for name, setup, inputs, launch, group in gen():
            try:
                res[name] = probe(name, setup, inputs, launch, group)
            except Exception as exc:  # noqa: BLE001
                res[name] = {"group": group, "status": "error", "reason": f"{type(exc).__name__}: {exc}"[:400],
                             "trace": traceback.format_exc()[-1200:]}
            r = res[name]
            print(name, {k: (v["complete_rate"], v["resolved_fraction"], v["width_over_ulp"]["median"]) for k, v in
                         r.get("outputs", {}).items()} if "outputs" in r else r["reason"][:200], flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"accumulation": MODE, "programs": res}, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
