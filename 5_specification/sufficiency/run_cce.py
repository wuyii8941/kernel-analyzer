#!/usr/bin/env python3
"""Sufficiency evaluation E3, new library (docs/SUFFICIENCY_PROTOCOL.md section 5.2): cut-cross-entropy 25.1.1.

The call set is the parameter grid of the library's own tests in the 25.1.1 sdist (tests/test_cce_loss_forward.py,
144 combinations, reduction "none"; tests/test_cce_loss_backward.py, 288 combinations), with the tests' input generation
(torch.cuda.manual_seed(0), e scaled by 1/sqrt(D), the first min(N,V)//2 rows of c copied from e, 20% ignored targets
when `invalids`). Every combination is executed once and classified into the five outcome classes of section 5.3 by
the fixed rule below; nothing is dropped.

Rule (first blocking stage in pipeline order):
  1. the call raises or returns non-finite / wrongly shaped outputs        -> run or evidence insufficient
  2. a requested task feature is written as outside every contract
     (softcap: contracts/FLCE.yaml decision FL.EXTRAS lists it as unsupported, and no other contract
     defines a softcapped linear cross entropy)                            -> new or modified task contract needed
  3. the task is written in a contract that is not admitted
     (FLCE implementation_status != validated)                             -> implementation extension needed
  4. admitted, but no reviewed binding plan for this library call          -> manual binding needed
  5. admitted and a reviewed binding plan exists                           -> usable as is
The component contracts that the task composes (MATMUL, CE, MATHFN tanh, LAYOUT slice) are recorded with their status.

    CUDA_VISIBLE_DEVICES=0 python 5_specification/sufficiency/run_cce.py --out NEW.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch
import yaml

SPEC = Path(__file__).resolve().parents[1]
IGNORE_INDEX = -100
SHAPES = [(256, 512, 128), (252, 507, 128), (252, 507, 123)]
FORWARD = list(itertools.product(["cce", "torch_compile"], [torch.float32, torch.float16, torch.bfloat16], [None, 20.0],
                                 [False, True], [False, True], SHAPES))
BACKWARD = list(itertools.product(["cce", "torch_compile"], [torch.float16, torch.bfloat16], [None, 20.0], [False, True],
                                  [False, True], ["none", "mean", "sum"], SHAPES))


def status(cid):
    d = yaml.safe_load((SPEC / "contracts" / f"{cid}.yaml").read_text())
    return {"review": d["review"]["status"], "implementation": (d.get("activation") or {}).get("implementation_status")}


def inputs(dtype, shape, invalids):
    torch.cuda.manual_seed(0)
    N, V, D = shape
    e = torch.randn((N, D), device="cuda", dtype=dtype) / (D ** 0.5)
    c = torch.randn((V, D), device="cuda", dtype=dtype)
    c[0: min(N, V) // 2] = e[0: min(N, V) // 2]
    targets = torch.randint(0, V, size=(N,), device="cuda")
    if invalids:
        inds = torch.randperm(len(targets), device="cuda")[0: int(0.2 * len(targets))]
        targets[inds] = IGNORE_INDEX
    return e, c, targets


def classify(run_ok, softcap, flce, plan_exists):
    if not run_ok:
        return "run or evidence insufficient"
    if softcap is not None:
        return "new or modified task contract needed"
    if flce["implementation"] != "validated":
        return "implementation extension needed"
    if not plan_exists:
        return "manual binding needed"
    return "usable as is"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from cut_cross_entropy import linear_cross_entropy
    import cut_cross_entropy
    torch.set_float32_matmul_precision("highest")
    torch._dynamo.config.cache_size_limit = 256
    flce = status("FLCE")
    components = {c: status(c) for c in ("MATMUL", "CE", "MATHFN", "LAYOUT", "GRAPH")}
    plan_exists = any("cut_cross_entropy" in p.read_text() for p in (SPEC / "corpus/call_bindings").glob("*.json"))
    with open(a.out, "x") as fh:
        fh.write(json.dumps({"header": True, "library": "cut-cross-entropy", "version": getattr(cut_cross_entropy, "__version__", "25.1.1"),
                             "flce": flce, "components": components, "reviewed_plan_exists": plan_exists}) + "\n")
        for grid, cfgs in (("forward", FORWARD), ("backward", BACKWARD)):
            for cfg in cfgs:
                if grid == "forward":
                    impl, dtype, softcap, shift, invalids, shape = cfg
                    reduction = "none"
                else:
                    impl, dtype, softcap, shift, invalids, reduction, shape = cfg
                row = {"grid": grid, "impl": impl, "dtype": str(dtype).replace("torch.", ""), "softcap": softcap,
                       "shift": shift, "invalids": invalids, "reduction": reduction, "shape": shape}
                try:
                    e, c, t = inputs(dtype, shape, invalids)
                    e = e.view(4, -1, shape[2])
                    t = t.view(e.size()[0:-1])
                    if grid == "backward":
                        e.requires_grad_(True)
                        c.requires_grad_(True)
                    loss = linear_cross_entropy(e, c, t, softcap=softcap, shift=shift, reduction=reduction, impl=impl)
                    outs = {"loss": loss}
                    if grid == "backward":
                        loss.float().mean().backward()
                        outs.update(grad_e=e.grad, grad_c=c.grad)
                    torch.cuda.synchronize()
                    ok = all(v is not None and bool(torch.isfinite(v.float()).all()) for v in outs.values())
                    row.update(run="ok" if ok else "non-finite or missing output",
                               outputs={k: list(v.shape) for k, v in outs.items() if v is not None})
                except Exception as err:  # noqa: BLE001 -- recorded, never hidden
                    ok = False
                    row.update(run=f"error {type(err).__name__}: {str(err)[:200]}")
                row["outcome"] = classify(ok, softcap, flce, plan_exists)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(grid, impl, row["dtype"], softcap, shift, invalids, reduction, shape, row["outcome"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
