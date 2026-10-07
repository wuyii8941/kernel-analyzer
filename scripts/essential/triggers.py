"""Item D (docs/protocol_essential_bugs_phase2_20261007.md): reusable trigger checks derived from B020-B022, pre-registered
for 2b.  Independent of the classifier; each returns {"check", "passed", "details"}.

D1 (B020 type): in an extremum reduction, a value that is EXCLUDED from the reduction but equal to its result must not
    receive gradient nor take any from the participants: d out / d excluded = 0, the participants' coefficients at each
    output sum to the upstream gradient, and (where forward-mode AD exists) <v, J u> = <B(v), u> at a unique extremum.
D2 (B021 type): legal padding / dilation that produce windows with no contributor: every returned index lies inside the
    input (or is a declared sentinel), and the backward writes nothing (all gradients zero) - run in a child process
    with MALLOC_CHECK_=3 so a heap overwrite is reported, not silently absorbed.
D3 (B022 type): sentinel indices that the forward produces are handled by the backward: no crash, zero gradient
    (child process; a device-side assert loses the CUDA context).
D4: inputs with deterministic equality and ties (small integer domains, forced repeats).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import torch


# ------------------------------------------------------------------------------------------------ D4 inputs

def tie_inputs(rng, shape, low=-2, high=2, force_equal=None):
    """integer-valued float64 array on [low, high]; ``force_equal``: list of (pos_a, pos_b) made equal."""
    x = rng.integers(low, high + 1, shape).astype(np.float64)
    for a, b in (force_equal or []):
        x[b] = x[a]
    return x


# ------------------------------------------------------------------------------------------------ D1

def d1_extremum_excluded(fn, reduce, device="cpu", dtype=torch.float64):
    """``fn(self, index, src, reduce, include_self)`` -> out, e.g. a wrapper of torch.scatter_reduce along dim 0.
    Construction: one target position; self (excluded) equals the result; the participants have a unique extremum."""
    big, small = (3.0, 1.0) if reduce == "amax" else (1.0, 3.0)
    s = torch.tensor([big, 5.0], device=device, dtype=dtype, requires_grad=True)
    src = torch.tensor([big, small], device=device, dtype=dtype, requires_grad=True)
    idx = torch.tensor([0, 0], device=device)
    out = fn(s, idx, src, reduce, False)
    up = torch.tensor([1.0, 1.0], device=device, dtype=dtype)
    gs, gsrc = torch.autograd.grad(out, (s, src), up)
    det = {"grad_excluded_self": float(gs[0]), "grad_src": gsrc.tolist(),
           "coefficient_sum_at_target": float(gsrc.sum())}
    ok = det["grad_excluded_self"] == 0.0 and abs(det["coefficient_sum_at_target"] - 1.0) < 1e-12 and gsrc.tolist() == [1.0, 0.0]
    try:                                                      # forward-mode consistency at the unique extremum
        u = (torch.tensor([0.0, 0.0], device=device, dtype=dtype), torch.tensor([1.0, 0.0], device=device, dtype=dtype))
        _, jv = torch.func.jvp(lambda a, b: fn(a, idx, b, reduce, False), (s.detach(), src.detach()), u)
        det["jvp_target"] = float(jv[0])
        ok = ok and float(jv[0]) == float(gsrc[0])
    except Exception as exc:  # noqa: BLE001
        det["jvp"] = f"not available: {type(exc).__name__}"
    return {"check": "D1", "passed": bool(ok), "details": det}


# ------------------------------------------------------------------------------------------------ D2 / D3 (child process)

_CHILD = r"""
import json, sys, torch, torch.nn.functional as F
nd, device, compiled = int(sys.argv[1]), sys.argv[2], sys.argv[3] == "1"
pool = {1: F.max_pool1d, 2: F.max_pool2d, 3: F.max_pool3d}[nd]
C = 4
x = torch.arange(1.0, C + 1, device=device).reshape((1, C) + (1,) * nd).requires_grad_(True)
f = lambda t: pool(t, 2, 1, 1, dilation=2, return_indices=True)
if compiled:
    f = torch.compile(f, fullgraph=True)
y, idx = f(x)
y.backward(torch.arange(10.0, 10.0 + C, device=device).reshape(y.shape))
if device == "cuda":
    torch.cuda.synchronize()
print(json.dumps({"out": y.flatten().tolist(), "index": idx.flatten().tolist(), "grad": x.grad.flatten().tolist()}))
"""


def _child(nd, device, compiled, python=None):
    env = {**os.environ, "MALLOC_CHECK_": "3", "CUDA_LAUNCH_BLOCKING": "1"}
    r = subprocess.run([python or sys.executable, "-c", _CHILD, str(nd), device, "1" if compiled else "0"],
                       capture_output=True, text=True, env=env, timeout=600)
    line = next((l for l in r.stdout.splitlines() if l.startswith("{")), None)
    return r.returncode, (json.loads(line) if line else None), (r.stderr.strip().splitlines() or [""])[-1][:200]


def d2_no_contributor_window(nd, device="cpu", python=None):
    """padding-only windows: indices inside the plane (size 1) or the sentinel -1, all gradients zero, clean exit."""
    rc, res, err = _child(nd, device, False, python)
    if res is None:
        return {"check": "D2", "passed": False, "details": {"exit": rc, "error": err}}
    bad_index = [i for i in res["index"] if not (i == -1 or 0 <= i < 1)]
    ok = rc == 0 and not bad_index and all(g == 0 for g in res["grad"])
    return {"check": "D2", "passed": bool(ok), "details": {**res, "exit": rc, "out_of_range_indices": bad_index}}


def d3_sentinel_backward(nd, device="cuda", python=None):
    """compiled: the backward of a forward that stored sentinel indices must not crash and must route nothing."""
    rc, res, err = _child(nd, device, True, python)
    if res is None:
        return {"check": "D3", "passed": False, "details": {"exit": rc, "error": err}}
    ok = rc == 0 and all(g == 0 for g in res["grad"])
    return {"check": "D3", "passed": bool(ok), "details": {**res, "exit": rc}}


def torch_scatter(s, idx, src, reduce, include_self):
    return torch.scatter_reduce(s, 0, idx, src, reduce, include_self=include_self)


def torch_index_reduce(s, idx, src, reduce, include_self):
    return torch.index_reduce(s, 0, idx, src, reduce, include_self=include_self)
