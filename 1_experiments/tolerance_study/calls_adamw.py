"""Group A, single optimizer step (protocol sections 2-3): calls, input sources and the specification for the unified
entry (``kernel_analyzer.measure``).

A unit is one fine-tuning run; its captured state at step t (``workload/finetune.py``) is the input: the pre-declared
coordinates of the 12 pre-declared tensors, concatenated in ``CAPTURE_TENSORS`` order into one vector per quantity
(parameter, gradient, exp_avg, exp_avg_sq, all bf16).  AdamW is elementwise, so one concatenated vector is the same
update as the 12 tensors separately.  Captures are read from ``$KA_TOL_CAPTURES/seed{SEED:04d}/capture_step{t:05d}.pt``.

Calls (the hyper-parameters are torchtune's: lr 2e-5, PyTorch AdamW defaults otherwise):
* ``adamw_compiled_s{t}``: torch.compile(AdamW(foreach=True).step) -- Inductor's Triton kernels (mode A / B);
* ``adamw_fused_s{t}``: AdamW(fused=True).step -- torchtune's default, an ATen kernel (black-box mode only).
Specification ``adamw_spec_s{t}``: the exact AdamW step in real arithmetic, enclosed with directed rounding
(``reference_eval.intervals``), constants as exact decimals.
"""
from __future__ import annotations

import os
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

LR, BETA1, BETA2, EPS, WD = "2e-5", "0.9", "0.999", "1e-8", "0.01"
CAPTURE_TENSORS = [f"model.layers.{i}.{n}" for i in (0, 11, 23) for n in
                   ("self_attn.q_proj.weight", "self_attn.q_proj.bias", "mlp.down_proj.weight", "input_layernorm.weight")]
QUANTITIES = ("param", "grad", "exp_avg", "exp_avg_sq")
_CACHE = {}


def _capture(seed: int, step: int) -> dict:
    key = (seed, step)
    if key not in _CACHE:
        root = Path(os.environ.get("KA_TOL_CAPTURES", "/data1/tzh/kernel-analyzer/.cache/round2/captures"))
        cap = torch.load(root / f"seed{seed:04d}" / f"capture_step{step:05d}.pt", weights_only=False)
        vec = {q: torch.cat([cap["tensors"][n][q].reshape(-1) for n in CAPTURE_TENSORS]) for q in QUANTITIES}
        vec["state_step"] = cap["tensors"][CAPTURE_TENSORS[0]]["state_step"]
        sizes = [cap["tensors"][n]["param"].numel() for n in CAPTURE_TENSORS]
        vec["segments"] = dict(zip(CAPTURE_TENSORS, np.cumsum([0] + sizes).tolist()))
        _CACHE.clear()
        _CACHE[key] = vec
    return _CACHE[key]


def segments(seed: int, step: int) -> dict:
    """Start offset of each tensor in the concatenated vector (for per-tensor summaries)."""
    return _capture(seed, step)["segments"]


def _source(quantity, step):
    def src(seed, shape, dtype):
        v = _capture(int(seed), step)[quantity]
        if tuple(shape) != (v.numel(),):
            raise ValueError(f"declared shape {shape}, captured {v.numel()} coordinates")
        return v.to(torch.bfloat16).cuda().clone()
    src.__name__ = f"{quantity}_s{step}"
    return src


def _step_source(step):
    def src(seed, shape, dtype):
        return torch.tensor([_capture(int(seed), step)["state_step"]], dtype=torch.float32)
    return src


class _Runner:
    """A persistent parameter and optimizer, refilled per call (a fresh optimizer per call would recompile)."""

    def __init__(self, n, fused):
        self.p = torch.nn.Parameter(torch.zeros(n, dtype=torch.bfloat16, device="cuda"))
        kw = {"fused": True} if fused else {"foreach": True}
        self.opt = torch.optim.AdamW([self.p], lr=float(LR), **kw)
        self.p.grad = torch.zeros_like(self.p)
        self.opt.step()                       # creates the state
        self.fused = fused
        self.step = self.opt.step if fused else torch.compile(self.opt.step)

    def __call__(self, inp):
        st = self.opt.state[self.p]
        with torch.no_grad():
            self.p.copy_(inp["param"])
            self.p.grad.copy_(inp["grad"])
            st["exp_avg"].copy_(inp["exp_avg"])
            st["exp_avg_sq"].copy_(inp["exp_avg_sq"])
            st["step"].fill_(float(inp["step"].item()))
        self.step()
        return {"param": self.p.data, "exp_avg": st["exp_avg"], "exp_avg_sq": st["exp_avg_sq"]}


def _runner(n, fused):
    key = ("runner", n, fused)
    if key not in _CACHE:
        _CACHE[key] = _Runner(n, fused)
    return _CACHE[key]


def _call(fused):
    def call(inp):
        return _runner(inp["param"].numel(), fused)(inp)
    return call


def adamw_exact(p, g, m, v, t):
    """Enclosures of (param, exp_avg, exp_avg_sq) after one AdamW step of step count t (1-based) in real arithmetic,
    as torch.optim.AdamW defines it: p <- p (1 - lr wd); m <- b1 m + (1 - b1) g; v <- b2 v + (1 - b2) g^2;
    p <- p - lr (m / (1 - b1^t)) / (sqrt(v / (1 - b2^t)) + eps)."""
    from kernel_analyzer.reference_eval import intervals as iv
    shape = np.shape(p)

    def c(x):
        q = Fraction(x)
        return np.full(shape, iv.rational_down(q)), np.full(shape, iv.rational_up(q))

    def pt(x):
        x = np.asarray(x, np.float64)
        return x, x
    b1, b2, lr, eps, wd = (Fraction(x) for x in (BETA1, BETA2, LR, EPS, WD))
    mp = iv.iadd(*iv.imul(*c(b1), *pt(m)), *iv.imul(*c(1 - b1), *pt(g)))
    vp = iv.iadd(*iv.imul(*c(b2), *pt(v)), *iv.imul(*c(1 - b2), *iv.isquare(*pt(g))))
    mhat = iv.idiv(*mp, *c(1 - b1 ** int(t)))
    vhat = iv.idiv(*vp, *c(1 - b2 ** int(t)))
    den = iv.iadd(*iv.isqrt(np.maximum(vhat[0], 0.0), vhat[1]), *c(eps))
    upd = iv.imul(*c(lr), *iv.idiv(*mhat, *den))
    decayed = iv.imul(*pt(p), *c(1 - lr * wd))
    pn = iv.isub(*decayed, *upd)
    return pn, mp, vp


def _spec(step):
    def spec(inp):
        p, g, m, v = (inp[q].detach().double().cpu().numpy() for q in QUANTITIES)
        t = int(inp["step"].item()) + 1
        pn, mp, vp = adamw_exact(p, g, m, v, t)
        return {"param": pn, "exp_avg": mp, "exp_avg_sq": vp}
    return spec


for _t in (10, 100, 300):
    for _q in QUANTITIES:
        globals()[f"{_q}_s{_t}"] = _source(_q, _t)
    globals()[f"step_s{_t}"] = _step_source(_t)
    globals()[f"adamw_spec_s{_t}"] = _spec(_t)
adamw_compiled = _call(False)
adamw_fused = _call(True)


def declaration(step: int, impl: str, mode: str, n: int, units: dict, seed_offset: int) -> dict:
    """The unified-entry declaration of one stratum (step) and implementation (protocol sections 2-4)."""
    inputs = {q: {"state": f"calls_adamw.py:{q}_s{step}", "shape": [n], "dtype": "bfloat16"} for q in QUANTITIES}
    inputs["step"] = {"state": f"calls_adamw.py:step_s{step}", "shape": [1], "dtype": "float32"}
    floor = {"relative_to_old": 2.0 ** -40}
    cmp_ = {"mode": mode, "measure": ["param", "exp_avg", "exp_avg_sq"],
            "observe": {q: {"layer": "increment", "old": q, "floor": floor} for q in ("param", "exp_avg", "exp_avg_sq")}}
    if mode in ("B", "black-box"):
        cmp_["spec"] = f"calls_adamw.py:adamw_spec_s{step}"
    return {"call": f"calls_adamw.py:adamw_{impl}", "inputs": inputs, "compare": cmp_,
            "budget": {"cpu_seconds": 7200, "gpu_seconds": 7200, "case_timeout": 3600,
                       "max_units": units["development"] + units["confirmation"]},
            "units": {**units, "seed_offset": seed_offset}, "alpha": 0.05,
            "resolution": {"ulp_fraction": 0.125, "max_level": 1}}
