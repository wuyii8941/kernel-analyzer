"""Group B, single optimizer step (protocol appendix B): calls, input sources and the specification for the unified
entry.  Same capture layout as group A (``calls_adamw``), parameters and states in fp32 (fp32 parameters with bf16
autocast), captured before steps 6, 7, 10, 20 and 100 of runs trained with eager RAdam.

* ``radam_compiled``: torch.compile(RAdam(foreach=True).step) -- Inductor's Triton kernels; the rectification term is
  computed from the step tensor in float32 (mode A / B);
* ``radam_eager``: RAdam(foreach=True).step -- the rectification term in Python float64 (ATen; black box).
Specification ``radam_spec_s{t}``: torch.optim.RAdam's step (weight_decay 0) in real arithmetic -- rho_t exactly as a
rational number, the rectification factor and the adaptive scale as strict enclosures.
"""
from __future__ import annotations

import os
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

LR, BETA1, BETA2, EPS = "2e-5", "0.9", "0.999", "1e-8"
CAPTURE_TENSORS = [f"model.layers.{i}.{n}" for i in (0, 11, 23) for n in
                   ("self_attn.q_proj.weight", "self_attn.q_proj.bias", "mlp.down_proj.weight", "input_layernorm.weight")]
QUANTITIES = ("param", "grad", "exp_avg", "exp_avg_sq")
STEPS = (6, 7, 10, 20, 100)
_CACHE = {}


def _capture(seed: int, step: int) -> dict:
    key = (seed, step)
    if key not in _CACHE:
        root = Path(os.environ.get("KA_TOL_CAPTURES_B", "/data1/tzh/kernel-analyzer/.cache/round2/captures_b"))
        cap = torch.load(root / f"seed{seed:04d}" / f"capture_step{step:05d}.pt", weights_only=False)
        vec = {q: torch.cat([cap["tensors"][n][q].reshape(-1) for n in CAPTURE_TENSORS]) for q in QUANTITIES}
        vec["state_step"] = cap["tensors"][CAPTURE_TENSORS[0]]["state_step"]
        _CACHE.clear()
        _CACHE[key] = vec
    return _CACHE[key]


def _source(quantity, step):
    def src(seed, shape, dtype):
        v = _capture(int(seed), step)[quantity]
        if tuple(shape) != (v.numel(),):
            raise ValueError(f"declared shape {shape}, captured {v.numel()} coordinates")
        return v.to(torch.float32).cuda().clone()
    return src


def _step_source(step):
    def src(seed, shape, dtype):
        return torch.tensor([_capture(int(seed), step)["state_step"]], dtype=torch.float32)
    return src


class _Runner:
    def __init__(self, n, compiled):
        self.p = torch.nn.Parameter(torch.zeros(n, dtype=torch.float32, device="cuda"))
        self.opt = torch.optim.RAdam([self.p], lr=float(LR), foreach=True)
        self.p.grad = torch.zeros_like(self.p)
        self.opt.step()
        self.step = torch.compile(self.opt.step) if compiled else self.opt.step

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


def _call(compiled):
    def call(inp):
        key = ("runner", inp["param"].numel(), compiled)
        if key not in _CACHE:
            _CACHE[key] = _Runner(inp["param"].numel(), compiled)
        return _CACHE[key](inp)
    return call


def rectification(t: int):
    """(rho_t exact, rectified?, enclosure of the rectification factor, enclosure of sqrt(1 - b2^t))."""
    from kernel_analyzer.reference_eval import intervals as iv
    b2 = Fraction(BETA2)
    rho_inf = 2 / (1 - b2) - 1
    rho_t = rho_inf - 2 * t * b2 ** t / (1 - b2 ** t)
    q = Fraction(1 - b2 ** t)
    sq = iv.isqrt(np.array([iv.rational_down(q)]), np.array([iv.rational_up(q)]))
    if rho_t <= 5:
        return rho_t, False, None, sq
    x = (rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t)
    r = iv.isqrt(np.array([iv.rational_down(x)]), np.array([iv.rational_up(x)]))
    return rho_t, True, r, sq


def radam_exact(p, g, m, v, t):
    """Enclosures of (param, exp_avg, exp_avg_sq) after one RAdam step of step count t (1-based)."""
    from kernel_analyzer.reference_eval import intervals as iv
    shape = np.shape(p)

    def c(x):
        q = Fraction(x)
        return np.full(shape, iv.rational_down(q)), np.full(shape, iv.rational_up(q))

    def pt(x):
        x = np.asarray(x, np.float64)
        return x, x
    b1, b2, lr, eps = (Fraction(x) for x in (BETA1, BETA2, LR, EPS))
    mp = iv.iadd(*iv.imul(*c(b1), *pt(m)), *iv.imul(*c(1 - b1), *pt(g)))
    vp = iv.iadd(*iv.imul(*c(b2), *pt(v)), *iv.imul(*c(1 - b2), *iv.isquare(*pt(g))))
    _, rectified, r, sq = rectification(int(t))
    bc1 = c(1 - b1 ** int(t))
    if not rectified:
        upd = iv.idiv(*iv.imul(*c(lr), *mp), *bc1)
    else:
        scale = iv.imul(*iv.imul(*c(lr), np.full(shape, r[0][0]), np.full(shape, r[1][0])),
                        np.full(shape, sq[0][0]), np.full(shape, sq[1][0]))
        den = iv.iadd(*iv.isqrt(np.maximum(vp[0], 0.0), vp[1]), *c(eps))
        upd = iv.idiv(*iv.imul(*scale, *iv.idiv(*mp, *den)), *bc1)
    pn = iv.isub(*pt(p), *upd)
    return pn, mp, vp


def _spec(step):
    def spec(inp):
        p, g, m, v = (inp[q].detach().double().cpu().numpy() for q in QUANTITIES)
        t = int(inp["step"].item()) + 1
        pn, mp, vp = radam_exact(p, g, m, v, t)
        return {"param": pn, "exp_avg": mp, "exp_avg_sq": vp}
    return spec


for _t in STEPS:
    for _q in QUANTITIES:
        globals()[f"{_q}_s{_t}"] = _source(_q, _t)
    globals()[f"step_s{_t}"] = _step_source(_t)
    globals()[f"radam_spec_s{_t}"] = _spec(_t)
radam_compiled = _call(True)
radam_eager = _call(False)


def declaration(step: int, impl: str, mode: str, n: int, units: dict, seed_offset: int) -> dict:
    inputs = {q: {"state": f"calls_radam.py:{q}_s{step}", "shape": [n], "dtype": "float32"} for q in QUANTITIES}
    inputs["step"] = {"state": f"calls_radam.py:step_s{step}", "shape": [1], "dtype": "float32"}
    floor = {"relative_to_old": 2.0 ** -40}
    cmp_ = {"mode": mode, "measure": ["param"], "observe": {"param": {"layer": "increment", "old": "param",
                                                                      "floor": floor}}}
    if mode in ("B", "black-box"):
        cmp_["spec"] = f"calls_radam.py:radam_spec_s{step}"
    return {"call": f"calls_radam.py:radam_{impl}", "inputs": inputs, "compare": cmp_,
            "budget": {"cpu_seconds": 7200, "gpu_seconds": 7200, "case_timeout": 3600,
                       "max_units": units["development"] + units["confirmation"]},
            "units": {**units, "seed_offset": seed_offset}, "alpha": 0.05,
            "resolution": {"ulp_fraction": 0.125, "max_level": 1}}
