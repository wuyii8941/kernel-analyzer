"""AdamW step on bf16 parameters and bf16 states, compiled with torch.compile (one Inductor Triton kernel).

This is what torch.optim.AdamW does when the parameters are bf16 (the states follow the parameter dtype), here in
the compiled form (math in FP32, one rounding of every state to bf16 per step, like the fused implementation).  The
state is drawn near its FP32 steady state: v ~ s^2 (1 + noise), g ~ N(0, s^2), with a log-normal scale s per
coordinate; step t = 1000 (bias corrections fixed)."""

import math

import torch
import torch._inductor.config as inductor_config

inductor_config.use_static_cuda_launcher = False  # every launch goes through Triton's CompiledKernel.run

N = 1 << 16
LR, B1, B2, EPS, T = 1e-4, 0.9, 0.999, 1e-8, 1000


@torch.compile(fullgraph=True)
def adamw_step(p, g, m, v):
    gf = g.float()
    m32 = m.float() * B1 + gf * (1 - B1)
    v32 = v.float() * B2 + gf * gf * (1 - B2)
    m.copy_(m32.to(m.dtype))
    v.copy_(v32.to(v.dtype))
    step = (m32 / (1 - B1 ** T)) / ((v32 / (1 - B2 ** T)).sqrt() + EPS)
    p.copy_((p.float() - LR * step).to(p.dtype))


def make_inputs(seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    s = torch.exp(torch.randn(N, device="cuda", generator=torch.Generator(device="cuda").manual_seed(12345)))
    p = torch.randn(N, device="cuda", generator=gen).to(torch.bfloat16)
    g = (torch.randn(N, device="cuda", generator=gen) * s).to(torch.bfloat16)
    m = (0.3 * torch.randn(N, device="cuda", generator=gen) * s).to(torch.bfloat16)
    v = (s * s * (1 + 0.1 * torch.randn(N, device="cuda", generator=gen)).abs()).to(torch.bfloat16)
    return {"p": p, "g": g, "m": m, "v": v}


def run(t):
    adamw_step(t["p"], t["g"], t["m"], t["v"])


# compile (and autotune) once before any unit is measured, so every unit sees the same single launch; the
# persistent recorder hook must exist before Inductor caches its launcher at the first launch
from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder  # noqa: E402

TritonLaunchRecorder.install_hook()
_w = make_inputs(10 ** 6)
run(_w)
torch.cuda.synchronize()
