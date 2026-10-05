#!/usr/bin/env python3
"""torch.optim.RAdam: the capturable path and torch.compile(opt.step) compute the rectification term in float32.

rho_t = rho_inf - 2 t beta2^t / (1 - beta2^t) is a difference of two numbers close to rho_inf = 2 / (1 - beta2) - 1.
The non-capturable eager path evaluates it with Python floats (float64); with capturable=True, and under
torch.compile (where _get_value returns the step tensor), `step` is a float32 tensor, so beta2 is rounded to
float32 and the cancellation happens in float32.  For beta2 close to 1 the decision rho_t > 5 flips and the update
is wrong.  Prints, per beta2: the steps whose rectification decision differs from the float64 formula, and the
relative error of the parameter update against eager float64 (max over steps 1-50 and at selected steps).

    python bugs/repro/B012_repro_radam_fp32_rho.py
"""

import torch


def rho(beta2, t, dtype):
    b = torch.tensor(beta2, dtype=dtype)
    tt = torch.tensor(float(t), dtype=dtype)
    rho_inf = 2 / (1 - beta2) - 1
    return float(rho_inf - 2 * tt * b**tt / (1 - b**tt))


def updates(beta2, mode, dtype, steps=50):
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(48, 64, device="cuda").to(dtype))
    opt = torch.optim.RAdam([p], lr=1e-2, betas=(0.9, beta2), capturable=(mode == "capturable"))
    step = torch.compile(opt.step) if mode == "compiled" else opt.step
    g = torch.Generator(device="cuda").manual_seed(1)
    out = []
    for _ in range(steps):
        p.grad = torch.randn(p.shape, device="cuda", generator=g).to(dtype)
        before = p.detach().clone()
        step()
        out.append((p.detach() - before).double())
    return out


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


print(torch.__version__)
for beta2 in (0.999, 0.9995, 0.9999, 0.99999):
    flips = [t for t in range(1, 51) if (rho(beta2, t, torch.float64) > 5) != (rho(beta2, t, torch.float32) > 5)]
    ref = updates(beta2, "eager", torch.float64)
    print(f"beta2={beta2}: rectification decision flipped in float32 at steps {flips}")
    for mode in ("eager", "capturable", "compiled"):
        d = updates(beta2, mode, torch.float32)
        errs = [rel(d[t], ref[t]) for t in range(50)]
        print(f"   {mode:10s} max rel update error {max(errs):.1e} (step {errs.index(max(errs)) + 1}); "
              f"steps 2/5/6/10/50: " + " ".join(f"{errs[t - 1]:.1e}" for t in (2, 5, 6, 10, 50)))
