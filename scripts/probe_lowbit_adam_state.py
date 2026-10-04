#!/usr/bin/env python3
"""Probe: torchao low-bit AdamW (8-bit, 4-bit, FP8 states) - does requantizing the second moment every step with
round-to-nearest bias it?  fp32 parameters (the intended use), lr = 0, the synthetic stationary gradient stream of
probe_bf16_adam_state.py; v is dequantized and compared with an fp32 torch AdamW fed the same gradients.

    python scripts/probe_lowbit_adam_state.py
"""

import torch
import torchao.optim as ao


def dequant(state):
    return state.dequantize().float() if hasattr(state, "dequantize") else state.float()


def main(steps=10000, n=1 << 16):
    g = torch.Generator(device="cuda").manual_seed(0)
    scale = torch.exp(torch.randn(n, device="cuda", generator=g))
    ref = torch.nn.Parameter(torch.zeros(n, device="cuda"))
    o_ref = torch.optim.AdamW([ref], lr=0.0, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    runs = {}
    for name in ("AdamW8bit", "AdamW4bit"):  # AdamWFp8 needs fp8e4nv (sm_89+)
        if hasattr(ao, name):
            p = torch.nn.Parameter(torch.zeros(n, device="cuda"))
            runs[name] = (p, getattr(ao, name)([p], lr=0.0, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0))
    print("step  | " + " | ".join(f"{k}: median v ratio (p10, p90)" for k in runs))
    for t in range(1, steps + 1):
        grad = torch.randn(n, device="cuda", generator=g) * scale
        ref.grad = grad.clone()
        o_ref.step()
        for p, o in runs.values():
            p.grad = grad.clone()
            o.step()
        if t in (100, 300, 1000, 3000, 10000):
            v32 = o_ref.state[ref]["exp_avg_sq"].float()
            cells = []
            for name, (p, o) in runs.items():
                r = dequant(o.state[p]["exp_avg_sq"]).flatten() / v32
                q = torch.quantile(r[torch.randperm(r.numel(), device=r.device)[:50000]], torch.tensor([0.1, 0.5, 0.9], device=r.device))
                cells.append(f"{float(q[1]):7.3f} ({float(q[0]):.3f}, {float(q[2]):.3f})")
            print(f"{t:5d} | " + " | ".join(cells), flush=True)


if __name__ == "__main__":
    main()
