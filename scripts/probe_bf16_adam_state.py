#!/usr/bin/env python3
"""Probe: PyTorch AdamW on bf16 parameters keeps exp_avg / exp_avg_sq in bf16.  Does rounding the state to bf16
every step bias the second moment (beta2 = 0.999 changes v by 0.1% per step, below half a bf16 ulp)?

The same gradient stream (stationary, per-coordinate scale drawn once) is fed to AdamW on an fp32 copy and on a
bf16 copy of the parameters (default implementation for the device: foreach / fused); both use lr = 0 so that the
parameters stay identical and only the states evolve.  Reported over steps: the ratio of exp_avg_sq (bf16 / fp32)
and of the Adam direction magnitude |m_hat| / (sqrt(v_hat) + eps).

    python scripts/probe_bf16_adam_state.py
"""

from __future__ import annotations

import torch


def run(steps=3000, n=1 << 16, implementation="foreach"):
    g = torch.Generator(device="cuda").manual_seed(0)
    scale = torch.exp(torch.randn(n, device="cuda", generator=g))  # per-coordinate gradient scale
    p32 = torch.nn.Parameter(torch.zeros(n, device="cuda"))
    p16 = torch.nn.Parameter(torch.zeros(n, device="cuda", dtype=torch.bfloat16))
    kw = dict(lr=0.0, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    kw.update({"foreach": True} if implementation == "foreach" else {"fused": True})
    o32 = torch.optim.AdamW([p32], **kw)
    o16 = torch.optim.AdamW([p16], **kw)
    rows = []
    for t in range(1, steps + 1):
        grad = torch.randn(n, device="cuda", generator=g) * scale
        p32.grad = grad.clone()
        p16.grad = grad.to(torch.bfloat16)
        o32.step()
        o16.step()
        if t in (10, 100, 300, 1000, 2000, 3000):
            s32, s16 = o32.state[p32], o16.state[p16]
            v32, v16 = s32["exp_avg_sq"].float(), s16["exp_avg_sq"].float()
            m32, m16 = s32["exp_avg"].float(), s16["exp_avg"].float()
            bc1, bc2 = 1 - 0.9 ** t, 1 - 0.999 ** t
            d32 = (m32 / bc1).abs() / ((v32 / bc2).sqrt() + 1e-8)
            d16 = (m16 / bc1).abs() / ((v16 / bc2).sqrt() + 1e-8)
            rows.append((t, float((v16 / v32).median()), float((v16 / v32).mean()),
                         float(d16.mean() / d32.mean()), float((v16 == v16).float().mean())))
    return rows


def main():
    for impl in ("foreach", "fused"):
        print(f"implementation: {impl}")
        print("  step | median v_bf16/v_fp32 | mean ratio | mean |dir| bf16 / fp32")
        for t, med, mean, d, _ in run(implementation=impl):
            print(f"  {t:5d} | {med:10.4f}           | {mean:8.4f}   | {d:8.4f}")


if __name__ == "__main__":
    main()
