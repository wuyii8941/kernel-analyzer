#!/usr/bin/env python3
"""Probe: bitsandbytes 8-bit AdamW (blockwise dynamic quantization of both moments) - is the dequantized second
moment biased against an fp32 AdamW fed the same gradients?  fp32 parameters, lr = 0, the synthetic stationary
stream of probe_bf16_adam_state.py.

    python scripts/probe_bnb_adam_state.py
"""

import torch
import bitsandbytes as bnb
import bitsandbytes.functional as F


def bnb_v(opt, p):
    st = opt.state[p]
    if "state2" in st and st["state2"].dtype == torch.uint8:
        if "absmax2" in st:
            return F.dequantize_blockwise(st["state2"], absmax=st["absmax2"], code=st["qmap2"], blocksize=256).float()
        return None
    return st["state2"].float()


def main(steps=10000, n=1 << 16):
    g = torch.Generator(device="cuda").manual_seed(0)
    scale = torch.exp(torch.randn(n, device="cuda", generator=g))
    ref = torch.nn.Parameter(torch.zeros(n, device="cuda"))
    o_ref = torch.optim.AdamW([ref], lr=0.0, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    p = torch.nn.Parameter(torch.zeros(n, device="cuda"))
    o = bnb.optim.AdamW8bit([p], lr=0.0, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    for t in range(1, steps + 1):
        grad = torch.randn(n, device="cuda", generator=g) * scale
        ref.grad, p.grad = grad.clone(), grad.clone()
        o_ref.step()
        o.step()
        if t == 1:
            print("bnb state keys:", {k: (tuple(v.shape), v.dtype) for k, v in o.state[p].items() if torch.is_tensor(v)})
        if t in (100, 300, 1000, 3000, 10000):
            v = bnb_v(o, p)
            r = (v / o_ref.state[ref]["exp_avg_sq"].float()).flatten()
            q = torch.quantile(r[torch.randperm(r.numel(), device=r.device)[:50000]], torch.tensor([0.1, 0.5, 0.9], device=r.device))
            print(f"{t:5d} | bnb AdamW8bit median v ratio {float(q[1]):.3f} (p10 {float(q[0]):.3f}, p90 {float(q[2]):.3f})", flush=True)


if __name__ == "__main__":
    main()
