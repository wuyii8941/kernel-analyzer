#!/usr/bin/env python3
"""beta2 sweep of the bf16 AdamW second-moment ratchet (prediction: docs/bf16_adam_beta2_prediction.md).

Same synthetic stream as probe_bf16_adam_state.py (stationary N(0, s^2), log-normal s per coordinate), lr = 0.

    python scripts/bf16_adam_beta2_sweep.py --out results/bf16_adam/beta2_sweep.json
"""

import argparse
import json
from pathlib import Path

import torch

BETA2 = (0.95, 0.98, 0.99, 0.995, 0.997, 0.998, 0.999, 0.9995)


def run(beta2, impl, steps=3000, n=1 << 16):
    g = torch.Generator(device="cuda").manual_seed(0)
    scale = torch.exp(torch.randn(n, device="cuda", generator=g))
    p32 = torch.nn.Parameter(torch.zeros(n, device="cuda"))
    p16 = torch.nn.Parameter(torch.zeros(n, device="cuda", dtype=torch.bfloat16))
    kw = dict(lr=0.0, betas=(0.9, beta2), eps=1e-8, weight_decay=0.0, **({"foreach": True} if impl == "foreach" else {"fused": True}))
    o32, o16 = torch.optim.AdamW([p32], **kw), torch.optim.AdamW([p16], **kw)
    out = {}
    for t in range(1, steps + 1):
        grad = torch.randn(n, device="cuda", generator=g) * scale
        p32.grad, p16.grad = grad.clone(), grad.to(torch.bfloat16)
        o32.step()
        o16.step()
        if t in (300, 1000, 3000):
            r = o16.state[p16]["exp_avg_sq"].float() / o32.state[p32]["exp_avg_sq"].float()
            out[t] = float(r.median())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    res = {}
    print("beta2   | foreach median ratio @300/1000/3000 | fused @300/1000/3000 | 1-beta2 vs 2^-8, 2^-9")
    for b2 in BETA2:
        res[b2] = {impl: run(b2, impl) for impl in ("foreach", "fused")}
        f, u = res[b2]["foreach"], res[b2]["fused"]
        zone = "no ratchet" if 1 - b2 >= 2 ** -8 else ("partial" if 1 - b2 > 2 ** -9 else "full")
        print(f"{b2:<7} | {f[300]:.3f} {f[1000]:.3f} {f[3000]:.3f}                 | {u[300]:.3f} {u[1000]:.3f} {u[3000]:.3f}    | predicted: {zone}", flush=True)
    args.out.write_text(json.dumps({str(k): v for k, v in res.items()}, indent=2) + "\n")


if __name__ == "__main__":
    main()
