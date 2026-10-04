#!/usr/bin/env python3
"""Screen: popular loss kernels against their documented math (float64 autograd reference).

FP32 inputs, so honest kernels agree with the reference to about 1e-6 relative; anything far above is a candidate
semantic deviation, to be confirmed with the reference evaluator (K_R vs the specification f).  Reported per
configuration: relative loss difference, max relative gradient difference (over the gradient's max), and the aligned
gradient bias <g_kernel - g_ref, g_ref> / |g_ref|^2.

    python scripts/screen_loss_kernels.py
"""

from __future__ import annotations

import itertools
import json
import sys

import torch
import torch.nn.functional as F


def reference_ce(logits, target, weight, ignore_index, label_smoothing, lse_square_scale, softcap, reduction,
                 logit_scaling=0.0):
    x = logits.double()
    if logit_scaling:
        x = x * logit_scaling
    if softcap:
        x = softcap * torch.tanh(x / softcap)
    loss = F.cross_entropy(x, target, weight=None if weight is None else weight.double(), ignore_index=ignore_index,
                           label_smoothing=label_smoothing, reduction=reduction)
    if lse_square_scale:
        lse = torch.logsumexp(x, dim=-1)
        valid = target != ignore_index
        z = lse_square_scale * (lse[valid] ** 2)
        loss = loss + (z.sum() / valid.sum() if reduction == "mean" else z.sum())
    return loss


def compare(name, loss_k, x_k, loss_r, x_r, rows):
    gk, gr = x_k.grad.double(), x_r.grad
    rel_loss = float((loss_k.double() - loss_r).abs() / loss_r.abs().clamp_min(1e-30))
    rel_grad = float((gk - gr).abs().max() / gr.abs().max().clamp_min(1e-30))
    aligned = float(((gk - gr) * gr).sum() / (gr * gr).sum().clamp_min(1e-300))
    flag = "  <== CHECK" if rel_loss > 1e-4 or rel_grad > 1e-3 else ""
    rows.append({"config": name, "rel_loss": rel_loss, "rel_grad_max": rel_grad, "aligned_grad_bias": aligned})
    print(f"{name:72s} loss {rel_loss:.1e}  grad {rel_grad:.1e}  aligned {aligned:+.1e}{flag}", flush=True)


def main():
    rows = []
    torch.manual_seed(0)
    B, V = 256, 4096
    from liger_kernel.transformers.functional import liger_cross_entropy, liger_fused_linear_cross_entropy

    logits0 = torch.randn(B, V, device="cuda") * 3
    target = torch.randint(0, V, (B,), device="cuda")
    target[::7] = -100
    weight = torch.rand(V, device="cuda") + 0.5
    print("== Liger cross_entropy (fp32 logits)")
    for ls, z, sc, red, w in itertools.product((0.0, 0.1), (0.0, 1e-4), (None, 30.0), ("mean", "sum"), (None, weight)):
        name = f"liger CE ls={ls} z={z} softcap={sc} red={red} weight={'yes' if w is not None else 'no'}"
        try:
            xk = logits0.clone().requires_grad_(True)
            lk = liger_cross_entropy(xk, target, weight=w, label_smoothing=ls, lse_square_scale=z, softcap=sc,
                                     reduction=red)
            lk.backward()
            xr = logits0.double().clone().requires_grad_(True)
            lr = reference_ce(xr, target, w, -100, ls, z, sc, red)
            lr.backward()
            compare(name, lk, xk, lr, xr, rows)
        except Exception as exc:  # noqa: BLE001
            print(f"{name:72s} ERROR {type(exc).__name__}: {str(exc)[:100]}")
            rows.append({"config": name, "error": str(exc)[:200]})
    print("== Liger fused_linear_cross_entropy (fp32)")
    H = 512
    h0 = torch.randn(B, H, device="cuda")
    W0 = torch.randn(V, H, device="cuda") * 0.05
    for ls, z, sc in itertools.product((0.0, 0.1), (0.0, 1e-4), (None, 30.0)):
        name = f"liger FLCE ls={ls} z={z} softcap={sc}"
        try:
            hk, wk = h0.clone().requires_grad_(True), W0.clone().requires_grad_(True)
            lk = liger_fused_linear_cross_entropy(hk, wk, target, label_smoothing=ls, lse_square_scale=z, softcap=sc)
            lk.backward()
            hr, wr = h0.double().clone().requires_grad_(True), W0.double().clone().requires_grad_(True)
            lr = reference_ce(hr @ wr.t(), target, None, -100, ls, z, sc, "mean")
            lr.backward()
            compare(name + " [dh]", lk, hk, lr, hr, rows)
            compare(name + " [dW]", lk, wk, lr, wr, rows)
        except Exception as exc:  # noqa: BLE001
            print(f"{name:72s} ERROR {type(exc).__name__}: {str(exc)[:100]}")
            rows.append({"config": name, "error": str(exc)[:200]})
    print("== unsloth fast_cross_entropy_loss (fp32 logits)")
    try:
        from unsloth.kernels.cross_entropy_loss import fast_cross_entropy_loss
        for sc, scale in itertools.product((0, 30.0), (0, 0.5)):
            name = f"unsloth CE softcap={sc} logit_scaling={scale}"
            xk = logits0.view(1, B, V).clone().requires_grad_(True)
            lk = fast_cross_entropy_loss(xk, target.view(1, B), logit_softcapping=sc, logit_scaling=scale)
            lk.backward()
            xr = logits0.double().view(1, B, V).clone().requires_grad_(True)
            lr = reference_ce(xr.view(B, V), target, None, -100, 0.0, 0.0, sc or None, "mean", logit_scaling=scale)
            lr.backward()
            compare(name, lk, xk, lr, xr, rows)
    except Exception as exc:  # noqa: BLE001
        print("unsloth CE ERROR", type(exc).__name__, str(exc)[:200])
    json.dump(rows, open(sys.argv[1] if len(sys.argv) > 1 else "/dev/null", "w"), indent=2)


if __name__ == "__main__":
    main()
