#!/usr/bin/env python3
"""Probe: Inductor's mix-order reduction (two reductions of one tensor along different dims fused into one kernel;
on by default outside fbcode; eligible when nrow*ncol >= 5*2^20, nrow >= max(4096, 2*ncol), ncol <= 16384, the other
reduction is a sum).  Compiled LayerNorm / RMSNorm backward and x.sum(0) + x.sum(1) against float64 eager, at row
counts on both sides of the split-size multiples.  Prints whether the mix-order path was generated.

    python scripts/probes/probe_mix_order_reduction.py
"""
import torch
import torch._inductor.metrics as metrics
import torch.nn.functional as F

dev = "cuda"


def rel(a, b):
    return ((a.double() - b).norm() / b.norm().clamp_min(1e-300)).item()


def ln_step(x, w, b, g):
    y = F.layer_norm(x, (x.shape[-1],), w, b, 1e-5)
    return torch.autograd.grad(y, (x, w, b), g)


def rms_step(x, w, b, g):
    y = F.rms_norm(x, (x.shape[-1],), w, 1e-6)
    return torch.autograd.grad(y, (x, w), g)


def sums(x, w, b, g):
    return (x * g).sum(0), (x * g).sum(1)


def run(name, fn, nrow, ncol, dtype=torch.float32):
    torch._dynamo.reset()
    metrics.reset()
    gen = torch.Generator(device=dev).manual_seed(nrow * 7 + ncol)
    x = torch.randn(nrow, ncol, device=dev, generator=gen, dtype=dtype).requires_grad_(fn is not sums)
    w = (1 + 0.1 * torch.randn(ncol, device=dev, generator=gen, dtype=dtype)).requires_grad_(fn is not sums)
    b = (0.1 * torch.randn(ncol, device=dev, generator=gen, dtype=dtype)).requires_grad_(fn is not sums)
    g = torch.randn(nrow, ncol, device=dev, generator=gen, dtype=dtype)
    out = torch.compile(fn)(x, w, b, g)
    mix = metrics.codegen_mix_order_reduction
    x64, w64, b64 = (t.detach().double().requires_grad_(fn is not sums) for t in (x, w, b))
    ref = fn(x64, w64, b64, g.double())
    errs = [f"{rel(o, r):.1e}" for o, r in zip(out, ref)]
    print(f"{name:8s} {nrow:6d} x {ncol:5d} {str(dtype)[6:]:8s} mix_order={mix} rel errors {errs}", flush=True)


for ncol in (768, 1000, 1024):
    for nrow in (8192, 8193, 8191, 8200, 8255, 12289):
        if nrow * ncol >= 5 * 2 ** 20:
            run("ln", ln_step, nrow, ncol)
            run("rms", rms_step, nrow, ncol)
            run("sums", sums, nrow, ncol)
for nrow in (8192, 8197):
    run("ln", ln_step, nrow, 768, torch.bfloat16)
