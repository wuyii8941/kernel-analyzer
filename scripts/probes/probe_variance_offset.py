#!/usr/bin/env python3
"""Probe: variance-based ops on inputs with a large mean (cancellation-safe Welford vs E[x^2] - E[x]^2).  Compiled
(Inductor) vs eager float32 vs float64, outputs and the batch-norm running variance, at sizes that make Inductor split
the reduction or loop over it.

    python scripts/probes/probe_variance_offset.py
"""
import torch
import torch.nn.functional as F

dev = "cuda"
torch.manual_seed(0)


def rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-300)).item()


def cmp(name, fn, *args):
    torch._dynamo.reset()
    try:
        yc = torch.compile(fn)(*[a.clone() if torch.is_tensor(a) else a for a in args])
    except Exception as e:  # noqa: BLE001
        print(f"{name:58s} ERROR {type(e).__name__}: {str(e).splitlines()[0][:70]}")
        return
    ye = fn(*[a.clone() if torch.is_tensor(a) else a for a in args])
    yd = fn(*[a.double().clone() if torch.is_tensor(a) and a.is_floating_point() else a for a in args])
    ys = yc if isinstance(yc, (tuple, list)) else (yc,)
    es = ye if isinstance(ye, (tuple, list)) else (ye,)
    ds = yd if isinstance(yd, (tuple, list)) else (yd,)
    c = [f"{rel(a, d):.1e}" for a, d in zip(ys, ds)]
    e = [f"{rel(a, d):.1e}" for a, d in zip(es, ds)]
    worse = any(float(ci) > 100 * max(float(ei), 1e-7) for ci, ei in zip(c, e))
    print(f"{name:58s} compiled {c}  eager {e}{'   <--' if worse else ''}", flush=True)


for offset in (1e2, 1e3, 1e4):
    # batch norm, training: output and running_var (unbiased variance over N*H*W)
    x = offset + torch.randn(64, 32, 28, 28, device=dev)
    def bn(x):
        rm, rv = torch.zeros(32, device=x.device, dtype=x.dtype), torch.ones(32, device=x.device, dtype=x.dtype)
        y = F.batch_norm(x, rm, rv, training=True, momentum=1.0)
        return y, rv
    cmp(f"batch_norm train (64,32,28,28) offset {offset:g}", bn, x)
    x1 = offset + torch.randn(8192, 64, device=dev)
    cmp(f"batch_norm1d train (8192,64) offset {offset:g}",
        lambda t: F.batch_norm(t, None, None, training=True), x1)
    cmp(f"group_norm (32,64,32,32) g=8 offset {offset:g}",
        lambda t: F.group_norm(t, 8), offset + torch.randn(32, 64, 32, 32, device=dev))
    cmp(f"instance_norm (16,32,64,64) offset {offset:g}",
        lambda t: F.instance_norm(t), offset + torch.randn(16, 32, 64, 64, device=dev))
    cmp(f"layer_norm (64,8192) offset {offset:g}",
        lambda t: F.layer_norm(t, (8192,)), offset + torch.randn(64, 8192, device=dev))
    big = offset + torch.randn(1 << 22, device=dev)
    cmp(f"var (4M,) offset {offset:g}", lambda t: t.var(), big)
    cmp(f"std_mean (4M,) offset {offset:g}", lambda t: torch.std_mean(t), big)
    cmp(f"var dim0 (65536,32) offset {offset:g}", lambda t: t.var(0), offset + torch.randn(65536, 32, device=dev))
    cmp(f"var_mean dim1 (32,65536) offset {offset:g}", lambda t: torch.var_mean(t, 1), offset + torch.randn(32, 65536, device=dev))
