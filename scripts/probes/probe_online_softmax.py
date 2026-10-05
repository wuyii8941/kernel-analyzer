#!/usr/bin/env python3
"""Probe: Inductor online softmax (default on) for large reductions: softmax / log_softmax / logsumexp / cross_entropy
over vocabulary-sized rows, with -inf masks, fully masked rows, large magnitudes and bf16 inputs, against float64
eager.  Reports max relative error over finite outputs and special-value (NaN / inf) class mismatches.

    python scripts/probes/probe_online_softmax.py
"""
import torch
import torch.nn.functional as F

dev = "cuda"


def cls(t):
    return torch.where(torch.isnan(t), 1, torch.where(t == float("inf"), 2, torch.where(t == float("-inf"), 3, 0)))


def compare(name, fn, x, *extra):
    torch._dynamo.reset()
    try:
        out = torch.compile(fn)(x, *extra)
    except Exception as e:  # noqa: BLE001
        print(f"{name:60s} COMPILE/RUN ERROR {type(e).__name__}: {str(e).splitlines()[0][:80]}", flush=True)
        return
    ref = fn(x.double(), *extra)
    eag = fn(x, *extra)
    fin = torch.isfinite(ref) & torch.isfinite(out.double())
    scale = ref[fin].abs().max().clamp_min(1e-30) if fin.any() else torch.tensor(1.0)
    err = ((out.double() - ref).abs()[fin].max() / scale).item() if fin.any() else 0.0
    eerr = ((eag.double() - ref).abs()[fin].max() / scale).item() if fin.any() else 0.0
    mism = int((cls(out.double()) != cls(ref)).sum())
    emism = int((cls(eag.double()) != cls(ref)).sum())
    flag = "  <-- " if (mism != emism or err > max(10 * eerr, 1e-5)) else ""
    print(f"{name:60s} compiled err {err:.1e} special mism {mism:4d} | eager err {eerr:.1e} special mism {emism:4d}{flag}",
          flush=True)


g = torch.Generator(device=dev).manual_seed(0)
for V in (4096, 32000, 50257, 131072):
    for dtype in (torch.float32, torch.bfloat16):
        x = (4 * torch.randn(16, V, device=dev, generator=g)).to(dtype)
        xm = x.clone()
        xm[:, V // 3:] = float("-inf")        # partially masked rows
        xm[3] = float("-inf")                 # a fully masked row
        xb = x.float().clone(); xb[:, 7] = 1e30  # one huge logit (fp32 only)
        tgt = torch.randint(0, V // 3, (16,), device=dev, generator=g)
        tag = f"V={V} {str(dtype)[6:]}"
        compare(f"softmax {tag}", lambda t: torch.softmax(t, -1), x)
        compare(f"softmax masked {tag}", lambda t: torch.softmax(t, -1), xm)
        compare(f"log_softmax masked {tag}", lambda t: torch.log_softmax(t, -1), xm)
        compare(f"logsumexp masked {tag}", lambda t: torch.logsumexp(t, -1), xm)
        compare(f"cross_entropy {tag}", lambda t, y: F.cross_entropy(t.float(), y, reduction="none"), x, tgt)
        compare(f"cross_entropy masked {tag}", lambda t, y: F.cross_entropy(t.float(), y, reduction="none"), xm, tgt)
        compare(f"cross_entropy label_smoothing {tag}", lambda t, y: F.cross_entropy(t.float(), y, reduction="none", label_smoothing=0.1), x, tgt)
        if dtype == torch.float32:
            compare(f"softmax huge logit {tag}", lambda t: torch.softmax(t, -1), xb)
            compare(f"log_softmax huge logit {tag}", lambda t: torch.log_softmax(t, -1), xb)
