"""B018: Inductor's attention fusion (torch/_inductor/fx_passes/fuse_attention.py) rewrites hand-written attention
into scaled_dot_product_attention when the result differs from the user's code.

A. Constants of the pattern are wildcards in torch 2.10: pattern 5 (`softmax(q @ k^T / math.sqrt(E) + mask) @ v`)
   matches any divisor and the replacement uses SDPA's default scale 1/sqrt(E); pattern 15 (DistilBERT) matches any
   masked_fill value and the replacement masks with -inf.  (main validates matches by re-tracing, #195383, merged
   2026-09-21; to be checked on a nightly.)
B. Patterns 15 / 17 / 20 mask where `attn_mask == 0` but the replacement keeps where `attn_mask == 1`: different for
   masks with other values (segment ids, soft masks); a fully masked row gives NaN (fill -inf) or a uniform average
   (finite fill) in eager, 0 after the rewrite.  The replacement code is unchanged on main.

    python bugs/repro/B018_repro_inductor_fuse_attention.py
"""
import math
import os

os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")

import torch  # noqa: E402
from torch._dynamo.utils import counters  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.__version__, dev)
g = torch.Generator(device=dev).manual_seed(0)


def run(label, fn, *args):
    torch._dynamo.reset()
    counters.clear()
    eager = fn(*args)
    comp = torch.compile(fn)(*args)
    fused = counters["inductor"]["fuse_attention"]
    nan_e, nan_c = bool(eager.isnan().any()), bool(comp.isnan().any())
    diff = (comp.nan_to_num(0) - eager.nan_to_num(0)).abs().max().item()
    print(f"  {label:58s} rewritten={fused}  max|compiled - eager| = {diff:.3g}"
          + (f"  (NaN: eager {nan_e}, compiled {nan_c})" if nan_e or nan_c else ""))


print("A. pattern 5: scores / c + mask, c is not sqrt(head_dim)")
B, H, L, E = 2, 4, 16, 64
q, k, v = (torch.randn(B, H, L, E, device=dev, generator=g) for _ in range(3))
mask = torch.randn(B, 1, L, L, device=dev, generator=g)
for label, c in [("c = sqrt(head_dim) = 8 (control)", math.sqrt(E)), ("c = sqrt(d_model) = 16", math.sqrt(256)),
                 ("c = 4.0 (temperature)", 4.0), ("c = 1.0 (no scaling)", 1.0)]:
    run(label, lambda q, k, v, m, c=c: torch.softmax((q @ k.transpose(-2, -1) / c) + m, dim=-1) @ v, q, k, v, mask)


def distilbert(fill):
    def attn(query, key, value, attn_mask, inv_scale):  # the form of _sfdp_pattern_15
        q_, k_, v_ = (t.permute([0, 2, 1, 3]) for t in (query, key, value))
        bs, k_len = q_.size(0), k_.size(-2)
        scores = (q_ @ k_.transpose(-2, -1)).div(inv_scale)
        fv = torch.full((), fill, dtype=query.dtype, device=query.device)
        m = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)
        return torch.softmax(scores.masked_fill(m, fv), dim=-1) @ v_
    return attn


bs, L, H, D = 2, 16, 4, 32
q, k, v = (torch.randn(bs, L, H, D, device=dev, generator=g) for _ in range(3))
pad = torch.tensor([[1] * 12 + [0] * 4, [1] * 16], device=dev)
print("A. pattern 15: masked_fill value other than -inf")
for fill in (-float("inf"), 0.0, 1.0):
    run(f"masked_fill(attn_mask == 0, {fill})", distilbert(fill), q, k, v, pad, 8.0)
print("B. pattern 15: masks with values other than 0 / 1, fully masked rows")
run("binary 0/1 padding mask (control)", distilbert(-float("inf")), q, k, v, pad, 8.0)
run("segment ids 1/2/3, 0 = padding", distilbert(-float("inf")), q, k, v,
    torch.tensor([[1] * 5 + [2] * 5 + [3] * 2 + [0] * 4, [1] * 8 + [2] * 8], device=dev), 8.0)
run("soft mask 0 / 0.5 / 1", distilbert(-float("inf")), q, k, v,
    torch.tensor([[1.0] * 6 + [0.5] * 6 + [0.0] * 4, [0.5] * 16], device=dev), 8.0)
allpad = torch.tensor([[1] * 12 + [0] * 4, [0] * 16], device=dev)
run("a padding-only sequence, fill -inf", distilbert(-float("inf")), q, k, v, allpad, 8.0)
run("a padding-only sequence, fill finfo.min", distilbert(torch.finfo(torch.float32).min), q, k, v, allpad, 8.0)
