"""Focus: rows with few finite logits (grammar / structured-output masks), top-k >= #finite, top-p < 1."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_topk_topp import apply_top_k_top_p_pytorch, spec_keep  # noqa: E402
from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton  # noqa: E402

dev = "cuda"
g = torch.Generator(device=dev).manual_seed(1)
V = 151936
for B in (4, 64, 65, 256):
    for nfin in (2, 3, 5, 10, 19):
        for k in (20, 50):
            for p in (0.8, 0.95):
                logits = torch.full((B, V), float("-inf"), device=dev)
                idx = torch.stack([torch.randperm(V, device=dev, generator=g)[:nfin] for _ in range(B)])
                logits.scatter_(1, idx, 2 * torch.randn(B, nfin, device=dev, generator=g))
                kk = torch.full((B,), k, device=dev, dtype=torch.int32)
                pp = torch.full((B,), p, device=dev)
                kt = apply_top_k_top_p_triton(logits.clone(), kk, pp) > float("-inf")
                kr = apply_top_k_top_p_pytorch(logits.clone(), kk, pp) > float("-inf")
                ks, amb = spec_keep(logits.cpu(), kk.cpu(), pp.cpu())
                ks = ks.to(dev)
                bad = (kt != ks).any(1) & ~amb.to(dev)
                probs = torch.softmax(logits.double(), -1)
                lost = ((ks & ~kt).double() * probs).sum(1)
                extra = ((kt & ~ks).double() * probs).sum(1)
                print(f"B={B:3d} finite={nfin:2d} k={k} p={p}: wrong rows {int(bad.sum()):3d}/{B} "
                      f"(pytorch wrong {int(((kr != ks).any(1) & ~amb.to(dev)).sum())}), "
                      f"max mass dropped {lost.max().item():.3f}, max mass added {extra.max().item():.3f}, "
                      f"kept triton/spec row0 {int(kt[0].sum())}/{int(ks[0].sum())}")
