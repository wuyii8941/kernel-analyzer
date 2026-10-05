"""Differential probe: vLLM apply_top_k_top_p_triton vs a float64 specification and vLLM's PyTorch path.

Spec (docstring of apply_top_k_top_p_triton): top-k by logit value first (ties at the k-th value kept, as the
PyTorch path does), then top-p on the remaining tokens by probability: a token is kept when the probability mass of
the tokens strictly above it is < p (the smallest set reaching p).  Rows whose decision depends on tie order or on a
mass within 1e-6 of p are reported as ambiguous, not as differences."""
import sys

import torch

import ast
from pathlib import Path

# vLLM's PyTorch path (apply_top_k_top_p_pytorch, apply_top_k_only), taken verbatim from the source file: the
# module itself imports FlashInfer and the rest of vLLM
_SRC = Path(__file__).resolve().parents[2] / ".cache/src/vllm-main/vllm/v1/sample/ops/topk_topp_sampler.py"
_tree = ast.parse(_SRC.read_text())
_ns = {"torch": torch}
for _node in _tree.body:
    if isinstance(_node, ast.FunctionDef) and _node.name in ("apply_top_k_top_p_pytorch", "apply_top_k_only"):
        exec(compile(ast.Module([_node], []), str(_SRC), "exec"), _ns)
apply_top_k_top_p_pytorch = _ns["apply_top_k_top_p_pytorch"]
from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton

dev = "cuda"


def spec_keep(logits, k, p):
    l = logits.double()
    B, Vn = l.shape
    keep = torch.isfinite(l) | (l == float("inf"))
    amb = torch.zeros(B, dtype=torch.bool)
    for i in range(B):
        row = l[i]
        kk = keep[i].clone()
        if k is not None and int(k[i]) < Vn:
            vals = row[kk]
            if int(k[i]) < vals.numel():
                th = vals.topk(int(k[i])).values[-1]
                kk &= row >= th
        if p is not None and float(p[i]) < 1.0:
            pr = torch.zeros_like(row)
            pr[kk] = torch.softmax(row[kk], 0)
            order = torch.argsort(row, descending=True, stable=True)
            ps = pr[order]
            above = torch.cumsum(ps, 0) - ps
            sel = above < float(p[i])
            k2 = torch.zeros_like(kk)
            k2[order[sel]] = True
            # ambiguity: the crossing token's value is tied with a dropped token, or mass within 1e-6 of p
            last = order[sel][-1]
            tied_drop = (~k2 & kk & (row == row[last])).any()
            near = ((above - float(p[i])).abs() < 1e-6).any()
            amb[i] = bool(tied_drop or near)
            kk &= k2
        keep[i] = kk
    return keep.to(logits.device), amb


def compare(name, logits, k, p):
    out_t = apply_top_k_top_p_triton(logits.clone(), k, p)
    out_r = apply_top_k_top_p_pytorch(logits.clone(), k, p)
    kt, kr = out_t > float("-inf"), out_r > float("-inf")
    ks, amb = spec_keep(logits.cpu(), None if k is None else k.cpu(), None if p is None else p.cpu())
    ks = ks.to(dev)
    amb = amb.to(dev)
    probs = torch.softmax(logits.double(), -1)
    diff_t = (kt != ks).any(1) & ~amb
    diff_r = (kr != ks).any(1) & ~amb
    worst = 0.0
    if diff_t.any():
        worst = float(((kt != ks).double() * probs).sum(1)[diff_t].max())
    print(f"{name:42s} rows={logits.shape[0]:4d} amb={int(amb.sum()):3d} triton!=spec={int(diff_t.sum()):3d} "
          f"pytorch!=spec={int(diff_r.sum()):3d} triton!=pytorch={int((kt != kr).any(1).sum()):3d} "
          f"worst mass diff={worst:.3g}  kept(t/s) e.g. {kt[diff_t].sum(1)[:3].tolist()}/{ks[diff_t].sum(1)[:3].tolist()}")
    return diff_t


def run(seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    for V in (32000, 151936):
        for B in (8, 200):
            def kp(kind):
                k = torch.randint(1, 200, (B,), device=dev, generator=g, dtype=torch.int32) if "k" in kind else None
                p = (torch.rand(B, device=dev, generator=g) * 0.98 + 0.01) if "p" in kind else None
                return k, p
            dists = {
                "gauss1": torch.randn(B, V, device=dev, generator=g),
                "gauss5": 5 * torch.randn(B, V, device=dev, generator=g),
                "bf16_ties": (4 * torch.randn(B, V, device=dev, generator=g)).bfloat16().float(),
                "peaked": torch.randn(B, V, device=dev, generator=g) + 30 * (torch.arange(V, device=dev) == 7),
                "gumbel": -torch.log(-torch.log(torch.rand(B, V, device=dev, generator=g).clamp_min(1e-12))) * 3,
                "masked_mostly": torch.where(torch.rand(B, V, device=dev, generator=g) < 0.999,
                                             torch.full((B, V), float("-inf"), device=dev),
                                             torch.randn(B, V, device=dev, generator=g)),
                "lognormal_heavy": torch.exp(torch.randn(B, V, device=dev, generator=g) * 1.5),
            }
            for dname, logits in dists.items():
                for kind in ("k", "p", "kp"):
                    k, p = kp(kind)
                    compare(f"V={V} B={B} {dname} {kind}", logits, k, p)
    # edge parameters
    V, B = 32000, 16
    logits = torch.randn(B, V, device=dev, generator=g) * 2
    for pv in (1e-4, 0.999, 0.9999999, 1.0):
        compare(f"edge p={pv}", logits, None, torch.full((B,), pv, device=dev))
    for kv in (1, V - 1, V):
        compare(f"edge k={kv}", logits, torch.full((B,), kv, device=dev, dtype=torch.int32), None)
    compare("edge k=V p=0.5", logits, torch.full((B,), V, device=dev, dtype=torch.int32),
            torch.full((B,), 0.5, device=dev))
    zeros = torch.zeros(B, V, device=dev)
    compare("all-equal k=5", zeros, torch.full((B,), 5, device=dev, dtype=torch.int32), None)
    compare("all-equal p=0.5", zeros, None, torch.full((B,), 0.5, device=dev))
    few = torch.full((B, V), float("-inf"), device=dev)
    few[:, :3] = torch.randn(B, 3, device=dev, generator=g)
    compare("3 finite, k=50", few, torch.full((B,), 50, device=dev, dtype=torch.int32), None)
    compare("3 finite, k=50 p=0.9", few, torch.full((B,), 50, device=dev, dtype=torch.int32),
            torch.full((B,), 0.9, device=dev))
    compare("3 finite, p=0.9", few, None, torch.full((B,), 0.9, device=dev))


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
