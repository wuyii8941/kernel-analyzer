"""Direct Triton 3.6 float32 tie-free top-k then top-p keep-mask core."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


BATCH, VOCAB = 64, 64


def make_unique_logits(seed, shape, dtype):
    """Make varied rows whose physical float32 values are provably tie-free."""
    if tuple(shape) != (BATCH, VOCAB) or dtype != "float32":
        raise ValueError("the reviewed input generator is fixed to float32 [64,64]")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    base = torch.linspace(-3.0, 3.0, VOCAB, dtype=torch.float32)
    rows = []
    for row in range(BATCH):
        scale = 0.5 + 1.5 * ((row + int(seed)) % 17) / 16.0
        shift = 0.03 * (((7 * row + int(seed)) % 11) - 5)
        values = base * scale + shift
        rows.append(values[torch.randperm(VOCAB, generator=generator)])
    return torch.stack(rows).cuda()


@triton.jit
def _topk_topp_keep(logits_ptr, k_ptr, p_ptr, keep_ptr, VOCAB: tl.constexpr):
    row = tl.program_id(0)
    token = tl.arange(0, VOCAB)
    logits = tl.load(logits_ptr + row * VOCAB + token)
    k = tl.load(k_ptr + row)
    p = tl.load(p_ptr + row)

    # higher[j, i] states that token i has a strictly larger logit than token j.
    higher = logits[None, :] > logits[:, None]
    rank = tl.sum(higher.to(tl.int32), axis=1)
    topk = rank < k

    maximum = tl.max(tl.where(topk, logits, -float("inf")), axis=0)
    mass = tl.where(topk, tl.exp(logits - maximum), 0.0)
    total = tl.sum(mass, axis=0)
    higher_mass = tl.sum(tl.where(higher & topk[None, :], mass[None, :], 0.0), axis=1) / total
    # p=1 keeps the complete top-k remainder.  Computing the complementary
    # mass as ``higher_mass`` can round to 1 for a very small final mass, so
    # preserve this source-level vLLM boundary explicitly.
    keep = topk & ((p == 1.0) | (higher_mass < p))
    tl.store(keep_ptr + row * VOCAB + token, keep)


def call(inputs):
    required = {"logits", "k", "p"}
    if set(inputs) != required:
        raise ValueError(f"expected {sorted(required)}, received {sorted(inputs)}")
    logits, k, p = inputs["logits"], inputs["k"], inputs["p"]
    if (not torch.is_tensor(logits) or tuple(logits.shape) != (BATCH, VOCAB)
            or logits.dtype != torch.float32 or not logits.is_cuda or not logits.is_contiguous()):
        raise ValueError(f"logits must be a contiguous CUDA float32 tensor of shape {(BATCH, VOCAB)}")
    if (not torch.is_tensor(k) or tuple(k.shape) != (BATCH,) or k.dtype != torch.int32
            or not k.is_cuda or not k.is_contiguous()):
        raise ValueError(f"k must be a contiguous CUDA int32 tensor of shape {(BATCH,)}")
    if (not torch.is_tensor(p) or tuple(p.shape) != (BATCH,) or p.dtype != torch.float32
            or not p.is_cuda or not p.is_contiguous()):
        raise ValueError(f"p must be a contiguous CUDA float32 tensor of shape {(BATCH,)}")
    if not bool(torch.isfinite(logits).all()) or bool((logits.abs() > 8).any()):
        raise ValueError("logits must contain finite values in [-8, 8]")
    ordered = torch.sort(logits, dim=1).values
    if bool((ordered[:, 1:] == ordered[:, :-1]).any()):
        raise ValueError("logits must be pairwise distinct within every row")
    if bool(((k < 1) | (k > VOCAB)).any()):
        raise ValueError(f"k must lie in [1, {VOCAB}]")
    if not bool(torch.isfinite(p).all()) or bool(((p <= 0) | (p > 1)).any()):
        raise ValueError("p must lie in (0, 1]")
    keep = torch.empty_like(logits, dtype=torch.bool)
    _topk_topp_keep[(BATCH,)](logits, k, p, keep, VOCAB=VOCAB, num_warps=4)
    return {"keep": keep}
