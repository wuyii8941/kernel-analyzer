import math

import torch
import triton
import triton.language as tl
from flag_gems.ops.flash_attention_backward import flash_attn_backward


@triton.jit
def make_mask(mask, seed, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(mask + i, tl.rand(seed, i) > 0.5, i < SIZE)


def test_varlen_dropout_query_gradient():
    m, n, d, seed = 128, 256, 32, 7331
    q = torch.zeros((m, 1, d), device="cuda", dtype=torch.float16)
    k = torch.full((n, 1, d), 0.125, device="cuda", dtype=q.dtype)
    v = torch.ones_like(k)
    dout = torch.full_like(q, 1 / q.numel())
    keep = torch.empty((m, n), device="cuda", dtype=torch.bool)
    make_mask[(triton.cdiv(keep.numel(), 256),)](
        keep, seed, keep.numel(), 256,
    )
    # All logits are equal. Apply exactly the backward core's fixed mask.
    out = (keep.float().sum(-1) * (2 / n))[:, None, None]
    out = out.expand_as(q).to(q.dtype).contiguous()
    lse = torch.full((m, 1), math.log(n), device="cuda", dtype=torch.float32)
    dq, _, _, _ = flash_attn_backward(
        dout, q, k, v, out, lse,
        cu_seq_q=torch.tensor([0, m], device="cuda", dtype=torch.int32),
        cu_seq_k=torch.tensor([0, n], device="cuda", dtype=torch.int32),
        max_seqlen_q=m, max_seqlen_k=n,
        softmax_scale=d**-0.5,
        is_dropout=True, dropout_p=0.5, rng_state=(seed, 0),
    )
    print("max |dQ|:", dq.abs().max().item())
    # Identical keys make the softmax probabilities independent of Q.
    torch.testing.assert_close(dq, torch.zeros_like(dq), rtol=0, atol=1e-6)
