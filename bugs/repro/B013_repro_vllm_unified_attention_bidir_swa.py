"""B013: vLLM unified_attention (TRITON_ATTN) drops keys for bidirectional sliding-window attention.

Runs against vLLM main (or the shim in .cache/pylibs/vllm_shim).  Reference: per-sequence semantics used by vLLM's
own tests (tests/kernels/attention/test_mixed_causal_attn.py in PR #51257): a causal sequence attends to keys with
0 <= q_pos - k_pos < W; a bidirectional sequence to keys with |q_pos - k_pos| < W.

    PYTHONPATH=.cache/pylibs/vllm_shim python bugs/repro/B013_repro_vllm_unified_attention_bidir_swa.py
"""

import torch

from vllm.v1.attention.ops.triton_unified_attention import unified_attention


def ref(query, kc, vc, qlens, kvlens, block_tables, scale, causal, W):
    out, start = [], 0
    for i, (ql, kl) in enumerate(zip(qlens, kvlens)):
        q = query[start:start + ql].float() * scale
        start += ql
        nb = (kl + kc.shape[1] - 1) // kc.shape[1]
        idx = block_tables[i, :nb].long()
        k = kc[idx].reshape(-1, *kc.shape[2:])[:kl].float()
        v = vc[idx].reshape(-1, *vc.shape[2:])[:kl].float()
        rep = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        s = torch.einsum("qhd,khd->hqk", q, k)
        dist = torch.arange(kl - ql, kl, device=q.device)[:, None] - torch.arange(kl, device=q.device)[None, :]
        keep = (dist < W) & (dist >= 0 if causal[i] else dist > -W)
        p = torch.softmax(s.masked_fill(~keep, float("-inf")), -1).to(v.dtype)
        out.append(torch.einsum("hqk,khd->qhd", p, v))
    return torch.cat(out)


def run(qlens, kvlens, causal, W, H, Hk, D=128, bs=16, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    nb = (max(kvlens) + bs - 1) // bs
    q = torch.randn(sum(qlens), H, D, device="cuda", dtype=dtype)
    kc = torch.randn(2048, bs, Hk, D, device="cuda", dtype=dtype)
    vc = torch.randn_like(kc)
    bt = torch.randint(0, 2048, (len(qlens), nb), device="cuda", dtype=torch.int32)
    cu = torch.tensor([0] + torch.tensor(qlens).cumsum(0).tolist(), device="cuda", dtype=torch.int32)
    sk = torch.tensor(kvlens, device="cuda", dtype=torch.int32)
    out = torch.empty_like(q)
    c = torch.tensor(causal, device="cuda", dtype=torch.bool) if isinstance(causal, list) else causal
    unified_attention(q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu, max_seqlen_q=max(qlens), seqused_k=sk,
                      max_seqlen_k=max(kvlens), softmax_scale=D ** -0.5, causal=c, window_size=(W - 1, 0),
                      block_table=bt, softcap=0.0, q_descale=None, k_descale=None, v_descale=None)
    flags = causal if isinstance(causal, list) else [causal] * len(qlens)
    r = ref(q, kc, vc, qlens, kvlens, bt, D ** -0.5, flags, W)
    errs, start = [], 0
    for ql in qlens:
        a, b = out[start:start + ql].float(), r[start:start + ql].float()
        errs.append(f"{((a - b).norm() / b.norm()).item():.2e}")
        start += ql
    return errs


if __name__ == "__main__":
    print("per-sequence relative error of the output (bf16 noise is ~4e-3)")
    # the configuration of PR #51257's new test (head_size 128, block 16, window 64)
    for H, Hk in ((4, 4), (8, 2), (16, 1)):
        for causal in ([False, True, False], [True, False, True], [True, True, False]):
            print(f"heads {H}/{Hk} causal {causal}: {run([129, 5, 65], [256, 64, 128], causal, 64, H, Hk)}")
    # plain causal=False (bool) with a 64-token window
    for H, Hk in ((4, 4), (8, 2), (16, 1)):
        print(f"heads {H}/{Hk} causal=False: {run([200, 70], [200, 70], False, 64, H, Hk)}")
