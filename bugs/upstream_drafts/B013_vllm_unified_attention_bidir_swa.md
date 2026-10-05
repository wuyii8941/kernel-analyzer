# [Bug]: TRITON_ATTN `unified_attention` drops in-window keys for bidirectional sliding-window attention (V mask uses the q-block's first row)

### Your current environment

vLLM main `51eeb0c` (2026-10-05); the same code is in v0.24.0 … v0.31.0. Reproduced on an RTX A6000 (sm_86) with
torch 2.10.0 / Triton 3.6.0, calling `vllm.v1.attention.ops.triton_unified_attention.unified_attention` directly.

### 🐛 Describe the bug

With a sliding window and non-causal attention (`causal=False`, or a per-sequence causal tensor with False entries),
`unified_attention` returns wrong outputs whenever a q-block holds more than one query position
(`BLOCK_Q = 16 // num_queries_per_kv > 1`) and the bidirectional query span of a sequence is longer than the window.
No error is raised.

Cause: besides the per-element score mask (`compute_kv_seq_mask`, which is correct), the kernel zeroes V rows outside
the window, and it computes that mask relative to the **first** query of the q-block:

```python
if SLIDING_WINDOW:
    qpos_lo = q_block_local_idx * BLOCK_Q
    dist = context_len + qpos_lo - seq_offset[:, None]
    ...
    else:
        sw_mask_v = (dist < SLIDING_WINDOW) & (dist > -SLIDING_WINDOW)
    V = tl.where(sw_mask_v, V, 0.0)
```

The left bound is valid for the whole block (a key left of the first row's window is left of every row's window), but
the right bound is not: row `qpos_lo + d` may attend to keys up to `qpos_lo + d + W - 1`, and the keys in
`[qpos_lo + W, qpos_lo + W + d - 1]` have their V zeroed. Their probabilities still enter the softmax denominator, so
the output loses up to `d` value contributions per row. This branch was added in #45163 (DiffusionGemma support);
before it the V mask had only the left bound.

### Reproduction

The configuration below is the one PR #51257 adds to `tests/kernels/attention/test_mixed_causal_attn.py` (head 128,
block 16, bf16, window 64, seq_lens (129, 256), (5, 64), (65, 128)); the reference is the per-sequence semantics used by
that test (causal: `0 <= q - k < W`; bidirectional: `|q - k| < W`).

```python
import torch
from vllm.v1.attention.ops.triton_unified_attention import unified_attention

def ref(q, kc, vc, qlens, kvlens, bt, scale, causal, W):
    out, s0 = [], 0
    for i, (ql, kl) in enumerate(zip(qlens, kvlens)):
        qq = q[s0:s0 + ql].float() * scale; s0 += ql
        nb = (kl + kc.shape[1] - 1) // kc.shape[1]
        k = kc[bt[i, :nb].long()].reshape(-1, *kc.shape[2:])[:kl].float()
        v = vc[bt[i, :nb].long()].reshape(-1, *vc.shape[2:])[:kl].float()
        r = qq.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(r, 1), v.repeat_interleave(r, 1)
        d = torch.arange(kl - ql, kl, device=q.device)[:, None] - torch.arange(kl, device=q.device)[None]
        keep = (d < W) & (d >= 0 if causal[i] else d > -W)
        p = torch.softmax(torch.einsum("qhd,khd->hqk", qq, k).masked_fill(~keep, float("-inf")), -1)
        out.append(torch.einsum("hqk,khd->qhd", p.to(v.dtype), v))
    return torch.cat(out)

torch.manual_seed(0)
qlens, kvlens, causal, W, H, Hk, D, bs = [129, 5, 65], [256, 64, 128], [False, True, False], 64, 4, 4, 128, 16
q = torch.randn(sum(qlens), H, D, device="cuda", dtype=torch.bfloat16)
kc = torch.randn(2048, bs, Hk, D, device="cuda", dtype=torch.bfloat16); vc = torch.randn_like(kc)
bt = torch.randint(0, 2048, (3, 16), device="cuda", dtype=torch.int32)
cu = torch.tensor([0, 129, 134, 199], device="cuda", dtype=torch.int32)
sk = torch.tensor(kvlens, device="cuda", dtype=torch.int32)
out = torch.empty_like(q)
unified_attention(q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu, max_seqlen_q=129, seqused_k=sk, max_seqlen_k=256,
                  softmax_scale=D ** -0.5, causal=torch.tensor(causal, device="cuda"), window_size=(W - 1, 0),
                  block_table=bt, softcap=0.0, q_descale=None, k_descale=None, v_descale=None)
r = ref(q, kc, vc, qlens, kvlens, bt, D ** -0.5, causal, W)
for a, b in zip(out.split(qlens), r.split(qlens)):
    print(f"{((a.float() - b.float()).norm() / b.float().norm()).item():.2e}")
```

Per-sequence relative error (bf16 noise is ~2e-3):

| heads (q/kv) | causal | seq 0 (129 queries) | seq 1 | seq 2 (65 queries) |
|---|---|---|---|---|
| 4/4 | [False, True, False] | **1.5e-1** | 2.1e-3 | **3.3e-2** |
| 8/2 | [False, True, False] | **7.8e-2** | 2.0e-3 | **8.8e-3** |
| 16/1 (BLOCK_Q = 1) | [False, True, False] | 2.1e-3 | 2.1e-3 | 2.1e-3 |
| 4/4, `causal=False` (bool), seq_lens (200, 200), (70, 70) | — | **2.0e-1** | **1.3e-1** | |

So the new test in #51257 would fail on the Triton path (its tolerance is 1e-2).

### Suggested fix

Keep only the left bound, as before #45163:

```python
if SLIDING_WINDOW:
    qpos_lo = q_block_local_idx * BLOCK_Q
    dist = context_len + qpos_lo - seq_offset[:, None]
    V = tl.where(dist < SLIDING_WINDOW, V, 0.0)
```

The right side does not need zeroing: slots at or beyond `seq_len` are already masked by `tile_mask` (loaded as 0), and
keys right of the query within the sequence are this step's freshly written KV. With this change all configurations
above are back at 2e-3. (Using the block's last query position for the right bound would also work.)

### Impact

The backend advertises both `supports_non_causal()` and `supports_sliding_window()`. Current in-tree users of
non-causal decoder attention (DiffusionGemma denoising with per-sequence causal flags, DFlash drafts) use canvases /
blocks shorter than their windows in the default configs, so they should not hit this today; any longer bidirectional
span on a sliding-window layer does.
