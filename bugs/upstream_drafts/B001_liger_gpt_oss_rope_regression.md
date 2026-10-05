# [Bug] #1451 breaks GPT-OSS RoPE on main (cos/sin are half width, inferred as partial rotary)

## Summary

Since #1451 (`feat(ops): support partial RoPE in Triton kernel`, merged 2026-10-02), `rope_forward` infers
`rotary_dim = min(cos.shape[-1], head_dim)`. GPT-OSS's rotary embedding returns cos/sin of width `head_dim / 2`
(`emb = freqs`, not `cat(freqs, freqs)`), and HF's `_apply_rotary_emb` rotates `(x[:d/2], x[d/2:])`. With the new
inference the kernel treats GPT-OSS as partial rotary with `rotary_dim = d/2`: it rotates only the first half of each
head, pairing `(i, i + d/4)`, and passes the second half through. No error is raised.

`apply_liger_kernel_to_gpt_oss` patches `modeling_gpt_oss.apply_rotary_pos_emb = liger_rotary_pos_emb` with
`rope=True` by default, so the next release would silently change GPT-OSS's attention for everyone who fine-tunes it
with `use_liger_kernel=True`. Release 0.8.4 is correct here (its kernel reads the first d/2 entries of cos/sin,
which matches GPT-OSS's half-width convention).

## Reproduction

```python
import torch
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding, apply_rotary_pos_emb
from liger_kernel.transformers.rope import liger_rotary_pos_emb

torch.manual_seed(0)
cfg = GptOssConfig()                      # head_dim 64
q = torch.randn(2, 8, 256, 64, device="cuda")
k = torch.randn(2, 2, 256, 64, device="cuda")
pos = torch.arange(256, device="cuda")[None].expand(2, -1)
cos, sin = GptOssRotaryEmbedding(cfg).cuda()(q, pos)     # cos.shape[-1] == 32
q_ref, _ = apply_rotary_pos_emb(q, k, cos, sin)
q_lig, _ = liger_rotary_pos_emb(q.clone(), k.clone(), cos, sin)
print(((q_lig - q_ref).norm() / q_ref.norm()).item())
```

| Liger | relative error of q (and of dq) |
|---|---|
| 0.8.4 (PyPI) | 3.3e-8 |
| main @ b297821787 | 0.97 |

Liger's own test also catches it: `pytest test/convergence/fp32/test_mini_models.py -k gpt_oss` fails on main and
passes when only `src/liger_kernel/ops/rope.py` is reverted to the 0.8.4 version (transformers 5.18.0, torch 2.10.0,
RTX A6000).

## Suggested direction

The rotary width cannot be inferred from `cos.shape[-1]` alone: Llama/Phi-3 style cos/sin are duplicated
(`width == rotary_dim`), GPT-OSS style cos/sin are not (`width == rotary_dim / 2`). Either let the caller pass the
convention / rotary_dim explicitly, or wrap GPT-OSS's patch so it duplicates cos/sin to full width before calling the
kernel. A regression check like the snippet above for each patched model's own `apply_rotary_pos_emb` would cover both
conventions.

Environment: transformers 5.18.0, torch 2.10.0+cu128, triton 3.6.0, RTX A6000 (sm_86).
