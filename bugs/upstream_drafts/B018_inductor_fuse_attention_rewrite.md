# [inductor] SDPA fusion of the DistilBERT pattern (`_sfdp_pattern_15/17/20`) keeps `attn_mask == 1` where the pattern masks `attn_mask == 0`: wrong results for masks with other values and for fully masked rows

### 🐛 Describe the bug

`_sfdp_pattern_15` (and the dropout variants 17 and 20) match

```python
mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)
out = torch.softmax(scores.masked_fill(mask, fill_value), dim=-1) @ v
```

but the replacement passes `(attn_mask == 1)` to SDPA as the boolean keep-mask. The two agree only when every mask
entry is 0 or 1. `_sfdp_params_check` does not constrain the mask's values or dtype for these patterns, so any mask is
accepted:

```python
import torch

def attn(query, key, value, attn_mask, inv_scale):          # the form of _sfdp_pattern_15
    q, k, v = (t.permute([0, 2, 1, 3]) for t in (query, key, value))
    bs, k_len = q.size(0), k.size(-2)
    scores = (q @ k.transpose(-2, -1)).div(inv_scale)
    fill = torch.full((), -float("inf"), dtype=query.dtype, device=query.device)
    m = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)
    return torch.softmax(scores.masked_fill(m, fill), dim=-1) @ v

torch.manual_seed(0)
q, k, v = (torch.randn(2, 16, 4, 32, device="cuda") for _ in range(3))
seg = torch.tensor([[1] * 5 + [2] * 5 + [3] * 2 + [0] * 4, [1] * 8 + [2] * 8], device="cuda")  # packed segments, 0 = pad
print((torch.compile(attn)(q, k, v, seg, 8.0) - attn(q, k, v, seg, 8.0)).abs().max())  # 1.88: keys of segments 2 and 3 dropped
```

Measured with torch 2.10 on CUDA (`counters["inductor"]["fuse_attention"] == 1` in every row):

| mask / input | max \|compiled - eager\| |
|---|---|
| 0/1 padding mask (control) | 7e-7 |
| segment ids 1/2/3, 0 = padding | 1.83 |
| soft mask 0 / 0.5 / 1 | 1.64 |
| a padding-only sequence, fill `-inf` | eager NaN, compiled 0 |
| a padding-only sequence, fill `torch.finfo(float32).min` (as in HF DistilBERT) | 0.70 (eager: uniform average of `v`; compiled: 0) |

The replacement code is the same on main (2026-10-06). #195383 (re-tracing matches) does not address this: the match is
structurally right, it is the replacement that is not equivalent.

Separately, on the 2.10 release the pattern constants are wildcards: pattern 5 (`/ math.sqrt(query.size(-1))`) matches
any divisor and is replaced with SDPA's default scale (dividing by `sqrt(d_model) = 16` instead of `sqrt(head_dim) = 8`
gives a max difference of 1.19; a temperature of 4.0 gives 1.7), and pattern 15 matches any `masked_fill` value
(`0.0` gives 0.55). We expect #195383 to fix this on main and will confirm on a nightly; it may be worth a backport.

### Suggested fix

- In `_sfdp_replacement_15/17/20`, keep `attn_mask != 0` (the exact complement of the pattern's `attn_mask == 0`).
- Only rewrite when `fill_value` is `-inf`, or reproduce the pattern's result on fully masked rows (NaN for `-inf`, a
  uniform average for a finite fill); SDPA returns 0 there.

### Versions

torch 2.10.0+cu128, RTX A6000. `fuse_attention.py` replacements unchanged on main (2026-10-06).
