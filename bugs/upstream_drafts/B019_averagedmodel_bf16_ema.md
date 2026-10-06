# `AveragedModel` EMA of a bfloat16 model stagnates, and an fp32 averaged copy is rejected

### 🐛 Describe the bug

`torch.optim.swa_utils.AveragedModel` deep-copies the model, so for a model whose parameters are bfloat16 the EMA
buffer is bfloat16 as well. `get_ema_multi_avg_fn(decay)` updates it with `torch._foreach_lerp_(ema, current, 1 - decay)`.
With a typical decay of 0.999 the increment `(1 - decay) * (p - ema)` is smaller than half a bf16 ulp of `ema` for almost
every element, so it is rounded away and the EMA stays close to the weights it started from.

Keeping the average in fp32 is not possible with the stock API: an fp32 copy of a bf16 model fails in
`update_parameters` (tensors are grouped by dtype before `multi_avg_fn` is called).

```python
import copy, torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

torch.manual_seed(0)
model = torch.nn.Linear(256, 256).cuda().to(torch.bfloat16)
target = copy.deepcopy(model).float()
with torch.no_grad():
    for p in target.parameters():
        p.add_(0.01 * torch.randn_like(p))          # the trained weights drift

ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))
ema.update_parameters(model)                         # first call copies
ref = [p.detach().float().clone() for p in model.parameters()]
with torch.no_grad():
    for p, t in zip(model.parameters(), target.parameters()):
        p.copy_(t)
for _ in range(1000):
    ema.update_parameters(model)
    with torch.no_grad():
        for r, p in zip(ref, model.parameters()):
            r.lerp_(p.float(), 1 - 0.999)

def dist(w):
    return ((w.float() - target.weight).norm() / target.weight.norm()).item()

print(dist(next(ema.parameters())), dist(ref[0]))    # 0.262 (bf16 AveragedModel) vs 0.098 (fp32 EMA)

fp32_copy = AveragedModel(copy.deepcopy(model).float(), multi_avg_fn=get_ema_multi_avg_fn(0.999))
fp32_copy.update_parameters(model)
fp32_copy.update_parameters(model)                   # RuntimeError (dtype mismatch)
```

Output with 2.10.0 and with the 2026-09-07 nightly: the fp32 EMA has closed the expected 63% of the gap
(1 − 0.999¹⁰⁰⁰; relative distance 0.268 → 0.098), the bf16 `AveragedModel` 2% (0.268 → 0.262); the fp32 copy raises
`expected dtype float for end but got dtype c10::BFloat16` (nightly) /
`Tensors of the same index must be on the same device and the same dtype ...` (2.10).

In a small pure-bf16 language-model training run (2,000 steps, 4 seeds), the averaged model from `AveragedModel` ends at
validation loss ≈ 9.0 versus ≈ 5.16 for an fp32 EMA of the same run (the trained model itself is at ≈ 4.89).

A workaround is an fp32 copy with a per-parameter `avg_fn` (which does not go through the dtype grouping):
`AveragedModel(copy.deepcopy(model).float(), avg_fn=lambda e, p, n: e.lerp(p.float(), 1 - decay))`.

Possible fixes: let the averaged copy have a different (wider) dtype than the model and do the lerp in the copy's dtype,
or add a `dtype=` argument; at least document that EMA of low-precision models should be kept in fp32.

### Versions

PyTorch 2.10.0+cu128 and 2.15.0.dev20260907+cu126 (CUDA, RTX A6000). The behaviour is not device specific.
