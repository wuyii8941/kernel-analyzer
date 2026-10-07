<!-- Proposed comment on pytorch/pytorch#198119 -->

`avg_pool3d` is affected too, on CUDA and CPU (torch 2.10.0, and the same code is on main). It only goes through
`fallback_avg_pool3d_backward` when the backward window exceeds 125 positions; for ordinary kernels the
`avg_pool3d_backward` lowering has the same `scale = kernel_size[0] * kernel_size[1] * kernel_size[2]` under
`count_include_pad`, so the overhanging `ceil_mode` windows are divided by the full kernel volume:

```python
import torch, torch.nn.functional as F

def grad_of(f, x, backend=None):
    x = x.detach().clone().requires_grad_()
    (f if backend is None else torch.compile(f, backend=backend))(x).sum().backward()
    torch._dynamo.reset()
    return x.grad

x = torch.arange(6.0, dtype=torch.float64).reshape(1, 1, 1, 1, 6).expand(1, 1, 6, 6, 6).contiguous()
f = lambda t: F.avg_pool3d(t, 3, stride=2, ceil_mode=True)
for backend in (None, "aot_eager", "inductor"):
    print(backend, grad_of(f, x, backend)[0, 0, 5, 5].tolist())
# eager / aot_eager: [0.0833, 0.0833, 0.1667, 0.0833, 0.2083, 0.125]
# inductor:          [0.0370, 0.0370, 0.0741, 0.0370, 0.0741, 0.0370]
```

Relative gradient error of the compiled backward (aot_eager is exact in every row):

| config | inductor |
|---|---|
| k=3, s=2, ceil_mode, 6x6x6 | 0.378 |
| k=3, s=2, p=1, ceil_mode, 6x6x6 | 0.248 |
| k=(2,3,3), s=2, ceil_mode, (2,3,7,7,7) | 0.316 |
| OpInfo sample k=(4,5,6), s=(2,3,2), p=2, ceil_mode | 0.161 |
| count_include_pad=False / ceil_mode=False | 0 |
| k=12, s=2, ceil_mode, 13^3 (window > 125 -> fallback) | 0 |

So the fix needs the same clipped window size in `avg_pool3d_backward` as in `avg_pool2d_backward`. (The OpInfo
samples hit this, but `test_torchinductor_opinfo.py` runs avg_pool1d/2d/3d on one sample only.)

Still present on nightly 2.15.0.dev20260907+cu126 (CUDA): in a grid of 384 avg_pool conditions, the compiled backward follows the
full-kernel divisor on all 22 overhanging `ceil_mode` + `count_include_pad` conditions - 14 1-d, 2 2-d and 6 3-d - while the compiled
forward and eager use the clipped window; `torch.autograd.gradcheck` through the compiled function fails on all 22.

<!-- search record (for the submitter), 2026-10-07: #198119 is the only report (open, 1d/2d, states 3d is right); no fix PR found;
test_torchinductor_opinfo runs avg_pool1d/2d/3d on one sample; nightly 2.15.0.dev20260907 reproduces 1d/2d/3d. -->
