"""B015: Inductor avg_pool3d backward with ceil_mode=True (count_include_pad=True, the default) divides the overhanging
last windows by the full kernel volume; compiled gradients are wrong while eager / aot_eager agree.

Same mechanism as pytorch/pytorch#198119 (avg_pool1d/2d), which states that avg_pool3d "goes through the fallback
kernel and is right": that holds only when the backward window exceeds 125 positions; smaller kernels are lowered by
avg_pool3d_backward in torch/_inductor/lowering.py with scale = kD * kH * kW.

    python bugs/repro/B015_repro_inductor_avg_pool3d_ceil_backward.py
"""
import os

os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def grad_of(f, x, backend=None):
    x = x.detach().clone().requires_grad_()
    (f if backend is None else torch.compile(f, backend=backend))(x).sum().backward()
    torch._dynamo.reset()
    return x.grad


# the smallest case: avg_pool3d(k=3, s=2, ceil_mode=True) on 6^3, windows [0,1,2] [2,3,4] [4,5] per dim
x = torch.arange(6.0, dtype=torch.float64).reshape(1, 1, 1, 1, 6).expand(1, 1, 6, 6, 6).contiguous()
f = lambda t: F.avg_pool3d(t, 3, stride=2, ceil_mode=True)  # noqa: E731
for backend in (None, "aot_eager", "inductor"):
    g = grad_of(f, x, backend)
    print(f"{str(backend or 'eager'):9s} grad[0,0,5,5,:] = {g[0, 0, 5, 5].tolist()}")

cfgs = [
    ("k3 s2 ceil, 6^3", (1, 1, 6, 6, 6), dict(kernel_size=3, stride=2, ceil_mode=True)),
    ("k3 s2 p1 ceil, 6^3", (1, 1, 6, 6, 6), dict(kernel_size=3, stride=2, padding=1, ceil_mode=True)),
    ("k(2,3,3) s2 ceil, (2,3,7,7,7)", (2, 3, 7, 7, 7), dict(kernel_size=(2, 3, 3), stride=2, ceil_mode=True)),
    ("OpInfo sample: k(4,5,6) s(2,3,2) p2 ceil", (1, 1, 6, 5, 6),
     dict(kernel_size=(4, 5, 6), stride=(2, 3, 2), padding=2, ceil_mode=True)),
    ("control: count_include_pad=False", (1, 1, 6, 6, 6), dict(kernel_size=3, stride=2, ceil_mode=True,
                                                               count_include_pad=False)),
    ("control: ceil_mode=False", (1, 1, 7, 7, 7), dict(kernel_size=3, stride=2)),
    # 6 pooled windows touch an input element per dim -> 216 > 125: fallback_avg_pool3d_backward (ATen)
    ("control: window > 125 (fallback)", (1, 1, 13, 13, 13), dict(kernel_size=12, stride=2, ceil_mode=True)),
]
for dev in ("cuda", "cpu"):
    for name, shape, kw in cfgs:
        torch.manual_seed(0)
        x = torch.randn(*shape, device=dev, dtype=torch.float64)
        f = lambda t: F.avg_pool3d(t, **kw)  # noqa: E731
        ge, ga, gi = grad_of(f, x), grad_of(f, x, "aot_eager"), grad_of(f, x, "inductor")
        print(f"{dev:4s} {name:42s} aot_eager rel {((ga - ge).norm() / ge.norm()).item():.1e}   "
              f"inductor rel {((gi - ge).norm() / ge.norm()).item():.3f}")
