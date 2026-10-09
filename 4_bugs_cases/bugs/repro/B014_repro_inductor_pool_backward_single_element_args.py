"""B014: torch.compile (Inductor) pooling backward crashes on 1-element kernel_size / stride / padding / dilation.

Eager accepts a 1-element sequence for these arguments (it means "the same value in every spatial dim") and the
forward lowering normalizes it (pad_listlike), but the backward lowerings avg_pool2d_backward, avg_pool3d_backward
and max_pool2d_with_indices_backward assert the full length, so training under torch.compile fails with
LoweringException: AssertionError.  --fixed applies the normalization to the three lowerings at runtime.

    python bugs/repro/B014_repro_inductor_pool_backward_single_element_args.py [--fixed]
"""
import os
import sys

os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")  # a cached compile would skip the lowering

import torch  # noqa: E402
import torch.nn.functional as F

if "--fixed" in sys.argv:
    from torch._inductor import lowering as L

    def normalized(op, n, positions):
        orig = L.lowerings[op]

        def wrapped(*args, **kwargs):
            args = list(args)
            for i in positions:
                if i < len(args) and args[i]:
                    args[i] = L.pad_listlike(args[i], n)
            return orig(*args, **kwargs)

        L.lowerings[op] = wrapped

    aten = torch.ops.aten
    normalized(aten.avg_pool2d_backward.default, 2, [2, 3, 4])
    normalized(aten.avg_pool3d_backward.default, 3, [2, 3, 4])
    normalized(aten.max_pool2d_with_indices_backward.default, 2, [2, 3, 4, 5])

x2 = torch.randn(2, 3, 9, 9, device="cuda", requires_grad=True)
x3 = torch.randn(2, 3, 7, 7, 7, device="cuda", requires_grad=True)
tests = {
    "nn.AvgPool2d(3, padding=(1,))": (x2, torch.nn.AvgPool2d(3, padding=(1,))),
    "F.avg_pool2d(x, (3,), (2,), (1,))": (x2, lambda t: F.avg_pool2d(t, (3,), (2,), (1,))),
    "F.max_pool2d(x, 3, 2, 1, dilation=(1,))": (x2, lambda t: F.max_pool2d(t, 3, 2, 1, (1,))),
    "F.max_pool2d(x, (3,), (2,), (1,))": (x2, lambda t: F.max_pool2d(t, (3,), (2,), (1,))),
    "F.avg_pool3d(x, (3,), (2,), (1,))": (x3, lambda t: F.avg_pool3d(t, (3,), (2,), (1,))),
    "F.max_pool3d(x, (3,), (2,), (1,))  [control]": (x3, lambda t: F.max_pool3d(t, (3,), (2,), (1,))),
}
for name, (x, fn) in tests.items():
    torch._dynamo.reset()
    ye = fn(x)
    ge, = torch.autograd.grad(ye.sum(), x)
    try:
        f = torch.compile(fn)
        y = f(x)
        g, = torch.autograd.grad(y.sum(), x)
        print(f"{name:46s} ok: forward diff {(y - ye).abs().max().item():.1e}, grad diff {(g - ge).abs().max().item():.1e}")
    except Exception as e:  # noqa: BLE001
        print(f"{name:46s} {type(e).__name__}: {str(e).splitlines()[0][:60]}")
