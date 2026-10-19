"""Direct Triton 3.6 float32 2x2 stride-2 max_pool2d values."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


N, C, H, W = 2, 3, 16, 16
KERNEL = (2, 2)
STRIDE = (2, 2)
OUT_H, OUT_W = H // STRIDE[0], W // STRIDE[1]
OUTPUT_SIZE = N * C * OUT_H * OUT_W


@triton.jit
def _max_pool2d_kernel(x_ptr, out_ptr,
                       C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                       OUT_H: tl.constexpr, OUT_W: tl.constexpr):
    pid = tl.program_id(0)
    ow = pid % OUT_W
    rest = pid // OUT_W
    oh = rest % OUT_H
    rest = rest // OUT_H
    channel = rest % C
    batch = rest // C
    ih = oh * 2
    iw = ow * 2
    base = ((batch * C + channel) * H + ih) * W + iw
    value_00 = tl.load(x_ptr + base)
    value_01 = tl.load(x_ptr + base + 1)
    value_10 = tl.load(x_ptr + base + W)
    value_11 = tl.load(x_ptr + base + W + 1)
    maximum = tl.maximum(tl.maximum(value_00, value_01),
                         tl.maximum(value_10, value_11))
    tl.store(out_ptr + pid, maximum)


def call(inputs):
    if set(inputs) != {"x"}:
        raise ValueError(f"expected input x, received {sorted(inputs)}")
    x = inputs["x"]
    if (not torch.is_tensor(x) or x.shape != (N, C, H, W) or x.dtype != torch.float32
            or not x.is_cuda or not x.is_contiguous()):
        raise ValueError(f"x must be a contiguous CUDA float32 tensor of shape {(N, C, H, W)}")
    if not bool(torch.isfinite(x).all()) or bool((x.abs() > 16).any()):
        raise ValueError("x must contain finite values in [-16, 16]")
    out = torch.empty((N, C, OUT_H, OUT_W), dtype=x.dtype, device=x.device)
    _max_pool2d_kernel[(OUTPUT_SIZE,)](
        x, out, C=C, H=H, W=W, OUT_H=OUT_H, OUT_W=OUT_W, num_warps=1)
    return {"out": out}
