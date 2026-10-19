"""Direct Triton 3.6 implementation of one ordinary grouped Conv2d call."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


N, C_IN, H, W = 2, 4, 8, 9
C_OUT, C_PER_GROUP, K_H, K_W, GROUPS = 6, 2, 3, 3, 2
STRIDE = (1, 2)
PADDING = (1, 1)
DILATION = (1, 1)
OUT_H = (H + 2 * PADDING[0] - DILATION[0] * (K_H - 1) - 1) // STRIDE[0] + 1
OUT_W = (W + 2 * PADDING[1] - DILATION[1] * (K_W - 1) - 1) // STRIDE[1] + 1


@triton.jit
def _conv2d_kernel(x_ptr, weight_ptr, bias_ptr, out_ptr,
                   N: tl.constexpr, C_IN: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                   C_OUT: tl.constexpr, C_PER_GROUP: tl.constexpr,
                   K_H: tl.constexpr, K_W: tl.constexpr, GROUPS: tl.constexpr,
                   STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
                   PAD_H: tl.constexpr, PAD_W: tl.constexpr,
                   DILATION_H: tl.constexpr, DILATION_W: tl.constexpr,
                   OUT_H: tl.constexpr, OUT_W: tl.constexpr):
    pid = tl.program_id(0)
    ow = pid % OUT_W
    q = pid // OUT_W
    oh = q % OUT_H
    q = q // OUT_H
    co = q % C_OUT
    n = q // C_OUT
    group = co // (C_OUT // GROUPS)

    acc = tl.load(bias_ptr + co).to(tl.float32)
    for local_ci in tl.static_range(0, C_PER_GROUP):
        ci = group * C_PER_GROUP + local_ci
        for kh in tl.static_range(0, K_H):
            ih = oh * STRIDE_H + kh * DILATION_H - PAD_H
            for kw in tl.static_range(0, K_W):
                iw = ow * STRIDE_W + kw * DILATION_W - PAD_W
                inside = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                x_offset = ((n * C_IN + ci) * H + ih) * W + iw
                weight_offset = ((co * C_PER_GROUP + local_ci) * K_H + kh) * K_W + kw
                x = tl.load(x_ptr + x_offset, mask=inside, other=0.0)
                weight = tl.load(weight_ptr + weight_offset)
                acc += x * weight
    tl.store(out_ptr + pid, acc)


def call(inputs):
    required = {"x", "weight", "bias", "stride", "padding", "dilation", "transposed", "output_padding", "groups"}
    if set(inputs) != required:
        raise ValueError(f"expected inputs {sorted(required)}, received {sorted(inputs)}")
    tensors = (("x", inputs["x"], (N, C_IN, H, W)),
               ("weight", inputs["weight"], (C_OUT, C_PER_GROUP, K_H, K_W)),
               ("bias", inputs["bias"], (C_OUT,)))
    for name, value, shape in tensors:
        if not torch.is_tensor(value) or value.shape != shape or value.dtype != torch.float32 or not value.is_cuda:
            raise ValueError(f"{name} must be a contiguous CUDA float32 tensor of shape {shape}")
        if not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    expected = {
        "stride": list(STRIDE), "padding": list(PADDING), "dilation": list(DILATION),
        "transposed": False, "output_padding": [0, 0], "groups": GROUPS,
    }
    if any(inputs[name] != value for name, value in expected.items()):
        raise ValueError("this call fixes stride, padding, dilation, transposed, output_padding and groups")

    out = torch.empty((N, C_OUT, OUT_H, OUT_W), dtype=torch.float32, device=inputs["x"].device)
    grid = (N * C_OUT * OUT_H * OUT_W,)
    _conv2d_kernel[grid](
        inputs["x"], inputs["weight"], inputs["bias"], out,
        N=N, C_IN=C_IN, H=H, W=W, C_OUT=C_OUT, C_PER_GROUP=C_PER_GROUP,
        K_H=K_H, K_W=K_W, GROUPS=GROUPS,
        STRIDE_H=STRIDE[0], STRIDE_W=STRIDE[1], PAD_H=PADDING[0], PAD_W=PADDING[1],
        DILATION_H=DILATION[0], DILATION_W=DILATION[1], OUT_H=OUT_H, OUT_W=OUT_W,
        num_warps=4,
    )
    return {"out": out}
