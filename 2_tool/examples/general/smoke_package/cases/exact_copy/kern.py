import torch
import triton
import triton.language as tl


@triton.jit
def _copy(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(y_ptr + offs, x, mask=m)                    # negative control: exact copy


def call(inp):
    x = inp["x"]
    y = torch.empty_like(x)
    _copy[(triton.cdiv(x.numel(), 256),)](x, y, x.numel(), BLOCK=256)
    return {"y": y}
