"""Direct Triton 3.6 float16 LoRA A projection with complete output writeback."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


TOKENS, LORAS, RANK, HIDDEN, SLICES = 64, 4, 8, 32, 2
SCALING = 0.5


def make_mapping(seed, shape, dtype):
    """Return a repeatable mix of no-LoRA and all four valid LoRA ids."""
    if tuple(shape) != (TOKENS,) or dtype != "int32":
        raise ValueError("the reviewed mapping generator requires int32 [64]")
    return (((torch.arange(TOKENS, dtype=torch.int32) + int(seed)) % (LORAS + 1)) - 1).cuda()


@triton.jit
def _lora_shrink_kernel(x_ptr, weight_ptr, mapping_ptr, out_ptr,
                        TOKENS: tl.constexpr, LORAS: tl.constexpr, RANK: tl.constexpr,
                        HIDDEN: tl.constexpr, SCALING: tl.constexpr):
    token = tl.program_id(0)
    rank = tl.program_id(1)
    slice_index = tl.program_id(2)
    lora = tl.load(mapping_ptr + token)
    if lora < 0:
        tl.store(out_ptr + (slice_index * TOKENS + token) * RANK + rank, 0.0)
        return
    hidden = tl.arange(0, HIDDEN)
    x = tl.load(x_ptr + token * HIDDEN + hidden).to(tl.float32)
    offset = (((slice_index * LORAS + lora) * RANK + rank) * HIDDEN + hidden)
    weight = tl.load(weight_ptr + offset).to(tl.float32)
    value = SCALING * tl.sum(weight * x, axis=0)
    tl.store(out_ptr + (slice_index * TOKENS + token) * RANK + rank, value)


def call(inputs):
    required = {"x", "weights", "mapping", "out_before"}
    if set(inputs) != required:
        raise ValueError(f"expected {sorted(required)}, received {sorted(inputs)}")
    x, weights = inputs["x"], inputs["weights"]
    mapping, output = inputs["mapping"], inputs["out_before"]
    expected = {
        "x": ((TOKENS, HIDDEN), torch.float16),
        "weights": ((SLICES, LORAS, RANK, HIDDEN), torch.float16),
        "mapping": ((TOKENS,), torch.int32),
        "out_before": ((SLICES, TOKENS, RANK), torch.float32),
    }
    for name, value in inputs.items():
        shape, dtype = expected[name]
        if (not torch.is_tensor(value) or tuple(value.shape) != shape or value.dtype != dtype
                or not value.is_cuda or not value.is_contiguous()):
            raise ValueError(f"{name} must be a contiguous CUDA {dtype} tensor of shape {shape}")
        if name != "mapping" and (not bool(torch.isfinite(value).all()) or bool((value.abs() > 8).any())):
            raise ValueError(f"{name} must contain finite values in [-8,8]")
    if bool(((mapping < -1) | (mapping >= LORAS)).any()):
        raise ValueError(f"mapping values must lie in [-1,{LORAS})")

    _lora_shrink_kernel[(TOKENS, RANK, SLICES)](
        x, weights, mapping, output,
        TOKENS=TOKENS, LORAS=LORAS, RANK=RANK, HIDDEN=HIDDEN, SCALING=SCALING,
        num_warps=1,
    )
    return {"y": output}
