"""Direct Triton 3.6 float32 Mamba selective state update with complete state writeback."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


BATCH, HEADS, DIM, DSTATE, GROUPS = 8, 4, 32, 8, 2


@triton.jit
def _mamba_ssu_kernel(state_ptr, x_ptr, dt_ptr, a_ptr, b_ptr, c_ptr,
                      d_ptr, z_ptr, bias_ptr, out_ptr,
                      HEADS: tl.constexpr, DIM: tl.constexpr, DSTATE: tl.constexpr,
                      GROUPS: tl.constexpr):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    channels = tl.arange(0, DIM)
    coordinates = tl.arange(0, DSTATE)
    group = head // (HEADS // GROUPS)

    value_offset = (batch * HEADS + head) * DIM + channels
    state_offset = value_offset[:, None] * DSTATE + coordinates[None, :]
    coefficient_offset = (head * DIM + channels[:, None]) * DSTATE + coordinates[None, :]
    group_offset = (batch * GROUPS + group) * DSTATE + coordinates

    x = tl.load(x_ptr + value_offset).to(tl.float32)
    delta = tl.load(dt_ptr + value_offset).to(tl.float32)
    delta += tl.load(bias_ptr + head * DIM + channels).to(tl.float32)
    positive = delta + tl.log(1.0 + tl.exp(-delta))
    negative = tl.log(1.0 + tl.exp(delta))
    softplus = tl.where(delta > 0.0, positive, negative)
    delta = tl.where(delta <= 20.0, softplus, delta)

    state = tl.load(state_ptr + state_offset).to(tl.float32)
    coefficient = tl.load(a_ptr + coefficient_offset).to(tl.float32)
    input_weight = tl.load(b_ptr + group_offset).to(tl.float32)
    output_weight = tl.load(c_ptr + group_offset).to(tl.float32)
    next_state = (
        state * tl.exp(delta[:, None] * coefficient)
        + delta[:, None] * input_weight[None, :] * x[:, None]
    )
    tl.store(state_ptr + state_offset, next_state)

    out = tl.sum(next_state * output_weight[None, :], axis=1)
    out += x * tl.load(d_ptr + head * DIM + channels).to(tl.float32)
    gate = tl.load(z_ptr + value_offset).to(tl.float32)
    out *= gate * tl.sigmoid(gate)
    tl.store(out_ptr + value_offset, out)


def call(inputs):
    names = ("state", "x", "dt", "A", "B", "C", "D", "z", "dt_bias")
    if set(inputs) != set(names):
        raise ValueError(f"expected {sorted(names)}, received {sorted(inputs)}")
    expected = {
        "state": (BATCH, HEADS, DIM, DSTATE),
        "x": (BATCH, HEADS, DIM),
        "dt": (BATCH, HEADS, DIM),
        "A": (HEADS, DIM, DSTATE),
        "B": (BATCH, GROUPS, DSTATE),
        "C": (BATCH, GROUPS, DSTATE),
        "D": (HEADS, DIM),
        "z": (BATCH, HEADS, DIM),
        "dt_bias": (HEADS, DIM),
    }
    for name in names:
        value = inputs[name]
        if (not torch.is_tensor(value) or tuple(value.shape) != expected[name]
                or value.dtype != torch.float32 or not value.is_cuda or not value.is_contiguous()):
            raise ValueError(f"{name} must be a contiguous CUDA float32 tensor of shape {expected[name]}")
        if not bool(torch.isfinite(value).all()) or bool((value.abs() > 8).any()):
            raise ValueError(f"{name} must contain finite values in [-8, 8]")

    out = torch.empty_like(inputs["x"])
    _mamba_ssu_kernel[(BATCH, HEADS)](
        inputs["state"], inputs["x"], inputs["dt"], inputs["A"], inputs["B"], inputs["C"],
        inputs["D"], inputs["z"], inputs["dt_bias"], out,
        HEADS=HEADS, DIM=DIM, DSTATE=DSTATE, GROUPS=GROUPS, num_warps=4,
    )
    return {"y": out, "state": inputs["state"]}
