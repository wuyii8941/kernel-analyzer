"""Direct Triton 3.6 float32 causal GQA SDPA forward."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


BATCH, HQ, HKV, LENGTH, SOURCE, HEAD_DIM, VALUE_DIM = 2, 4, 2, 16, 32, 16, 16
SCALE = 0.25


@triton.jit
def _sdpa_forward(q_ptr, k_ptr, v_ptr, out_ptr,
                  HQ: tl.constexpr, HKV: tl.constexpr, LENGTH: tl.constexpr,
                  SOURCE: tl.constexpr, HEAD_DIM: tl.constexpr, VALUE_DIM: tl.constexpr,
                  SCALE: tl.constexpr):
    program = tl.program_id(0)
    query_index = program % LENGTH
    head = (program // LENGTH) % HQ
    batch = program // (LENGTH * HQ)
    kv_head = head // (HQ // HKV)

    source = tl.arange(0, SOURCE)
    feature = tl.arange(0, HEAD_DIM)
    q_base = ((batch * HQ + head) * LENGTH + query_index) * HEAD_DIM
    k_base = ((batch * HKV + kv_head) * SOURCE) * HEAD_DIM
    q = tl.load(q_ptr + q_base + feature)
    k = tl.load(k_ptr + k_base + source[:, None] * HEAD_DIM + feature[None, :])
    scores = tl.sum(k * q[None, :], axis=1) * SCALE
    scores = tl.where(source <= query_index, scores, -float("inf"))
    maximum = tl.max(scores, axis=0)
    numerator = tl.exp(scores - maximum)
    probabilities = numerator / tl.sum(numerator, axis=0)

    value_feature = tl.arange(0, VALUE_DIM)
    v_base = ((batch * HKV + kv_head) * SOURCE) * VALUE_DIM
    values = tl.load(v_ptr + v_base + source[:, None] * VALUE_DIM + value_feature[None, :])
    result = tl.sum(probabilities[:, None] * values, axis=0)
    out_base = ((batch * HQ + head) * LENGTH + query_index) * VALUE_DIM
    tl.store(out_ptr + out_base + value_feature, result)


def call(inputs):
    required = {"q", "k", "v"}
    if set(inputs) != required:
        raise ValueError(f"expected {sorted(required)}, received {sorted(inputs)}")
    q, k, v = (inputs[name] for name in ("q", "k", "v"))
    expected = {"q": (BATCH, HQ, LENGTH, HEAD_DIM), "k": (BATCH, HKV, SOURCE, HEAD_DIM),
                "v": (BATCH, HKV, SOURCE, VALUE_DIM)}
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if (not torch.is_tensor(tensor) or tuple(tensor.shape) != expected[name] or tensor.dtype != torch.float32
                or not tensor.is_cuda or not tensor.is_contiguous()):
            raise ValueError(f"{name} must be a contiguous CUDA float32 tensor of shape {expected[name]}")
        if not bool(torch.isfinite(tensor).all()) or bool((tensor.abs() > 8).any()):
            raise ValueError(f"{name} must contain finite values in [-8, 8]")
    out = torch.empty((BATCH, HQ, LENGTH, VALUE_DIM), dtype=torch.float32, device=q.device)
    _sdpa_forward[(BATCH * HQ * LENGTH,)](
        q, k, v, out, HQ=HQ, HKV=HKV, LENGTH=LENGTH, SOURCE=SOURCE,
        HEAD_DIM=HEAD_DIM, VALUE_DIM=VALUE_DIM, SCALE=SCALE, num_warps=4,
    )
    return {"out": out}
