"""Direct Triton 3.6 float32 causal paged GQA attention."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


TOKENS, HQ, HKV, HEAD_DIM = 5, 4, 2, 16
BLOCKS, BLOCK_SIZE = 6, 8
SCALE, LEFT_WINDOW = 0.25, 4


def make_block_table(seed, shape, dtype):
    if tuple(shape) != (2, 2) or dtype != "int32":
        raise ValueError("the reviewed block-table generator requires int32 [2,2]")
    shift = int(seed) % BLOCKS
    values = [[shift, (shift + 2) % BLOCKS],
              [(shift + 3) % BLOCKS, (shift + 5) % BLOCKS]]
    return torch.tensor(values, dtype=torch.int32, device="cuda")


@triton.jit
def _paged_attention(q_ptr, k_ptr, v_ptr, table_ptr, out_ptr,
                     HQ: tl.constexpr, HKV: tl.constexpr, HEAD_DIM: tl.constexpr,
                     BLOCK_SIZE: tl.constexpr, SCALE: tl.constexpr, LEFT_WINDOW: tl.constexpr):
    token = tl.program_id(0)
    head = tl.program_id(1)
    sequence = tl.where(token < 3, 0, 1)
    local_query = tl.where(token < 3, token, token - 3)
    query_len = tl.where(sequence == 0, 3, 2)
    seq_len = tl.where(sequence == 0, 13, 10)
    position = seq_len - query_len + local_query
    kv_head = head // (HQ // HKV)

    feature = tl.arange(0, HEAD_DIM)
    source = tl.arange(0, 16)
    query = tl.load(q_ptr + (token * HQ + head) * HEAD_DIM + feature)
    logical_block = source // BLOCK_SIZE
    physical_block = tl.load(table_ptr + sequence * 2 + logical_block, mask=source < seq_len, other=0)
    slot = source % BLOCK_SIZE
    cache_offset = ((((physical_block[:, None] * BLOCK_SIZE + slot[:, None]) * HKV + kv_head) * HEAD_DIM)
                    + feature[None, :])
    keys = tl.load(k_ptr + cache_offset, mask=(source < seq_len)[:, None], other=0.0)
    scores = tl.sum(keys * query[None, :], axis=1) * SCALE
    allowed = (source < seq_len) & (source <= position) & (position - source <= LEFT_WINDOW)
    scores = tl.where(allowed, scores, -float("inf"))
    maximum = tl.max(scores, axis=0)
    weights = tl.exp(scores - maximum)
    weights = weights / tl.sum(weights, axis=0)
    values = tl.load(v_ptr + cache_offset, mask=(source < seq_len)[:, None], other=0.0)
    result = tl.sum(weights[:, None] * values, axis=0)
    tl.store(out_ptr + (token * HQ + head) * HEAD_DIM + feature, result)


def call(inputs):
    required = {"query", "key_cache", "value_cache", "block_table"}
    if set(inputs) != required:
        raise ValueError(f"expected {sorted(required)}, received {sorted(inputs)}")
    expected = {
        "query": ((TOKENS, HQ, HEAD_DIM), torch.float32),
        "key_cache": ((BLOCKS, BLOCK_SIZE, HKV, HEAD_DIM), torch.float32),
        "value_cache": ((BLOCKS, BLOCK_SIZE, HKV, HEAD_DIM), torch.float32),
        "block_table": ((2, 2), torch.int32),
    }
    for name, value in inputs.items():
        shape, dtype = expected[name]
        if (not torch.is_tensor(value) or tuple(value.shape) != shape or value.dtype != dtype
                or not value.is_cuda or not value.is_contiguous()):
            raise ValueError(f"{name} must be a contiguous CUDA {dtype} tensor of shape {shape}")
        if name != "block_table" and (not bool(torch.isfinite(value).all()) or bool((value.abs() > 4).any())):
            raise ValueError(f"{name} must contain finite values in [-4,4]")
    table = inputs["block_table"]
    if bool(((table < 0) | (table >= BLOCKS)).any()):
        raise ValueError("block_table entries must name physical cache blocks")
    out = torch.empty_like(inputs["query"])
    _paged_attention[(TOKENS, HQ)](
        inputs["query"], inputs["key_cache"], inputs["value_cache"], table, out,
        HQ=HQ, HKV=HKV, HEAD_DIM=HEAD_DIM, BLOCK_SIZE=BLOCK_SIZE,
        SCALE=SCALE, LEFT_WINDOW=LEFT_WINDOW, num_warps=4,
    )
    return {"out": out}
