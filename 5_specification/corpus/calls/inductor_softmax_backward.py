"""PyTorch 2.10 Inductor saved-output softmax backward over the final dimension."""
from __future__ import annotations

import torch


ROWS, COLS, DIM = 16, 128, -1
INPUT_DTYPE = "f32"


def make_saved_output(seed, shape, dtype):
    if tuple(shape) != (ROWS, COLS) or dtype != "float32":
        raise ValueError("saved-output generator requires shape [16,128] and dtype float32")
    generator = torch.Generator(device="cpu").manual_seed(20261019 + int(seed))
    logits = torch.randn(ROWS, COLS, generator=generator, dtype=torch.float32)
    return torch.softmax(logits, dim=DIM).cuda()


def _backward(upstream, saved_output):
    return torch.ops.aten._softmax_backward_data.default(
        upstream, saved_output, DIM, torch.float32
    )


_compiled = torch.compile(_backward, fullgraph=True, backend="inductor")


def call(inputs):
    required = {"upstream", "saved_output", "dim", "input_dtype"}
    if set(inputs) != required:
        raise ValueError(f"expected {sorted(required)}, received {sorted(inputs)}")
    if inputs["dim"] != DIM or inputs["input_dtype"] != INPUT_DTYPE:
        raise ValueError("this call fixes dim=-1 and input_dtype=f32")
    shape = (ROWS, COLS)
    for name in ("upstream", "saved_output"):
        tensor = inputs[name]
        if (not torch.is_tensor(tensor) or tuple(tensor.shape) != shape or tensor.dtype != torch.float32
                or not tensor.is_cuda or not tensor.is_contiguous()):
            raise ValueError(f"{name} must be a contiguous CUDA float32 tensor of shape {shape}")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must contain finite values")
    saved_output = inputs["saved_output"]
    if bool((saved_output <= 0).any()):
        raise ValueError("saved_output must contain positive softmax probabilities")
    if not bool(torch.allclose(saved_output.sum(dim=DIM), torch.ones(ROWS, device=saved_output.device),
                               rtol=0.0, atol=2e-6)):
        raise ValueError("each saved_output row must sum to one within float32 accumulation error")
    return {"dx": _compiled(inputs["upstream"], saved_output)}
