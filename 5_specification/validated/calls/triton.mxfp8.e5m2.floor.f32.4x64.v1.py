"""Retained Triton MXFP8 E5M2 ROUND_DOWN conversion on finite float32 input."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
import triton
from triton.language.target_info import cuda_capability_geq


ROWS, COLUMNS, MICROBLOCK = 4, 64, 32
SPECIFICATION = Path(__file__).resolve().parents[2]
SOURCE = SPECIFICATION / "sources" / "MX_LOCAL_TRITON_DOWNCAST.py"

# The retained CUDA kernel imports only this capability predicate from the
# newer triton_kernels package.  Triton 3.6 exposes the same predicate here;
# supplying it directly avoids importing unrelated AMD-only names.
bridge = types.ModuleType("triton_kernels.target_info")
bridge.cuda_capability_geq = cuda_capability_geq
sys.modules[bridge.__name__] = bridge
spec = importlib.util.spec_from_file_location("retained_mxfp8_downcast", SOURCE)
retained = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = retained
spec.loader.exec_module(retained)


def make_safe_input(seed, shape, dtype):
    """Finite blocks whose fixed maximum keeps FLOOR normalization in E5M2 range."""
    if tuple(shape) != (ROWS, COLUMNS) or dtype != "float32":
        raise ValueError("the reviewed generator requires float32 [4,64]")
    generator = torch.Generator(device="cuda")
    generator.manual_seed(int(seed))
    value = torch.empty(shape, dtype=torch.float32, device="cuda")
    value.uniform_(-56, 56, generator=generator)
    value[:, 0] = 64
    value[:, 32] = -64
    return value


def call(inputs):
    if set(inputs) != {"x"}:
        raise ValueError("expected exactly x")
    x = inputs["x"]
    if (not torch.is_tensor(x) or tuple(x.shape) != (ROWS, COLUMNS)
            or x.dtype != torch.float32 or not x.is_cuda or not x.is_contiguous()):
        raise ValueError("x must be a contiguous CUDA float32 tensor of shape [4,64]")
    if not bool(torch.isfinite(x).all()) or bool((x.abs() > 128).any()):
        raise ValueError("x must contain finite values in [-128,128]")

    elements = torch.empty_like(x, dtype=torch.float8_e5m2)
    scale_codes = torch.empty((ROWS, COLUMNS // MICROBLOCK), dtype=torch.uint8, device=x.device)
    retained._downcast_to_mxfp[(1, 1)](
        elements, *elements.stride(),
        scale_codes, *scale_codes.stride(),
        x, *x.stride(), *x.shape,
        32, 128, MICROBLOCK, 1,
        num_warps=4,
    )
    return {"elements": elements, "scale_codes": scale_codes}
