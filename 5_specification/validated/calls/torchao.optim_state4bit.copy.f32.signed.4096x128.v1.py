"""torchao 0.16 OptimState4bit copy compiled by PyTorch 2.10 into three Triton launches."""
from __future__ import annotations

import importlib.metadata
import torch
from torchao.optim.subclass_4bit import OptimState4bit


if importlib.metadata.version("torchao") != "0.16.0" or torch.__version__ != "2.10.0+cu128" \
        or importlib.metadata.version("triton") != "3.6.0":
    raise RuntimeError("this retained call requires torchao 0.16.0, PyTorch 2.10.0+cu128 and Triton 3.6.0")


SIZE, BLOCK_SIZE = 4096, 128
SIGNED_QMAP = [float.fromhex(value) for value in (
    "-0x1.c666660000000p-1", "-0x1.5333340000000p-1", "-0x1.c000000000000p-2", "-0x1.b333320000000p-3",
    "-0x1.3d70a40000000p-4", "-0x1.0a3d720000000p-5", "-0x1.6872b00000000p-8", "0x0.0p+0",
    "0x1.6872b00000000p-8", "0x1.0a3d720000000p-5", "0x1.3d70a40000000p-4", "0x1.b333320000000p-3",
    "0x1.c000000000000p-2", "0x1.5333340000000p-1", "0x1.c666660000000p-1", "0x1.0000000000000p+0",
)]


def make_signed_qmap(seed, shape, dtype):
    if tuple(shape) != (16,) or dtype != "float32":
        raise ValueError("the reviewed qmap generator requires float32[16]")
    return torch.tensor(SIGNED_QMAP, dtype=torch.float32, device="cuda")


@torch.compile(fullgraph=True, dynamic=False)
def _copy_state(dst, src):
    dst.copy_(src)
    return dst


def call(inputs):
    if set(inputs) != {"x", "qmap"}:
        raise ValueError("expected exactly x and qmap")
    x, qmap = inputs["x"], inputs["qmap"]
    if (not torch.is_tensor(x) or tuple(x.shape) != (SIZE,) or x.dtype != torch.float32
            or not x.is_cuda or not x.is_contiguous()):
        raise ValueError("x must be a contiguous CUDA float32 tensor of shape [4096]")
    if not bool(torch.isfinite(x).all()) or bool((x.abs() > 16).any()):
        raise ValueError("x must contain finite values in [-16,16]")
    tiny = torch.finfo(torch.float32).tiny
    if bool(((x != 0) & (x.abs() < tiny)).any()):
        raise ValueError("float32 subnormal source values are outside the reviewed profile")
    if (not torch.is_tensor(qmap) or tuple(qmap.shape) != (16,) or qmap.dtype != torch.float32
            or not qmap.is_cuda or not qmap.is_contiguous()
            or qmap.cpu().tolist() != SIGNED_QMAP):
        raise ValueError("qmap must be the canonical contiguous CUDA signed float32 table")

    codes = torch.zeros(SIZE // 2, dtype=torch.uint8, device=x.device)
    scale = torch.zeros(SIZE // BLOCK_SIZE, dtype=torch.float32, device=x.device)
    state = OptimState4bit(codes, scale, qmap, signed=True, shape=x.shape)
    _copy_state(state, x)
    return {"codes": state.codes, "scale": state.scale}
