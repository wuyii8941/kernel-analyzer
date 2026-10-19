"""Original FlagGems 5.0.2 experimental MSE forward, with explicit task parameters."""
from __future__ import annotations

import importlib.metadata
import importlib.util
from pathlib import Path
import sys

import torch


_distribution = importlib.metadata.distribution("flag_gems")
if _distribution.version != "5.0.2" or torch.__version__ != "2.10.0+cu128":
    raise RuntimeError("this call requires FlagGems 5.0.2 and PyTorch 2.10.0+cu128")
_source = Path(_distribution.locate_file("flag_gems/experimental_ops/mse_loss.py"))
_retained = Path(__file__).resolve().parents[2] / "sources/FLAGGEMS502_EXPERIMENTAL_MSE.py"
if _source.read_bytes() != _retained.read_bytes():
    raise RuntimeError("installed MSE source differs from the reviewed source")
_spec = importlib.util.spec_from_file_location("_ka_flaggems502_mse", _source)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)


def call(inputs):
    if set(inputs) != {"x", "target", "reduction"}:
        raise ValueError("expected exactly x, target and reduction")
    reduction = inputs["reduction"]
    if reduction not in ("none", "sum", "mean"):
        raise ValueError("reduction must be none, sum or mean")
    for name in ("x", "target"):
        value = inputs[name]
        if (not torch.is_tensor(value) or value.shape != (7, 41) or value.dtype != torch.float32
                or not value.is_cuda or not value.is_contiguous()):
            raise ValueError(f"{name} must be a contiguous CUDA float32 tensor of shape [7,41]")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")
    return {"loss": _module.mse_loss(inputs["x"], inputs["target"], reduction=reduction)}
