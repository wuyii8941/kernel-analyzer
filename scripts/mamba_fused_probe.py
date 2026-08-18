#!/usr/bin/env python3
"""Probe the official Mamba1 selective-scan CUDA F+B inside its container."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(7)
    device = "cuda"
    batch, width, length, state = 1, 64, 32, 16
    u = torch.randn(batch, width, length, device=device, dtype=torch.bfloat16, requires_grad=True)
    delta = torch.randn(batch, width, length, device=device, dtype=torch.bfloat16, requires_grad=True)
    A = (-torch.rand(width, state, device=device, dtype=torch.float32)).requires_grad_()
    B = torch.randn(batch, state, length, device=device, dtype=torch.bfloat16, requires_grad=True)
    C = torch.randn(batch, state, length, device=device, dtype=torch.bfloat16, requires_grad=True)
    D = torch.randn(width, device=device, dtype=torch.float32, requires_grad=True)
    output = selective_scan_fn(u, delta, A, B, C, D, delta_softplus=True)
    loss = output.float().square().mean()
    loss.backward()
    result = {
        "status": "COMPLETE",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "loss": float(loss.detach()),
        "all_finite": bool(torch.isfinite(output).all()) and all(
            bool(torch.isfinite(value.grad).all()) for value in (u, delta, A, B, C, D)
        ),
        "gradient_norms": {
            name: float(value.grad.float().norm())
            for name, value in (("u", u), ("delta", delta), ("A", A), ("B", B), ("C", C), ("D", D))
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
