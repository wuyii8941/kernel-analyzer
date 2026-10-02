#!/usr/bin/env python3
"""Capture Liger Triton kernels with whole-storage operands (liger env).

Workloads: the fused linear cross-entropy loss (which launches
``liger_cross_entropy_kernel`` once per chunk, writing dlogits in place) and
RMSNorm forward/backward.  Packages are written for analysis in ka_main.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder, check_replay, save_launch  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    import importlib.metadata as md

    torch.manual_seed(20261002)
    dev = "cuda"
    V, H, BT = 32000, 256, 64
    lin = torch.nn.Linear(H, V, bias=False, device=dev)
    h = torch.randn(BT, H, device=dev, requires_grad=True)
    target = torch.randint(0, V, (BT,), device=dev)
    target[5] = -100  # one ignored position exercises the early-exit branch
    norm = LigerRMSNorm(H).to(dev)
    with torch.no_grad():
        norm.weight.copy_(1.0 + 0.1 * torch.randn(H, device=dev))
    x = torch.randn(32, H, device=dev, requires_grad=True)
    recorder = TritonLaunchRecorder()
    with recorder:
        LigerFusedLinearCrossEntropyLoss()(lin.weight, h, target).backward()
        norm(x).pow(2).sum().backward()
        torch.cuda.synchronize()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for j, launch in enumerate(recorder.launches):
        replay = check_replay(launch)
        save_launch(launch, args.out / f"launch{j:03d}")
        rows.append({"launch": j, "kernel": launch.kernel_name, "grid": list(launch.grid),
                     "bitwise_replay": replay["bitwise_reproduced"]})
    torch.save({"weight": lin.weight.detach().cpu(), "hidden": h.detach().cpu(), "target": target.cpu(),
                "norm_weight": norm.weight.detach().cpu(), "norm_x": x.detach().cpu()}, args.out / "inputs.pt")
    (args.out / "manifest.json").write_text(json.dumps({"liger_kernel": md.version("liger-kernel"),
                                                        "V": V, "H": H, "BT": BT, "launches": rows}, indent=2) + "\n")
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
