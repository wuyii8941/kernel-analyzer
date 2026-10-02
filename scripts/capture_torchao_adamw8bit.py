#!/usr/bin/env python3
"""Capture torchao AdamW8bit optimizer-step kernels (TorchInductor Triton) as packages.

The recorder is active before the first step so that Inductor compiles and
launches through Triton's launcher.  The first step initializes the 8-bit
moments; later steps (non-zero moments) are captured with whole-storage
operand copies and compiled artifacts.  Runs in an environment with torchao.
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
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--cols", type=int, default=256)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    from torchao.optim import AdamW8bit
    import torchao

    torch.manual_seed(20261002)
    p = torch.nn.Parameter(torch.randn(args.rows, args.cols, device="cuda") * 0.02)
    opt = AdamW8bit([p], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    grads = [torch.randn_like(p) * 1e-3 for _ in range(args.steps)]
    recorder = TritonLaunchRecorder()
    with recorder:
        for step, g in enumerate(grads):
            recorder_count = len(recorder.launches)
            p.grad = g
            opt.step()
            torch.cuda.synchronize()
            for launch in recorder.launches[recorder_count:]:
                launch.step = step
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for j, launch in enumerate(recorder.launches):
        replay = check_replay(launch)
        save_launch(launch, args.out / f"launch{j:03d}")
        rows.append({"launch": j, "step": launch.step, "kernel": launch.kernel_name, "grid": list(launch.grid),
                     "bitwise_replay": replay["bitwise_reproduced"], "replays_identical": replay["replays_identical"]})
    (args.out / "manifest.json").write_text(json.dumps({
        "torchao": torchao.__version__, "shape": [args.rows, args.cols], "steps": args.steps,
        "optimizer": "torchao.optim.AdamW8bit(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)",
        "launches": rows}, indent=2) + "\n")
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
