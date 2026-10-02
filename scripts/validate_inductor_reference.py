#!/usr/bin/env python3
"""Automatic reference on TorchInductor-generated Triton kernels (generalization check).

Each workload runs under the recorder with operand copies; the first launch
of every distinct kernel is evaluated in both reference modes.

    python scripts/validate_inductor_reference.py --out results/reference_eval/validation/inductor_kernels.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder, check_replay  # noqa: E402
from kernel_analyzer.reference_eval.ttir_eval import KernelReferenceEvaluator, NumericMode  # noqa: E402
from kernel_analyzer.reference_eval.ttir_mapping import kernel_coverage  # noqa: E402
from kernel_analyzer.reference_eval.ttir_parser import parse_ttir  # noqa: E402


def workloads():
    dev = "cuda"

    def chunk_sum():
        f = torch.compile(lambda *c: ((((((c[0] + c[1]) + c[2]) + c[3]) + c[4]) + c[5]) + c[6]) + c[7])
        return lambda: f(*(torch.randn(4096, 64, device=dev) for _ in range(8)))

    def cross_entropy():
        f = torch.compile(lambda logits, target: torch.nn.functional.cross_entropy(logits, target))

        def run():
            logits = torch.randn(64, 1000, device=dev, requires_grad=True)
            f(logits, torch.randint(0, 1000, (64,), device=dev)).backward()
        return run

    def rmsnorm_bf16():
        f = torch.compile(lambda x, w: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w)
        return lambda: f(torch.randn(64, 512, device=dev, dtype=torch.bfloat16),
                         torch.randn(512, device=dev, dtype=torch.bfloat16))

    def adamw():
        p = torch.nn.Parameter(torch.randn(8192, device=dev))
        opt = torch.optim.AdamW([p], lr=1e-3, foreach=False)
        step = torch.compile(lambda: opt.step())

        def run():
            for _ in range(2):
                p.grad = torch.randn_like(p)
                step()
        return run

    return {"chunk_sum": chunk_sum, "cross_entropy": cross_entropy, "rmsnorm_bf16": rmsnorm_bf16, "adamw": adamw}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(20261002)
    report = {}
    for name, make in workloads().items():
        torch._dynamo.reset()
        recorder = TritonLaunchRecorder()
        with recorder:
            run = make()
            run()
            torch.cuda.synchronize()
        seen = set()
        for launch in recorder.launches:
            key = (launch.kernel_name, launch.asm["ttir"])
            if key in seen:
                continue
            seen.add(key)
            module = parse_ttir(launch.asm["ttir"])
            coverage = kernel_coverage(module)
            entry = {"workload": name, "kernel": launch.kernel_name, "grid": list(launch.grid),
                     "operations": coverage["operations"], "coverage_complete": coverage["complete"],
                     "bitwise_replay": check_replay(launch)["bitwise_reproduced"]}
            for mode in (NumericMode.NUMERICAL_DIFFERENCE, NumericMode.ROUNDING_CHECK):
                t0 = time.time()
                result = KernelReferenceEvaluator(module, mode=mode).evaluate(launch)
                entry[mode] = {"seconds": round(time.time() - t0, 3),
                               "aborted": sorted(set(result.aborted.values()))[:2],
                               "outputs": {k: {kk: v[kk] for kk in (
                                   "classes", "residual_positive", "residual_negative", "residual_contains_zero",
                                   "integer_mismatches", "max_reference_width", "max_abs_residual") if kk in v}
                                   for k, v in result.compare().items()}}
            report[f"{name}:{launch.kernel_name}:{len(seen)}"] = entry
            print(json.dumps({"kernel": launch.kernel_name[:60], "coverage": coverage["complete"],
                              "replay": entry["bitwise_replay"],
                              "nd": entry[NumericMode.NUMERICAL_DIFFERENCE]["outputs"],
                              "rc_aborted": entry[NumericMode.ROUNDING_CHECK]["aborted"]})[:600])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
