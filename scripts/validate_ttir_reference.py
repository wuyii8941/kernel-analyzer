"""Run exercise kernels on the GPU, capture them, and evaluate the TTIR reference.

For each kernel both reference modes are evaluated and compared with the
captured device output.  In rounding-check mode, kernels made only of
correctly rounded IEEE operations must be reproduced bit for bit; in
numerical-difference mode the residual K - K_R shows where the implementation
differs from the declared real computation.

    python scripts/validate_ttir_reference.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder  # noqa: E402


def workloads():
    from scripts import reference_eval_kernels as k

    g = torch.Generator(device="cuda").manual_seed(0)
    n = 1000
    x = torch.randn(n, device="cuda", generator=g)
    a, b, c, d = (torch.randn(n, device="cuda", generator=g) for _ in range(4))
    yield "scale_masked", lambda: k.scale_masked[(4,)](x, torch.empty_like(x), n, 0.1, BLOCK=256)
    yield "sequential_sum4", lambda: k.sequential_sum4[(4,)](a, b, c, d, torch.empty_like(a), n, BLOCK=256)
    yield "add_then_mul", lambda: k.add_then_mul[(4,)](a, b, c, torch.empty_like(a), n, BLOCK=256)
    rows = torch.randn(8, 100, device="cuda", generator=g)
    yield "softmax_rows", lambda: k.softmax_rows[(8,)](rows, torch.empty_like(rows), 100, 100, BLOCK=128)
    w, bb = torch.randn(100, device="cuda", generator=g), torch.randn(100, device="cuda", generator=g)
    yield "layernorm_rows", lambda: k.layernorm_rows[(8,)](rows, w, bb, torch.empty_like(rows), 100, 1e-5,
                                                           BLOCK=128)
    A = torch.randn(64, 48, device="cuda", generator=g)
    B = torch.randn(48, 32, device="cuda", generator=g)
    for precision in ("tf32", "ieee"):
        C = torch.empty(64, 32, device="cuda")
        yield f"matmul_{precision}", (lambda C=C, precision=precision: k.matmul[(2, 2)](
            A, B, C, 64, 32, 48, *A.stride(), *B.stride(), *C.stride(), BM=32, BN=16, BK=16,
            PRECISION=precision))
    out = torch.zeros(4, device="cuda")
    yield "atomic_unused", lambda: k.atomic_accumulate[(4,)](x, out, torch.empty_like(x), n, BLOCK=256,
                                                             USE_OLD=False)
    yield "atomic_used", lambda: k.atomic_accumulate[(4,)](x, torch.zeros(4, device="cuda"), torch.empty_like(x),
                                                           n, BLOCK=256, USE_OLD=True)
    yield "branch_on_scalar", lambda: k.branch_on_scalar[(4,)](x, torch.empty_like(x), 0.0, n, BLOCK=256)
    yield "conversions", lambda: k.conversions[(4,)](
        x, torch.empty(n, device="cuda", dtype=torch.float16), torch.empty(n, device="cuda", dtype=torch.bfloat16),
        torch.empty(n, device="cuda", dtype=torch.float8_e5m2), torch.empty_like(x), n, BLOCK=256)
    yield "elementary", lambda: k.elementary[(4,)](x, torch.empty_like(x), n, BLOCK=256)
    xn = x.clone()
    xn[::7] = float("nan")
    yield "nan_rules", lambda: k.nan_rules[(4,)](xn, torch.empty_like(x), n, BLOCK=256)
    yield "scan_and_argmax", lambda: k.scan_and_argmax[(1,)](x, torch.empty_like(x),
                                                             torch.empty(1, device="cuda", dtype=torch.int32),
                                                             200, BLOCK=256)
    yield "bit_level", lambda: k.bit_level[(4,)](x, torch.empty_like(x), n, BLOCK=256)
    yield "inline_asm", lambda: k.inline_asm[(4,)](x, torch.empty_like(x), n, BLOCK=256)
    yield "while_loop", lambda: k.while_loop[(4,)](x, torch.empty_like(x), n, BLOCK=256)
    yield "store_load_chain", lambda: k.store_load_chain[(4,)](x, torch.empty_like(x), torch.empty_like(x), n,
                                                               BLOCK=256)
    yield "exp_sum_divide", lambda: k.exp_sum_divide[(1,)](x, torch.empty_like(x), 200, BLOCK=256)


def evaluate_launch(launch, mode):
    from kernel_analyzer.reference_eval.ttir_eval import KernelReferenceEvaluator
    from kernel_analyzer.reference_eval.ttir_mapping import kernel_coverage
    from kernel_analyzer.reference_eval.ttir_parser import parse_ttir

    module = parse_ttir(launch.asm["ttir"])
    coverage = kernel_coverage(module)
    t0 = time.time()
    result = KernelReferenceEvaluator(module, mode=mode).evaluate(launch)
    return {"seconds": round(time.time() - t0, 3), "coverage_complete": coverage["complete"],
            "aborted": {str(k): v for k, v in result.aborted.items()}, "outputs": result.compare(), "notes": result.notes,
            "reasons": sorted(result.reasons)[:8]}


def main():
    from kernel_analyzer.reference_eval.ttir_eval import NumericMode

    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    report = {}
    for name, fn in workloads():
        recorder = TritonLaunchRecorder()
        with recorder:
            fn()
            torch.cuda.synchronize()
        launch = recorder.launches[-1]
        report[name] = {mode: evaluate_launch(launch, mode)
                        for mode in (NumericMode.NUMERICAL_DIFFERENCE, NumericMode.ROUNDING_CHECK)}
    for name, modes in report.items():
        for mode, r in modes.items():
            outs = {k: {kk: v[kk] for kk in ("classes", "residual_positive", "residual_negative",
                                             "residual_contains_zero", "special_mismatches",
                                             "integer_mismatches") if kk in v}
                    for k, v in r["outputs"].items()}
            print(f"{name:18s} {mode[:9]:9s} {r['seconds']:6.2f}s aborted={list(r['aborted'].values())[:1]} {outs}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
