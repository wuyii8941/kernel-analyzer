"""Run the lightweight public bias checker on unfamiliar Triton callables.

This is an executable smoke/example runner, not a claim that the kernels are
representative of all Triton programs.  Every result is conditional on the
declared input generator and compares a real Triton callable with a PyTorch
reference.  No source inspection or hidden reference is used.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", help="Torch device used by the examples")
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/property/bias_checker_examples_v1/triton_kernel_bias_checks_20260914.json"),
    )
    return parser.parse_args()


def _build_kernels() -> dict[str, tuple[Callable[..., Any], Callable[..., Any], Callable[[int], Any], str]]:
    import torch
    import triton
    import triton.language as tl

    # Triton recompiles the function from its source and resolves annotation
    # names in the defining module globals.  The kernels live in this helper
    # so expose the language namespace explicitly for that recompilation.
    globals()["tl"] = tl

    @triton.jit
    def add_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
        b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, a + b, mask=mask)

    @triton.jit
    def low_precision_row_sum_kernel(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        values = tl.load(x_ptr + row * n_cols + offsets, mask=offsets < n_cols, other=0.0)
        # This is a deliberately explicit implementation variant: the input
        # is rounded to fp16 before the reduction, while the reference keeps
        # the input in fp32.  It is useful as a positive control for bias
        # detection, not as an automatically discovered root cause.
        values = values.to(tl.float16)
        total = tl.sum(values, axis=0)
        tl.store(out_ptr + row, total.to(tl.float32))

    @triton.jit
    def ordered_fp32_row_sum_kernel(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        values = tl.load(x_ptr + row * n_cols + offsets, mask=offsets < n_cols, other=0.0)
        # Keep every operand and accumulator in fp32, but force a two-chunk
        # reassociation.  The reference below uses torch's native reduction.
        midpoint = BLOCK // 2
        left = tl.where(offsets < midpoint, values, 0.0)
        right = tl.where(offsets >= midpoint, values, 0.0)
        total = tl.sum(left, axis=0) + tl.sum(right, axis=0)
        tl.store(out_ptr + row, total)

    @triton.jit
    def softmax_kernel(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_cols
        values = tl.load(x_ptr + row * n_cols + offsets, mask=mask, other=-float("inf"))
        row_max = tl.max(values, axis=0)
        exps = tl.exp(values - row_max)
        denom = tl.sum(exps, axis=0)
        result = exps / denom
        tl.store(out_ptr + row * n_cols + offsets, result, mask=mask)

    @triton.jit
    def rms_normalized_kernel(x_ptr, out_ptr, n_cols, EPS: tl.constexpr,
                              BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_cols
        values = tl.load(x_ptr + row * n_cols + offsets, mask=mask, other=0.0).to(tl.float32)
        mean_square = tl.sum(values * values, axis=0) / n_cols
        result = values * tl.rsqrt(mean_square + EPS)
        tl.store(out_ptr + row * n_cols + offsets, result, mask=mask)

    def add_candidate(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(a)
        n = a.numel()
        add_kernel[(triton.cdiv(n, 128),)](a, b, out, n, BLOCK=128, num_warps=4)
        return out

    def add_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b

    def add_inputs(index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Keep the generator deterministic without relying on global RNG
        # state, so the pair is exactly matched by check_bias.
        positions = torch.arange(257, device=device, dtype=torch.float32)
        shift = float(index) * 0.001
        return torch.sin(positions * 0.013 + shift), torch.cos(positions * 0.017 - shift)

    def row_sum_candidate(x: torch.Tensor) -> torch.Tensor:
        rows, cols = x.shape
        out = torch.empty((rows,), device=x.device, dtype=torch.float32)
        low_precision_row_sum_kernel[(rows,)](x, out, cols, BLOCK=128, num_warps=4)
        return out

    def row_sum_reference(x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=-1)

    def ordered_row_sum_candidate(x: torch.Tensor) -> torch.Tensor:
        rows, cols = x.shape
        out = torch.empty((rows,), device=x.device, dtype=torch.float32)
        ordered_fp32_row_sum_kernel[(rows,)](x, out, cols, BLOCK=128, num_warps=4)
        return out

    def row_sum_inputs(index: int) -> tuple[torch.Tensor]:
        rows, cols = 7, 113
        positions = torch.arange(cols, device=device, dtype=torch.float32)
        # Positive, nearly equal terms make the fp16 accumulation choice a
        # reproducible signed scaling control rather than random cancellation.
        row_offsets = torch.arange(rows, device=device, dtype=torch.float32)[:, None]
        x = 1.0 + 0.0007 * ((positions[None, :] + row_offsets + float(index)) % 17.0)
        return (x,)

    def softmax_candidate(x: torch.Tensor) -> torch.Tensor:
        rows, cols = x.shape
        out = torch.empty_like(x)
        softmax_kernel[(rows,)](x, out, cols, BLOCK=128, num_warps=4)
        return out

    def softmax_reference(x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=-1)

    def softmax_inputs(index: int) -> tuple[torch.Tensor]:
        rows, cols = 9, 97
        positions = torch.arange(cols, device=device, dtype=torch.float32)
        row_offsets = torch.arange(rows, device=device, dtype=torch.float32)[:, None]
        x = 0.4 * torch.sin(positions[None, :] * 0.17 + row_offsets * 0.11 + float(index) * 0.03)
        return (x,)

    def rms_candidate(x: torch.Tensor) -> torch.Tensor:
        rows, cols = x.shape
        out = torch.empty_like(x)
        rms_normalized_kernel[(rows,)](x, out, cols, EPS=1e-6, BLOCK=128, num_warps=4)
        return out

    def rms_native_reference(x: torch.Tensor) -> torch.Tensor:
        values = x.float()
        mean_square = values.square().sum(dim=-1, keepdim=True) / float(values.shape[-1])
        return (values * torch.rsqrt(mean_square + 1e-6)).to(x.dtype)

    def rms_reverse_reference(x: torch.Tensor) -> torch.Tensor:
        values = x.float()
        mean_square = values.square().flip(-1).sum(dim=-1, keepdim=True) / float(values.shape[-1])
        return (values * torch.rsqrt(mean_square + 1e-6)).to(x.dtype)

    def rms_inputs(index: int) -> tuple[torch.Tensor]:
        rows, cols = 11, 127
        positions = torch.arange(cols, device=device, dtype=torch.float32)
        row_offsets = torch.arange(rows, device=device, dtype=torch.float32)[:, None]
        x = (0.35 * torch.sin(positions[None, :] * 0.13 + row_offsets * 0.19 + float(index) * 0.07))
        x = x + 0.01 * ((positions[None, :] + row_offsets + index) % 11.0)
        return (x.to(torch.bfloat16),)

    return {
        "triton_elementwise_add": (
            add_candidate,
            add_reference,
            add_inputs,
            "same-fp32 elementwise Triton add versus torch add; expected no systematic bias",
        ),
        "triton_low_precision_row_sum": (
            row_sum_candidate,
            row_sum_reference,
            row_sum_inputs,
            "Triton row reduction with explicit fp16 input rounding versus fp32 torch sum; positive control",
        ),
        "triton_ordered_fp32_row_sum": (
            ordered_row_sum_candidate,
            row_sum_reference,
            row_sum_inputs,
            "same-fp32 Triton sequential row reduction versus torch reduction; order-only control",
        ),
        "triton_softmax": (
            softmax_candidate,
            softmax_reference,
            softmax_inputs,
            "stable Triton row softmax versus torch.softmax; tests a distinct unfamiliar family",
        ),
        "triton_rms_normalized_native_reference": (
            rms_candidate,
            rms_native_reference,
            rms_inputs,
            "Triton RMS normalization versus native FP32 torch reduction",
        ),
        "triton_rms_normalized_reverse_reference": (
            rms_candidate,
            rms_reverse_reference,
            rms_inputs,
            "same Triton RMS normalization versus FP32 reverse-feature reference; source intervention",
        ),
    }


def main() -> None:
    args = _parse_args()
    if args.samples < 4:
        raise SystemExit("--samples must be >= 4 for calibration/confirmation")
    os.environ.setdefault("TRITON_CACHE_DIR", "/data1/tzh/cache/triton_bias_checker")
    import torch

    global device
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the Triton example runner")

    from kernel_analyzer import check_bias

    results: dict[str, Any] = {
        "schema": "kernel-analyzer-triton-bias-check-examples-v1",
        "device": str(device),
        "samples": args.samples,
        "calibration_samples": args.samples // 2,
        "cases": {},
        "scope": "DECLARED_INPUT_GENERATORS_ONLY",
        "note": (
            "These are callable-level checks. The references and input generators are explicitly supplied; "
            "the runner does not infer Triton semantics or a training consequence."
        ),
    }
    for name, (candidate, reference, make_inputs, description) in _build_kernels().items():
        report = check_bias(
            candidate,
            reference,
            make_inputs,
            samples=args.samples,
            calibration_samples=args.samples // 2,
            check_backward=False,
        )
        results["cases"][name] = {"description": description, "report": report}
        print(
            f"{name}: status={report['status']} "
            f"measurement={report['measurement_status']} "
            f"total_rms={report['stages']['OUTPUT'].get('total_rms')}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
