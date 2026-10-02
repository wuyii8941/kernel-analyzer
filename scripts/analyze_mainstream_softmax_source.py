#!/usr/bin/env python3
"""Validate the source of a nonzero mainstream BF16 softmax discrepancy.

The public bias scan compares an Inductor candidate with eager softmax.  This
follow-up keeps the logits fixed and separates three effects: eager BF16
softmax, the same operation with FP32 intermediates, and the compiled route
with/without an autograd-enabled input.  It is deliberately a behavioural
source check.  It does not claim a particular generated instruction or a
training-level consequence.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _metric(left: Any, right: Any) -> dict[str, Any]:
    import torch

    delta = (left.detach().float() - right.detach().float()).double()
    right_norm = float(torch.linalg.vector_norm(right.detach().float()).item())
    return {
        "exact": bool(torch.count_nonzero(delta).item() == 0),
        "nonzero_coordinates": int(torch.count_nonzero(delta).item()),
        "relative_l2": float(torch.linalg.vector_norm(delta).item()) / right_norm
        if right_norm else 0.0,
        "max_abs": float(delta.abs().max().item()) if delta.numel() else 0.0,
        "signed_mean": float(delta.mean().item()) if delta.numel() else 0.0,
    }


def _make_logits(torch: Any, index: int, *, requires_grad: bool) -> Any:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(3500 + index)
    logits = torch.randn(
        (2, 8, 32, 32), generator=generator, device="cuda", dtype=torch.bfloat16
    ) * 4
    logits[..., 0] = logits[..., 0] + 12
    logits.requires_grad_(requires_grad)
    return logits


def _fp32_formula(logits: Any) -> Any:
    import torch

    return torch.softmax(logits.float(), dim=-1).to(torch.bfloat16)


def _run(samples: int, device: Any) -> dict[str, Any]:
    import torch

    class Softmax(torch.nn.Module):
        def forward(self, logits):
            return torch.softmax(logits, dim=-1)

    reference = Softmax().to(device=device).eval()
    candidate = torch.compile(reference, backend="inductor", fullgraph=True, dynamic=False)
    # Compile both the no-grad and autograd-enabled execution paths before
    # recording rows, so compilation itself is not mistaken for an effect.
    candidate(_make_logits(torch, 0, requires_grad=False))
    candidate(_make_logits(torch, 0, requires_grad=True))

    rows = []
    for index in range(samples):
        logits_nograd = _make_logits(torch, index, requires_grad=False)
        logits_grad = _make_logits(torch, index, requires_grad=True)
        with torch.no_grad():
            reference_nograd = reference(logits_nograd)
            candidate_nograd = candidate(logits_nograd)
            fp32_nograd = _fp32_formula(logits_nograd)
        reference_grad = reference(logits_grad)
        candidate_grad = candidate(logits_grad)
        rows.append({
            "sample": index,
            "reference_vs_fp32_formula": _metric(reference_nograd, fp32_nograd),
            "candidate_nograd_vs_reference": _metric(candidate_nograd, reference_nograd),
            "candidate_grad_vs_reference": _metric(candidate_grad, reference_grad),
            "candidate_grad_vs_candidate_nograd": _metric(candidate_grad, candidate_nograd),
            "reference_grad_vs_reference_nograd": _metric(reference_grad, reference_nograd),
            "candidate_row_sum_error": {
                "max_abs": float(
                    (candidate_grad.float().sum(dim=-1) - 1.0).abs().max().item()
                ),
                "reference_max_abs": float(
                    (reference_grad.float().sum(dim=-1) - 1.0).abs().max().item()
                ),
            },
        })
    return {
        "samples": samples,
        "shape": [2, 8, 32, 32],
        "dtype": "bfloat16",
        "rows": rows,
        "all_reference_matches_fp32_formula": all(
            row["reference_vs_fp32_formula"]["exact"] for row in rows
        ),
        "all_candidate_nograd_matches_reference": all(
            row["candidate_nograd_vs_reference"]["nonzero_coordinates"] == 0
            for row in rows
        ),
        "candidate_grad_differs_from_reference_rows": [
            row["sample"] for row in rows
            if row["candidate_grad_vs_reference"]["nonzero_coordinates"] > 0
        ],
        "candidate_grad_differs_from_nograd_rows": [
            row["sample"] for row in rows
            if row["candidate_grad_vs_candidate_nograd"]["nonzero_coordinates"] > 0
        ],
        "reference_grad_equals_reference_nograd": all(
            row["reference_grad_vs_reference_nograd"]["exact"] for row in rows
        ),
        "max_candidate_grad_reference_relative_l2": max(
            row["candidate_grad_vs_reference"]["relative_l2"] for row in rows
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 4:
        raise SystemExit("--samples must be >= 4")
    os.environ.setdefault("TRITON_CACHE_DIR", "/data1/tzh/cache/triton_mainstream_bias_scan")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data1/tzh/cache/torchinductor_mainstream_bias_scan")
    import torch

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("a CUDA device is required")
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not output.is_relative_to(root):
        raise ValueError("output must be inside the repository")
    if output.exists():
        raise FileExistsError(output)
    payload: dict[str, Any] = {
        "schema": "mainstream-softmax-source-analysis-v1",
        "status": "COMPLETE",
        "candidate_route": "torch.compile backend=inductor fullgraph=True",
        "reference_route": "eager torch.softmax",
        "input_policy": "identical generated BF16 logits; output and autograd-enabled paths are separated",
        "interpretation": (
            "If eager matches the FP32-intermediate formula and the no-grad compiled route, "
            "while a sparse difference appears only for the autograd-enabled compiled route, "
            "the discrepancy is localized to that compiled BF16/autograd evaluation path. "
            "The evidence is consistent with a reduction/materialization choice, but is not "
            "an instruction-level proof and does not establish a training consequence."
        ),
    }
    try:
        payload["analysis"] = _run(args.samples, device)
    except Exception as error:
        payload["status"] = "UNRESOLVED_MEASUREMENT"
        payload["error"] = f"{type(error).__name__}: {error}"
        raise
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
