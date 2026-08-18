#!/usr/bin/env python3
"""GPU negative control for the ABI-correct typed Triton reference path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch._inductor.async_compile import AsyncCompile

from scripts.typed_triton_reference import (
    embedded_triton_programs,
    fp32_pointer_program,
)


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = (
    ROOT / "results/coverage/runtime_releases/qwen_seq64_r1/trace"
    / "model__0_forward_segment0_executed/output_code.py"
)
SYMBOL = "triton_poi_fused_clone_transpose_view_8"
OUTPUT = ROOT / "results/coverage/typed_triton_reference_pilot.json"


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the typed Triton pilot")
    programs = embedded_triton_programs(WRAPPER)
    source = programs[SYMBOL]
    typed_source, metadata = fp32_pointer_program(source, SYMBOL)
    compiler = AsyncCompile()
    kernels = {
        "candidate": compiler.triton(SYMBOL, source, device_str="cuda"),
        "reference": compiler.triton(SYMBOL, typed_source, device_str="cuda"),
    }
    compiler.wait(kernels)
    device = torch.device("cuda:0")
    elements = 131072
    generator = torch.Generator(device=device).manual_seed(39173)
    candidate_input = torch.randn(
        elements, device=device, dtype=torch.bfloat16, generator=generator,
    )
    candidate_output = torch.empty_like(candidate_input)
    reference_input = candidate_input.float()
    reference_output = torch.empty(elements, device=device, dtype=torch.float32)
    stream = torch._C._cuda_getCurrentRawStream(device.index)
    kernels["candidate"].run(
        candidate_input, candidate_output, elements, stream=stream,
    )
    kernels["reference"].run(
        reference_input, reference_output, elements, stream=stream,
    )
    torch.cuda.synchronize(device)
    shaped = candidate_input.reshape(1, 16, 64, 128)
    analytic = shaped.permute(0, 2, 1, 3).contiguous().reshape(-1)
    candidate_error = candidate_output.float() - analytic.float()
    typed_error = reference_output - analytic.float()
    cross_error = candidate_output.float() - reference_output
    payload = {
        "schema": "kernel-analyzer-typed-triton-reference-pilot-v1",
        "status": "PASS_TYPED_ABI_PURE_REORDER_NEGATIVE_CONTROL",
        "frozen_wrapper": str(WRAPPER.relative_to(ROOT)),
        "symbol": SYMBOL,
        "program": metadata,
        "runtime": {
            "torch_version": torch.__version__,
            "cuda_version": str(torch.version.cuda),
            "device": torch.cuda.get_device_name(device),
        },
        "denominator": {"coordinates": elements, "runs": 2},
        "checks": {
            "candidate_matches_analytic": bool(torch.count_nonzero(candidate_error) == 0),
            "typed_reference_matches_analytic": bool(torch.count_nonzero(typed_error) == 0),
            "candidate_matches_typed_reference": bool(torch.count_nonzero(cross_error) == 0),
            "candidate_max_abs": float(candidate_error.abs().max()),
            "typed_reference_max_abs": float(typed_error.abs().max()),
            "cross_max_abs": float(cross_error.abs().max()),
            "candidate_pointer_abi": dict(kernels["candidate"].triton_meta["signature"]),
            "reference_pointer_abi": dict(kernels["reference"].triton_meta["signature"]),
        },
        "claim_boundary": (
            "The exact frozen pure-reorder program was independently compiled with physical "
            "BF16 and FP32 pointer ABIs. Both agree with the analytic permutation on every "
            "coordinate. This validates the typed-reference mechanism, not all kernels."
        ),
    }
    if not all(payload["checks"][key] for key in (
        "candidate_matches_analytic", "typed_reference_matches_analytic",
        "candidate_matches_typed_reference",
    )):
        payload["status"] = "FAIL_TYPED_ABI_NEGATIVE_CONTROL"
    payload["result_sha256"] = digest(payload)
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(OUTPUT)
    print(json.dumps({"output": str(OUTPUT.relative_to(ROOT)),
                      "status": payload["status"]}, sort_keys=True))
    if not payload["status"].startswith("PASS"):
        raise RuntimeError(payload["status"])


if __name__ == "__main__":
    main()
