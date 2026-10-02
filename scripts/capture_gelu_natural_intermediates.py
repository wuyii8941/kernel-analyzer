#!/usr/bin/env python3
"""Capture the reviewed Gemma tanh-GELU backward expression on real operands."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data1/tzh/cache/torchinductor/natural_gelu_probe")
os.environ.setdefault("TRITON_CACHE_DIR", "/data1/tzh/cache/triton/natural_gelu_probe")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import triton
import triton.language as tl
from torch._inductor.codecache import PyCodeCache
from torch._inductor.runtime.triton_helpers import libdevice

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_generated_fp32_screen import load_model  # noqa: E402
from scripts.generated_fp32_observer import runtime_signature  # noqa: E402

MODEL = Path("/data1/tzh/models/google/gemma-4-E2B")
INPUT_BANK = ROOT / "results/property/tcmp_allop_v1/input_banks/gemma4_e2b_text128_trajectory32.json"
SYMBOL_FRAGMENT = "triton_poi_fused__unsafe_view_gelu_gelu_backward_mul_view_"
OUT = ROOT / "results/property/root_cause_closure_v1/gelu_natural_intermediates_v1.json"


@triton.jit
def _instrumented(
    in_out_ptr0, in_ptr0, in_ptr1, out_ptr0,
    saved_out, saved_tanh, saved_poly, saved_factor, saved_derivative,
    xnumel: tl.constexpr, XBLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    mask = xindex < xnumel
    grad = tl.load(in_ptr0 + xindex, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(in_ptr1 + xindex, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(in_out_ptr0 + xindex, mask=mask, other=0.0).to(tl.float32)
    half = tl.full([1], 0.5, tl.float32)
    one = tl.full([1], 1.0, tl.float32)
    cubic = gate * gate
    cubic = cubic * gate
    cubic = cubic * tl.full([1], 0.044715, tl.float32)
    poly = gate + cubic
    scaled = poly * tl.full([1], 0.7978845608028654, tl.float32)
    tanh_value = libdevice.tanh(scaled)
    one_plus_tanh = tanh_value + one
    forward_factor = half * gate * one_plus_tanh
    output = grad * forward_factor
    saved_grad_up = grad * up
    first = one_plus_tanh * half
    tanh_sq = tanh_value * tanh_value
    second = half * gate * (one - tanh_sq)
    derivative_factor = first + second * (
        one + (gate * gate) * tl.full([1], 0.134145, tl.float32)
    ) * tl.full([1], 0.7978845608028654, tl.float32)
    derivative = saved_grad_up * derivative_factor
    tl.store(saved_out + xindex, output, mask=mask)
    tl.store(saved_tanh + xindex, tanh_value, mask=mask)
    tl.store(saved_poly + xindex, poly, mask=mask)
    tl.store(saved_factor + xindex, derivative_factor, mask=mask)
    tl.store(saved_derivative + xindex, derivative, mask=mask)


def metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    delta = left.double() - right.double()
    return {
        "exact": bool(torch.count_nonzero(delta).item() == 0 and not bool(torch.isnan(delta).any().item())),
        "nonzero_coordinates": int(torch.count_nonzero(delta).item()),
        "relative_l2": float(torch.linalg.vector_norm(delta).item()) / float(torch.linalg.vector_norm(left.double()).item()) if left.numel() and torch.linalg.vector_norm(left.double()).item() else 0.0,
        "max_abs": float(delta.abs().max().item()) if delta.numel() else 0.0,
        "signed_mean": float(delta.mean().item()) if delta.numel() else 0.0,
    }


def run(device: torch.device, count: int) -> dict[str, Any]:
    bank = json.loads(INPUT_BANK.read_text())
    states = bank.get("states", bank.get("records"))
    configure_candidate_runtime(24000)
    model = load_model("gemma4", MODEL, device)
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=True, dynamic=False)
    warm = torch.tensor([states[0].get("token_ids", states[0].get("input_ids"))], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    candidate(warm).backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    target = None
    for module in modules:
        for name, value in vars(module).items():
            if SYMBOL_FRAGMENT in str(name) and callable(getattr(value, "run", None)):
                target = value
                break
        if target is not None:
            break
    if target is None:
        raise RuntimeError(f"generated GELU kernel not found: {SYMBOL_FRAGMENT}")
    original_run = target.run
    current: dict[str, torch.Tensor] | None = None

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        nonlocal current
        names = [str(name) for name, annotation in runtime_signature(target) if str(annotation).startswith("*")]
        tensors = [value for value in args if isinstance(value, torch.Tensor)]
        if len(names) != len(tensors):
            raise RuntimeError("candidate pointer ABI changed")
        before = {name: value.detach().clone() for name, value in zip(names, tensors)}
        result = original_run(*args, **kwargs)
        if current is not None and not current:
            current.update(before)
            current["candidate_out_post"] = tensors[3].detach().clone()
            current["candidate_derivative_post"] = tensors[0].detach().clone()
        return result

    target.run = wrapped
    rows = []
    try:
        for index, state in enumerate(states[:count]):
            values = torch.tensor([state.get("token_ids", state.get("input_ids"))], dtype=torch.long, device=device)
            current = {}
            model.zero_grad(set_to_none=True)
            candidate(values).backward()
            torch.cuda.synchronize(device)
            required = {"in_out_ptr0", "in_ptr0", "in_ptr1", "out_ptr0"}
            if not required.issubset(current):
                raise RuntimeError(f"natural operands not captured: {sorted(current)}")
            n = int(current["in_ptr0"].numel())
            outputs = [torch.empty_like(current["in_ptr0"], dtype=torch.float32) for _ in range(5)]
            _instrumented[(triton.cdiv(n, 1024),)](
                current["in_out_ptr0"], current["in_ptr0"], current["in_ptr1"], current["out_ptr0"],
                *outputs, xnumel=n, XBLOCK=1024,
            )
            torch.cuda.synchronize(device)
            saved_out, saved_tanh, saved_poly, saved_factor, saved_derivative = outputs
            gate = current["in_ptr1"].float()
            grad = current["in_ptr0"].float()
            up = current["in_out_ptr0"].float()
            half = torch.full_like(gate, 0.5)
            one = torch.ones_like(gate)
            cubic = gate * gate
            cubic = cubic * gate
            cubic = cubic * 0.044715
            poly = gate + cubic
            scaled = poly * 0.7978845608028654
            tanh_value = torch.tanh(scaled)
            one_plus = tanh_value + one
            forward_factor = half * gate * one_plus
            direct_out = grad * forward_factor
            grad_up = grad * up
            first = one_plus * half
            tanh_sq = tanh_value * tanh_value
            second = half * gate * (one - tanh_sq)
            direct_factor = first + second * (one + (gate * gate) * 0.134145) * 0.7978845608028654
            direct_derivative = grad_up * direct_factor
            rows.append({
                "state_index": index,
                "final_output_alignment": metrics(saved_out.to(current["candidate_out_post"].dtype).float(), current["candidate_out_post"].float()),
                "final_derivative_alignment": metrics(saved_derivative.to(current["candidate_derivative_post"].dtype).float(), current["candidate_derivative_post"].float()),
                "intermediates": {
                    "tanh": metrics(saved_tanh, tanh_value),
                    "polynomial": metrics(saved_poly, poly),
                    "derivative_factor": metrics(saved_factor, direct_factor),
                    "derivative": metrics(saved_derivative, direct_derivative),
                },
            })
    finally:
        target.run = original_run
    final_ok = all(row["final_output_alignment"]["exact"] and row["final_derivative_alignment"]["exact"] for row in rows)
    first = {
        name: any(row["intermediates"][name]["nonzero_coordinates"] > 0 for row in rows)
        for name in ("tanh", "polynomial", "derivative_factor", "derivative")
    }
    return {
        "schema": "kernel-analyzer-gelu-natural-intermediates-v1",
        "status": "COMPLETE_NATURAL_INTERMEDIATE_PROBE" if final_ok else "REJECTED_FINAL_ALIGNMENT",
        "kernel_symbol_fragment": SYMBOL_FRAGMENT,
        "state_count": len(rows),
        "input_policy": "natural Gemma model operands captured at the reviewed generated kernel boundary",
        "rows": rows,
        "first_difference_candidates": first,
        "claim_boundary": "Natural source localization for the reviewed endpoint only; no population mean or training-quality claim.",
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    payload = run(torch.device("cuda:0"), 4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output": str(OUT), "status": payload["status"], "first_difference_candidates": payload["first_difference_candidates"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
