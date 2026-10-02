#!/usr/bin/env python3
"""Try one-intermediate-at-a-time GELU capture without changing the target ABI."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data1/tzh/cache/torchinductor/natural_gelu_component_probe")
os.environ.setdefault("TRITON_CACHE_DIR", "/data1/tzh/cache/triton/natural_gelu_component_probe")
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
OUT = ROOT / "results/property/root_cause_closure_v1/gelu_componentwise_intermediates_v1.json"


@triton.jit
def _one_stage(
    in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, saved_stage, saved_derivative,
    stage: tl.constexpr, xnumel: tl.constexpr, XBLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    mask = xindex < xnumel
    grad = tl.load(in_ptr0 + xindex, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(in_ptr1 + xindex, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(in_out_ptr0 + xindex, mask=mask, other=0.0).to(tl.float32)
    half = tl.full([1], 0.5, tl.float32)
    one = tl.full([1], 1.0, tl.float32)
    gate_sq = gate * gate
    cubic = gate_sq * gate
    cubic = cubic * tl.full([1], 0.044715, tl.float32)
    polynomial = gate + cubic
    scaled = polynomial * tl.full([1], 0.7978845608028654, tl.float32)
    tanh_value = libdevice.tanh(scaled)
    one_plus = tanh_value + one
    forward_factor = (gate * half) * one_plus
    _unused_output = grad * forward_factor
    grad_up = grad * up
    first = one_plus * half
    tanh_sq = tanh_value * tanh_value
    second = (gate * half) * (one - tanh_sq)
    tmp25 = tl.full([1], 0.134145, tl.float32)
    tmp26 = gate_sq * tmp25
    tmp27 = tmp26 + one
    tmp28 = tmp27 * tl.full([1], 0.7978845608028654, tl.float32)
    tmp29 = second * tmp28
    derivative_factor = first + tmp29
    derivative = grad_up * derivative_factor
    selected = tl.where(stage == 0, tanh_value,
                        tl.where(stage == 1, polynomial,
                                 tl.where(stage == 2, derivative_factor, derivative)))
    tl.store(saved_stage + xindex, selected, mask=mask)
    tl.store(saved_derivative + xindex, derivative, mask=mask)


def metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    left = left.reshape(-1)
    right = right.reshape(-1)
    delta = left.double() - right.double()
    norm = float(torch.linalg.vector_norm(left.double()).item())
    return {
        "exact": bool(torch.count_nonzero(delta).item() == 0 and not bool(torch.isnan(delta).any().item())),
        "nonzero_coordinates": int(torch.count_nonzero(delta).item()),
        "relative_l2": float(torch.linalg.vector_norm(delta).item()) / norm if norm else 0.0,
        "max_abs": float(delta.abs().max().item()) if delta.numel() else 0.0,
        "signed_mean": float(delta.mean().item()) if delta.numel() else 0.0,
    }


def run(device: torch.device, count: int) -> dict[str, Any]:
    bank = json.loads(INPUT_BANK.read_text())
    states = bank.get("states", bank.get("records"))
    configure_candidate_runtime(26000)
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
            current["candidate_derivative_post"] = tensors[0].detach().clone()
        return result

    target.run = wrapped
    rows = []
    stage_names = ("tanh", "polynomial", "derivative_factor", "derivative")
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
            gate = current["in_ptr1"].float()
            grad = current["in_ptr0"].float()
            up = current["in_out_ptr0"].float()
            one = torch.ones_like(gate)
            half = torch.full_like(gate, 0.5)
            gate_sq = gate * gate
            cubic = gate_sq * gate * 0.044715
            polynomial = gate + cubic
            scaled = polynomial * 0.7978845608028654
            tanh_value = torch.tanh(scaled)
            one_plus = tanh_value + one
            tanh_sq = tanh_value * tanh_value
            tmp25 = torch.full_like(gate, 0.134145)
            tmp26 = gate_sq * tmp25
            tmp27 = tmp26 + one
            tmp28 = tmp27 * 0.7978845608028654
            tmp29 = ((gate * half) * (one - tanh_sq)) * tmp28
            derivative_factor = half * one_plus + tmp29
            direct = (grad * up) * derivative_factor
            direct_stages = (tanh_value, polynomial, derivative_factor, direct)
            candidate_order_torch_tanh = torch.tanh(scaled)
            candidate_order_one_plus = candidate_order_torch_tanh + one
            candidate_order_first = candidate_order_one_plus * half
            candidate_order_tanh_sq = candidate_order_torch_tanh * candidate_order_torch_tanh
            candidate_order_second = (gate * half) * (one - candidate_order_tanh_sq)
            candidate_order_tmp26 = gate_sq * 0.134145
            candidate_order_tmp27 = candidate_order_tmp26 + one
            candidate_order_tmp28 = candidate_order_tmp27 * 0.7978845608028654
            candidate_order_tmp29 = candidate_order_second * candidate_order_tmp28
            candidate_order_torch = (grad * up) * (candidate_order_first + candidate_order_tmp29)
            candidate_order_exp_tanh = 2.0 / (1.0 + torch.exp(-2.0 * scaled)) - 1.0
            exp_one_plus = candidate_order_exp_tanh + one
            exp_first = exp_one_plus * half
            exp_tanh_sq = candidate_order_exp_tanh * candidate_order_exp_tanh
            exp_second = (gate * half) * (one - exp_tanh_sq)
            exp_tmp29 = exp_second * ((gate_sq * 0.134145 + one) * 0.7978845608028654)
            exp_order_torch = (grad * up) * (exp_first + exp_tmp29)
            stage_rows = {}
            for mode, (name, direct_stage) in enumerate(zip(stage_names, direct_stages)):
                saved_stage = torch.empty((n,), dtype=torch.float32, device=device)
                saved_derivative = torch.empty((n,), dtype=torch.float32, device=device)
                _one_stage[(triton.cdiv(n, 1024),)](
                    current["in_out_ptr0"], current["in_ptr0"], current["in_ptr1"], current["out_ptr0"],
                    saved_stage, saved_derivative, stage=mode, xnumel=n, XBLOCK=1024,
                )
                torch.cuda.synchronize(device)
                candidate_derivative = current["candidate_derivative_post"].float().reshape(-1)
                stage_rows[name] = {
                    "intermediate_vs_torch": metrics(saved_stage, direct_stage.reshape(-1)),
                    "derivative_alignment": metrics(
                        saved_derivative.to(current["candidate_derivative_post"].dtype).float(),
                        candidate_derivative,
                    ),
                }
            stage_rows["reference_variant_alignment"] = {
                "candidate_order_torch_tanh": metrics(candidate_order_torch, candidate_derivative),
                "candidate_order_exp_tanh": metrics(exp_order_torch, candidate_derivative),
            }
            rows.append({"state_index": index, "state_id": state.get("state_id"), "stages": stage_rows})
    finally:
        target.run = original_run
    derivative_ok = {
        name: all(row["stages"][name]["derivative_alignment"]["exact"] for row in rows)
        for name in stage_names
    }
    return {
        "schema": "kernel-analyzer-gelu-componentwise-intermediates-v1",
        "status": "COMPLETE_COMPONENTWISE_PROBE" if any(derivative_ok.values()) else "REJECTED_ALL_COMPONENT_ALIGNMENT",
        "kernel_symbol": next((name for module in modules for name in vars(module) if SYMBOL_FRAGMENT in str(name)), None),
        "state_count": len(rows),
        "input_policy": "natural Gemma operands captured at the reviewed generated GELU boundary",
        "rows": rows,
        "derivative_alignment_by_component": derivative_ok,
        "claim_boundary": "A component source is interpretable only for modes preserving exact candidate derivative writes; no population claim.",
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    payload = run(torch.device("cuda:0"), 4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output": str(OUT), "status": payload["status"], "derivative_alignment_by_component": payload["derivative_alignment_by_component"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
