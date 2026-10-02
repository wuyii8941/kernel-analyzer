#!/usr/bin/env python3
"""Capture a reviewed SiLU Triton expression on natural model operands.

The model run supplies the real pointer operands of the frozen generated
kernel.  A separate instrumented kernel evaluates the reviewed expression and
stores its intermediate values.  The final stores are compared with the
captured candidate stores before any source conclusion is emitted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data1/tzh/cache/torchinductor/natural_silu_probe")
os.environ.setdefault("TRITON_CACHE_DIR", "/data1/tzh/cache/triton/natural_silu_probe")
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

SYMBOL = "triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_4"
MODEL = Path("/data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
INPUT_BANK = ROOT / "results/coverage/deepseek8b_seq128_input_bank.json"
OUT = ROOT / "results/property/root_cause_closure_v1/silu_natural_intermediates_v1.json"


@triton.jit
def _instrumented(
    in_out_ptr0, in_ptr0, in_ptr1, out_ptr0,
    saved_out, saved_exp, saved_denom, saved_sigmoid, saved_product,
    saved_factor, saved_derivative,
    xnumel: tl.constexpr, XBLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    mask = xindex < xnumel
    grad = tl.load(in_ptr0 + xindex, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(in_ptr1 + xindex, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(in_out_ptr0 + xindex, mask=mask, other=0.0).to(tl.float32)
    neg_gate = -gate
    exp_value = libdevice.exp(neg_gate)
    one = tl.full([1], 1.0, tl.float32)
    denominator = exp_value + one
    gate_over_denominator = gate / denominator
    sigmoid = one / denominator
    grad_up = grad * up
    product = grad_up * sigmoid
    factor = gate * (one - sigmoid) + one
    derivative = product * factor
    output = grad * gate_over_denominator
    tl.store(saved_out + xindex, output, mask=mask)
    tl.store(saved_exp + xindex, exp_value, mask=mask)
    tl.store(saved_denom + xindex, denominator, mask=mask)
    tl.store(saved_sigmoid + xindex, sigmoid, mask=mask)
    tl.store(saved_product + xindex, product, mask=mask)
    tl.store(saved_factor + xindex, factor, mask=mask)
    tl.store(saved_derivative + xindex, derivative, mask=mask)


def metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    delta = left.double() - right.double()
    left_norm = float(torch.linalg.vector_norm(left.double()).item())
    return {
        "exact": bool(torch.count_nonzero(delta).item() == 0 and not bool(torch.isnan(delta).any().item())),
        "nonzero_coordinates": int(torch.count_nonzero(delta).item()),
        "relative_l2": float(torch.linalg.vector_norm(delta).item()) / left_norm if left_norm else 0.0,
        "max_abs": float(delta.abs().max().item()) if delta.numel() else 0.0,
        "signed_mean": float(delta.mean().item()) if delta.numel() else 0.0,
    }


def capture_states(device: torch.device, count: int) -> list[dict[str, Any]]:
    bank = json.loads(INPUT_BANK.read_text())
    states = bank.get("states", bank.get("records"))
    configure_candidate_runtime(24000)
    model = load_model("deepseek8", MODEL, device)
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=True, dynamic=False)
    warm = torch.tensor([states[0].get("token_ids", states[0].get("input_ids"))], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    candidate(warm).backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    target = None
    for module in modules:
        value = getattr(module, SYMBOL, None)
        if value is not None and callable(getattr(value, "run", None)):
            target = value
            break
    if target is None:
        raise RuntimeError(f"generated kernel not found: {SYMBOL}")
    original_run = target.run
    captured: list[dict[str, torch.Tensor]] = []
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
    try:
        for state in states[:count]:
            values = torch.tensor([state.get("token_ids", state.get("input_ids"))], dtype=torch.long, device=device)
            current = {}
            model.zero_grad(set_to_none=True)
            candidate(values).backward()
            torch.cuda.synchronize(device)
            if set(current) < {"in_out_ptr0", "in_ptr0", "in_ptr1", "out_ptr0"}:
                raise RuntimeError(f"natural operands not captured: {sorted(current)}")
            captured.append({key: value for key, value in current.items()})
    finally:
        target.run = original_run

    rows = []
    for index, operands in enumerate(captured):
        n = int(operands["in_ptr0"].numel())
        outputs = [torch.empty_like(operands["in_ptr0"], dtype=torch.float32) for _ in range(7)]
        _instrumented[(triton.cdiv(n, 1024),)](
            operands["in_out_ptr0"], operands["in_ptr0"], operands["in_ptr1"], operands["out_ptr0"],
            *outputs, xnumel=n, XBLOCK=1024,
        )
        torch.cuda.synchronize(device)
        saved_out, saved_exp, saved_denom, saved_sigmoid, saved_product, saved_factor, saved_derivative = outputs
        candidate_out = operands["candidate_out_post"].float()
        candidate_derivative = operands["candidate_derivative_post"].float()
        direct_gate = operands["in_ptr1"].float()
        direct_grad = operands["in_ptr0"].float()
        direct_up = operands["in_out_ptr0"].float()
        one = torch.ones_like(direct_gate)
        direct_exp = torch.exp(-direct_gate)
        direct_denom = direct_exp + one
        direct_sigmoid = one / direct_denom
        direct_product = (direct_grad * direct_up) * direct_sigmoid
        direct_factor = direct_gate * (one - direct_sigmoid) + one
        direct_derivative = direct_product * direct_factor
        rows.append({
            "state_index": index,
            "final_output_alignment": metrics(saved_out.to(operands["candidate_out_post"].dtype).float(), candidate_out),
            "final_derivative_alignment": metrics(saved_derivative.to(operands["candidate_derivative_post"].dtype).float(), candidate_derivative),
            "intermediates": {
                "exp": metrics(saved_exp, direct_exp),
                "denominator": metrics(saved_denom, direct_denom),
                "sigmoid": metrics(saved_sigmoid, direct_sigmoid),
                "product": metrics(saved_product, direct_product),
                "factor": metrics(saved_factor, direct_factor),
                "derivative": metrics(saved_derivative, direct_derivative),
            },
        })
    return rows


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    rows = capture_states(device, 4)
    names = ("exp", "denominator", "sigmoid", "product", "factor", "derivative")
    first = {name: any(row["intermediates"][name]["nonzero_coordinates"] > 0 for row in rows) for name in names}
    final_ok = all(row["final_output_alignment"]["exact"] and row["final_derivative_alignment"]["exact"] for row in rows)
    payload = {
        "schema": "kernel-analyzer-silu-natural-intermediates-v1",
        "status": "COMPLETE_NATURAL_INTERMEDIATE_PROBE" if final_ok else "REJECTED_FINAL_ALIGNMENT",
        "state_count": len(rows),
        "kernel_symbol": SYMBOL,
        "input_policy": "natural model operands captured at the reviewed generated kernel boundary",
        "rows": rows,
        "first_difference_candidates": first,
        "claim_boundary": (
            "Only if final candidate stores align exactly does this identify a source class on natural operands; "
            "it still does not establish a population mean or training-quality claim."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output": str(OUT), "status": payload["status"], "first_difference_candidates": first}, ensure_ascii=False))


if __name__ == "__main__":
    main()
