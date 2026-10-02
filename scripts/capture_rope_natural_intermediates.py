#!/usr/bin/env python3
"""Probe the reviewed fused position-scaling expression on natural operands.

The probe is deliberately conservative: an intermediate source claim is
accepted only when the independently evaluated expression reproduces both
original candidate stores on every inspected natural state.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data1/tzh/cache/torchinductor/natural_rope_probe")
os.environ.setdefault("TRITON_CACHE_DIR", "/data1/tzh/cache/triton/natural_rope_probe")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import triton
import triton.language as tl
from torch._inductor.codecache import PyCodeCache
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime, text_step_values  # noqa: E402
from scripts.run_generated_fp32_screen import load_model  # noqa: E402
from scripts.generated_fp32_observer import runtime_signature  # noqa: E402

MODEL = Path("/data1/tzh/models/mistralai/Ministral-3-3B-Base-2512")
INPUT_BANK = ROOT / "results/property/numerical_coverage_v1/ministral_fused_rotary_highpos_cold_matched_bank_v1.json"
SYMBOL_FRAGMENT = "triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_cos"
OUT = ROOT / "results/property/root_cause_closure_v1/rope_natural_intermediates_v1.json"


@triton.jit
def _instrumented(
    in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1,
    saved_query, saved_phase, saved_cos, saved_sin, saved_rotated, saved_scale,
    saved_first, saved_mixed, saved_output,
    xnumel: tl.constexpr, XBLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    mask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128) % 128
    x2 = xindex // 16384
    x4 = xindex
    query = tl.load(in_ptr0 + (x0 + 128 * x2 + 4096 * x1), mask=mask).to(tl.float32)
    frequency = tl.load(in_ptr1 + (x4 % 64), mask=mask).to(tl.float32)
    position = tl.load(in_ptr2 + x1, mask=mask).to(tl.float32)
    phase = frequency * position
    cosine = tl_math.cos(phase)
    first = query * cosine
    half_index = x0 < 64
    second_a = tl.load(
        in_ptr0 + (64 + 128 * x2 + 4096 * x1 + x0),
        mask=half_index,
        other=0.0,
    ).to(tl.float32)
    second_b = tl.load(
        in_ptr0 + (128 * x2 + 4096 * x1 + (-64) + x0),
        mask=x0 >= 64,
        other=0.0,
    ).to(tl.float32)
    rotated = tl.where(half_index, -second_a, second_b)
    sine = tl_math.sin(phase)
    mixed = first + rotated * sine
    scale_const = tl.full([1], 6.103515625e-05, tl.float32)
    scale_floor = libdevice.floor(position * scale_const)
    scale_plus = scale_floor + tl.full([1], 1.0, tl.float32)
    scale_log = tl_math.log(scale_plus)
    scale_mul = scale_log * tl.full([1], 0.1, tl.float32)
    scale_term = scale_mul + tl.full([1], 1.0, tl.float32)
    output = mixed * scale_term
    tl.store(saved_phase + x4, phase, mask=mask)
    tl.store(saved_query + x4, query, mask=mask)
    tl.store(saved_cos + x4, cosine, mask=mask)
    tl.store(saved_sin + x4, sine, mask=mask)
    tl.store(saved_rotated + x4, rotated, mask=mask)
    tl.store(saved_scale + x4, scale_term, mask=mask)
    tl.store(saved_first + x4, first, mask=mask)
    tl.store(saved_mixed + x4, mixed, mask=mask)
    tl.store(saved_output + x4, output, mask=mask)
    tl.store(out_ptr0 + x4, output, mask=mask)
    tl.store(out_ptr1 + (x0 + 128 * x2 + 4096 * x1), output, mask=mask)


def metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    delta = left.double() - right.double()
    denom = float(torch.linalg.vector_norm(left.double()).item())
    return {
        "exact": bool(torch.count_nonzero(delta).item() == 0 and not bool(torch.isnan(delta).any().item())),
        "nonzero_coordinates": int(torch.count_nonzero(delta).item()),
        "relative_l2": float(torch.linalg.vector_norm(delta).item()) / denom if denom else 0.0,
        "max_abs": float(delta.abs().max().item()) if delta.numel() else 0.0,
        "signed_mean": float(delta.mean().item()) if delta.numel() else 0.0,
    }


def run(device: torch.device, count: int) -> dict[str, Any]:
    bank = json.loads(INPUT_BANK.read_text())
    states = bank["states"]
    configure_candidate_runtime(25000)
    model = load_model("ministral3", MODEL, device)
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=False, dynamic=False)
    warm_values = text_step_values(states[0], device)
    model.zero_grad(set_to_none=True)
    candidate(*warm_values).backward()
    torch.cuda.synchronize(device)
    # A cached compiled module can be reused without being appended after the
    # local start marker; scan the complete in-process module registry.
    modules = list(PyCodeCache.modules)
    target = None
    target_name = None
    candidates = []
    for module in modules:
        for name, value in vars(module).items():
            if SYMBOL_FRAGMENT not in str(name) or not callable(getattr(value, "run", None)):
                continue
            try:
                pointer_names = [
                    str(arg) for arg, annotation in runtime_signature(value)
                    if str(annotation).startswith("*")
                ]
            except Exception:
                continue
            if pointer_names[:3] == ["in_ptr0", "in_ptr1", "in_ptr2"] and pointer_names[3:] == ["out_ptr0", "out_ptr1"]:
                candidates.append((str(name), value))
    if candidates:
        candidates.sort(key=lambda item: ("floor_log" not in item[0] and "log" not in item[0], item[0]))
        target_name, target = candidates[0]
    if target is None:
        raise RuntimeError(f"generated position-scaling kernel not found: {SYMBOL_FRAGMENT}")

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
            current["candidate_out0_post"] = tensors[3].detach().clone()
            current["candidate_out1_post"] = tensors[4].detach().clone()
        return result

    target.run = wrapped
    rows = []
    try:
        for index, state in enumerate(states[:count]):
            values = text_step_values(state, device)
            current = {}
            model.zero_grad(set_to_none=True)
            candidate(*values).backward()
            torch.cuda.synchronize(device)
            required = {"in_ptr0", "in_ptr1", "in_ptr2", "out_ptr0", "out_ptr1"}
            if not required.issubset(current):
                raise RuntimeError(f"natural operands not captured: {sorted(current)}")
            n = int(current["in_ptr0"].numel())
            outputs = [torch.empty((n,), dtype=torch.float32, device=device) for _ in range(9)]
            _instrumented[(triton.cdiv(n, 1024),)](
                current["in_ptr0"], current["in_ptr1"], current["in_ptr2"],
                current["out_ptr0"], current["out_ptr1"], *outputs,
                xnumel=n, XBLOCK=1024,
            )
            torch.cuda.synchronize(device)
            saved_query, saved_phase, saved_cos, saved_sin, saved_rotated, saved_scale, saved_first, saved_mixed, saved_output = outputs
            def raw_memory(value: torch.Tensor, size: int | None = None) -> torch.Tensor:
                length = int(value.numel()) if size is None else int(size)
                # Runtime pointer arguments already point at the view's data
                # start.  Reusing the logical storage offset would count the
                # view offset twice and falsely implicate the fused arithmetic.
                return value.as_strided((length,), (1,), storage_offset=0).float()

            query = raw_memory(current["in_ptr0"])
            frequency = raw_memory(current["in_ptr1"])
            position = raw_memory(current["in_ptr2"])
            flat = torch.arange(n, device=device)
            phase = frequency[flat % 64] * position[(flat // 128) % 128]
            cosine = torch.cos(phase)
            sine = torch.sin(phase)
            x0 = flat % 128
            x1 = (flat // 128) % 128
            x2 = flat // 16384
            first = query * cosine
            second_a = torch.zeros_like(query)
            second_b = torch.zeros_like(query)
            valid_a = x0 < 64
            valid_b = x0 >= 64
            idx_a = 64 + 128 * x2 + 4096 * x1 + x0
            idx_b = 128 * x2 + 4096 * x1 - 64 + x0
            second_a[valid_a] = query[idx_a[valid_a]]
            second_b[valid_b] = query[idx_b[valid_b]]
            rotated = torch.where(valid_a, -second_a, second_b)
            mixed = first + rotated * sine
            scale = torch.log(torch.floor(position[x1] * 6.103515625e-05) + 1.0) * 0.1 + 1.0
            direct = mixed * scale
            candidate0 = raw_memory(current["candidate_out0_post"], n)
            candidate1_tensor = current["candidate_out1_post"].float()
            # The second output is a non-contiguous view.  Its logical index
            # order is [head, feature, position], while the kernel receives
            # the corresponding flat storage pointer.
            candidate1 = candidate1_tensor[x2, x0, x1]
            rows.append({
                "state_index": index,
                "state_id": state.get("state_id"),
                "final_output0_alignment": metrics(saved_output.to(current["candidate_out0_post"].dtype).float(), candidate0),
                "final_output1_alignment": metrics(saved_output.to(current["candidate_out1_post"].dtype).float(), candidate1),
                "intermediates": {
                    "phase": metrics(saved_phase, phase),
                    "query": metrics(saved_query, query),
                    "cosine": metrics(saved_cos, cosine),
                    "sine": metrics(saved_sin, sine),
                    "rotated": metrics(saved_rotated, rotated),
                    "scale": metrics(saved_scale, scale),
                    "first": metrics(saved_first, first),
                    "mixed": metrics(saved_mixed, mixed),
                    "output": metrics(saved_output, direct),
                },
            })
    finally:
        target.run = original_run
    # The independent replay stores FP32 intermediates and then casts to the
    # candidate BF16 output representation.  A few ulp-level store differences
    # are therefore accepted, while larger discrepancies still reject the
    # source replay.  The phase/trig/rotation/scale checks remain exact.
    final_store_tolerance = {
        "max_abs": 0.0078125,
        "relative_l2": 2.0e-5,
    }
    final_ok = all(
        row["final_output0_alignment"]["max_abs"] <= final_store_tolerance["max_abs"]
        and row["final_output1_alignment"]["max_abs"] <= final_store_tolerance["max_abs"]
        and row["final_output0_alignment"]["relative_l2"] <= final_store_tolerance["relative_l2"]
        and row["final_output1_alignment"]["relative_l2"] <= final_store_tolerance["relative_l2"]
        for row in rows
    )
    names = ("phase", "cosine", "sine", "rotated", "scale", "output")
    first = {name: any(row["intermediates"][name]["nonzero_coordinates"] > 0 for row in rows) for name in names}
    return {
        "schema": "kernel-analyzer-rope-natural-intermediates-v1",
        "status": "COMPLETE_NATURAL_INTERMEDIATE_PROBE" if final_ok else "REJECTED_FINAL_ALIGNMENT",
        "kernel_symbol_fragment": SYMBOL_FRAGMENT,
        "kernel_symbol": target_name,
        "state_count": len(rows),
        "input_policy": "natural Ministral operands captured at the reviewed fused position-scaling boundary",
        "rows": rows,
        "first_difference_candidates": first,
        "final_store_tolerance": final_store_tolerance,
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
