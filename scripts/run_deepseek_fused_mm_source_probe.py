#!/usr/bin/env python3
"""Isolate upstream BF16 matrix-product outputs entering a fused backward call.

The selected DeepSeek fused embedding/NLL boundary receives three BF16 matrix
products.  This probe keeps the compiled call and all later arithmetic fixed,
but replaces one selected product at a time with the same operands evaluated in
FP32.  It is intended to decide whether the large region-level effect is an
upstream matrix-product materialisation effect or a remaining producer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl
from torch._inductor.codecache import PyCodeCache
from transformers import AutoModelForCausalLM

from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime


ROOT = Path(__file__).resolve().parents[1]
SYMBOL = "triton_red_fused__to_copy_add_div_embedding_dense_backward_expand_mul_nll_loss_forward_pow_sum_view_22"


@triton.jit
def _mm_variant(
    in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7,
    out_ptr1, xnumel: tl.constexpr, r0_numel: tl.constexpr,
    XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr, MODE: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xlinear = xoffset + tl.arange(0, XBLOCK)
    xindex = xlinear[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, R0_BLOCK)[None, :]
    partial = tl.zeros([XBLOCK, R0_BLOCK], dtype=tl.float32)
    for offset in tl.range(0, r0_numel, R0_BLOCK):
        r = offset + rbase
        mask = (r < r0_numel) & xmask
        a = tl.load(in_ptr0 + r + r0_numel * xindex, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(in_ptr1 + r + r0_numel * xindex, mask=mask, other=0.0).to(tl.float32)
        c = tl.load(in_ptr2 + r + r0_numel * xindex, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(in_ptr3 + r, mask=r < r0_numel, other=0.0).to(tl.float32)
        emb = tl.load(in_ptr4 + r + r0_numel * xindex, mask=mask, other=0.0).to(tl.float32)
        term = (a + b + c) * scale * emb
        partial += term
    first = tl.sum(partial, 1)[:, None]
    token = tl.load(in_ptr5 + xlinear, mask=xlinear < xnumel, other=0)
    inv = tl.load(in_ptr7 + xlinear, mask=xlinear < xnumel, other=0.0).to(tl.float32)
    for offset in tl.range(0, r0_numel, R0_BLOCK):
        r = offset + rbase
        mask = (r < r0_numel) & xmask
        grad = tl.load(in_ptr6 + r + r0_numel * xindex, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(in_ptr0 + r + r0_numel * xindex, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(in_ptr1 + r + r0_numel * xindex, mask=mask, other=0.0).to(tl.float32)
        c = tl.load(in_ptr2 + r + r0_numel * xindex, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(in_ptr3 + r, mask=r < r0_numel, other=0.0).to(tl.float32)
        emb = tl.load(in_ptr4 + r + r0_numel * xindex, mask=mask, other=0.0).to(tl.float32)
        base = (a + b + c) * scale
        term = first * (-0.5) * inv[:, None] * inv[:, None] * inv[:, None]
        term = term * 0.000244140625 * (2.0 * emb)
        value = grad + base * inv[:, None] + term
        value = tl.where(token[:, None] == -1, 0.0, value)
        tl.store(out_ptr1 + r + r0_numel * xindex, value, mask=mask)


def _write(g: torch.Tensor, lr: float = 1e-4) -> torch.Tensor:
    x = g.float()
    return -lr * x / (x.abs() + 1e-8)


def _ratio(effect: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(effect).item()) / max(float(torch.linalg.vector_norm(reference).item()), 1e-30)


def _interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    if x.numel() < 2:
        value = float(x.mean().item())
        return [value, value]
    mean = float(x.mean().item())
    half = 1.96 * float(x.std(unbiased=True).item()) / (x.numel() ** 0.5)
    return [mean - half, mean + half]


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    bank = json.loads(args.input_bank.read_text(encoding="utf-8"))
    states = bank["states"][: args.states]
    if len(states) != args.states or args.states < 4 or args.states % 2:
        raise ValueError("states must be an even number available in the input bank")
    configure_candidate_runtime(27890)
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True,
    ).to(device).train()
    model.config.use_cache = False
    target = dict(model.named_parameters())["model.embed_tokens.weight"]
    start = len(PyCodeCache.modules)
    compiled = torch.compile(LossStep(model), backend="inductor", fullgraph=True, dynamic=False)
    warm = torch.tensor([states[0].get("token_ids", states[0].get("input_ids"))], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    compiled(warm).backward()
    torch.cuda.synchronize(device)

    target_kernel = None
    target_module = None
    for module in list(PyCodeCache.modules)[start:]:
        for name, value in vars(module).items():
            if SYMBOL in str(name) and callable(getattr(value, "run", None)):
                target_kernel = value
                target_module = module
                break
        if target_kernel is not None:
            break
    if target_kernel is None:
        raise RuntimeError("generated fused boundary kernel not found")
    extern = getattr(target_module, "extern_kernels", None)
    if extern is None or not hasattr(extern, "mm"):
        raise RuntimeError("compiled module does not expose extern_kernels.mm")
    original_mm = extern.mm
    original_run = target_kernel.run
    mode = "native"
    fp32_products: list[torch.Tensor] = []

    def mm_capture(*call_args: Any, **kwargs: Any) -> Any:
        out = kwargs.get("out")
        result = original_mm(*call_args, **kwargs)
        if isinstance(out, torch.Tensor) and len(out.shape) == 2 and int(out.shape[1]) == 4096:
            # The selected generated graph has exactly three [tokens,4096]
            # products immediately before the fused embedding boundary.
            if len(fp32_products) < 3:
                left, right = call_args[:2]
                fp32_products.append(torch.mm(left.float(), right.float()).detach())
        return result

    def wrapped(*call_args: Any, **kwargs: Any) -> Any:
        if mode == "native":
            return original_run(*call_args, **kwargs)
        if len(fp32_products) != 3:
            raise RuntimeError(f"expected three captured products, got {len(fp32_products)}")
        tensors = list(call_args)
        for index in range(3):
            if mode == f"mm{index}_fp32" or mode == "all_mm_fp32":
                tensors[index] = fp32_products[index]
        xnumel = int(tensors[-2])
        r0_numel = int(tensors[-1])
        _mm_variant[(triton.cdiv(xnumel, 1),)](
            *tensors, XBLOCK=1, R0_BLOCK=128,
            MODE=0,
        )
        return None

    variants = ("mm0_fp32", "mm1_fp32", "mm2_fp32", "all_mm_fp32")
    rows: list[dict[str, Any]] = []
    try:
        extern.mm = mm_capture
        target_kernel.run = wrapped
        for state in states:
            ids = torch.tensor([state.get("token_ids", state.get("input_ids"))], dtype=torch.long, device=device)
            row: dict[str, Any] = {"state_id": state.get("state_id")}
            model.zero_grad(set_to_none=True)
            mode = "native"
            fp32_products.clear()
            native_loss = compiled(ids)
            native_loss.backward()
            torch.cuda.synchronize(device)
            native_write = _write(target.grad.detach().float().cpu().clone())
            row["loss_native"] = float(native_loss.detach().cpu().item())
            for variant in variants:
                model.zero_grad(set_to_none=True)
                mode = variant
                fp32_products.clear()
                variant_loss = compiled(ids)
                variant_loss.backward()
                torch.cuda.synchronize(device)
                variant_write = _write(target.grad.detach().float().cpu().clone())
                effect = native_write - variant_write
                row[f"loss_{variant}"] = float(variant_loss.detach().cpu().item())
                row[f"write_effect_rms_over_{variant}"] = _ratio(effect, variant_write)
                row[f"write_aligned_over_{variant}"] = float(torch.sum(effect * variant_write).item()) / max(float(torch.sum(variant_write * variant_write).item()), 1e-30)
            rows.append(row)
            del ids, native_loss
    finally:
        extern.mm = original_mm
        target_kernel.run = original_run

    summary = {"state_count": len(rows), "calibration_count": len(rows) // 2, "confirmation_count": len(rows) - len(rows) // 2, "variants": {}}
    for variant in variants:
        values = [row[f"write_aligned_over_{variant}"] for row in rows]
        summary["variants"][variant] = {
            "write_effect_rms_mean": float(torch.tensor([row[f"write_effect_rms_over_{variant}"] for row in rows]).mean()),
            "aligned_write_mean": float(torch.tensor(values).mean()),
            "aligned_write_interval_normal_95": _interval(values),
            "positive_count": sum(value > 0 for value in values),
            "negative_count": sum(value < 0 for value in values),
            "loss_difference_mean": float(torch.tensor([row["loss_native"] - row[f"loss_{variant}"] for row in rows]).mean()),
        }
    return {
        "schema": "kernel-analyzer-deepseek-fused-mm-source-probe-v1",
        "status": "COMPLETE_SOURCE_ISOLATION_PROBE",
        "model": str(args.model),
        "input_bank": str(args.input_bank),
        "operator": "fused embedding/NLL/normalization backward boundary upstream MM outputs",
        "candidate": "native BF16 MM outputs entering the fused boundary",
        "reference": "same MM operands evaluated in FP32, one selected product at a time",
        "comparison_scope": {"same_model_weights": True, "same_input_ids": True, "same_compiled_call_boundary": True, "single_changed_boundary": "one of three upstream MM output materialisations"},
        "rows": rows,
        "summary": summary,
        "claim_boundary": "This isolates only the three visible upstream MM output materialisations; it does not claim that the entire fused-region effect has one source unless the observed direction and scale are reproduced.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
