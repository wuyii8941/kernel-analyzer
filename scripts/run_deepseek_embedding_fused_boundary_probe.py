#!/usr/bin/env python3
"""Probe the DeepSeek fused embedding/NLL backward boundary on real states.

The generated backward region combines normalization, NLL and embedding
gradient terms.  This probe keeps the compiled call boundary and all inputs
fixed, then replaces only the reduction arithmetic with explicitly declared
variants.  It is a source-isolation experiment, not a claim that the whole
fused region has one root cause.
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
DEFAULT_MODEL = ROOT / "../models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
DEFAULT_BANK = ROOT / "results/coverage/deepseek8b_seq128_input_bank.json"
SYMBOL = "triton_red_fused__to_copy_add_div_embedding_dense_backward_expand_mul_nll_loss_forward_pow_sum_view_22"


@triton.jit
def _fused_boundary_variant(
    in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7,
    out_ptr1, xnumel: tl.constexpr, r0_numel: tl.constexpr,
    XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr, VARIANT: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xlinear = xoffset + tl.arange(0, XBLOCK)
    xindex = xlinear[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, R0_BLOCK)[None, :]

    # Variant 0 follows the generated expression with FP32 accumulation.
    # Variant 1 materializes every partial term in BF16 before accumulation.
    # Variant 2 keeps the reduction in FP64 and casts only at the store.
    # Variant 3 keeps FP32 throughout but reverses the reduction element order.
    # Variant 4 materializes only the normalization derivative term before the
    # final value expression; variant 5 materializes only the base term; and
    # variant 6 materializes the final value before the generated FP32 store.
    # These are source-isolation controls, not claims that any one spelling is
    # a production repair.
    if VARIANT == 2:
        partial = tl.zeros([XBLOCK, R0_BLOCK], dtype=tl.float64)
    else:
        partial = tl.zeros([XBLOCK, R0_BLOCK], dtype=tl.float32)
    for offset in tl.range(0, r0_numel, R0_BLOCK):
        if VARIANT == 3:
            r = r0_numel - 1 - (offset + rbase)
        else:
            r = offset + rbase
        mask = (r < r0_numel) & xmask
        a = tl.load(in_ptr0 + r + r0_numel * xindex, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(in_ptr1 + r + r0_numel * xindex, mask=mask, other=0.0).to(tl.float32)
        c = tl.load(in_ptr2 + r + r0_numel * xindex, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(in_ptr3 + r, mask=r < r0_numel, other=0.0).to(tl.float32)
        emb = tl.load(in_ptr4 + r + r0_numel * xindex, mask=mask, other=0.0).to(tl.float32)
        term = (a + b + c) * scale * emb
        if VARIANT == 1:
            term = term.to(tl.bfloat16).to(tl.float32)
        partial += term
    if VARIANT == 2:
        first = tl.sum(partial, 1).to(tl.float32)[:, None]
    else:
        first = tl.sum(partial, 1)[:, None]

    token = tl.load(in_ptr5 + xlinear, mask=xlinear < xnumel, other=0)
    inv = tl.load(in_ptr7 + xlinear, mask=xlinear < xnumel, other=0.0).to(tl.float32)
    for offset in tl.range(0, r0_numel, R0_BLOCK):
        if VARIANT == 3:
            r = r0_numel - 1 - (offset + rbase)
        else:
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
        # The generated kernel scales this normalization derivative by the
        # reciprocal hidden width before multiplying the embedding carrier.
        term = term * 0.000244140625 * (2.0 * emb)
        if VARIANT == 4:
            term = term.to(tl.bfloat16).to(tl.float32)
        if VARIANT == 5:
            base = base.to(tl.bfloat16).to(tl.float32)
        value = grad + base * inv[:, None] + term
        if VARIANT == 1:
            value = value.to(tl.bfloat16).to(tl.float32)
        if VARIANT == 6:
            value = value.to(tl.bfloat16).to(tl.float32)
        # The generated kernel writes FP32 and a later index_put accumulates it.
        tl.store(out_ptr1 + r + r0_numel * xindex, tl.where(token[:, None] == -1, 0.0, value), mask=mask)


def _write(gradient: torch.Tensor, lr: float = 1e-4) -> torch.Tensor:
    g = gradient.float()
    return -lr * g / (g.abs() + 1e-8)


def _ratio(effect: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(effect)) / max(float(torch.linalg.vector_norm(reference)), 1e-30)


def _interval(values: list[float]) -> list[float] | None:
    if not values:
        return None
    x = torch.tensor(values, dtype=torch.float64)
    if x.numel() < 2:
        v = float(x.mean())
        return [v, v]
    half = 1.96 * float(x.std(unbiased=True)) / (x.numel() ** 0.5)
    mean = float(x.mean())
    return [mean - half, mean + half]


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    bank = json.loads(args.input_bank.read_text(encoding="utf-8"))
    states = bank["states"][: args.states]
    if len(states) != args.states or args.states < 4 or args.states % 2:
        raise ValueError("states must be an even number available in the input bank")
    configure_candidate_runtime(27890)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True,
    ).to(device).train()
    model.config.use_cache = False
    target_name = "model.embed_tokens.weight"
    target = dict(model.named_parameters()).get(target_name)
    if target is None:
        raise RuntimeError(f"missing target parameter: {target_name}")
    start = len(PyCodeCache.modules)
    compiled = torch.compile(LossStep(model), backend="inductor", fullgraph=True, dynamic=False)
    warm = torch.tensor([states[0].get("token_ids", states[0].get("input_ids"))], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    compiled(warm).backward()
    torch.cuda.synchronize(device)
    target_kernel = None
    for module in list(PyCodeCache.modules)[start:]:
        for name, value in vars(module).items():
            if SYMBOL in str(name) and callable(getattr(value, "run", None)):
                target_kernel = value
                break
        if target_kernel is not None:
            break
    if target_kernel is None:
        raise RuntimeError(f"generated kernel not found: {SYMBOL}")
    original_run = target_kernel.run
    mode = "native"

    def wrapped(*call_args: Any, **kwargs: Any) -> Any:
        tensors = [value for value in call_args if isinstance(value, torch.Tensor)]
        if len(tensors) != 9:
            raise RuntimeError(f"unexpected fused embedding ABI: {len(tensors)} tensors")
        if mode == "native":
            return original_run(*call_args, **kwargs)
        variant = {
            "fp32_replay": 0,
            "bf16_partial": 1,
            "fp64_reduction": 2,
            "fp32_reverse_order": 3,
            "bf16_normalization_term": 4,
            "bf16_base_term": 5,
            "bf16_final_value": 6,
        }[mode]
        n = int(tensors[0].shape[0])
        r = int(tensors[0].shape[1])
        _fused_boundary_variant[(triton.cdiv(n, 1),)](
            *tensors, n, r, XBLOCK=1, R0_BLOCK=128, VARIANT=variant,
        )
        return None

    target_kernel.run = wrapped
    all_variants = (
        "fp32_replay", "bf16_partial", "fp64_reduction", "fp32_reverse_order",
        "bf16_normalization_term", "bf16_base_term", "bf16_final_value",
    )
    variant_names = tuple(args.variants) if args.variants else all_variants
    rows: list[dict[str, Any]] = []
    try:
        for index, state in enumerate(states):
            ids = torch.tensor([state.get("token_ids", state.get("input_ids"))], dtype=torch.long, device=device)
            model.zero_grad(set_to_none=True)
            mode = "native"
            native_loss = compiled(ids)
            native_loss.backward()
            torch.cuda.synchronize(device)
            native_grad = target.grad.detach().float().cpu().clone()
            native_write = _write(native_grad)
            row: dict[str, Any] = {"state_index": index, "loss_native": float(native_loss.detach().cpu())}
            for variant_name in variant_names:
                mode = variant_name
                model.zero_grad(set_to_none=True)
                variant_loss = compiled(ids)
                variant_loss.backward()
                torch.cuda.synchronize(device)
                variant_grad = target.grad.detach().float().cpu().clone()
                variant_write = _write(variant_grad)
                effect = native_write - variant_write
                row[f"loss_{variant_name}"] = float(variant_loss.detach().cpu())
                row[f"write_effect_rms_over_{variant_name}"] = _ratio(effect, variant_write)
                row[f"write_aligned_over_{variant_name}"] = float(torch.sum(effect * variant_write)) / max(float(torch.sum(variant_write * variant_write)), 1e-30)
            rows.append(row)
    finally:
        target_kernel.run = original_run
    return {
        "schema": "kernel-analyzer-deepseek-embedding-fused-boundary-probe-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "operator": "fused embedding/NLL/normalization backward region",
        "candidate": "native generated fused reduction",
        "reference_variants": list(variant_names),
        "input_source": str(args.input_bank),
        "comparison_scope": {
            "same_model_weights": True,
            "same_input_ids": True,
            "same_compiled_call_boundary": True,
            "single_changed_boundary": "declared fused reduction arithmetic",
        },
        "rows": rows,
        "summary": {
            "state_count": len(rows),
            "calibration_count": len(rows) // 2,
            "confirmation_count": len(rows) - len(rows) // 2,
            "variants": {
                name: {
                    "write_effect_rms_mean": float(torch.tensor([r[f"write_effect_rms_over_{name}"] for r in rows]).mean()),
                    "aligned_write_mean": float(torch.tensor([r[f"write_aligned_over_{name}"] for r in rows]).mean()),
                    "aligned_write_interval_normal_95": _interval([r[f"write_aligned_over_{name}"] for r in rows]),
                    "loss_difference_mean": float(torch.tensor([r["loss_native"] - r[f"loss_{name}"] for r in rows]).mean()),
                }
                for name in variant_names
            },
        },
        "claim_boundary": "This probe only tests the declared arithmetic variants at one generated fused boundary; it does not assign the original region effect to a source unless a variant reproduces the observed direction and scale.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--input-bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--states", type=int, default=4)
    parser.add_argument(
        "--variants", nargs="+",
        choices=(
            "fp32_replay", "bf16_partial", "fp64_reduction", "fp32_reverse_order",
            "bf16_normalization_term", "bf16_base_term", "bf16_final_value",
        ),
        default=None,
        help="optional subset of source-isolation variants; default runs all",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
