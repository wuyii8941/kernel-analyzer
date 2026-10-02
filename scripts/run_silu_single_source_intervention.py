#!/usr/bin/env python3
"""Path-preserving DeepSeek SiLU reciprocal intervention on natural states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl
from torch._inductor.codecache import PyCodeCache
from torch._inductor.runtime.triton_helpers import libdevice
from transformers import AutoModelForCausalLM

from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime
from kernel_analyzer.silu_backward_reference import evaluate as evaluate_silu_backward


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "../models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
DEFAULT_BANK = ROOT / "results/coverage/deepseek8b_seq128_input_bank.json"
SYMBOL = "triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_4"


@triton.jit
def _silu_intervention(
    in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, sigmoid_override_ptr,
    product_override_ptr, factor_override_ptr,
    xnumel: tl.constexpr, XBLOCK: tl.constexpr, VARIANT: tl.constexpr,
):
    offset = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    mask = offset < xnumel
    # Reproduce the generated kernel's temporary association.  The variants
    # below replace only one declared boundary; they do not silently change
    # the other products/divisions or the two output stores.
    tmp0 = tl.load(in_ptr0 + offset, mask=mask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + offset, mask=mask, other=0.0).to(tl.float32)
    tmp10 = tl.load(in_out_ptr0 + offset, mask=mask, other=0.0).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = -tmp2
    if VARIANT == 7:
        tmp4 = tl.exp(tmp3)
    else:
        tmp4 = libdevice.exp(tmp3)
    tmp5 = tl.full([1], 1.0, tl.float32)
    tmp6 = tmp4 + tmp5
    reciprocal = libdevice.rcp_rn(tmp6)
    if VARIANT == 0 or VARIANT == 1:
        tmp7 = tmp2 * reciprocal
    elif VARIANT == 11:
        tmp7 = tmp2 * tl.load(sigmoid_override_ptr + offset, mask=mask, other=0.0)
    else:
        tmp7 = tmp2 / tmp6
    tmp8 = tmp7.to(tl.float32)
    if VARIANT == 2:
        tmp9 = libdevice.fma(tmp0, tmp8, 0.0)
    else:
        tmp9 = tmp0 * tmp8
    tmp11 = tmp0 * tmp10
    tmp12 = tmp11.to(tl.float32)
    if VARIANT == 1 or VARIANT == 9:
        tmp13 = reciprocal
    elif VARIANT == 8:
        tmp13 = tl.sigmoid(tmp2)
    elif VARIANT == 10:
        tmp13 = tl.load(sigmoid_override_ptr + offset, mask=mask, other=0.0)
    else:
        tmp13 = tmp5 / tmp6
    tmp14 = tmp13 * tmp5
    if VARIANT == 6:
        # Preserve the ABI and downstream path while replacing only the saved
        # forward materialization with a recomputation from the captured gate.
        tmp12 = tmp2 * tmp14
    if VARIANT == 5:
        tmp15 = libdevice.fma(tmp12, tmp14, 0.0)
    elif VARIANT == 12 or VARIANT == 14:
        tmp15 = tl.load(product_override_ptr + offset, mask=mask, other=0.0)
    else:
        tmp15 = tmp12 * tmp14
    if VARIANT == 3:
        tmp16 = libdevice.fma(-tmp14, tmp5, tmp5)
    elif VARIANT == 15:
        tmp16 = tl.load(factor_override_ptr + offset, mask=mask, other=0.0)
    else:
        tmp16 = tmp5 - tmp14
    tmp17 = tmp2 * tmp16
    if VARIANT == 16:
        tmp17 = tl.load(factor_override_ptr + offset, mask=mask, other=0.0)
    tmp18 = tmp17 + tmp5
    if VARIANT == 13 or VARIANT == 14:
        tmp18 = tl.load(factor_override_ptr + offset, mask=mask, other=0.0)
    if VARIANT == 4:
        tmp19 = libdevice.fma(tmp15, tmp18, 0.0)
    else:
        tmp19 = tmp15 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(out_ptr0 + offset, tmp9, mask=mask)
    tl.store(in_out_ptr0 + offset, tmp20, mask=mask)


def _write(grad: torch.Tensor, lr: float = 1e-4) -> torch.Tensor:
    g = grad.float()
    return -lr * g / (g.abs() + 1e-8)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    bank = json.loads(args.input_bank.read_text(encoding="utf-8"))
    bank_states = list(bank.get("states", bank.get("records")))
    if args.state_indices:
        selected_indices = [int(value) for value in args.state_indices.split(",") if value.strip()]
        if not selected_indices:
            raise ValueError("--state-indices must contain at least one integer")
        if any(index < 0 or index >= len(bank_states) for index in selected_indices):
            raise IndexError("state index is outside the supplied input bank")
        states = [bank_states[index] for index in selected_indices]
    else:
        selected_indices = list(range(min(args.states, len(bank_states))))
        states = [bank_states[index] for index in selected_indices]
    configure_candidate_runtime(27002)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True,
    ).to(device).train()
    model.config.use_cache = False
    target_name = "model.layers.35.mlp.gate_proj.weight"
    named = dict(model.named_parameters())
    if target_name not in named:
        candidates = [(n, p) for n, p in named.items() if ".mlp.gate_proj.weight" in n]
        if not candidates:
            raise RuntimeError("DeepSeek gate projection parameter was not found")
        target_name, target_param = candidates[-1]
    else:
        target_param = named[target_name]
    start = len(PyCodeCache.modules)
    compiled = torch.compile(LossStep(model), backend="inductor", fullgraph=True, dynamic=False)
    warm = torch.tensor([states[0].get("token_ids", states[0].get("input_ids"))], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    compiled(warm).backward()
    torch.cuda.synchronize(device)
    target = None
    for module in list(PyCodeCache.modules)[start:]:
        for name, value in vars(module).items():
            if SYMBOL in str(name) and callable(getattr(value, "run", None)):
                target = value
                break
        if target is not None:
            break
    if target is None:
        raise RuntimeError(f"generated SiLU kernel not found: {SYMBOL}")
    original_run = target.run
    mode = "native"

    def wrapped(*call_args: Any, **kwargs: Any) -> Any:
        tensors = [value for value in call_args if isinstance(value, torch.Tensor)]
        if len(tensors) != 4:
            raise RuntimeError(f"unexpected SiLU pointer ABI: {len(tensors)} tensors")
        if mode == "native":
            return original_run(*call_args, **kwargs)
        if mode == "torch_reference":
            # Clone the aliased saved forward value before writing either
            # output. This is the same-input mathematical reference already
            # used by the retained coverage case, now carried through the
            # identical downstream graph for vector mediation checks.
            up = tensors[0].detach().float().clone()
            grad = tensors[1].detach().float()
            gate = tensors[2].detach().float()
            derivative = evaluate_silu_backward(
                grad, gate, up, variant="NATIVE_SIGMOID_COMPACT"
            ).to(tensors[0].dtype)
            output = (grad * torch.nn.functional.silu(gate)).to(tensors[3].dtype)
            tensors[0].copy_(derivative)
            tensors[3].copy_(output)
            return None
        n = int(tensors[1].numel())
        variant = {
            "reciprocal": 0,
            "all_divisions": 1,
            "derivative_reciprocal": 9,
            "reference_derivative_sigmoid": 10,
            "reference_output_sigmoid": 11,
            "reference_product": 12,
            "reference_factor": 13,
            "reference_product_factor": 14,
            "reference_one_minus_sigmoid": 15,
            "reference_gate_times_one_minus": 16,
            "output_fma": 2,
            "factor_fma": 3,
            "derivative_fma": 4,
            "product_fma": 5,
            "saved_forward_fp32": 6,
            "tl_exp": 7,
            "tl_sigmoid": 8,
        }[mode]
        if variant in (10, 11, 12, 13, 14, 15, 16):
            gate_fp32 = tensors[2].detach().float()
            grad_fp32 = tensors[1].detach().float()
            up_fp32 = tensors[0].detach().float()
            sigmoid_override = (1.0 / (1.0 + torch.exp(-gate_fp32))).contiguous()
            product_override = ((grad_fp32 * up_fp32) * sigmoid_override).contiguous()
            factor_override = (gate_fp32 * (1.0 - sigmoid_override) + 1.0).contiguous()
            if variant == 15:
                factor_override = (1.0 - sigmoid_override).contiguous()
            elif variant == 16:
                factor_override = (gate_fp32 * (1.0 - sigmoid_override)).contiguous()
        else:
            sigmoid_override = torch.empty_like(tensors[2], dtype=torch.float32)
            product_override = torch.empty_like(tensors[2], dtype=torch.float32)
            factor_override = torch.empty_like(tensors[2], dtype=torch.float32)
        _silu_intervention[(triton.cdiv(n, args.intervention_block),)](
            tensors[0], tensors[1], tensors[2], tensors[3], sigmoid_override,
            product_override, factor_override,
            xnumel=n, XBLOCK=args.intervention_block,
            VARIANT=variant,
        )
        return None

    target.run = wrapped
    rows: list[dict[str, Any]] = []
    all_variant_names = (
        "torch_reference", "reciprocal", "all_divisions", "derivative_reciprocal",
        "reference_derivative_sigmoid", "reference_output_sigmoid", "output_fma",
        "reference_product", "reference_factor", "reference_product_factor",
        "reference_one_minus_sigmoid", "reference_gate_times_one_minus",
        "factor_fma", "derivative_fma", "product_fma", "saved_forward_fp32",
        "tl_exp", "tl_sigmoid",
    )
    measured_variant_names = ("torch_reference",) if args.mean_only else all_variant_names
    try:
        for index, state in enumerate(states):
            values = torch.tensor([state.get("token_ids", state.get("input_ids"))], dtype=torch.long, device=device)
            mode = "native"
            model.zero_grad(set_to_none=True)
            native_loss = compiled(values)
            native_loss.backward()
            torch.cuda.synchronize(device)
            native_grad = target_param.grad.detach().float().cpu().clone()
            native_write = _write(native_grad)
            row = {
                "state_index": index,
                "source_state_index": selected_indices[index],
                "source_state_id": str(state.get("state_id", selected_indices[index])),
                "loss_native": float(native_loss.detach().cpu()),
            }
            variant_names = measured_variant_names
            variant_writes: dict[str, torch.Tensor] = {}
            variant_grads: dict[str, torch.Tensor] = {}
            for variant_name in variant_names:
                mode = variant_name
                model.zero_grad(set_to_none=True)
                repaired_loss = compiled(values)
                repaired_loss.backward()
                torch.cuda.synchronize(device)
                repaired_grad = target_param.grad.detach().float().cpu().clone()
                repaired_write = _write(repaired_grad)
                variant_grads[variant_name] = repaired_grad
                variant_writes[variant_name] = repaired_write
                effect = native_write - repaired_write
                row[f"loss_{variant_name}"] = float(repaired_loss.detach().cpu())
                row[f"gradient_effect_rms_over_{variant_name}"] = float(torch.linalg.vector_norm(native_grad - repaired_grad)) / max(float(torch.linalg.vector_norm(repaired_grad)), 1e-30)
                row[f"write_effect_rms_over_{variant_name}"] = float(torch.linalg.vector_norm(effect)) / max(float(torch.linalg.vector_norm(repaired_write)), 1e-30)
                row[f"write_aligned_over_{variant_name}"] = float(torch.sum(effect * repaired_write)) / max(float(torch.sum(repaired_write * repaired_write)), 1e-30)
                row[f"write_reference_energy_{variant_name}"] = float(torch.sum(repaired_write * repaired_write))
                row[f"write_effect_reference_inner_product_{variant_name}"] = float(torch.sum(effect * repaired_write))
            target_effect = native_write - variant_writes["torch_reference"]
            target_norm = float(torch.linalg.vector_norm(target_effect))
            row["candidate_reference_write_effect_norm"] = target_norm
            for variant_name in variant_names[1:]:
                intervention_effect = native_write - variant_writes[variant_name]
                intervention_norm = float(torch.linalg.vector_norm(intervention_effect))
                inner = float(torch.sum(target_effect * intervention_effect))
                residual = target_effect - intervention_effect
                row[f"mediation_{variant_name}"] = {
                    "effect_norm_ratio_to_candidate_reference": intervention_norm / max(target_norm, 1e-30),
                    "cosine_to_candidate_reference": inner / max(target_norm * intervention_norm, 1e-30),
                    "residual_norm_ratio": float(torch.linalg.vector_norm(residual)) / max(target_norm, 1e-30),
                    "exact_parameter_write": bool(torch.equal(native_write - target_effect, native_write - intervention_effect)),
                }
            rows.append(row)
    finally:
        target.run = original_run
    intervention_text = f"same generated boundary and expression order (XBLOCK={args.intervention_block}), with one reciprocal/division/product/derivative arithmetic boundary changed at a time"
    return {
        "schema": "kernel-analyzer-silu-single-source-intervention-v7",
        "status": "COMPLETE_PATH_PRESERVING_SILU_MEDIATION_INTERVENTIONS_V7",
        "model": str(args.model), "target_parameter": target_name,
        "operator": "DeepSeek generated SiLU backward",
        "candidate": "native generated sigmoid reciprocal/division expression",
        "variant": "generated-association-preserving-arithmetic-boundaries-plus-saved-forward-factorial",
        "intervention": intervention_text + "; plus one saved-forward FP32 recomputation with the same gate",
        "state_count": len(rows), "rows": rows,
        "sampling": {
            "input_bank": str(args.input_bank),
            "source_bank_state_count": len(bank_states),
            "source_state_indices": selected_indices,
            "with_replacement": bool(args.state_indices),
            "mean_only": bool(args.mean_only),
        },
        "summary": {
            variant: {
                "write_effect_rms_mean": sum(row[f"write_effect_rms_over_{variant}"] for row in rows) / len(rows),
                "write_aligned_interval": [min(row[f"write_aligned_over_{variant}"] for row in rows), max(row[f"write_aligned_over_{variant}"] for row in rows)],
                "loss_difference_mean": sum(row["loss_native"] - row[f"loss_{variant}"] for row in rows) / len(rows),
            }
            for variant in measured_variant_names
        },
        "mediation_summary": {
            variant: _aggregate_mediation(rows, variant)
            for variant in measured_variant_names[1:]
        },
        "claim_boundary": "Same-checkpoint natural-state path intervention that preserves the generated temporary association and changes only the declared reciprocal/division/product/derivative boundary; it can close only that candidate source if it explains the original native/reference profile.",
}


def _aggregate_mediation(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    """Aggregate vector mediation by concatenated norms, not mean ratios.

    Per-state ratios give every state the same weight and can misrepresent a
    high-dimensional write effect when state magnitudes differ.  The aggregate
    is the ratio of squared norms and inner products over all state vectors.
    """
    target_sq = 0.0
    intervention_sq = 0.0
    residual_sq = 0.0
    inner = 0.0
    exact = 0
    for row in rows:
        target = float(row["candidate_reference_write_effect_norm"])
        mediation = row[f"mediation_{variant}"]
        ratio = float(mediation["effect_norm_ratio_to_candidate_reference"])
        cosine = float(mediation["cosine_to_candidate_reference"])
        residual_ratio = float(mediation["residual_norm_ratio"])
        target_sq += target * target
        intervention_sq += (target * ratio) ** 2
        residual_sq += (target * residual_ratio) ** 2
        inner += (target * target) * ratio * cosine
        exact += int(bool(mediation["exact_parameter_write"]))
    denom = max(target_sq, 1e-30)
    cosine_aggregate = inner / max((target_sq * intervention_sq) ** 0.5, 1e-30)
    return {
        "effect_norm_ratio_aggregate": (intervention_sq / denom) ** 0.5,
        "cosine_aggregate": max(-1.0, min(1.0, cosine_aggregate)),
        "residual_norm_ratio_aggregate": (residual_sq / denom) ** 0.5,
        "effect_norm_ratio_mean": sum(
            float(row[f"mediation_{variant}"]["effect_norm_ratio_to_candidate_reference"])
            for row in rows
        ) / max(len(rows), 1),
        "cosine_mean": sum(
            float(row[f"mediation_{variant}"]["cosine_to_candidate_reference"])
            for row in rows
        ) / max(len(rows), 1),
        "residual_norm_ratio_mean": sum(
            float(row[f"mediation_{variant}"]["residual_norm_ratio"])
            for row in rows
        ) / max(len(rows), 1),
        "exact_write_count": exact,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--input-bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--states", type=int, default=4)
    parser.add_argument(
        "--state-indices", default=None,
        help="comma-separated indices sampled from the input bank; repeats are allowed",
    )
    parser.add_argument(
        "--mean-only", action="store_true",
        help="measure only the native/reference write effect for a population mean probe",
    )
    parser.add_argument("--variant", choices=("reciprocal", "all_divisions"), default="reciprocal", help="retained for compatibility; v4 runs all arithmetic interventions")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--intervention-block", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
