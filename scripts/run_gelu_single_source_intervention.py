#!/usr/bin/env python3
"""Path-preserving Gemma GELU source intervention on natural training states.

The generated backward call is kept at the same boundary and only one declared
arithmetic boundary is changed at a time.
This is a causal source probe, not a family-wide or population claim.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl
from torch._inductor.codecache import PyCodeCache
from torch._inductor.runtime.triton_helpers import libdevice
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime
from kernel_analyzer.gelu_product_reference import evaluate as evaluate_gelu_backward


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "../models/google/gemma-4-E2B"
DEFAULT_BANK = ROOT / "results/property/tcmp_allop_v1/input_banks/gemma4_e2b_text128_trajectory32.json"
TARGET_ELEMENTS = 32768
TARGET_WIDTH = 256


@triton.jit
def _gelu_intervention(
    in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    tanh_override_ptr, derivative_override_ptr, product_override_ptr,
    mode: tl.constexpr, xnumel: tl.constexpr, XBLOCK: tl.constexpr,
    multiplier_offset: tl.constexpr, multiplier_stride: tl.constexpr,
):
    offset = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    mask = offset < xnumel
    x0 = offset % 256
    x1 = offset // 256
    # Keep the generated expression association and temporary boundaries from
    # the reviewed native kernel.  Each intervention changes only one named
    # operation; all loads, casts, launch geometry and the final BF16 store stay
    # at the same boundary.
    tmp0 = tl.load(in_ptr0 + offset, mask=mask, other=0.0).to(tl.float32)
    tmp1 = tl.load(
        in_ptr1 + (multiplier_offset + x0 + multiplier_stride * x1),
        mask=mask, other=0.0,
    ).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + offset, mask=mask, other=0.0).to(tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp5 * tmp5
    tmp7 = tmp6 * tmp5
    tmp8 = tl.full([1], 0.044715, tl.float32)
    if mode == 2:
        tmp9 = libdevice.fma(tmp7, tmp8, 0.0)
    else:
        tmp9 = tmp7 * tmp8
    tmp10 = tmp5 + tmp9
    tmp11 = tl.full([1], 0.7978845608028654, tl.float32)
    tmp12 = tmp10 * tmp11
    if mode == 0 or mode == 5:
        tmp13 = libdevice.tanh(tmp12)
    elif mode == 6:
        tmp13 = tl.load(tanh_override_ptr + offset, mask=mask, other=0.0)
    else:
        # This is intentionally a separate tanh-source intervention.  It does
        # not claim to be the compiler's exact exp lowering.
        tmp13 = 2.0 / (1.0 + libdevice.exp(-2.0 * tmp12)) - 1.0
    tmp14 = tl.full([1], 1.0, tl.float32)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0.5, tl.float32)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp5 * tmp16
    tmp19 = tmp13 * tmp13
    tmp20 = tmp14 - tmp19
    tmp21 = tmp18 * tmp20
    tmp22 = tl.full([1], 0.134145, tl.float32)
    tmp23 = tmp6 * tmp22
    if mode == 3:
        tmp24 = libdevice.fma(tmp6, tmp22, tmp14)
    else:
        tmp24 = tmp23 + tmp14
    tmp25 = tmp24 * tmp11
    tmp26 = tmp21 * tmp25
    tmp27 = tmp17 + tmp26
    if mode == 7 or mode == 9:
        tmp27 = tl.load(derivative_override_ptr + offset, mask=mask, other=0.0)
    if mode == 4:
        tmp28 = libdevice.fma(tmp3, tmp27, 0.0)
    elif mode == 8 or mode == 9:
        tmp28 = tl.load(product_override_ptr + offset, mask=mask, other=0.0)
    else:
        tmp28 = tmp3 * tmp27
    tmp29 = tmp28.to(tl.float32)
    tl.store(out_ptr0 + offset, tmp29, mask=mask)


def _write_from_grad(grad: torch.Tensor, lr: float = 1e-4) -> torch.Tensor:
    return -lr * grad.float() / (grad.float().abs() + 1e-8)


def _find_target(model: torch.nn.Module) -> tuple[str, torch.nn.Parameter]:
    named = list(model.named_parameters())
    preferred = [
        (name, value) for name, value in named
        if name == "model.language_model.per_layer_model_projection.weight"
    ]
    if not preferred:
        preferred = [
            (name, value) for name, value in named
            if ".mlp.gate_proj.weight" in name and ".layers." in name
        ]
    if not preferred:
        raise RuntimeError("Gemma4 gate projection parameter was not found")
    return preferred[-1]


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
    configure_candidate_runtime(27001)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True,
    ).to(device).train()
    model.config.use_cache = False
    target_name, target_param = _find_target(model)
    start = len(PyCodeCache.modules)
    compiled = torch.compile(LossStep(model), backend="inductor", fullgraph=True, dynamic=False)
    warm = torch.tensor([states[0].get("token_ids", states[0].get("input_ids"))], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    compiled(warm).backward()
    torch.cuda.synchronize(device)
    target = None
    target_source = None
    target_offset = None
    target_stride = None
    for module in list(PyCodeCache.modules)[start:]:
        for name, value in vars(module).items():
            if "triton_poi_fused__unsafe_view_gelu_gelu_backward_mul_select_view_" in str(name) and callable(getattr(value, "run", None)):
                source = Path(getattr(module, "__file__", "")).read_text(errors="ignore")
                if f"xnumel = {TARGET_ELEMENTS}" not in source:
                    continue
                match = re.search(r"in_ptr1 \+ \((\d+) \+ x0 \+ (\d+)\*x1\)", source)
                if match is None:
                    continue
                target = value
                target_source = source
                target_offset = int(match.group(1))
                target_stride = int(match.group(2))
                break
        if target is not None:
            break
    if target is None:
        raise RuntimeError("Gemma4 GELU backward kernel was not found")
    original_run = target.run
    current_mode = "native"
    current_state_index = -1
    clone_checks: list[dict[str, float]] = []
    output_checks: list[dict[str, Any]] = []
    override_checks: list[dict[str, Any]] = []
    native_output_snapshot: torch.Tensor | None = None
    reference_output_snapshot: torch.Tensor | None = None

    def wrapped(*call_args: Any, **kwargs: Any) -> Any:
        nonlocal native_output_snapshot, reference_output_snapshot
        tensors = [value for value in call_args if isinstance(value, torch.Tensor)]
        if len(tensors) != 4:
            raise RuntimeError(f"unexpected GELU pointer ABI: {len(tensors)} tensors")
        if current_mode in ("native", "native_repeat"):
            result = original_run(*call_args, **kwargs)
            if current_mode == "native":
                torch.cuda.synchronize(tensors[3].device)
                native_output_snapshot = tensors[3].detach().clone()
            return result
        assert target_offset is not None and target_stride is not None
        required_multiplier_storage = target_offset + target_stride * ((TARGET_ELEMENTS // TARGET_WIDTH) - 1) + TARGET_WIDTH
        if tensors[1].numel() < required_multiplier_storage:
            raise RuntimeError(
                "TARGET_LAYOUT_NOT_AVAILABLE: the freshly compiled same-family "
                "kernel does not expose the historical packed multiplier layout"
            )
        n = int(tensors[0].numel())
        if n != 32768:
            raise RuntimeError(f"TARGET_LAYOUT_NOT_AVAILABLE: unexpected endpoint extent {n}")
        if current_mode == "torch_reference":
            rows = TARGET_ELEMENTS // TARGET_WIDTH
            multiplier = tensors[1].detach().reshape(rows, target_stride)[
                :, target_offset:target_offset + TARGET_WIDTH
            ].reshape(-1)
            derivative = evaluate_gelu_backward(
                tensors[0].detach().reshape(-1),
                multiplier,
                tensors[2].detach().reshape(-1),
                variant="FP32_NATIVE",
                output_dtype=tensors[3].dtype,
            )
            tensors[3].copy_(derivative.reshape_as(tensors[3]))
            torch.cuda.synchronize(tensors[3].device)
            reference_output_snapshot = tensors[3].detach().clone()
            return None
        reference_output = None
        input_before = None
        if current_mode == "native_clone":
            input_before = [tensor.detach().clone() for tensor in tensors[:3]]
            reference_output = torch.empty_like(tensors[3])
            reference_call = list(call_args)
            tensor_positions = [i for i, value in enumerate(reference_call) if isinstance(value, torch.Tensor)]
            reference_call[tensor_positions[-1]] = reference_output
            original_run(*reference_call, **kwargs)
            torch.cuda.synchronize(tensors[3].device)
            # Keep the real output tensor connected to the compiled graph.
            # Redirecting the output pointer without copying it back makes a
            # seemingly successful clone bypass every downstream consumer and
            # is therefore not a valid path-preserving intervention.
            tensors[3].copy_(reference_output)
        mode = {"native_clone": 5, "exp_tanh": 1, "polynomial_fma": 2, "derivative_factor_fma": 3, "derivative_fma": 4, "reference_tanh": 6, "reference_derivative": 7, "reference_product": 8, "reference_derivative_product": 9}[current_mode]
        # Match the reviewed generated kernel's 256-element pointwise tile;
        # using a different launch geometry can perturb downstream atomic
        # accumulation even when the stored BF16 output is unchanged.
        intervention_block = int(args.intervention_block)
        rows = TARGET_ELEMENTS // TARGET_WIDTH
        multiplier = tensors[1].detach().reshape(rows, target_stride)[
            :, target_offset:target_offset + TARGET_WIDTH
        ].reshape(-1).float()
        saved_input = tensors[2].detach().reshape(-1).float()
        gradient = tensors[0].detach().reshape(-1).float()
        x2 = saved_input * saved_input
        # This is sqrt(2/pi), not 2/sqrt(pi).  The latter is a superficially
        # similar but materially different constant and would make the host
        # override a different GELU than the reviewed generated kernel.
        gelu_scale = math.sqrt(2.0 / math.pi)
        argument = (saved_input + 0.044715 * (x2 * saved_input)) * gelu_scale
        tanh_override = torch.tanh(argument).contiguous()
        one = torch.ones_like(saved_input)
        polynomial = one + 0.134145 * x2
        derivative_override = (
            0.5 * (one + tanh_override)
            + (0.5 * saved_input) * (one - tanh_override * tanh_override)
            * polynomial * gelu_scale
        ).contiguous()
        product_override = ((gradient * multiplier) * derivative_override).contiguous()
        expected_reference = evaluate_gelu_backward(
            gradient, multiplier, saved_input, variant="FP32_NATIVE", output_dtype=tensors[3].dtype,
        ).reshape(-1)
        override_checks.append({
            "state_index": current_state_index,
            "max_abs_product_vs_reference": float((product_override.to(tensors[3].dtype) - expected_reference).abs().max().cpu()),
            "relative_l2_product_vs_reference": float(
                torch.linalg.vector_norm(product_override.to(tensors[3].dtype) - expected_reference)
                / torch.clamp(torch.linalg.vector_norm(expected_reference), min=1e-30)
            ),
        })
        _gelu_intervention[(triton.cdiv(n, intervention_block),)](
            tensors[0], tensors[1], tensors[2], tensors[3],
            tanh_override, derivative_override, product_override,
            mode=mode, xnumel=n, XBLOCK=intervention_block,
            # The historical generated kernel uses a packed multiplier view;
            # pass its layout through the kernel constants rather than assuming
            # one fixed offset across checkpoints.
            multiplier_offset=target_offset,
            multiplier_stride=target_stride,
        )
        if reference_output is not None:
            torch.cuda.synchronize(tensors[3].device)
            delta = tensors[3].float() - reference_output.float()
            input_deltas = [
                float((tensor.float() - before.float()).abs().max().cpu())
                for tensor, before in zip(tensors[:3], input_before or [])
            ]
            clone_checks.append({
                "max_abs": float(delta.abs().max().cpu()),
                "relative_l2": float(torch.linalg.vector_norm(delta) / torch.clamp(torch.linalg.vector_norm(reference_output.float()), min=1e-30)),
                "input_max_abs": max(input_deltas) if input_deltas else 0.0,
                "data_ptrs": [int(tensor.data_ptr()) for tensor in tensors],
            })
        torch.cuda.synchronize(tensors[3].device)
        output = tensors[3].detach().float()
        check = {
            "state_index": current_state_index,
            "mode": current_mode,
            "max_abs_vs_native": None,
            "max_abs_vs_reference": None,
            "relative_l2_vs_native": None,
            "relative_l2_vs_reference": None,
        }
        if native_output_snapshot is not None:
            delta = output - native_output_snapshot.float()
            check["max_abs_vs_native"] = float(delta.abs().max().cpu())
            check["relative_l2_vs_native"] = float(
                torch.linalg.vector_norm(delta)
                / torch.clamp(torch.linalg.vector_norm(native_output_snapshot.float()), min=1e-30)
            )
        if reference_output_snapshot is not None:
            delta = output - reference_output_snapshot.float()
            check["max_abs_vs_reference"] = float(delta.abs().max().cpu())
            check["relative_l2_vs_reference"] = float(
                torch.linalg.vector_norm(delta)
                / torch.clamp(torch.linalg.vector_norm(reference_output_snapshot.float()), min=1e-30)
            )
        output_checks.append(check)
        return None

    target.run = wrapped
    rows: list[dict[str, Any]] = []
    try:
        for index, state in enumerate(states):
            current_state_index = index
            values = torch.tensor([state.get("token_ids", state.get("input_ids"))], dtype=torch.long, device=device)
            current_mode = "native"
            model.zero_grad(set_to_none=True)
            native_loss = compiled(values)
            native_loss.backward()
            torch.cuda.synchronize(device)
            native_grad = target_param.grad.detach().float().cpu().clone()
            native_write = _write_from_grad(native_grad)
            row = {
                "state_index": index,
                "source_state_index": selected_indices[index],
                "source_state_id": str(state.get("state_id", selected_indices[index])),
                "loss_native": float(native_loss.detach().cpu()),
            }
            all_variant_names = (
                "native_repeat", "torch_reference", "native_clone", "exp_tanh",
                "polynomial_fma", "derivative_factor_fma", "derivative_fma",
                "reference_tanh", "reference_derivative", "reference_product",
                "reference_derivative_product",
            )
            variant_names = ("torch_reference",) if args.mean_only else all_variant_names
            variant_writes: dict[str, torch.Tensor] = {}
            for variant in variant_names:
                current_mode = variant
                model.zero_grad(set_to_none=True)
                repaired_loss = compiled(values)
                repaired_loss.backward()
                torch.cuda.synchronize(device)
                repaired_grad = target_param.grad.detach().float().cpu().clone()
                repaired_write = _write_from_grad(repaired_grad)
                variant_writes[variant] = repaired_write
                effect = native_write - repaired_write
                row[f"loss_{variant}"] = float(repaired_loss.detach().cpu())
                row[f"gradient_effect_rms_over_{variant}"] = float(torch.linalg.vector_norm(native_grad - repaired_grad)) / max(float(torch.linalg.vector_norm(repaired_grad)), 1e-30)
                row[f"write_effect_rms_over_{variant}"] = float(torch.linalg.vector_norm(effect)) / max(float(torch.linalg.vector_norm(repaired_write)), 1e-30)
                row[f"write_aligned_over_{variant}"] = float(torch.sum(effect * repaired_write)) / max(float(torch.sum(repaired_write * repaired_write)), 1e-30)
            target_effect = native_write - variant_writes["torch_reference"]
            target_norm = float(torch.linalg.vector_norm(target_effect))
            row["candidate_reference_write_effect_norm"] = target_norm
            for variant in variant_names[2:]:
                intervention_effect = native_write - variant_writes[variant]
                intervention_norm = float(torch.linalg.vector_norm(intervention_effect))
                inner = float(torch.sum(target_effect * intervention_effect))
                row[f"mediation_{variant}"] = {
                    "effect_norm_ratio_to_candidate_reference": intervention_norm / max(target_norm, 1e-30),
                    "cosine_to_candidate_reference": inner / max(target_norm * intervention_norm, 1e-30),
                    "residual_norm_ratio": float(torch.linalg.vector_norm(target_effect - intervention_effect)) / max(target_norm, 1e-30),
                    "exact_parameter_write": bool(torch.equal(variant_writes[variant], variant_writes["torch_reference"])),
                }
            rows.append(row)
    finally:
        target.run = original_run
    return {
        "schema": "kernel-analyzer-gelu-single-source-intervention-v4",
        "status": "COMPLETE_PATH_PRESERVING_GELU_MEDIATION_INTERVENTIONS_V4",
        "model": str(args.model),
        "target_parameter": target_name,
        "operator": "Gemma4 generated GELU backward",
        "candidate": "native generated tanh-GELU expression",
        "intervention": "same generated temporary association and write boundary with one arithmetic boundary changed at a time, plus same-input FP32 tanh, derivative-factor and product intermediate overrides",
        "state_count": len(rows),
        "target_layout": {
            "elements": TARGET_ELEMENTS,
            "width": TARGET_WIDTH,
            "multiplier_offset": target_offset,
            "multiplier_stride": target_stride,
        },
        "sampling": {
            "input_bank": str(args.input_bank),
            "source_bank_state_count": len(bank_states),
            "source_state_indices": selected_indices,
            "with_replacement": bool(args.state_indices),
            "mean_only": bool(args.mean_only),
        },
        "native_clone_validation": clone_checks,
        "override_checks": override_checks,
        "output_checks": output_checks,
        "rows": rows,
        "summary": {
            variant: {
                "write_effect_rms_mean": sum(row[f"write_effect_rms_over_{variant}"] for row in rows) / len(rows),
                "write_aligned_interval": [min(row[f"write_aligned_over_{variant}"] for row in rows), max(row[f"write_aligned_over_{variant}"] for row in rows)],
                "loss_difference_mean": sum(row["loss_native"] - row[f"loss_{variant}"] for row in rows) / len(rows),
            }
            for variant in (("torch_reference",) if args.mean_only else ("native_repeat", "torch_reference", "native_clone", "exp_tanh", "polynomial_fma", "derivative_factor_fma", "derivative_fma", "reference_tanh", "reference_derivative", "reference_product", "reference_derivative_product"))
        },
        "mediation_summary": {
            variant: _aggregate_mediation(rows, variant)
            for variant in (() if args.mean_only else ("native_clone", "exp_tanh", "polynomial_fma", "derivative_factor_fma", "derivative_fma", "reference_tanh", "reference_derivative", "reference_product", "reference_derivative_product"))
        },
        "claim_boundary": "This is a same-checkpoint natural-state path intervention; it can close only the tested tanh source if the intervention changes the original profile as predicted.",
}


def _aggregate_mediation(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    """Aggregate mediation over vectors, rather than averaging ratios per state.

    A state with a zero candidate/reference effect has no defined direction.  A
    plain mean of per-state ratios silently turns such rows into zeros and can
    make a complete mediation look partial (the previous report showed 0.125
    for one non-zero row out of eight).  The aggregate below is the ratio of
    concatenated squared norms and inner products; the exact-write count and
    nonzero target count remain explicit.
    """
    target_sq = 0.0
    intervention_sq = 0.0
    residual_sq = 0.0
    inner = 0.0
    exact = 0
    nonzero = 0
    for row in rows:
        target = float(row["candidate_reference_write_effect_norm"])
        med = row[f"mediation_{variant}"]
        ratio = float(med["effect_norm_ratio_to_candidate_reference"])
        residual_ratio = float(med["residual_norm_ratio"])
        target_sq += target * target
        intervention_sq += (target * ratio) ** 2
        residual_sq += (target * residual_ratio) ** 2
        inner += (target * target) * float(med["cosine_to_candidate_reference"]) * ratio
        exact += int(bool(med["exact_parameter_write"]))
        nonzero += int(target > 0.0)
    denom = max(target_sq, 1e-30)
    cosine = inner / max((target_sq * intervention_sq) ** 0.5, 1e-30)
    return {
        "effect_norm_ratio_aggregate": (intervention_sq / denom) ** 0.5,
        "cosine_aggregate": max(-1.0, min(1.0, cosine)),
        "residual_norm_ratio_aggregate": (residual_sq / denom) ** 0.5,
        "exact_write_count": exact,
        "nonzero_target_count": nonzero,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--input-bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument(
        "--state-indices", default=None,
        help="comma-separated indices sampled from the input bank; repeats are allowed",
    )
    parser.add_argument(
        "--mean-only", action="store_true",
        help="measure only the native/reference write effect for a population mean probe",
    )
    parser.add_argument("--intervention-block", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
