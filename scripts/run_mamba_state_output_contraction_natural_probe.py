#!/usr/bin/env python3
"""Natural Mamba state-to-output contraction materialization probe.

The candidate is the native sequential selective-scan path.  The reference
keeps the recurrent state and C vector in FP32 for the state-to-output dot
product, then casts that output back before the original D/z/output path.  The
softplus, state recurrence, and all model inputs are otherwise unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, MambaForCausalLM
from transformers.models.mamba import modeling_mamba


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/data1/tzh/models/state-spaces/mamba-130m-hf")
DEFAULT_TEXT = ROOT / "README.md"


def windows(tokenizer: Any, text_path: Path, seq_len: int, count: int) -> list[torch.Tensor]:
    tokens = tokenizer(text_path.read_text(encoding="utf-8"), add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    stride = seq_len * 2
    need = count * stride + seq_len
    if tokens.numel() < need:
        raise RuntimeError(f"text source has {tokens.numel()} tokens, need {need}")
    return [tokens[i * stride : i * stride + seq_len].clone() for i in range(count)]


def interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    if x.numel() < 2:
        m = float(x.mean().item()); return [m, m]
    m = float(x.mean().item())
    h = 1.96 * float(x.std(unbiased=True).item()) / (x.numel() ** 0.5)
    return [m - h, m + h]


def write(g: torch.Tensor, lr: float, eps: float) -> torch.Tensor:
    g = g.float()
    return -lr * g / (g.abs() + eps)


def relative(effect: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(effect).item()) / max(float(torch.linalg.vector_norm(reference).item()), 1e-30)


def fp32_state_output_scan(
    hidden_states: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
    **kwargs: Any,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Sequential scan with only the state-to-C contraction kept in FP32."""
    batch_size, intermediate_size, seq_len = hidden_states.shape
    input_dtype = hidden_states.dtype
    if delta_bias is not None:
        dt = dt + delta_bias.to(dt.dtype)[..., None]
    if delta_softplus:
        dt = modeling_mamba.F.softplus(dt)
    B = B.transpose(1, 2)
    C = C.transpose(1, 2)
    discrete_A = torch.exp(A[None, :, None, :] * dt[:, :, :, None])
    discrete_B = dt[:, :, :, None] * B[:, None, :, :].float()
    deltaB_u = discrete_B * hidden_states[:, :, :, None].float()
    ssm_state = torch.zeros(batch_size, intermediate_size, A.shape[-1], dtype=input_dtype, device=hidden_states.device)
    scan_outputs = []
    for index in range(seq_len):
        ssm_state = discrete_A[:, :, index] * ssm_state + deltaB_u[:, :, index]
        # The sole intervention: FP32 state/C dot, then the same native-dtype
        # materialization used before the downstream D and z paths.
        scan_output = torch.matmul(ssm_state.float(), C[:, index, :].float().unsqueeze(-1)).squeeze(-1)
        scan_outputs.append(scan_output.to(input_dtype))
    scan_output = torch.stack(scan_outputs, dim=-1)
    if D is not None:
        scan_output = scan_output + hidden_states * D[None, :, None]
    if z is not None:
        scan_output = scan_output * modeling_mamba.F.silu(z)
    if return_last_state:
        return scan_output, ssm_state
    return scan_output


def fp32_skip_scan(
    hidden_states: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
    **kwargs: Any,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Sequential scan with only the residual D*hidden_states product in FP32."""
    batch_size, intermediate_size, seq_len = hidden_states.shape
    input_dtype = hidden_states.dtype
    if delta_bias is not None:
        dt = dt + delta_bias.to(dt.dtype)[..., None]
    if delta_softplus:
        dt = modeling_mamba.F.softplus(dt)
    B = B.transpose(1, 2)
    C = C.transpose(1, 2)
    discrete_A = torch.exp(A[None, :, None, :] * dt[:, :, :, None])
    discrete_B = dt[:, :, :, None] * B[:, None, :, :].float()
    deltaB_u = discrete_B * hidden_states[:, :, :, None].float()
    ssm_state = torch.zeros(batch_size, intermediate_size, A.shape[-1], dtype=input_dtype, device=hidden_states.device)
    scan_outputs = []
    for index in range(seq_len):
        ssm_state = discrete_A[:, :, index] * ssm_state + deltaB_u[:, :, index]
        scan_output = torch.matmul(ssm_state.to(input_dtype), C[:, index, :].unsqueeze(-1)).squeeze(-1)
        scan_outputs.append(scan_output)
    scan_output = torch.stack(scan_outputs, dim=-1)
    if D is not None:
        # The sole intervention: preserve the native state/C contraction, but
        # materialize only the residual skip product in FP32 before write-back.
        scan_output = scan_output + (hidden_states.float() * D.float()[None, :, None]).to(input_dtype)
    if z is not None:
        scan_output = scan_output * modeling_mamba.F.silu(z)
    if return_last_state:
        return scan_output, ssm_state
    return scan_output


def run_once(model: torch.nn.Module, target: torch.nn.Parameter, ids: torch.Tensor, mode: str) -> tuple[float, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    original = modeling_mamba.mamba_selective_scan
    if mode == "fp32":
        modeling_mamba.mamba_selective_scan = fp32_state_output_scan
    elif mode == "fp32_d":
        modeling_mamba.mamba_selective_scan = fp32_skip_scan
    try:
        loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        loss.backward()
    finally:
        modeling_mamba.mamba_selective_scan = original
    if target.grad is None:
        raise RuntimeError("target gradient is missing")
    return float(loss.detach().cpu().item()), target.grad.detach().float().cpu().clone()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    bank = windows(tokenizer, args.text_source, args.sequence_length, args.states)
    model = MambaForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, local_files_only=True).to(args.device).eval()
    model.config.use_cache = False
    named = dict(model.named_parameters())
    parameter = f"backbone.layers.{args.layer}.mixer.{args.parameter}"
    target = named[parameter]
    effects: list[torch.Tensor] = []
    writes: list[torch.Tensor] = []
    refs: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    for state_id, tokens in enumerate(bank):
        ids = tokens.unsqueeze(0).to(args.device)
        native_loss, native_grad = run_once(model, target, ids, "native")
        intervention_mode = "fp32_d" if args.boundary == "d_skip" else "fp32"
        fp32_loss, fp32_grad = run_once(model, target, ids, intervention_mode)
        native_write = write(native_grad, args.learning_rate, args.eps)
        fp32_write = write(fp32_grad, args.learning_rate, args.eps)
        grad_effect = fp32_grad - native_grad
        write_effect = fp32_write - native_write
        effects.append(grad_effect.double().reshape(-1))
        writes.append(write_effect.double().reshape(-1))
        refs.append(native_write.double().reshape(-1))
        rows.append({
            "state_id": state_id,
            "native_loss": native_loss,
            "fp32_state_output_loss": fp32_loss,
            "loss_difference_fp32_minus_native": fp32_loss - native_loss,
            "gradient_effect_rms_over_native": relative(grad_effect, native_grad),
            "write_effect_rms_over_native": relative(write_effect, native_write),
        })
        del ids
        torch.cuda.empty_cache()
    split = args.states // 2
    direction = torch.stack(effects[:split]).mean(dim=0)
    norm = float(torch.linalg.vector_norm(direction).item())
    projections: list[float] = []
    if norm:
        direction = direction / norm
        projections = [float(torch.dot(value, direction).item()) for value in effects[split:]]
    aligned = [float(torch.dot(effect, ref).item()) / max(float(torch.dot(ref, ref).item()), 1e-30) for effect, ref in zip(writes[split:], refs[split:])]
    return {
        "schema": "kernel-analyzer-mamba-state-output-contraction-natural-probe-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "operator": (
            "Mamba recurrent residual D skip-product materialization"
            if args.boundary == "d_skip"
            else "Mamba recurrent state-to-output C contraction materialization"
        ),
        "parameter": parameter,
        "candidate": "native sequential Mamba selective scan",
        "reference": (
            "same sequential scan with FP32 residual D*hidden_states product, native state/C path"
            if args.boundary == "d_skip"
            else "same sequential scan with FP32 state/C output contraction, cast before D/z path"
        ),
        "input_source": str(args.text_source),
        "comparison_scope": {"same_model_weights": True, "same_input_ids": True, "same_recurrence": True, "single_changed_boundary": args.boundary},
        "state_count": len(rows),
        "calibration_count": split,
        "confirmation_count": len(rows) - split,
        "rows": rows,
        "summary": {
            "state_count": len(rows),
            "calibration_count": split,
            "confirmation_count": len(rows) - split,
            "gradient_effect_rms_mean": sum(r["gradient_effect_rms_over_native"] for r in rows) / len(rows),
            "write_effect_rms_mean": sum(r["write_effect_rms_over_native"] for r in rows) / len(rows),
            "confirmation_projection_mean": sum(projections) / len(projections) if projections else None,
            "confirmation_projection_interval_normal_95": interval(projections) if projections else None,
            "confirmation_projection_positive": sum(v > 0 for v in projections),
            "confirmation_projection_negative": sum(v < 0 for v in projections),
            "aligned_write_mean": sum(aligned) / len(aligned) if aligned else None,
            "aligned_write_interval_normal_95": interval(aligned) if aligned else None,
            "direction_norm": norm,
        },
        "claim_boundary": "One real Mamba checkpoint, one sequential state-to-output boundary and the declared text bank; not a population or long-run loss claim.",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--text-source", type=Path, default=DEFAULT_TEXT)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--parameter", default="A_log")
    p.add_argument("--boundary", choices=("state_output", "d_skip"), default="state_output")
    p.add_argument("--sequence-length", type=int, default=64)
    p.add_argument("--states", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **result["summary"]}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
