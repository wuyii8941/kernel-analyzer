#!/usr/bin/env python3
"""Cut the actual Mamba fused layer at the selective-scan boundary.

The fused branch keeps the installed causal-convolution and projection calls,
but replaces only the scan kernel with an explicit differentiable recurrence.
If the candidate-versus-cut effect agrees with the local fused-versus-sequential
effect, the source can be assigned to the scan/reassociation boundary rather
than to the entire Mamba region.
"""

from __future__ import annotations

import argparse
import json
import math
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import MambaForCausalLM
from transformers.models.mamba import modeling_mamba


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/data1/tzh/models/state-spaces/mamba-130m-hf")
DEFAULT_BANK = ROOT / "results/coverage/mamba_seq128_input_bank.json"


def interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    if x.numel() < 2:
        value = float(x.mean().item()) if x.numel() else 0.0
        return [value, value]
    mean = float(x.mean().item())
    half = 1.96 * float(x.std(unbiased=True).item()) / math.sqrt(x.numel())
    return [mean - half, mean + half]


def write(gradient: torch.Tensor, learning_rate: float, eps: float) -> torch.Tensor:
    value = gradient.float()
    return -learning_rate * value / (value.abs() + eps)


def relative(effect: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(effect).item()) / max(float(torch.linalg.vector_norm(reference).item()), 1e-30)


def load_states(path: Path, count: int) -> list[torch.Tensor]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    states = payload.get("states", payload if isinstance(payload, list) else None)
    if not states or len(states) < count:
        raise RuntimeError(f"input bank has fewer than {count} states")
    return [torch.tensor(row["token_ids"], dtype=torch.long) for row in states[:count]]


def python_scan_inner(
    xz: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv1d_bias: torch.Tensor | None,
    x_proj_weight: torch.Tensor,
    delta_proj_weight: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor | None,
    A: torch.Tensor,
    B: torch.Tensor | None = None,
    C: torch.Tensor | None = None,
    D: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    B_proj_bias: torch.Tensor | None = None,
    C_proj_bias: torch.Tensor | None = None,
    delta_softplus: bool = True,
    **_: Any,
) -> torch.Tensor:
    """MambaInnerFn-compatible path with only selective scan replaced."""
    _, dim2, length = xz.shape
    dtype = xz.dtype
    rank = delta_proj_weight.shape[1]
    d_state = A.shape[-1]
    conv_weight = conv1d_weight.reshape(conv1d_weight.shape[0], conv1d_weight.shape[2])
    x, z = xz.chunk(2, dim=1)
    from causal_conv1d import causal_conv1d_fn

    conv_out = causal_conv1d_fn(x, conv_weight, conv1d_bias, None, None, False, None, "silu")
    x_dbl = F.linear(rearrange(conv_out, "b d l -> (b l) d"), x_proj_weight)
    delta = rearrange(delta_proj_weight @ x_dbl[:, :rank].t(), "d (b l) -> b d l", l=length)
    B = x_dbl[:, rank : rank + d_state]
    C = x_dbl[:, -d_state:]
    if B_proj_bias is not None:
        B = B + B_proj_bias.to(B.dtype)
    if C_proj_bias is not None:
        C = C + C_proj_bias.to(C.dtype)
    B = rearrange(B, "(b l) s -> b 1 s l", l=length)
    C = rearrange(C, "(b l) s -> b 1 s l", l=length)
    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = F.softplus(delta)

    state = torch.zeros(
        (xz.shape[0], conv_out.shape[1], d_state), device=xz.device, dtype=torch.float32
    )
    A_float = A.float()
    D_float = D.float() if D is not None else None
    outputs = []
    for index in range(length):
        dt = delta[:, :, index].float()
        transition = torch.exp(dt[:, :, None] * A_float[None, :, :])
        state = transition * state + dt[:, :, None] * B[:, :, :, index].float() * conv_out[:, :, index].float()[:, :, None]
        value = torch.sum(state * C[:, :, :, index].float(), dim=-1)
        if D_float is not None:
            value = value + D_float[None, :] * conv_out[:, :, index].float()
        outputs.append(value.to(dtype))
    out = torch.stack(outputs, dim=-1)
    out_z = out * F.silu(z)
    return F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)


def install_branch(model: MambaForCausalLM, layer: int, branch: str) -> list[tuple[torch.nn.Module, Any]]:
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update

    modeling_mamba.mamba_inner_fn = mamba_inner_fn if branch == "fused" else python_scan_inner
    modeling_mamba.selective_scan_fn = selective_scan_fn
    modeling_mamba.selective_state_update = selective_state_update
    modeling_mamba._causal_conv1d_cache = None
    saved: list[tuple[torch.nn.Module, Any]] = []
    for index, block in enumerate(model.backbone.layers):
        mixer = block.mixer
        original = mixer.forward
        method = mixer.cuda_kernels_forward if index == layer else mixer.slow_forward
        mixer.forward = types.MethodType(
            lambda self, hidden_states, cache_params=None, cache_position=None, attention_mask=None, _method=method: _method(
                hidden_states, cache_params, cache_position, attention_mask
            ),
            mixer,
        )
        saved.append((mixer, original))
    return saved


def restore(saved: list[tuple[torch.nn.Module, Any]]) -> None:
    for module, original in saved:
        module.forward = original


def run_once(model: MambaForCausalLM, target: torch.nn.Parameter, ids: torch.Tensor, layer: int, branch: str) -> tuple[float, torch.Tensor]:
    saved = install_branch(model, layer, branch)
    try:
        model.train(True)
        model.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        loss.backward()
        if target.grad is None:
            raise RuntimeError("target gradient is missing")
        return float(loss.detach().cpu()), target.grad.detach().float().cpu().clone()
    finally:
        restore(saved)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    states = load_states(args.input_bank, args.states)
    model = MambaForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, local_files_only=True).to(args.device)
    model.config.use_cache = False
    target_name = args.parameter or f"backbone.layers.{args.layer}.mixer.out_proj.weight"
    target = dict(model.named_parameters())[target_name]
    rows: list[dict[str, Any]] = []
    effects: list[torch.Tensor] = []
    cut_effects: list[torch.Tensor] = []
    calibration_count = min(args.calibration_count, args.states // 2)
    for state_id, token_ids in enumerate(states):
        ids = token_ids.unsqueeze(0).to(args.device)
        # Explicit sequential all-layer reference.
        model.train(False)
        model.zero_grad(set_to_none=True)
        ref_loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        ref_loss.backward()
        reference_grad = target.grad.detach().float().cpu().clone()
        reference_loss = float(ref_loss.detach().cpu())
        fused_loss, fused_grad = run_once(model, target, ids, args.layer, "fused")
        cut_loss, cut_grad = run_once(model, target, ids, args.layer, "cut")
        fused_write = write(fused_grad, args.learning_rate, args.eps)
        cut_write = write(cut_grad, args.learning_rate, args.eps)
        reference_write = write(reference_grad, args.learning_rate, args.eps)
        effect = fused_write - reference_write
        cut_effect = cut_write - reference_write
        effects.append(effect.double().reshape(-1))
        cut_effects.append(cut_effect.double().reshape(-1))
        rows.append({
            "state_id": state_id,
            "reference_loss": reference_loss,
            "fused_loss": fused_loss,
            "cut_loss": cut_loss,
            "fused_minus_reference_loss": fused_loss - reference_loss,
            "cut_minus_reference_loss": cut_loss - reference_loss,
            "fused_write_effect_rms_over_reference": relative(effect, reference_write),
            "cut_write_effect_rms_over_reference": relative(cut_effect, reference_write),
            "fused_aligned_write_ratio": float(torch.dot(effect.double().reshape(-1), reference_write.double().reshape(-1)).item())
            / max(float(torch.dot(reference_write.double().reshape(-1), reference_write.double().reshape(-1)).item()), 1e-30),
            "cut_aligned_write_ratio": float(torch.dot(cut_effect.double().reshape(-1), reference_write.double().reshape(-1)).item())
            / max(float(torch.dot(reference_write.double().reshape(-1), reference_write.double().reshape(-1)).item()), 1e-30),
            "fused_vs_cut_effect_relative_l2": relative(effect - cut_effect, effect),
            "fused_vs_cut_effect_cosine": float(torch.dot(effect.double().reshape(-1), cut_effect.double().reshape(-1)).item()) / max(float(torch.linalg.vector_norm(effect.double().reshape(-1)).item()) * float(torch.linalg.vector_norm(cut_effect.double().reshape(-1)).item()), 1e-30),
        })
        del ids, reference_grad, fused_grad, cut_grad
        torch.cuda.empty_cache()

    raw = sum(effects[:calibration_count], torch.zeros_like(effects[0]))
    norm = float(torch.linalg.vector_norm(raw).item())
    direction = raw / norm if norm else None
    heldout_rows = rows[calibration_count:]
    fused_aligned = [row["fused_aligned_write_ratio"] for row in heldout_rows]
    cut_aligned = [row["cut_aligned_write_ratio"] for row in heldout_rows]
    fused_proj = [float(torch.dot(effect, direction).item()) for effect in effects[calibration_count:]] if direction is not None else []
    output = {
        "schema": "kernel-analyzer-mamba-scan-source-cut-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "input_bank": str(args.input_bank),
        "layer": args.layer,
        "target_parameter": target_name,
        "candidate": "actual installed mamba_ssm fused scan in selected layer",
        "source_cut": "same selected-layer causal convolution/projection inputs, explicit differentiable sequential selective scan",
        "reference": "explicit sequential recurrence in every layer",
        "state_count": len(rows),
        "calibration_count": calibration_count,
        "confirmation_count": len(heldout_rows),
        "rows": rows,
        "summary": {
            "fused_aligned_mean": sum(fused_aligned) / len(fused_aligned),
            "fused_aligned_interval_normal_95": interval(fused_aligned),
            "fused_aligned_negative": sum(value < 0 for value in fused_aligned),
            "fused_aligned_positive": sum(value > 0 for value in fused_aligned),
            "cut_aligned_mean": sum(cut_aligned) / len(cut_aligned),
            "cut_aligned_interval_normal_95": interval(cut_aligned),
            "cut_aligned_negative": sum(value < 0 for value in cut_aligned),
            "cut_aligned_positive": sum(value > 0 for value in cut_aligned),
            "fused_vs_cut_effect_relative_l2_mean": sum(row["fused_vs_cut_effect_relative_l2"] for row in rows) / len(rows),
            "fused_vs_cut_effect_cosine_mean": sum(row["fused_vs_cut_effect_cosine"] for row in rows) / len(rows),
            "fused_frozen_direction_mean": sum(fused_proj) / len(fused_proj) if fused_proj else None,
            "fused_frozen_direction_interval_normal_95": interval(fused_proj) if fused_proj else None,
        },
        "claim_boundary": "The cut identifies the selective-scan boundary only if the cut reproduces the fused effect while preserving the selected-layer convolution and projections; it remains a fixed-bank source result, not a population or loss claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": output["summary"]}, ensure_ascii=False, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--input-bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--parameter", default=None)
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--calibration-count", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
