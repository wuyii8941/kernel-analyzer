#!/usr/bin/env python3
"""Natural Mamba causal-convolution accumulation probe.

The candidate uses the model's native causal_conv1d_fn.  The reference only
changes the convolution accumulation to FP32 and casts back before applying
the original activation, so this is a single-boundary probe rather than a
whole-recurrence replacement.
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
    text = text_path.read_text(encoding="utf-8")
    tokens = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
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


def write(gradient: torch.Tensor, lr: float, eps: float) -> torch.Tensor:
    g = gradient.float()
    return -lr * g / (g.abs() + eps)


def relative(effect: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(effect).item()) / max(float(torch.linalg.vector_norm(reference).item()), 1e-30)


def fp32_conv_preserving_activation(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """FP32 convolution, then native-dtype activation and return."""
    _, _, seq_len = hidden_states.shape
    input_dtype = hidden_states.dtype
    out = torch.nn.functional.conv1d(
        hidden_states.float(),
        weight=weight.float().unsqueeze(1),
        bias=bias.float() if bias is not None else None,
        padding=weight.shape[-1] - 1,
        groups=hidden_states.shape[1],
    )[:, :, :seq_len]
    out = out.to(input_dtype)
    if activation is not None:
        out = modeling_mamba.ACT2FN[activation](out)
    return out.to(input_dtype)


def run_once(
    model: torch.nn.Module,
    target: torch.nn.Parameter,
    ids: torch.Tensor,
    mode: str,
) -> tuple[float, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    original = modeling_mamba.causal_conv1d_fn
    if mode == "fp32":
        modeling_mamba.causal_conv1d_fn = fp32_conv_preserving_activation
    try:
        loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        loss.backward()
    finally:
        modeling_mamba.causal_conv1d_fn = original
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
    target = model.backbone.layers[args.layer].mixer.conv1d.weight
    effects: list[torch.Tensor] = []
    writes: list[torch.Tensor] = []
    refs: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    for state_id, tokens in enumerate(bank):
        ids = tokens.unsqueeze(0).to(args.device)
        native_loss, native_grad = run_once(model, target, ids, "native")
        fp32_loss, fp32_grad = run_once(model, target, ids, "fp32")
        native_write = write(native_grad, args.learning_rate, args.eps)
        fp32_write = write(fp32_grad, args.learning_rate, args.eps)
        grad_effect = fp32_grad - native_grad
        write_effect = fp32_write - native_write
        effects.append(grad_effect.double().reshape(-1))
        writes.append(write_effect.double().reshape(-1))
        refs.append(native_write.double().reshape(-1))
        rows.append(
            {
                "state_id": state_id,
                "native_loss": native_loss,
                "fp32_conv_loss": fp32_loss,
                "loss_difference_fp32_minus_native": fp32_loss - native_loss,
                "gradient_effect_rms_over_native": relative(grad_effect, native_grad),
                "write_effect_rms_over_native": relative(write_effect, native_write),
            }
        )
        del ids
        torch.cuda.empty_cache()

    split = args.states // 2
    direction = torch.stack(effects[:split]).mean(dim=0)
    norm = float(torch.linalg.vector_norm(direction).item())
    projections: list[float] = []
    if norm:
        direction = direction / norm
        projections = [float(torch.dot(value, direction).item()) for value in effects[split:]]
    aligned = [
        float(torch.dot(effect, ref).item()) / max(float(torch.dot(ref, ref).item()), 1e-30)
        for effect, ref in zip(writes[split:], refs[split:])
    ]
    return {
        "schema": "kernel-analyzer-mamba-causal-conv-materialisation-natural-probe-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "operator": "Mamba causal depthwise convolution accumulation",
        "parameter": f"backbone.layers.{args.layer}.mixer.conv1d.weight",
        "candidate": "native Mamba causal_conv1d_fn",
        "reference": "same causal convolution with FP32 accumulation, cast before native activation",
        "input_source": str(args.text_source),
        "comparison_scope": {"same_model_weights": True, "same_input_ids": True, "single_changed_boundary": "causal convolution accumulation"},
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
        "claim_boundary": "One real Mamba checkpoint, one causal convolution boundary and the declared text bank; not a population or long-run loss claim.",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--text-source", type=Path, default=DEFAULT_TEXT)
    p.add_argument("--layer", type=int, default=0)
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
