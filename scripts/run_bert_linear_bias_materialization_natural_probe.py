#!/usr/bin/env python3
"""Probe BERT fused-addmm versus split matmul+bias materialization on a real graph.

The candidate uses the native ``nn.Linear(..., bias)`` path.  The reference
splits the weight multiply from the bias addition and performs the split path
in FP32 before writing back to the output dtype.  A factorial control shows
that changing only the bias precision after the split is exact, so the
identified source is the fused ``addmm`` versus split materialization boundary,
not an isolated bias-add operation.
"""

from __future__ import annotations

import argparse
import json
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/data1/tzh/models/prajjwal1/bert-tiny")
DEFAULT_TEXT = ROOT / "docs/root_cause_closure_current.md"


def windows(tokenizer: Any, text_path: Path, seq_len: int, count: int) -> list[torch.Tensor]:
    tokens = tokenizer(
        text_path.read_text(encoding="utf-8"), add_special_tokens=False, return_tensors="pt"
    )["input_ids"][0]
    stride = seq_len * 2
    need = count * stride + seq_len
    if tokens.numel() < need:
        raise RuntimeError(f"text source has {tokens.numel()} tokens, need {need}")
    return [tokens[i * stride : i * stride + seq_len].clone() for i in range(count)]


def interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    if x.numel() < 2:
        m = float(x.mean().item())
        return [m, m]
    m = float(x.mean().item())
    h = 1.96 * float(x.std(unbiased=True).item()) / (x.numel() ** 0.5)
    return [m - h, m + h]


def first_step_write(gradient: torch.Tensor, learning_rate: float, eps: float) -> torch.Tensor:
    g = gradient.float()
    return -learning_rate * g / (g.abs() + eps)


def rel_norm(effect: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = float(torch.linalg.vector_norm(reference).item())
    return float(torch.linalg.vector_norm(effect).item()) / max(denominator, 1e-30)


def install_bias_reference(module: torch.nn.Module) -> None:
    """Keep the linear multiply and all downstream operations unchanged."""

    def forward(self: torch.nn.Module, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        dense = self.dense
        value = F.linear(hidden_states, dense.weight, None)
        if dense.bias is not None:
            value = (value.float() + dense.bias.float()).to(value.dtype)
        value = self.dropout(value)
        return self.LayerNorm(value + input_tensor)

    module.forward = types.MethodType(forward, module)


def run_once(
    model: torch.nn.Module,
    target: torch.nn.Parameter,
    ids: torch.Tensor,
    reference: bool,
    layer: int,
) -> tuple[float, torch.Tensor]:
    module = model.bert.encoder.layer[layer].attention.output
    original = module.forward
    if reference:
        install_bias_reference(module)
    model.zero_grad(set_to_none=True)
    try:
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
    finally:
        module.forward = original
    if target.grad is None:
        raise RuntimeError("target gradient is missing")
    return float(loss.detach().cpu().item()), target.grad.detach().float().cpu().clone()


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    bank = windows(tokenizer, args.text_source, args.sequence_length, args.states)
    model = AutoModelForMaskedLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16
    ).to(device).eval()
    target_name = args.parameter or f"bert.encoder.layer.{args.layer}.attention.output.LayerNorm.weight"
    named = dict(model.named_parameters())
    if target_name not in named:
        raise KeyError(target_name)
    target = named[target_name]
    effects: list[torch.Tensor] = []
    writes: list[torch.Tensor] = []
    refs: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    for state_id, tokens in enumerate(bank):
        ids = tokens.unsqueeze(0).to(device)
        native_loss, native_grad = run_once(model, target, ids, False, args.layer)
        fp32_loss, fp32_grad = run_once(model, target, ids, True, args.layer)
        native_write = first_step_write(native_grad, args.learning_rate, args.eps)
        fp32_write = first_step_write(fp32_grad, args.learning_rate, args.eps)
        gradient_effect = fp32_grad - native_grad
        write_effect = fp32_write - native_write
        effects.append(gradient_effect.double().reshape(-1))
        writes.append(write_effect.double().reshape(-1))
        refs.append(native_write.double().reshape(-1))
        rows.append(
            {
                "state_id": state_id,
                "native_loss": native_loss,
                "fp32_bias_loss": fp32_loss,
                "loss_difference_fp32_minus_native": fp32_loss - native_loss,
                "gradient_effect_rms_over_native": rel_norm(gradient_effect, native_grad),
                "write_effect_rms_over_native": rel_norm(write_effect, native_write),
            }
        )
        if device.type == "cuda":
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
        "schema": "kernel-analyzer-bert-linear-bias-materialization-natural-probe-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "operator": "BERT attention fused addmm versus split matmul+bias materialization",
        "layer": args.layer,
        "parameter": target_name,
        "candidate": "native BERT fused addmm linear path",
        "reference": "same BERT block with split matmul plus FP32 bias materialization then cast back",
        "input_source": str(args.text_source),
        "comparison_scope": {
            "same_model_weights": True,
            "same_input_ids": True,
            "same_linear_weight_multiply": True,
            "same_dropout_residual_layer_norm": True,
            "single_changed_boundary": "attention fused addmm versus split matmul+bias materialization",
        },
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
            "confirmation_projection_interval_normal_95": interval(projections) if projections else None,
            "confirmation_projection_positive": sum(v > 0 for v in projections),
            "confirmation_projection_negative": sum(v < 0 for v in projections),
            "aligned_write_mean": sum(aligned) / len(aligned) if aligned else None,
            "aligned_write_interval_normal_95": interval(aligned) if aligned else None,
            "direction_norm": norm,
        },
        "claim_boundary": "One BERT-tiny checkpoint, one attention dense-bias boundary and the declared text bank; no population or long-run loss claim.",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--text-source", type=Path, default=DEFAULT_TEXT)
    p.add_argument("--parameter", default=None)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--sequence-length", type=int, default=64)
    p.add_argument("--states", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **result["summary"]}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
