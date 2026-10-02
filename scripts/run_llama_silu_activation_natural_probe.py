#!/usr/bin/env python3
"""Natural Llama SwiGLU SiLU-activation materialization probe.

The gate and up projections, model weights, inputs, and downstream projection
are held fixed.  The only changed boundary is the SiLU evaluation:
native model-dtype activation versus FP32 SiLU followed by one original-dtype
write-back.  This is a direct natural training source intervention.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    if len(values) < 2:
        return [float(x.mean()), float(x.mean())]
    m = float(x.mean())
    h = 1.96 * float(x.std(unbiased=True)) / math.sqrt(len(values))
    return [m - h, m + h]


def make_windows(tokenizer: Any, source: Path, length: int, count: int) -> list[list[int]]:
    ids = tokenizer(source.read_text(encoding="utf-8"), add_special_tokens=True, return_tensors="pt")["input_ids"][0]
    stride = max(1, length // 2)
    need = (count - 1) * stride + length + 1
    if ids.numel() < need:
        raise RuntimeError("text source has {} tokens, need {}".format(ids.numel(), need))
    return [ids[i * stride : i * stride + length].tolist() for i in range(count)]


def run_branch(model, mlp, ids, labels, target, fp32_activation: bool):
    original = mlp.forward

    def patched(hidden_states):
        gate = mlp.gate_proj(hidden_states)
        up = mlp.up_proj(hidden_states)
        if fp32_activation:
            activated = F.silu(gate.float()).to(gate.dtype)
        else:
            activated = mlp.act_fn(gate)
        return mlp.down_proj(activated * up)

    mlp.forward = patched
    model.zero_grad(set_to_none=True)
    try:
        loss = model(input_ids=ids, labels=labels, use_cache=False).loss
        loss.backward()
        if target.grad is None:
            raise RuntimeError("target gradient missing")
        return float(loss.detach()), target.grad.detach().float().cpu().clone()
    finally:
        mlp.forward = original


def ratio(effect: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(effect)) / max(float(torch.linalg.vector_norm(reference)), 1e-30)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("/data1/tzh/models/meta-llama/Llama-3.2-3B"))
    p.add_argument("--text-source", type=Path, default=ROOT / "docs/root_cause_closure_current.md")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--sequence-length", type=int, default=64)
    p.add_argument("--states", type=int, default=16)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.pad_token
    bank = make_windows(tokenizer, args.text_source, args.sequence_length, args.states)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, dtype=torch.bfloat16).to(device).eval()
    mlp = model.model.layers[args.layer].mlp
    target = mlp.down_proj.weight
    rows: list[dict[str, Any]] = []
    effects: list[torch.Tensor] = []
    for state_id, ids_list in enumerate(bank):
        ids = torch.tensor([ids_list], dtype=torch.long, device=device)
        labels = ids.clone()
        native_loss, native_grad = run_branch(model, mlp, ids, labels, target, False)
        fp32_loss, fp32_grad = run_branch(model, mlp, ids, labels, target, True)
        effect = fp32_grad - native_grad
        native_write = -args.learning_rate * native_grad / (native_grad.abs() + args.eps)
        fp32_write = -args.learning_rate * fp32_grad / (fp32_grad.abs() + args.eps)
        write_effect = fp32_write - native_write
        aligned = float(torch.dot(write_effect.reshape(-1), native_write.reshape(-1))) / max(
            float(torch.dot(native_write.reshape(-1), native_write.reshape(-1))), 1e-30
        )
        effects.append(effect.double().reshape(-1))
        rows.append({
            "state_id": state_id,
            "native_loss": native_loss,
            "fp32_activation_loss": fp32_loss,
            "loss_difference_fp32_minus_native": fp32_loss - native_loss,
            "gradient_effect_rms_over_native": ratio(effect, native_grad),
            "write_effect_rms_over_native": ratio(write_effect, native_write),
            "write_aligned_fp32_minus_native": aligned,
        })
    split = args.states // 2
    direction = torch.stack(effects[:split]).mean(dim=0)
    direction_norm = float(torch.linalg.vector_norm(direction))
    projections: list[float] = []
    if direction_norm > 0:
        direction = direction / direction_norm
        projections = [float(torch.dot(value, direction)) for value in effects[split:]]
    aligned = [row["write_aligned_fp32_minus_native"] for row in rows[split:]]
    result = {
        "schema": "kernel-analyzer-llama-silu-activation-natural-probe-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "operator": "Llama SwiGLU SiLU activation materialization",
        "parameter": "model.layers.{}.mlp.down_proj.weight".format(args.layer),
        "candidate": "native model-dtype SiLU gate activation",
        "reference": "same gate tensor evaluated by FP32 SiLU followed by one BF16 write-back",
        "layer": args.layer,
        "input_source": "real document-derived token windows retokenized with the Llama tokenizer",
        "comparison_scope": {
            "same_model_weights": True,
            "same_input_ids": True,
            "same_gate_and_up_projections": True,
            "single_changed_boundary": "SiLU activation evaluation/materialization",
        },
        "claim_boundary": "One Llama-3.2 checkpoint, one MLP layer and declared natural text bank; source and fixed-suite update evidence only.",
        "rows": rows,
        "summary": {
            "state_count": len(rows),
            "calibration_count": split,
            "confirmation_count": len(rows) - split,
            "gradient_effect_rms_mean": sum(r["gradient_effect_rms_over_native"] for r in rows) / len(rows),
            "write_effect_rms_mean": sum(r["write_effect_rms_over_native"] for r in rows) / len(rows),
            "confirmation_aligned_write_mean": sum(aligned) / len(aligned),
            "confirmation_aligned_write_interval_normal_95": interval(aligned),
            "confirmation_projection_interval_normal_95": interval(projections),
            "confirmation_projection_positive": sum(x > 0 for x in projections),
            "confirmation_projection_negative": sum(x < 0 for x in projections),
            "loss_difference_interval_normal_95": interval([r["loss_difference_fp32_minus_native"] for r in rows]),
            "calibration_direction_norm": direction_norm,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **result["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
