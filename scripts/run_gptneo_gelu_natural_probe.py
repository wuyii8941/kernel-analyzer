#!/usr/bin/env python3
"""Natural GPT-Neo GELU evaluation probe.

The probe compares the model's native NewGELUActivation with an explicit
FP32 tanh-GELU expression followed by the original activation dtype writeback.
The two paths share weights, token sequences, loss, and optimizer proxy.  It
is intended to decide whether this native implementation is a distinct,
closed training problem or only a repetition of an existing activation case.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    mean = float(x.mean())
    if len(values) < 2:
        return [mean, mean]
    half = 1.96 * float(x.std(unbiased=True)) / math.sqrt(len(values))
    return [mean - half, mean + half]


def ratio(effect: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(effect)) / max(float(torch.linalg.vector_norm(reference)), 1e-30)


def write(gradient: torch.Tensor, lr: float, eps: float) -> torch.Tensor:
    g = gradient.float()
    return -lr * g / (g.abs() + eps)


class ExplicitTanhGELU(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        x = value.float()
        y = 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
        return y.to(value.dtype)


def text_sequences(tokenizer: AutoTokenizer, qwen_bank: Path, count: int, length: int) -> list[list[int]]:
    bank = json.loads(qwen_bank.read_text(encoding="utf-8"))
    qwen_tokenizer = AutoTokenizer.from_pretrained(
        "/data1/tzh/models/Qwen/Qwen3-1.7B", local_files_only=True, use_fast=True,
    )
    rows = []
    for state in bank["states"]:
        ids = state.get("input_ids", state.get("token_ids"))
        text = qwen_tokenizer.decode(ids, skip_special_tokens=True)
        encoded = tokenizer(text, add_special_tokens=True, truncation=True,
                            max_length=length, return_tensors="pt")["input_ids"][0].tolist()
        if len(encoded) >= 8:
            rows.append(encoded)
        if len(rows) == count:
            break
    if len(rows) != count:
        raise RuntimeError(f"only {len(rows)} usable natural text states")
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.pad_token
    sequences = text_sequences(tokenizer, args.qwen_bank, args.states, args.length)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
    ).to(device).train()
    model.config.use_cache = False
    layer = model.transformer.h[args.layer]
    activation = layer.mlp.act
    target = layer.mlp.c_proj.weight
    rows: list[dict[str, Any]] = []
    effects: list[torch.Tensor] = []
    writes: list[torch.Tensor] = []
    references: list[torch.Tensor] = []
    for state_index, ids_list in enumerate(sequences):
        ids = torch.tensor([ids_list], dtype=torch.long, device=device)
        labels = ids.clone()
        captured: dict[str, torch.Tensor] = {}

        def capture(_module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
            captured["input"] = inputs[0].detach().clone()
            captured["output"] = output.detach().clone()
            return output

        model.zero_grad(set_to_none=True)
        layer.mlp.act = activation
        handle = layer.mlp.act.register_forward_hook(capture)
        native_loss = model(input_ids=ids, labels=labels, use_cache=False).loss
        handle.remove()
        native_loss.backward()
        native_gradient = target.grad.detach().float().cpu().clone()
        native_write = write(native_gradient, args.learning_rate, args.eps)

        model.zero_grad(set_to_none=True)
        layer.mlp.act = ExplicitTanhGELU()
        reference_loss = model(input_ids=ids, labels=labels, use_cache=False).loss
        reference_loss.backward()
        reference_gradient = target.grad.detach().float().cpu().clone()
        reference_write = write(reference_gradient, args.learning_rate, args.eps)
        effect = native_gradient - reference_gradient
        write_effect = native_write - reference_write
        effects.append(effect.reshape(-1).double())
        writes.append(write_effect.reshape(-1).double())
        references.append(reference_write.reshape(-1).double())
        rows.append({
            "state_index": state_index,
            "token_count": len(ids_list),
            "loss_difference": float((native_loss - reference_loss).detach().cpu()),
            "gradient_effect_rms_over_reference": ratio(effect, reference_gradient),
            "write_effect_rms_over_reference": ratio(write_effect, reference_write),
            "write_aligned": float(torch.dot(write_effect.reshape(-1), reference_write.reshape(-1)))
            / max(float(torch.dot(reference_write.reshape(-1), reference_write.reshape(-1))), 1e-30),
            "source": {
                "native_formula_bf16_relative_l2": ratio(
                    captured["output"].float()
                    - (0.5 * captured["input"] * (1.0 + torch.tanh(
                        0.7978845608028654 * (captured["input"] + 0.044715 * captured["input"] ** 3)
                    ))),
                    captured["output"].float(),
                ),
                "native_vs_fp32_formula_relative_l2": ratio(
                    captured["output"].float()
                    - ExplicitTanhGELU()(captured["input"]),
                    ExplicitTanhGELU()(captured["input"]),
                ),
            },
        })
        del ids, labels, native_loss, reference_loss
        torch.cuda.empty_cache()
    layer.mlp.act = activation

    split = len(effects) // 2
    direction = torch.stack(effects[:split]).mean(dim=0)
    norm = float(torch.linalg.vector_norm(direction))
    projections: list[float] = []
    if norm > 0:
        direction = direction / norm
        projections = [float(torch.dot(x, direction)) for x in effects[split:]]
    aligned = [
        float(torch.dot(effect, reference)) / max(float(torch.dot(reference, reference)), 1e-30)
        for effect, reference in zip(writes[split:], references[split:])
    ]
    return {
        "schema": "kernel-analyzer-gptneo-gelu-natural-probe-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "operator": "GPT-Neo native NewGELUActivation",
        "candidate": "native GPT-Neo NewGELUActivation",
        "reference": "explicit FP32 tanh-GELU with original dtype writeback",
        "layer": args.layer,
        "input_source": "real text decoded from the declared Qwen natural bank and retokenized with GPT-Neo",
        "comparison_scope": {
            "same_model_weights": True,
            "same_input_text": True,
            "single_changed_boundary": "GELU evaluation",
        },
        "claim_boundary": "One GPT-Neo checkpoint, one MLP layer and declared natural text bank; source and fixed-suite update evidence only.",
        "rows": rows,
        "summary": {
            "state_count": len(rows),
            "calibration_count": split,
            "confirmation_count": len(rows) - split,
            "gradient_effect_rms_mean": sum(r["gradient_effect_rms_over_reference"] for r in rows) / len(rows),
            "write_effect_rms_mean": sum(r["write_effect_rms_over_reference"] for r in rows) / len(rows),
            "aligned_write_mean": sum(aligned) / len(aligned),
            "aligned_write_interval_normal_95": interval(aligned),
            "projection_mean": sum(projections) / len(projections) if projections else None,
            "projection_interval_normal_95": interval(projections) if projections else None,
            "projection_positive": sum(x > 0 for x in projections),
            "projection_negative": sum(x < 0 for x in projections),
            "loss_difference_interval_normal_95": interval([r["loss_difference"] for r in rows]),
            "calibration_direction_norm": norm,
            "source_native_formula_bf16_relative_l2_mean": sum(
                r["source"]["native_formula_bf16_relative_l2"] for r in rows
            ) / len(rows),
            "source_native_vs_fp32_formula_relative_l2_mean": sum(
                r["source"]["native_vs_fp32_formula_relative_l2"] for r in rows
            ) / len(rows),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/EleutherAI/gpt-neo-125m"))
    parser.add_argument("--qwen-bank", type=Path, default=Path("results/coverage/qwen_seq128_input_bank.json"))
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
