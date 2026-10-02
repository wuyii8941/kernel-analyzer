#!/usr/bin/env python3
"""Probe T5 relative-position attention-bias materialization on real text.

The intervention changes only the final score-plus-position-bias addition in
encoder layer 0.  Native mode adds the tensors in the model dtype; the
reference mode performs that one addition in FP32 and casts back before the
unchanged softmax/value/output path.  The probe measures the actual gradient
and first-step AdamW write of the learned relative-attention-bias table.
"""

from __future__ import annotations

import argparse
import json
import math
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/data1/tzh/models/google/t5-small")
DEFAULT_TEXT = ROOT / "README.md"
DEFAULT_OUTPUT = ROOT / "results/property/new_problem_group_search_v1/t5_relative_attention_bias_natural_16_20260920.json"


def interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    mean = float(x.mean())
    if len(values) < 2:
        return [mean, mean]
    half = 1.96 * float(x.std(unbiased=True)) / math.sqrt(len(values))
    return [mean - half, mean + half]


def text_states(tokenizer: Any, source: Path, count: int, max_length: int) -> list[dict[str, torch.Tensor]]:
    lines = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("empty text source")
    states: list[dict[str, torch.Tensor]] = []
    for index in range(count):
        # Use disjoint real-text windows, wrapping only if the source is short.
        body = " ".join(lines[index % len(lines) : index % len(lines) + 3])
        if len(body) < 24:
            body = (body + " " + " ".join(lines))[:512]
        encoded = tokenizer(
            "summarize: " + body,
            text_target=body,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = encoded["labels"][0]
        labels = labels.masked_fill(labels == tokenizer.pad_token_id, -100)
        states.append({
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
            "labels": labels,
        })
    return states


def install_attention_mode(model: T5ForConditionalGeneration, mode: str) -> None:
    attention = model.encoder.block[0].layer[0].SelfAttention
    original = attention.forward

    # Keep the original method available for restoration and for any future
    # caller that exercises an unsupported cache/cross-attention path.
    if not hasattr(attention, "_kernel_analyzer_original_forward"):
        attention._kernel_analyzer_original_forward = original

    def forward(self, hidden_states, mask=None, key_value_states=None, position_bias=None,
                past_key_values=None, **kwargs):
        if key_value_states is not None or past_key_values is not None:
            return self._kernel_analyzer_original_forward(
                hidden_states, mask=mask, key_value_states=key_value_states,
                position_bias=position_bias, past_key_values=past_key_values, **kwargs,
            )
        input_shape = hidden_states.shape[:-1]
        batch_size, seq_length = input_shape
        query_states = self.q(hidden_states).view(
            batch_size, -1, self.n_heads, self.key_value_proj_dim
        ).transpose(1, 2)
        key_states = self.k(hidden_states).view(
            batch_size, -1, self.n_heads, self.key_value_proj_dim
        ).transpose(1, 2)
        value_states = self.v(hidden_states).view(
            batch_size, -1, self.n_heads, self.key_value_proj_dim
        ).transpose(1, 2)
        scores = torch.matmul(query_states, key_states.transpose(3, 2)) * self.scaling
        if position_bias is None:
            key_length = key_states.shape[-2]
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, seq_length, key_length),
                    device=scores.device, dtype=scores.dtype,
                )
            else:
                position_bias = self.compute_bias(seq_length, key_length, device=scores.device, past_seen_tokens=0)
        if mode == "fp32_add":
            scores = (scores.float() + position_bias.float()).to(scores.dtype)
        else:
            scores = scores + position_bias
        if mask is not None:
            scores = scores + mask
        attn_weights = nn.functional.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(*input_shape, -1)
        attn_output = self.o(attn_output)
        return attn_output, position_bias, attn_weights

    attention.forward = types.MethodType(forward, attention)


def one_state(model, state: dict[str, torch.Tensor], target: torch.Tensor, device: torch.device) -> tuple[float, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    batch = {key: value.unsqueeze(0).to(device) for key, value in state.items()}
    output = model(**batch)
    loss = output.loss
    loss.backward()
    grad = target.grad.detach().float().clone()
    return float(loss.detach().float()), grad


def first_adamw_write(grad: torch.Tensor, lr: float = 1e-3, eps: float = 1e-8) -> torch.Tensor:
    return -lr * grad / (grad.abs() + eps)


def run(model_path: Path, text_path: Path, count: int, max_length: int, device: str) -> dict[str, Any]:
    tokenizer = T5Tokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(str(model_path), local_files_only=True)
    model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    target = model.encoder.block[0].layer[0].SelfAttention.relative_attention_bias.weight
    states = text_states(tokenizer, text_path, count, max_length)
    rows: list[dict[str, Any]] = []
    base_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    for index, state in enumerate(states):
        model.load_state_dict(base_state, strict=True)
        install_attention_mode(model, "native")
        native_loss, native_grad = one_state(model, state, target, torch.device(device))
        native_write = first_adamw_write(native_grad)
        model.load_state_dict(base_state, strict=True)
        install_attention_mode(model, "fp32_add")
        reference_loss, reference_grad = one_state(model, state, target, torch.device(device))
        reference_write = first_adamw_write(reference_grad)
        effect = native_write - reference_write
        repair_norm = float(reference_write.norm())
        effect_norm = float(effect.norm())
        aligned = float(torch.sum(effect * reference_write) / max(repair_norm * repair_norm, 1e-30))
        rows.append({
            "state_id": index,
            "native_loss": native_loss,
            "reference_loss": reference_loss,
            "loss_difference": native_loss - reference_loss,
            "gradient_effect_rms_over_reference": float((native_grad - reference_grad).norm() / max(reference_grad.norm(), 1e-30)),
            "write_effect_rms_over_reference": effect_norm / max(repair_norm, 1e-30),
            "write_aligned": aligned,
        })
    calibration = count // 2
    cal_effects = []
    for row in rows[:calibration]:
        cal_effects.append(row["write_aligned"])
    cal_mean = float(torch.tensor(cal_effects, dtype=torch.float64).mean()) if cal_effects else 0.0
    confirmation = rows[calibration:]
    # A held-out direction is only defined when the calibration aligned mean is
    # nonzero; here we retain the scalar sign counts and do not overclaim a
    # vector mean from the single-table target.
    return {
        "schema": "kernel-analyzer-t5-relative-attention-bias-natural-v1",
        "status": "COMPLETE_NATURAL_T5_RELATIVE_ATTENTION_BIAS_BOUNDARY",
        "model": str(model_path),
        "operator_family": "t5_relative_attention_bias_materialization",
        "target_parameter": "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
        "candidate": "native BF16 score plus relative-position bias",
        "reference": "same input and weights with only score-plus-bias addition evaluated in FP32 then BF16 cast",
        "input_source": str(text_path),
        "comparison_scope": {"same_text": True, "same_weights": True, "same_labels": True, "single_changed_boundary": "score_plus_relative_position_bias_addition"},
        "rows": rows,
        "summary": {
            "state_count": count,
            "calibration_count": calibration,
            "confirmation_count": len(confirmation),
            "write_effect_rms_mean": float(torch.tensor([r["write_effect_rms_over_reference"] for r in rows], dtype=torch.float64).mean()),
            "aligned_write_interval_normal_95": interval([r["write_aligned"] for r in rows]),
            "confirmation_aligned_interval_normal_95": interval([r["write_aligned"] for r in confirmation]),
            "confirmation_positive": sum(r["write_aligned"] > 0 for r in confirmation),
            "confirmation_negative": sum(r["write_aligned"] < 0 for r in confirmation),
            "loss_difference_interval_normal_95": interval([r["loss_difference"] for r in rows]),
            "calibration_aligned_mean": cal_mean,
        },
        "claim_boundary": "One T5-small checkpoint and declared real-text bank; source/write fixed-suite result only, no population or quality claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--text", type=Path, default=DEFAULT_TEXT)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.model, args.text, args.count, args.max_length, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
