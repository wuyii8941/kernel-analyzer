#!/usr/bin/env python3
"""Natural BERT attention-score materialization probe.

The candidate computes the attention score QK^T in the model dtype.  The
reference computes only that score product in FP32 and casts it back before
the unchanged scaling, masking, softmax, value contraction and classifier.
This is a real checkpoint/text-window probe; it is not a claim about every
attention implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import types
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, BertForSequenceClassification


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/data1/tzh/models/prajjwal1/bert-tiny")
DEFAULT_TEXT = ROOT / "docs/root_cause_closure_current.md"


def windows(tokenizer: Any, source: Path, seq_len: int, count: int) -> list[torch.Tensor]:
    ids = tokenizer(
        source.read_text(encoding="utf-8"),
        add_special_tokens=True,
        return_tensors="pt",
    )["input_ids"][0]
    stride = max(1, seq_len // 2)
    need = (count - 1) * stride + seq_len
    if ids.numel() < need:
        raise RuntimeError(f"text source has {ids.numel()} tokens, need {need}")
    return [ids[i * stride : i * stride + seq_len].clone() for i in range(count)]


def interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    if x.numel() < 2:
        mean = float(x.mean().item()) if x.numel() else 0.0
        return [mean, mean]
    mean = float(x.mean().item())
    half = 1.96 * float(x.std(unbiased=True).item()) / math.sqrt(x.numel())
    return [mean - half, mean + half]


def relative(effect: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = float(torch.linalg.vector_norm(reference).item())
    return float(torch.linalg.vector_norm(effect).item()) / max(denominator, 1e-30)


def write(gradient: torch.Tensor, learning_rate: float, eps: float) -> torch.Tensor:
    g = gradient.float()
    return -learning_rate * g / (g.abs() + eps)


def install_score_variant(attention: torch.nn.Module, fp32_score: bool, fp32_value: bool) -> None:
    def forward(
        self: torch.nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        past_key_values: Any = None,
        output_attentions: bool = False,
        cache_position: torch.Tensor | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, _ = hidden_states.shape
        query_layer = self.query(hidden_states).view(
            batch_size, -1, self.num_attention_heads, self.attention_head_size
        ).transpose(1, 2)
        current_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key_layer = self.key(current_states).view(
            batch_size, -1, self.num_attention_heads, self.attention_head_size
        ).transpose(1, 2)
        value_layer = self.value(current_states).view(
            batch_size, -1, self.num_attention_heads, self.attention_head_size
        ).transpose(1, 2)
        if fp32_score:
            attention_scores = torch.matmul(query_layer.float(), key_layer.float().transpose(-1, -2)).to(query_layer.dtype)
        else:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        if fp32_value:
            context_layer = torch.matmul(attention_probs.float(), value_layer.float()).to(value_layer.dtype)
        else:
            context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(context_layer.size()[:-2] + (self.all_head_size,))
        return context_layer, attention_probs

    attention.forward = types.MethodType(forward, attention)


def run_once(
    model: torch.nn.Module,
    ids: torch.Tensor,
    label: torch.Tensor,
    target: torch.nn.Parameter,
    boundary: str,
) -> tuple[float, torch.Tensor]:
    attention = model.bert.encoder.layer[0].attention.self
    original = attention.forward
    install_score_variant(attention, boundary == "score", boundary == "value")
    model.zero_grad(set_to_none=True)
    try:
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=label)
        output.loss.backward()
        if target.grad is None:
            raise RuntimeError("target gradient missing")
        return float(output.loss.detach()), target.grad.detach().float().cpu().clone()
    finally:
        attention.forward = original


def run(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    bank = windows(tokenizer, args.text_source, args.sequence_length, args.states)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    model = BertForSequenceClassification.from_pretrained(
        args.model, local_files_only=True, torch_dtype=dtype
    ).to(args.device).eval()
    named = dict(model.named_parameters())
    target_name = args.parameter or "bert.encoder.layer.0.attention.self.query.weight"
    if target_name not in named:
        raise KeyError(target_name)
    target = named[target_name]
    effects: list[torch.Tensor] = []
    writes: list[torch.Tensor] = []
    references: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    for state_id, tokens in enumerate(bank):
        ids = tokens.unsqueeze(0).to(args.device)
        label = torch.tensor([state_id % 2], dtype=torch.long, device=args.device)
        native_loss, native_grad = run_once(model, ids, label, target, "native")
        fp32_loss, fp32_grad = run_once(model, ids, label, target, args.boundary)
        native_write = write(native_grad, args.learning_rate, args.eps)
        fp32_write = write(fp32_grad, args.learning_rate, args.eps)
        gradient_effect = fp32_grad - native_grad
        write_effect = fp32_write - native_write
        effects.append(gradient_effect.double().reshape(-1))
        writes.append(write_effect.double().reshape(-1))
        references.append(native_write.double().reshape(-1))
        write_effect_flat = write_effect.double().reshape(-1)
        native_write_flat = native_write.double().reshape(-1)
        rows.append({
            "state_id": state_id,
            "native_loss": native_loss,
            "fp32_score_loss": fp32_loss,
            "loss_difference_fp32_minus_native": fp32_loss - native_loss,
            "gradient_effect_rms_over_native": relative(gradient_effect, native_grad),
            "write_effect_rms_over_native": relative(write_effect, native_write),
            "aligned_write_ratio": float(torch.dot(write_effect_flat, native_write_flat).item())
            / max(float(torch.dot(native_write_flat, native_write_flat).item()), 1e-30),
        })
        print(json.dumps({"event": "BERT_SCORE_STATE", "state": state_id}), flush=True)

    split = len(effects) // 2
    direction = torch.stack(effects[:split]).mean(dim=0)
    direction_norm = float(direction.norm().item())
    projections: list[float] = []
    if direction_norm > 0:
        direction = direction / direction_norm
        projections = [float(torch.dot(value, direction).item()) for value in effects[split:]]
    aligned = [row["aligned_write_ratio"] for row in rows[split:]]
    return {
        "schema": "kernel-analyzer-bert-attention-score-materialization-natural-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "operator": f"BERT attention {args.boundary} matmul materialization",
        "parameter": target_name,
        "candidate": f"native model-dtype attention {args.boundary} matmul",
        "reference": f"FP32 attention {args.boundary} matmul followed by one original-dtype write",
        "input_source": str(args.text_source),
        "comparison_scope": {
            "same_model_weights": True,
            "same_input_ids": True,
            "same_attention_mask": True,
            "single_changed_boundary": f"attention {args.boundary} matmul",
        },
        "state_count": len(rows),
        "calibration_count": split,
        "confirmation_count": len(rows) - split,
        "rows": rows,
        "summary": {
            "gradient_effect_rms_mean": sum(r["gradient_effect_rms_over_native"] for r in rows) / len(rows),
            "write_effect_rms_mean": sum(r["write_effect_rms_over_native"] for r in rows) / len(rows),
            "aligned_write_mean_confirmation": sum(aligned) / len(aligned),
            "aligned_write_interval_normal_95": interval(aligned),
            "aligned_positive_count": sum(v > 0 for v in aligned),
            "aligned_negative_count": sum(v < 0 for v in aligned),
            "heldout_projection_interval_normal_95": interval(projections),
            "heldout_projection_positive": sum(v > 0 for v in projections),
            "heldout_projection_negative": sum(v < 0 for v in projections),
            "calibration_direction_norm": direction_norm,
        },
        "claim_boundary": "One BERT-tiny checkpoint, one attention-score boundary, and the declared text bank; not a population or long-run quality claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--text-source", type=Path, default=DEFAULT_TEXT)
    parser.add_argument("--parameter", default=None)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--states", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--boundary", choices=("score", "value"), default="score")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
