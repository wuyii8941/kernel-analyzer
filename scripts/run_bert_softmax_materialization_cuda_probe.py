#!/usr/bin/env python3
"""Real BERT-tiny CUDA probe for attention-softmax materialization.

Only the layer-0 attention softmax is changed.  All Q/K/V projections, masks,
dropout (evaluation mode), encoder blocks and the selected parameter carrier
are shared between native BF16 softmax and explicit FP32 softmax followed by
the original dtype write-back.
"""

from __future__ import annotations

import argparse
import json
import math
import types
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/data1/tzh/models/prajjwal1/bert-tiny")
DEFAULT_TEXT = ROOT / "docs/root_cause_closure_current.md"


def windows(tokenizer: Any, source: Path, seq_len: int, count: int) -> list[torch.Tensor]:
    ids = tokenizer(source.read_text(encoding="utf-8"), add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    stride = seq_len * 2
    need = (count - 1) * stride + seq_len
    if ids.numel() < need:
        raise RuntimeError(f"text source has {ids.numel()} tokens, need {need}")
    return [ids[i * stride : i * stride + seq_len].clone() for i in range(count)]


def interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    mean = float(x.mean())
    if x.numel() < 2:
        return [mean, mean]
    half = 1.96 * float(x.std(unbiased=True)) / math.sqrt(x.numel())
    return [mean - half, mean + half]


def relative(effect: torch.Tensor, reference: torch.Tensor) -> float:
    return float(effect.float().norm()) / max(float(reference.float().norm()), 1e-30)


def first_step_write(gradient: torch.Tensor, lr: float, eps: float) -> torch.Tensor:
    value = gradient.float()
    return -lr * value / (value.abs() + eps)


def install_attention_variant(attention: torch.nn.Module, fp32_softmax: bool) -> None:
    def forward(self: torch.nn.Module, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None,
                head_mask: torch.Tensor | None = None, encoder_hidden_states: torch.Tensor | None = None,
                past_key_values: Any = None, output_attentions: bool = False, cache_position: Any = None):
        if encoder_hidden_states is not None or past_key_values is not None:
            raise RuntimeError("cross-attention/cache is outside this probe")
        batch, seq_len, _ = hidden_states.shape
        query = self.query(hidden_states).view(batch, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key = self.key(hidden_states).view(batch, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value = self.value(hidden_states).view(batch, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            scores = scores + attention_mask
        if fp32_softmax:
            probs = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        else:
            probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        if head_mask is not None:
            probs = probs * head_mask
        context = torch.matmul(probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size()[:-2] + (self.all_head_size,))
        return context, probs

    attention.forward = types.MethodType(forward, attention)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    bank = windows(tokenizer, args.text_source, args.sequence_length, args.states)
    model = AutoModelForMaskedLM.from_pretrained(args.model, local_files_only=True, dtype=torch.bfloat16).to(device).eval()
    model.config.use_cache = False
    attention = model.bert.encoder.layer[0].attention.self
    target = dict(model.named_parameters())[args.parameter]
    rows: list[dict[str, Any]] = []
    effects: list[torch.Tensor] = []
    writes: list[torch.Tensor] = []
    references: list[torch.Tensor] = []
    for state_id, tokens in enumerate(bank):
        ids = tokens.unsqueeze(0).to(device)
        outputs: list[tuple[float, torch.Tensor]] = []
        original = attention.forward
        try:
            for fp32 in (False, True):
                install_attention_variant(attention, fp32)
                model.zero_grad(set_to_none=True)
                loss = model(input_ids=ids, labels=ids).loss
                loss.backward()
                if target.grad is None:
                    raise RuntimeError("target gradient missing")
                outputs.append((float(loss.detach()), target.grad.detach().float().cpu().clone()))
        finally:
            attention.forward = original
        native_loss, native_grad = outputs[0]
        reference_loss, reference_grad = outputs[1]
        effect = native_grad - reference_grad
        native_write = first_step_write(native_grad, args.learning_rate, args.eps)
        reference_write = first_step_write(reference_grad, args.learning_rate, args.eps)
        write_effect = native_write - reference_write
        effects.append(effect.double().reshape(-1))
        writes.append(write_effect.double().reshape(-1))
        references.append(reference_write.double().reshape(-1))
        rows.append({
            "state_id": state_id,
            "native_loss": native_loss,
            "fp32_softmax_loss": reference_loss,
            "loss_difference_native_minus_fp32": native_loss - reference_loss,
            "gradient_effect_rms_over_reference": relative(effect, reference_grad),
            "write_effect_rms_over_reference": relative(write_effect, reference_write),
        })
        del ids
        torch.cuda.empty_cache()

    split = len(rows) // 2
    direction = torch.stack(effects[:split]).mean(dim=0)
    direction_norm = float(direction.norm())
    projections: list[float] = []
    if direction_norm:
        unit = direction / direction_norm
        projections = [float(torch.dot(value, unit)) for value in effects[split:]]
    aligned = [
        float(torch.dot(effect, reference)) / max(float(torch.dot(reference, reference)), 1e-30)
        for effect, reference in zip(writes[split:], references[split:])
    ]
    return {
        "schema": "kernel-analyzer-bert-softmax-materialization-cuda-probe-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "operator": "BERT layer-0 attention softmax materialization",
        "parameter": args.parameter,
        "candidate": "native BF16 attention softmax",
        "reference": "same attention with FP32 softmax followed by one BF16 write-back",
        "input_source": str(args.text_source),
        "comparison_scope": {"same_model_weights": True, "same_input_ids": True, "same_qkv_mask": True,
                              "single_changed_boundary": "attention softmax evaluation/materialization"},
        "rows": rows,
        "summary": {
            "state_count": len(rows), "calibration_count": split, "confirmation_count": len(rows) - split,
            "gradient_effect_rms_mean": sum(r["gradient_effect_rms_over_reference"] for r in rows) / len(rows),
            "write_effect_rms_mean": sum(r["write_effect_rms_over_reference"] for r in rows) / len(rows),
            "projection_interval_normal_95": interval(projections),
            "projection_positive": sum(x > 0 for x in projections),
            "projection_negative": sum(x < 0 for x in projections),
            "aligned_write_mean": sum(aligned) / len(aligned),
            "aligned_write_interval_normal_95": interval(aligned),
            "loss_difference_interval_normal_95": interval([r["loss_difference_native_minus_fp32"] for r in rows]),
            "calibration_direction_norm": direction_norm,
        },
        "claim_boundary": "One BERT-tiny CUDA checkpoint, layer-0 attention softmax boundary and declared text bank; no population or long-run quality claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--text-source", type=Path, default=DEFAULT_TEXT)
    parser.add_argument("--parameter", default="bert.encoder.layer.0.attention.self.query.weight")
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--states", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
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
