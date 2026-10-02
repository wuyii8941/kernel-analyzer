#!/usr/bin/env python3
"""Natural T5 relative-position bias materialisation probe.

This is deliberately a one-variable, same-input intervention.  T5 computes
attention scores in the model dtype and then adds a learned relative-position
bias.  The candidate path adds both tensors in that dtype; the reference arm
performs the same addition in FP32 and casts the score back once.  All model
weights, inputs, masks and downstream operations are held fixed.

The output is a scoped source/parameter-write result.  It is not a population
or long-horizon training claim.  A nonzero norm alone never promotes a new
problem group; the confirmation aligned interval must also be one-sided.
"""

from __future__ import annotations

import argparse
import json
import types
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "../models/google/t5-small"

TEXTS = [
    "The river crossed the old stone bridge before sunrise.",
    "A small model can still expose a precise numerical boundary.",
    "The researcher compared two implementations on identical inputs.",
    "Attention scores combine content similarity with relative position.",
    "A compiler may materialize the same mathematical expression differently.",
    "The measured update depends on the state used by the optimizer.",
    "A held out input should not be selected after seeing its result.",
    "The wind moved through the trees beside the quiet road.",
    "We record the source boundary before interpreting a numerical effect.",
    "A reference implementation is a declared comparison, not absolute truth.",
    "The model reads a sequence and predicts the next sequence of tokens.",
    "Finite precision addition is not associative in general.",
    "The same tensor values can produce different writes after rounding.",
    "A stable direction is stronger evidence than a single large norm.",
    "The experiment keeps the attention mask and model weights unchanged.",
    "An unresolved source is retained as open rather than counted as a result.",
]


def _interval(values: list[float]) -> list[float] | None:
    if not values:
        return None
    x = torch.tensor(values, dtype=torch.float64)
    if x.numel() < 2:
        v = float(x.mean().item())
        return [v, v]
    half = 1.96 * float(x.std(unbiased=True).item()) / (x.numel() ** 0.5)
    mean = float(x.mean().item())
    return [mean - half, mean + half]


def _adamw_write(gradient: torch.Tensor, learning_rate: float = 1e-4) -> torch.Tensor:
    g = gradient.float()
    return -learning_rate * g / (g.abs() + 1e-8)


def _ratio(effect: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(effect).item()) / max(
        float(torch.linalg.vector_norm(reference).item()), 1e-30
    )


def _attention_forward(self: Any, hidden_states: torch.Tensor, mask: torch.Tensor | None = None,
                       key_value_states: torch.Tensor | None = None,
                       position_bias: torch.Tensor | None = None,
                       past_key_values: Any = None, layer_head_mask: torch.Tensor | None = None,
                       query_length: int | None = None, use_cache: bool = False,
                       output_attentions: bool = False, cache_position: torch.Tensor | None = None,
                       *, fp32_score_bias: bool = False):
    """T5Attention.forward with only the score+bias addition factored out."""
    batch_size, seq_length = hidden_states.shape[:2]
    is_cross_attention = key_value_states is not None
    query_states = self.q(hidden_states)
    query_states = query_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    current_states = key_value_states if is_cross_attention else hidden_states
    key_states = self.k(current_states)
    value_states = self.v(current_states)
    key_states = key_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    scores = torch.matmul(query_states, key_states.transpose(3, 2))

    if position_bias is None:
        key_length = key_states.shape[-2]
        if query_length is not None:
            real_seq_length = query_length
        elif cache_position is not None:
            real_seq_length = int(cache_position[-1].item()) + 1
        else:
            real_seq_length = key_length
        if not self.has_relative_attention_bias:
            position_bias = torch.zeros(
                (1, self.n_heads, seq_length, key_length), device=scores.device, dtype=scores.dtype
            )
        else:
            position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device,
                                              cache_position=cache_position)
            position_bias = position_bias[:, :, -seq_length:, :]
        if mask is not None:
            position_bias = position_bias + mask[:, :, :, : key_states.shape[-2]]

    if self.pruned_heads:
        keep = torch.ones(position_bias.shape[1], device=position_bias.device, dtype=torch.bool)
        keep[list(self.pruned_heads)] = False
        position_bias_masked = position_bias[:, keep]
    else:
        position_bias_masked = position_bias

    if fp32_score_bias:
        scores = (scores.float() + position_bias_masked.float()).to(scores.dtype)
    else:
        scores = scores + position_bias_masked
    attn_weights = torch.nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
    attn_weights = torch.nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    if layer_head_mask is not None:
        attn_weights = attn_weights * layer_head_mask
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    attn_output = self.o(attn_output)
    outputs = (attn_output, position_bias)
    if output_attentions:
        outputs = outputs + (attn_weights,)
    return outputs


def _install(attention: Any, fp32_score_bias: bool) -> Any:
    original = attention.forward

    def wrapped(self: Any, *args: Any, **kwargs: Any):
        return _attention_forward(self, *args, **kwargs, fp32_score_bias=fp32_score_bias)

    attention.forward = types.MethodType(wrapped, attention)
    return original


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    model_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(
        args.model, local_files_only=True, torch_dtype=model_dtype
    ).to(device)
    model.eval()
    model.config.use_cache = False
    encoded = tokenizer(
        TEXTS[: args.states], padding="max_length", truncation=True, max_length=args.length,
        return_tensors="pt",
    )
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    # A fixed, natural text target makes this a genuine training loss path;
    # dropout is disabled above so the two arms share the exact computation.
    labels = input_ids.clone()
    target = model.encoder.block[0].layer[0].SelfAttention.q.weight
    attention = model.encoder.block[0].layer[0].SelfAttention
    original = attention.forward
    rows: list[dict[str, Any]] = []
    try:
        for i in range(args.states):
            ids = input_ids[i:i + 1]
            mask = attention_mask[i:i + 1]
            labs = labels[i:i + 1]
            model.zero_grad(set_to_none=True)
            attention.forward = original
            native = model(input_ids=ids, attention_mask=mask, labels=labs, use_cache=False).loss
            native.backward()
            native_grad = target.grad.detach().float().cpu().clone()
            native_write = _adamw_write(native_grad)

            model.zero_grad(set_to_none=True)
            _install(attention, True)
            repaired = model(input_ids=ids, attention_mask=mask, labels=labs, use_cache=False).loss
            repaired.backward()
            repaired_grad = target.grad.detach().float().cpu().clone()
            repaired_write = _adamw_write(repaired_grad)
            effect = native_write - repaired_write
            rows.append({
                "state_index": i,
                "loss_native": float(native.detach().cpu()),
                "loss_fp32_score_bias": float(repaired.detach().cpu()),
                "loss_difference": float((native - repaired).detach().cpu()),
                "gradient_effect_rms_over_reference": _ratio(native_grad - repaired_grad, repaired_grad),
                "write_effect_rms_over_reference": _ratio(effect, repaired_write),
                "write_aligned_over_reference": float(torch.sum(effect * repaired_write).item()) / max(
                    float(torch.sum(repaired_write * repaired_write).item()), 1e-30
                ),
            })
    finally:
        attention.forward = original

    half = args.states // 2
    aligned_confirmation = [row["write_aligned_over_reference"] for row in rows[half:]]
    return {
        "schema": "kernel-analyzer-t5-relative-bias-materialization-natural-v1",
        "status": "COMPLETE_NATURAL_SOURCE_ISOLATION",
        "model": str(args.model),
        "operator": "T5 encoder relative-position bias addition to attention scores",
        "candidate": "native model-dtype score plus relative-bias addition",
        "reference": "same score and relative bias added in FP32, cast back once",
        "model_dtype": args.dtype,
        "target_parameter": "encoder.block.0.layer.0.SelfAttention.q.weight",
        "input_source": "16 real tokenizer-encoded text sentences",
        "comparison_scope": {
            "same_model_weights": True,
            "same_input_ids": True,
            "same_attention_mask": True,
            "same_downstream_path": True,
            "single_changed_boundary": "attention score plus relative-position bias arithmetic",
        },
        "rows": rows,
        "summary": {
            "state_count": len(rows),
            "calibration_count": half,
            "confirmation_count": len(rows) - half,
            "write_effect_rms_mean": sum(row["write_effect_rms_over_reference"] for row in rows) / len(rows),
            "aligned_write_mean_confirmation": sum(aligned_confirmation) / len(aligned_confirmation),
            "aligned_write_interval_confirmation": _interval(aligned_confirmation),
            "aligned_positive_confirmation": sum(value > 0 for value in aligned_confirmation),
            "aligned_negative_confirmation": sum(value < 0 for value in aligned_confirmation),
            "loss_difference_mean": sum(row["loss_difference"] for row in rows) / len(rows),
        },
        "claim_boundary": (
            "Fixed T5-small checkpoint, encoder layer 0 self-attention, and declared text bank. "
            "A one-sided confirmation aligned interval is required before this can be promoted "
            "as a new problem group; no population or long-horizon claim is made."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--states", type=int, default=16)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.states < 4 or args.states % 2:
        raise ValueError("states must be an even number >= 4")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
