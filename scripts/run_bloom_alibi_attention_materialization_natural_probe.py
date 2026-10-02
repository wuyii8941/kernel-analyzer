#!/usr/bin/env python3
"""Natural Bloom ALiBi attention-score materialization probe.

This keeps a real Bloom checkpoint, real text windows, the generated causal
mask and ALiBi tensor fixed.  Only the selected attention layer changes its
score computation: native ``baddbmm`` versus an explicit FP32 QK product and
FP32 ALiBi addition before the original probability cast.  The result is a
scoped natural training-path comparison, not a claim about all ALiBi models.
"""

from __future__ import annotations

import argparse
import json
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BloomForCausalLM
from transformers.models.bloom import modeling_bloom


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/data1/tzh/models/bigscience/bloom-560m")
DEFAULT_TEXT = ROOT / "README.md"


def interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    mean = float(x.mean().item())
    if x.numel() < 2:
        return [mean, mean]
    half = 1.96 * float(x.std(unbiased=True).item()) / (x.numel() ** 0.5)
    return [mean - half, mean + half]


def ratio(effect: torch.Tensor, reference: torch.Tensor) -> float:
    den = float(torch.linalg.vector_norm(reference).item())
    return float(torch.linalg.vector_norm(effect).item()) / max(den, 1e-30)


def first_step_write(gradient: torch.Tensor, learning_rate: float, eps: float) -> torch.Tensor:
    g = gradient.float()
    return -learning_rate * g / (g.abs() + eps)


def make_windows(tokenizer: Any, source: Path, states: int, sequence_length: int) -> list[dict[str, Any]]:
    text = source.read_text(encoding="utf-8", errors="ignore")
    ids = tokenizer(text, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
    windows: list[dict[str, Any]] = []
    stride = max(1, sequence_length - 8)
    for index, start in enumerate(range(0, max(1, ids.numel() - sequence_length + 1), stride)):
        row = ids[start : start + sequence_length]
        if row.numel() < sequence_length:
            break
        windows.append({"state_id": f"readme_window_{index}", "token_ids": [int(x) for x in row]})
        if len(windows) >= states:
            break
    if len(windows) < states:
        raise RuntimeError(f"text source supplied only {len(windows)} windows")
    return windows


def fp32_score_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    alibi: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_past: Any = None,
    head_mask: torch.Tensor | None = None,
    use_cache: bool = False,
    output_attentions: bool = False,
    cache_position: torch.Tensor | None = None,
):
    batch_size, q_length, _ = hidden_states.shape
    hidden_shape = (batch_size, q_length, -1, self.head_dim)
    fused_qkv = self.query_key_value(hidden_states)
    query_layer, key_layer, value_layer = self._reshape(fused_qkv)
    if layer_past is not None:
        cache_kwargs = {"cache_position": cache_position}
        key_layer, value_layer = layer_past.update(key_layer, value_layer, self.layer_idx, cache_kwargs)
    query_layer = query_layer.reshape(batch_size * self.num_heads, -1, self.head_dim)
    key_layer = key_layer.reshape(batch_size * self.num_heads, -1, self.head_dim).transpose(-1, -2)
    value_layer = value_layer.reshape(batch_size * self.num_heads, -1, self.head_dim)

    # The only intended change from BloomAttention.forward: perform the QK
    # product and ALiBi addition in FP32 before the native probability cast.
    score = torch.bmm(query_layer.float(), key_layer.float())
    score = score * float(self.inv_norm_factor)
    alibi_fp32 = alibi.to(dtype=torch.float32)
    score = score + alibi_fp32
    attn_weights = score.view(batch_size, self.num_heads, q_length, -1)
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_layer.shape[-1]]
        attn_weights = attn_weights + causal_mask
    attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_layer.dtype)
    attention_probs = self.attention_dropout(attention_probs)
    if head_mask is not None:
        attention_probs = attention_probs * head_mask
    attention_probs_reshaped = attention_probs.view(batch_size * self.num_heads, q_length, -1)
    context_layer = torch.bmm(attention_probs_reshaped, value_layer)
    context_layer = self._merge_heads(context_layer)
    if self.pretraining_tp > 1 and self.slow_but_exact:
        slices = self.hidden_size / self.pretraining_tp
        output_tensor = torch.zeros_like(context_layer)
        for i in range(self.pretraining_tp):
            output_tensor = output_tensor + F.linear(
                context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
                self.dense.weight[:, int(i * slices) : int((i + 1) * slices)],
            )
    else:
        output_tensor = self.dense(context_layer)
    output_tensor = modeling_bloom.dropout_add(
        output_tensor, residual, self.hidden_dropout, self.training
    )
    return output_tensor, attention_probs


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    states = make_windows(tokenizer, args.text_source, args.states, args.sequence_length)
    model = BloomForCausalLM.from_pretrained(
        str(args.model), local_files_only=True, torch_dtype=torch.bfloat16
    ).to(device).train()
    model.config.use_cache = False
    attention = model.transformer.h[args.layer].self_attention
    target = attention.query_key_value.weight
    original_forward = attention.forward
    rows: list[dict[str, Any]] = []
    effects: list[torch.Tensor] = []
    references: list[torch.Tensor] = []
    try:
        for state in states:
            ids = torch.tensor([state["token_ids"]], dtype=torch.long, device=device)
            labels = ids.clone()
            model.zero_grad(set_to_none=True)
            attention.forward = original_forward
            native = model(input_ids=ids, labels=labels, use_cache=False)
            native_loss = native.loss
            native_loss.backward()
            native_grad = target.grad.detach().float().cpu().clone()
            native_write = first_step_write(native_grad, args.learning_rate, args.eps)

            model.zero_grad(set_to_none=True)
            attention.forward = types.MethodType(fp32_score_forward, attention)
            reference = model(input_ids=ids, labels=labels, use_cache=False)
            reference_loss = reference.loss
            reference_loss.backward()
            reference_grad = target.grad.detach().float().cpu().clone()
            reference_write = first_step_write(reference_grad, args.learning_rate, args.eps)
            effect = native_write - reference_write
            effects.append(effect)
            references.append(reference_write)
            rows.append({
                "state_id": state["state_id"],
                "loss_difference": float((native_loss - reference_loss).detach().cpu()),
                "gradient_effect_rms_over_reference": ratio(native_grad - reference_grad, reference_grad),
                "write_effect_rms_over_reference": ratio(effect, reference_write),
                "write_aligned": float(torch.sum(effect * reference_write).item())
                / max(float(torch.sum(reference_write * reference_write).item()), 1e-30),
            })
            del ids, labels, native, reference, native_loss, reference_loss
            torch.cuda.empty_cache()
    finally:
        attention.forward = original_forward

    split = len(rows) // 2
    calibration = torch.stack([x.double() for x in effects[:split]], dim=0).mean(dim=0)
    direction_norm = float(torch.linalg.vector_norm(calibration).item())
    projections: list[float] = []
    if direction_norm > 0:
        direction = calibration / direction_norm
        projections = [float(torch.sum(x.double() * direction).item()) for x in effects[split:]]
    aligned = [rows[i]["write_aligned"] for i in range(split, len(rows))]
    return {
        "schema": "kernel-analyzer-bloom-alibi-attention-materialization-natural-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "layer": args.layer,
        "target": f"transformer.h[{args.layer}].self_attention.query_key_value.weight",
        "operator": "ALiBi attention score materialization",
        "candidate": "native Bloom baddbmm QK score with ALiBi",
        "reference": "same Bloom attention with explicit FP32 QK product and ALiBi addition",
        "input_source": f"real text windows from {args.text_source}",
        "optimizer": {"name": "AdamW first-step proxy", "zero_moments": True, "learning_rate": args.learning_rate},
        "comparison_scope": {"same_model_weights": True, "same_input_ids": True, "same_alibi_and_mask": True, "single_changed_boundary": "QK and ALiBi score materialization"},
        "claim_boundary": "One Bloom checkpoint/layer and declared real-text bank; not a population or loss-quality guarantee.",
        "rows": rows,
        "summary": {
            "state_count": len(rows), "calibration_count": split, "confirmation_count": len(rows) - split,
            "gradient_effect_rms_mean": sum(r["gradient_effect_rms_over_reference"] for r in rows) / len(rows),
            "write_effect_rms_mean": sum(r["write_effect_rms_over_reference"] for r in rows) / len(rows),
            "write_aligned_mean": sum(aligned) / len(aligned),
            "write_aligned_interval_normal_95": interval(aligned),
            "projection_mean": sum(projections) / len(projections) if projections else None,
            "projection_interval_normal_95": interval(projections) if projections else None,
            "projection_positive": sum(x > 0 for x in projections),
            "projection_negative": sum(x < 0 for x in projections),
            "loss_difference_interval_normal_95": interval([r["loss_difference"] for r in rows]),
            "calibration_direction_norm": direction_norm,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--text-source", type=Path, default=DEFAULT_TEXT)
    parser.add_argument("--states", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
