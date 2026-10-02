#!/usr/bin/env python3
"""Natural GPT-Neo local causal-attention mask probe.

The native GPT-Neo local attention uses a boolean causal/window mask through
``torch.where``.  The reference keeps Q/K/V and the model path fixed but
materializes the same mask with an explicit additive masked-fill operation.
This is a deliberately narrow test: a nonzero result would identify a local
window-mask materialization boundary; an exact-zero result is retained as a
negative control rather than promoted to a problem group.
"""

from __future__ import annotations

import argparse
import json
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, GPTNeoForCausalLM


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/data1/tzh/models/EleutherAI/gpt-neo-125m")
DEFAULT_TEXT = ROOT / "docs/method.md"


def interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    mean = float(x.mean().item())
    if x.numel() < 2:
        return [mean, mean]
    half = 1.96 * float(x.std(unbiased=True).item()) / (x.numel() ** 0.5)
    return [mean - half, mean + half]


def ratio(effect: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(effect).item()) / max(
        float(torch.linalg.vector_norm(reference).item()), 1e-30
    )


def first_step_write(gradient: torch.Tensor, learning_rate: float, eps: float) -> torch.Tensor:
    g = gradient.float()
    return -learning_rate * g / (g.abs() + eps)


def make_windows(tokenizer: Any, source: Path, states: int, sequence_length: int) -> list[torch.Tensor]:
    ids = tokenizer(source.read_text(encoding="utf-8", errors="ignore"), return_tensors="pt")["input_ids"][0]
    stride = max(1, sequence_length - 8)
    windows: list[torch.Tensor] = []
    for start in range(0, max(1, ids.numel() - sequence_length + 1), stride):
        row = ids[start : start + sequence_length]
        if row.numel() < sequence_length:
            break
        windows.append(row.clone())
        if len(windows) >= states:
            break
    if len(windows) < states:
        raise RuntimeError(f"text source supplied only {len(windows)} windows")
    return windows


def explicit_mask_attn(
    self: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    head_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Match GPT-Neo's reviewed _attn path, changing only boolean-mask
    # materialization from torch.where to an explicit masked fill.
    query = query.to(torch.float32)
    key = key.to(torch.float32)
    scores = torch.matmul(query, key.transpose(-1, -2))
    query_length, key_length = query.size(-2), key.size(-2)
    causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    mask_value = torch.tensor(
        torch.finfo(scores.dtype).min, dtype=scores.dtype, device=scores.device
    )
    scores = scores.masked_fill(~causal_mask, mask_value)
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : key.shape[-2]]
    weights = F.softmax(scores, dim=-1).to(value.dtype)
    weights = self.attn_dropout(weights)
    if head_mask is not None:
        weights = weights * head_mask
    return torch.matmul(weights, value), weights


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    bank = make_windows(tokenizer, args.text_source, args.states, args.sequence_length)
    model = GPTNeoForCausalLM.from_pretrained(
        str(args.model), local_files_only=True, torch_dtype=torch.bfloat16
    ).to(device).train()
    model.config.use_cache = False
    attention = model.transformer.h[args.layer].attn.attention
    target = attention.q_proj.weight
    original_attn = attention._attn
    effects: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    try:
        for index, tokens in enumerate(bank):
            ids = tokens.unsqueeze(0).to(device)
            labels = ids.clone()
            model.zero_grad(set_to_none=True)
            attention._attn = original_attn
            native = model(input_ids=ids, labels=labels, use_cache=False)
            native_loss = native.loss
            native_loss.backward()
            native_grad = target.grad.detach().float().cpu().clone()
            native_write = first_step_write(native_grad, args.learning_rate, args.eps)

            model.zero_grad(set_to_none=True)
            attention._attn = types.MethodType(explicit_mask_attn, attention)
            reference = model(input_ids=ids, labels=labels, use_cache=False)
            reference_loss = reference.loss
            reference_loss.backward()
            reference_grad = target.grad.detach().float().cpu().clone()
            reference_write = first_step_write(reference_grad, args.learning_rate, args.eps)
            effect = reference_write - native_write
            effects.append(effect.double().reshape(-1))
            rows.append({
                "state_id": index,
                "loss_difference": float((reference_loss - native_loss).detach().cpu()),
                "gradient_effect_rms_over_native": ratio(reference_grad - native_grad, native_grad),
                "write_effect_rms_over_native": ratio(effect, native_write),
                "write_aligned": float(torch.sum(effect * native_write).item()) /
                max(float(torch.sum(native_write * native_write).item()), 1e-30),
                "exact_gradient_effect": bool(torch.count_nonzero(reference_grad - native_grad) == 0),
                "exact_write_effect": bool(torch.count_nonzero(effect) == 0),
            })
            del ids, labels, native, reference
            torch.cuda.empty_cache()
    finally:
        attention._attn = original_attn
    split = len(rows) // 2
    calibration = torch.stack(effects[:split], dim=0).mean(dim=0)
    norm = float(torch.linalg.vector_norm(calibration).item())
    projections: list[float] = []
    if norm > 0:
        direction = calibration / norm
        projections = [float(torch.dot(value, direction).item()) for value in effects[split:]]
    aligned = [rows[i]["write_aligned"] for i in range(split, len(rows))]
    return {
        "schema": "kernel-analyzer-gptneo-local-attention-materialization-natural-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "layer": args.layer,
        "target": f"transformer.h[{args.layer}].attn.attention.q_proj.weight",
        "operator": "GPT-Neo local causal/window attention mask materialization",
        "candidate": "native GPT-Neo boolean local causal mask via torch.where",
        "reference": "same GPT-Neo attention with explicit additive masked-fill materialization",
        "input_source": str(args.text_source),
        "comparison_scope": {"same_model_weights": True, "same_input_ids": True, "same_qkv": True, "single_changed_boundary": "local causal/window mask materialization"},
        "claim_boundary": "One GPT-Neo checkpoint/layer and declared real-text bank; negative-control probe, not a population or loss-quality claim.",
        "rows": rows,
        "summary": {
            "state_count": len(rows), "calibration_count": split, "confirmation_count": len(rows) - split,
            "gradient_effect_rms_mean": sum(r["gradient_effect_rms_over_native"] for r in rows) / len(rows),
            "write_effect_rms_mean": sum(r["write_effect_rms_over_native"] for r in rows) / len(rows),
            "write_aligned_mean": sum(aligned) / len(aligned),
            "write_aligned_interval_normal_95": interval(aligned),
            "projection_mean": sum(projections) / len(projections) if projections else None,
            "projection_interval_normal_95": interval(projections) if projections else None,
            "exact_gradient_effect_count": sum(r["exact_gradient_effect"] for r in rows),
            "exact_write_effect_count": sum(r["exact_write_effect"] for r in rows),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--text-source", type=Path, default=DEFAULT_TEXT)
    parser.add_argument("--states", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=0)
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
