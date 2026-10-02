#!/usr/bin/env python3
"""Source decomposition for the Qwen3 eager/SDPA attention boundary.

The model, inputs, Q/K/V tensors and downstream graph are held fixed.  Each
variant changes one arithmetic boundary in the eager attention implementation:
score contraction, softmax/value materialization, or value contraction.  The
result is a source probe, not a population or loss-quality claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.qwen3 import modeling_qwen3


def ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a)) / max(float(torch.linalg.vector_norm(b)), 1e-30)


def write(g: torch.Tensor, lr: float, eps: float) -> torch.Tensor:
    g = g.float()
    return -lr * g / (g.abs() + eps)


def interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    if x.numel() < 2:
        m = float(x.mean())
        return [m, m]
    m = float(x.mean())
    h = 1.96 * float(x.std(unbiased=True)) / (x.numel() ** 0.5)
    return [m - h, m + h]


def component_attention(
    variant: str,
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **_: Any,
):
    key_states = modeling_qwen3.repeat_kv(key, module.num_key_value_groups)
    value_states = modeling_qwen3.repeat_kv(value, module.num_key_value_groups)

    if variant in {"score_fp32", "full_fp32"}:
        attn_weights = torch.matmul(query.float(), key_states.float().transpose(2, 3)) * scaling
        attn_weights = attn_weights.to(query.dtype)
    else:
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    if variant in {"softmax_value_fp32", "full_fp32"}:
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states.float()).to(query.dtype)
    elif variant == "value_fp32":
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights.float(), value_states.float()).to(query.dtype)
    else:
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)

    return attn_output.transpose(1, 2).contiguous(), attn_weights


def run(args: argparse.Namespace) -> dict[str, Any]:
    bank = json.loads(args.input_bank.read_text(encoding="utf-8"))
    states = bank["states"][: args.states]
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device).train()
    model.config.use_cache = False
    target_module = model.model.layers[args.layer].self_attn
    target = target_module.q_proj.weight if args.parameter == "q_proj" else target_module.v_proj.weight
    original = modeling_qwen3.eager_attention_forward
    variants = ["sdpa", "score_fp32", "value_fp32", "softmax_value_fp32", "full_fp32"]
    rows: list[dict[str, Any]] = []
    effect_vectors: dict[str, list[torch.Tensor]] = {variant: [] for variant in variants}
    try:
        for state in states:
            values = state.get("token_ids", state.get("input_ids"))
            ids = torch.tensor([values], dtype=torch.long, device=device)
            labels = ids.clone()
            model.zero_grad(set_to_none=True)
            native_inputs: dict[str, torch.Tensor] = {}

            def capture_native(*call_args: Any, **call_kwargs: Any):
                if not call_args or call_args[0] is not target_module:
                    return original(*call_args, **call_kwargs)
                query = call_kwargs.get("query", call_args[1] if len(call_args) > 1 else None)
                key = call_kwargs.get("key", call_args[2] if len(call_args) > 2 else None)
                value = call_kwargs.get("value", call_args[3] if len(call_args) > 3 else None)
                if query is not None:
                    native_inputs["query"] = query.detach().float().cpu()
                if key is not None:
                    native_inputs["key"] = key.detach().float().cpu()
                if value is not None:
                    native_inputs["value"] = value.detach().float().cpu()
                return original(*call_args, **call_kwargs)

            modeling_qwen3.eager_attention_forward = capture_native
            native_loss = model(input_ids=ids, labels=labels, use_cache=False).loss
            native_loss.backward()
            native_grad = target.grad.detach().float().cpu().clone()
            native_write = write(native_grad, args.learning_rate, args.eps)
            row: dict[str, Any] = {
                "state_id": state.get("state_id", state.get("sequence_id")),
                "native_loss": float(native_loss.detach().cpu()),
                "variants": {},
            }
            for variant in variants:
                model.zero_grad(set_to_none=True)
                variant_inputs: dict[str, torch.Tensor] = {}

                def capture_variant(*call_args: Any, _variant: str = variant, **call_kwargs: Any):
                    if not call_args or call_args[0] is not target_module:
                        return original(*call_args, **call_kwargs)
                    query = call_kwargs.get("query", call_args[1] if len(call_args) > 1 else None)
                    key = call_kwargs.get("key", call_args[2] if len(call_args) > 2 else None)
                    value = call_kwargs.get("value", call_args[3] if len(call_args) > 3 else None)
                    if query is not None:
                        variant_inputs["query"] = query.detach().float().cpu()
                    if key is not None:
                        variant_inputs["key"] = key.detach().float().cpu()
                    if value is not None:
                        variant_inputs["value"] = value.detach().float().cpu()
                    if _variant == "sdpa":
                        return sdpa_attention_forward(*call_args, **call_kwargs)
                    return component_attention(_variant, *call_args, **call_kwargs)

                if variant == "sdpa":
                    modeling_qwen3.eager_attention_forward = capture_variant
                else:
                    modeling_qwen3.eager_attention_forward = capture_variant
                changed_loss = model(input_ids=ids, labels=labels, use_cache=False).loss
                changed_loss.backward()
                changed_grad = target.grad.detach().float().cpu().clone()
                changed_write = write(changed_grad, args.learning_rate, args.eps)
                effect = native_write - changed_write
                effect_vectors[variant].append(effect)
                row["variants"][variant] = {
                    "loss_difference": float((native_loss - changed_loss).detach().cpu()),
                    "gradient_effect_rms_over_variant": ratio(native_grad - changed_grad, changed_grad),
                    "write_effect_rms_over_variant": ratio(effect, changed_write),
                    "write_aligned_over_variant": float(torch.sum(effect * changed_write))
                    / max(float(torch.sum(changed_write * changed_write)), 1e-30),
                    "attention_input_max_abs_diff": max(
                        (
                            float(torch.max(torch.abs(variant_inputs[name] - native_inputs[name])))
                            for name in ("query", "key", "value")
                            if name in variant_inputs and name in native_inputs
                        ),
                        default=0.0,
                    ),
                }
            rows.append(row)
            del ids, labels, native_loss
            torch.cuda.empty_cache()
    finally:
        modeling_qwen3.eager_attention_forward = original

    summary: dict[str, Any] = {"state_count": len(rows), "variants": {}}
    for variant in variants:
        vals = [r["variants"][variant] for r in rows]
        summary["variants"][variant] = {
            "write_effect_rms_mean": sum(x["write_effect_rms_over_variant"] for x in vals) / len(vals),
            "write_aligned_interval_normal_95": interval([x["write_aligned_over_variant"] for x in vals]),
            "loss_difference_interval_normal_95": interval([x["loss_difference"] for x in vals]),
        }
    sdpa_effects = effect_vectors["sdpa"]
    summary["relative_to_sdpa"] = {}
    for variant in variants:
        if variant == "sdpa":
            continue
        residuals = [effect_vectors[variant][i] - sdpa_effects[i] for i in range(len(rows))]
        denom = [max(float(torch.linalg.vector_norm(x).item()), 1e-30) for x in sdpa_effects]
        summary["relative_to_sdpa"][variant] = {
            "effect_vector_difference_rms_over_sdpa_mean": sum(
                float(torch.linalg.vector_norm(residuals[i]).item()) / denom[i]
                for i in range(len(rows))
            ) / len(rows),
            "effect_vector_cosine_mean": sum(
                float(torch.dot(effect_vectors[variant][i].reshape(-1), sdpa_effects[i].reshape(-1)).item())
                / max(float(torch.linalg.vector_norm(effect_vectors[variant][i]).item()) * denom[i], 1e-30)
                for i in range(len(rows))
            ) / len(rows),
        }
    return {
        "schema": "kernel-analyzer-qwen3-attention-component-source-probe-v1",
        "status": "COMPLETE_SOURCE_DECOMPOSITION_PROBE",
        "model": str(args.model), "layer": args.layer, "parameter": args.parameter,
        "input_source": "declared real Qwen3 text input bank",
        "reference": "native eager attention at the same model and inputs",
        "variants": variants,
        "comparison_scope": {
            "same_model_weights": True,
            "same_input_ids": True,
            "same_attention_inputs": True,
            "single_changed_boundaries": variants,
        },
        "claim_boundary": "The variants isolate arithmetic boundaries inside eager attention on one checkpoint and bank; a matching effect would support source attribution, not a population or loss-quality claim.",
        "rows": rows,
        "summary": summary,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    p.add_argument("--input-bank", type=Path, default=Path("results/coverage/qwen_seq128_input_bank.json"))
    p.add_argument("--layer", type=int, default=13)
    p.add_argument("--parameter", choices=("q_proj", "v_proj"), default="q_proj")
    p.add_argument("--states", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
