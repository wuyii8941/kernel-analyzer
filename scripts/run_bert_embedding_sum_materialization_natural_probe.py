#!/usr/bin/env python3
"""Natural BERT embedding-sum materialization probe.

The native BERT embedding block adds word, position and token-type embeddings
in the model dtype before LayerNorm.  The reference replaces only that sum by
an FP32 sum followed by a cast, leaving LayerNorm and the encoder unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
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


def write(g: torch.Tensor, lr: float, eps: float) -> torch.Tensor:
    g = g.float()
    return -lr * g / (g.abs() + eps)


def relative(effect: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(effect).item()) / max(
        float(torch.linalg.vector_norm(reference).item()), 1e-30
    )


def run_once(
    model: torch.nn.Module,
    target: torch.nn.Parameter,
    ids: torch.Tensor,
    fp32_addition: bool,
) -> tuple[float, torch.Tensor]:
    embeddings = model.bert.embeddings
    captured: dict[str, torch.Tensor] = {}

    def save(name: str):
        def hook(_module: torch.nn.Module, _args: tuple[Any, ...], output: torch.Tensor):
            captured[name] = output
            return output
        return hook

    def replace_sum(_module: torch.nn.Module, args: tuple[Any, ...]):
        if not fp32_addition:
            return args
        required = ("word", "position", "token_type")
        if any(name not in captured for name in required):
            raise RuntimeError("embedding hooks did not capture all three operands")
        hidden = sum((captured[name].float() for name in required))
        return (hidden.to(args[0].dtype),) + tuple(args[1:])

    hooks = [
        embeddings.word_embeddings.register_forward_hook(save("word")),
        embeddings.position_embeddings.register_forward_hook(save("position")),
        embeddings.token_type_embeddings.register_forward_hook(save("token_type")),
        embeddings.LayerNorm.register_forward_pre_hook(replace_sum),
    ]
    model.zero_grad(set_to_none=True)
    try:
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
    finally:
        for hook in hooks:
            hook.remove()
    if target.grad is None:
        raise RuntimeError("target gradient is missing")
    return float(loss.detach().cpu().item()), target.grad.detach().float().cpu().clone()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    bank = windows(tokenizer, args.text_source, args.sequence_length, args.states)
    model = AutoModelForMaskedLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16
    ).to(args.device).eval()
    named = dict(model.named_parameters())
    target = named[args.parameter]
    effects: list[torch.Tensor] = []
    writes: list[torch.Tensor] = []
    refs: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    for state_id, tokens in enumerate(bank):
        ids = tokens.unsqueeze(0).to(args.device)
        native_loss, native_grad = run_once(model, target, ids, False)
        fp32_loss, fp32_grad = run_once(model, target, ids, True)
        native_write = write(native_grad, args.learning_rate, args.eps)
        fp32_write = write(fp32_grad, args.learning_rate, args.eps)
        grad_effect = fp32_grad - native_grad
        write_effect = fp32_write - native_write
        effects.append(grad_effect.double().reshape(-1))
        writes.append(write_effect.double().reshape(-1))
        refs.append(native_write.double().reshape(-1))
        rows.append({
            "state_id": state_id,
            "native_loss": native_loss,
            "fp32_addition_loss": fp32_loss,
            "loss_difference_fp32_minus_native": fp32_loss - native_loss,
            "gradient_effect_rms_over_native": relative(grad_effect, native_grad),
            "write_effect_rms_over_native": relative(write_effect, native_write),
        })
        del ids
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
        "schema": "kernel-analyzer-bert-embedding-sum-materialization-natural-probe-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "operator": "BERT word-plus-position-plus-token-type embedding addition materialization",
        "parameter": args.parameter,
        "candidate": "native BERT embedding additions in model dtype",
        "reference": "same BERT path with only the three-embedding sum evaluated in FP32 then cast back",
        "input_source": str(args.text_source),
        "comparison_scope": {
            "same_model_weights": True,
            "same_input_ids": True,
            "same_layer_norm_and_encoder": True,
            "single_changed_boundary": "word/position/token-type embedding sum materialization",
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
            "confirmation_projection_mean": sum(projections) / len(projections) if projections else None,
            "confirmation_projection_interval_normal_95": interval(projections) if projections else None,
            "confirmation_projection_positive": sum(v > 0 for v in projections),
            "confirmation_projection_negative": sum(v < 0 for v in projections),
            "aligned_write_mean": sum(aligned) / len(aligned) if aligned else None,
            "aligned_write_interval_normal_95": interval(aligned) if aligned else None,
            "direction_norm": norm,
        },
        "claim_boundary": "One BERT-tiny checkpoint, one embedding-sum boundary and the declared text bank; no population or long-run loss claim.",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--text-source", type=Path, default=DEFAULT_TEXT)
    p.add_argument("--parameter", default="bert.embeddings.word_embeddings.weight")
    p.add_argument("--sequence-length", type=int, default=64)
    p.add_argument("--states", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **result["summary"]}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
