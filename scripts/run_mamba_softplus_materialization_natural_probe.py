#!/usr/bin/env python3
"""Path-preserving natural probe for Mamba softplus materialisation.

The model, token windows, recurrent implementation, and all parameters are
held fixed.  The only intervention is whether the ``delta_softplus`` value is
computed in the input dtype or in FP32 before being cast back.  This is a
source-boundary probe, not a claim that every Mamba recurrence difference is a
softplus problem.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, MambaForCausalLM
from transformers.models.mamba import modeling_mamba


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/data1/tzh/models/state-spaces/mamba-130m-hf")
DEFAULT_ARROW = Path(
    "/data1/tzh/cache/huggingface/datasets/Salesforce___wikitext/"
    "wikitext-103-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/"
    "wikitext-validation.arrow"
)
DEFAULT_OUTPUT = ROOT / "results/property/new_problem_group_search_v1/mamba_softplus_materialization_natural_16_20260918.json"


def windows(tokenizer: Any, arrow: Path | None, seq_len: int, count: int, text_source: Path | None) -> list[torch.Tensor]:
    if text_source is not None:
        text = text_source.read_text(encoding="utf-8")
    else:
        try:
            from datasets import Dataset
        except ModuleNotFoundError as exc:
            raise RuntimeError("install datasets or pass --text-source") from exc
        if arrow is None:
            raise RuntimeError("--validation-arrow is required without --text-source")
        dataset = Dataset.from_file(str(arrow))
        text = "\n".join(row["text"] for row in dataset if row["text"].strip())
    tokens = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    # For a small local text source use separated, but still feasible, windows.
    # The source is a declared probe bank rather than a population sample.
    stride = seq_len * 2
    need = count * stride + seq_len
    if tokens.numel() < need:
        raise RuntimeError(f"validation corpus has {tokens.numel()} tokens, need {need}")
    return [tokens[i * stride : i * stride + seq_len].clone() for i in range(count)]


def first_step_write(gradient: torch.Tensor, learning_rate: float, eps: float) -> torch.Tensor:
    grad = gradient.float()
    return -learning_rate * grad / (grad.abs() + eps)


def relative_rms(effect: torch.Tensor, reference: torch.Tensor) -> float:
    denom = float(torch.linalg.vector_norm(reference).item())
    return 0.0 if denom == 0.0 else float(torch.linalg.vector_norm(effect).item()) / denom


def run_once(model: torch.nn.Module, target: torch.nn.Parameter, ids: torch.Tensor, mode: str) -> tuple[float, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    original_softplus = modeling_mamba.F.softplus
    if mode == "fp32":
        def fp32_softplus(x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            return original_softplus(x.float(), *args, **kwargs).to(x.dtype)
        modeling_mamba.F.softplus = fp32_softplus
    try:
        loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        loss.backward()
    finally:
        modeling_mamba.F.softplus = original_softplus
    if target.grad is None:
        raise RuntimeError("target gradient is missing")
    return float(loss.detach().cpu().item()), target.grad.detach().float().cpu().clone()


def normal_interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    mean = float(x.mean().item())
    if len(values) < 2:
        return [mean, mean]
    half = 1.96 * float(x.std(unbiased=True).item()) / (len(values) ** 0.5)
    return [mean - half, mean + half]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--validation-arrow", type=Path, default=DEFAULT_ARROW)
    parser.add_argument("--text-source", type=Path, default=ROOT / "README.md")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--parameter", default="mixer.dt_proj.weight")
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--states", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    state_bank = windows(tokenizer, args.validation_arrow, args.sequence_length, args.states, args.text_source)
    model = MambaForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, local_files_only=True).to(args.device).eval()
    model.config.use_cache = False
    layer = model.backbone.layers[args.layer]
    if args.parameter.startswith("mixer."):
        target = dict(layer.mixer.named_parameters())[args.parameter.split(".", 1)[1]]
    else:
        target = dict(model.named_parameters())[args.parameter]

    rows = []
    effects = []
    writes = []
    refs = []
    for state_id, tokens in enumerate(state_bank):
        ids = tokens.unsqueeze(0).to(args.device)
        native_loss, native_grad = run_once(model, target, ids, "native")
        fp32_loss, fp32_grad = run_once(model, target, ids, "fp32")
        grad_effect = fp32_grad - native_grad
        native_write = first_step_write(native_grad, args.learning_rate, args.eps)
        fp32_write = first_step_write(fp32_grad, args.learning_rate, args.eps)
        write_effect = fp32_write - native_write
        effects.append(grad_effect.double().reshape(-1))
        writes.append(write_effect.double().reshape(-1))
        refs.append(native_write.double().reshape(-1))
        rows.append({
            "state_id": state_id,
            "native_loss": native_loss,
            "fp32_softplus_loss": fp32_loss,
            "loss_difference_fp32_minus_native": fp32_loss - native_loss,
            "gradient_effect_rms_over_native": relative_rms(grad_effect, native_grad),
            "write_effect_rms_over_native": relative_rms(write_effect, native_write),
            "gradient_effect_signed_mean": float(grad_effect.mean().item()),
            "write_effect_signed_mean": float(write_effect.mean().item()),
        })
        del ids
        torch.cuda.empty_cache()

    split = args.states // 2
    calibration = torch.stack(effects[:split])
    direction = calibration.mean(dim=0)
    direction_norm = float(torch.linalg.vector_norm(direction).item())
    projections: list[float] = []
    if direction_norm:
        direction = direction / direction_norm
        projections = [float(torch.dot(value, direction).item()) for value in effects[split:]]
    aligned = [
        float(torch.dot(effect, ref).item()) / max(float(torch.dot(ref, ref).item()), 1e-30)
        for effect, ref in zip(writes[split:], refs[split:])
    ]
    output = {
        "schema": "kernel-analyzer-mamba-softplus-materialisation-natural-probe-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "operator": "Mamba selective-scan delta softplus materialisation",
        "parameter": args.parameter,
        "candidate": "sequential Mamba path with native input-dtype softplus",
        "reference": "same sequential path with FP32 softplus cast back to input dtype",
        "input_source": str(args.text_source if args.text_source is not None else args.validation_arrow),
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
            "confirmation_projection_interval_normal_95": normal_interval(projections) if projections else None,
            "confirmation_projection_positive": sum(v > 0 for v in projections),
            "confirmation_projection_negative": sum(v < 0 for v in projections),
            "aligned_write_mean": sum(aligned) / len(aligned) if aligned else None,
            "aligned_write_interval_normal_95": normal_interval(aligned) if aligned else None,
            "direction_norm": direction_norm,
        },
        "claim_boundary": "One real Mamba checkpoint and declared validation bank; this is a path-preserving softplus precision probe, not a population or long-run loss claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), **output["summary"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
