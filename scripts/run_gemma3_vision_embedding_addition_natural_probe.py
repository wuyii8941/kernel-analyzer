#!/usr/bin/env python3
"""Natural Gemma-3 vision patch-plus-position embedding probe.

The native SigLIP vision embedding path adds patch embeddings and learned
position embeddings in the model dtype.  The reference changes only that
addition to FP32 before casting back; the convolution, position lookup and
all downstream vision/text computation remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import types
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/data1/tzh/models/google/gemma-3-4b-pt")
DEFAULT_BANK = ROOT / "results/property/tcmp_allop_v1/input_banks/gemma3_4b_image_text128.json"


def prepare_values(state: dict[str, Any], processor: Any, device: torch.device):
    image = Image.open(state["image_path"]).convert("RGB")
    prepared = processor(text=state["prompt"], images=image, return_tensors="pt")
    labels = prepared["input_ids"].clone()
    labels[prepared["token_type_ids"] == 1] = -100
    return (
        prepared["input_ids"].to(device),
        prepared["pixel_values"].to(device, dtype=torch.bfloat16),
        prepared["attention_mask"].to(device),
        prepared["token_type_ids"].to(device),
        labels.to(device),
    )


def install_fp32_addition(module: torch.nn.Module) -> None:
    def forward(self: torch.nn.Module, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False):
        if interpolate_pos_encoding:
            raise RuntimeError("this probe requires the fixed learned-position path")
        _, _, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        position = self.position_embedding(self.position_ids)
        return (embeddings.float() + position.float()).to(dtype=embeddings.dtype)

    module.forward = types.MethodType(forward, module)


def adamw_write(gradient: torch.Tensor, lr: float, eps: float) -> torch.Tensor:
    grad = gradient.float()
    return -lr * grad / (grad.abs() + eps)


def relative(effect: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(effect).item()) / max(
        float(torch.linalg.vector_norm(reference).item()), 1e-30
    )


def interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    mean = float(x.mean().item())
    if x.numel() < 2:
        return [mean, mean]
    half = 1.96 * float(x.std(unbiased=True).item()) / (x.numel() ** 0.5)
    return [mean - half, mean + half]


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    bank = json.loads(args.input_bank.read_text(encoding="utf-8"))
    states = bank["states"][: args.states]
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        str(args.model), dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True,
    ).to(device).train()
    model.gradient_checkpointing_enable()
    vision_embeddings = model.model.vision_tower.vision_model.embeddings
    original_forward = vision_embeddings.forward
    target = vision_embeddings.position_embedding.weight
    rows: list[dict[str, Any]] = []
    effects: list[torch.Tensor] = []
    write_effects: list[torch.Tensor] = []
    references: list[torch.Tensor] = []
    try:
        for state in states:
            values = prepare_values(state, processor, device)
            model.zero_grad(set_to_none=True)
            vision_embeddings.forward = original_forward
            candidate_loss = model(
                input_ids=values[0], pixel_values=values[1], attention_mask=values[2],
                token_type_ids=values[3], labels=values[4], use_cache=False,
            ).loss
            candidate_loss.backward()
            candidate_gradient = target.grad.detach().float().cpu().clone()
            candidate_write = adamw_write(candidate_gradient, args.learning_rate, args.eps)

            model.zero_grad(set_to_none=True)
            install_fp32_addition(vision_embeddings)
            reference_loss = model(
                input_ids=values[0], pixel_values=values[1], attention_mask=values[2],
                token_type_ids=values[3], labels=values[4], use_cache=False,
            ).loss
            reference_loss.backward()
            reference_gradient = target.grad.detach().float().cpu().clone()
            reference_write = adamw_write(reference_gradient, args.learning_rate, args.eps)

            effect = candidate_gradient - reference_gradient
            write_effect = candidate_write - reference_write
            effects.append(effect)
            write_effects.append(write_effect)
            references.append(reference_write)
            rows.append({
                "state_id": state["state_id"],
                "loss_difference_native_minus_fp32_addition": float((candidate_loss - reference_loss).detach().cpu().item()),
                "gradient_effect_rms_over_reference": relative(effect, reference_gradient),
                "write_effect_rms_over_reference": relative(write_effect, reference_write),
                "finite": bool(torch.isfinite(candidate_gradient).all() and torch.isfinite(reference_gradient).all()),
            })
            vision_embeddings.forward = original_forward
            del values, candidate_loss, reference_loss
            torch.cuda.empty_cache()
    finally:
        vision_embeddings.forward = original_forward

    split = len(effects) // 2
    direction = torch.stack([x.double().reshape(-1) for x in effects[:split]]).mean(dim=0)
    direction_norm = float(torch.linalg.vector_norm(direction).item())
    projections: list[float] = []
    if direction_norm:
        direction = direction / direction_norm
        projections = [float(torch.dot(x.double().reshape(-1), direction).item()) for x in effects[split:]]
    aligned = [
        float(torch.sum(effect.double() * ref.double()).item())
        / max(float(torch.sum(ref.double() ** 2).item()), 1e-30)
        for effect, ref in zip(write_effects[split:], references[split:])
    ]
    return {
        "schema": "kernel-analyzer-gemma3-vision-embedding-addition-natural-probe-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "operator": "Gemma-3 SigLIP patch-plus-position embedding addition",
        "parameter": "model.vision_tower.vision_model.embeddings.position_embedding.weight",
        "candidate": "native model-dtype patch embedding plus learned position embedding",
        "reference": "same path with only patch-plus-position addition evaluated in FP32 then cast back",
        "input_source": "real Gemma-3 image/text input bank",
        "comparison_scope": {
            "same_model_weights": True,
            "same_images_and_text": True,
            "same_patch_convolution": True,
            "single_changed_boundary": "vision patch-plus-position embedding addition",
        },
        "rows": rows,
        "summary": {
            "state_count": len(rows),
            "calibration_count": split,
            "confirmation_count": len(rows) - split,
            "gradient_effect_rms_mean": sum(r["gradient_effect_rms_over_reference"] for r in rows) / len(rows),
            "write_effect_rms_mean": sum(r["write_effect_rms_over_reference"] for r in rows) / len(rows),
            "projection_mean": sum(projections) / len(projections) if projections else None,
            "projection_interval_normal_95": interval(projections) if projections else None,
            "projection_positive": sum(x > 0 for x in projections),
            "projection_negative": sum(x < 0 for x in projections),
            "aligned_write_mean": sum(aligned) / len(aligned),
            "aligned_write_interval_normal_95": interval(aligned),
            "loss_difference_interval_normal_95": interval([r["loss_difference_native_minus_fp32_addition"] for r in rows]),
            "calibration_direction_norm": direction_norm,
        },
        "claim_boundary": "One Gemma-3 checkpoint, fixed vision embedding boundary and declared image/text bank; no population or quality guarantee.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--input-bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--states", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **result["summary"]}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
