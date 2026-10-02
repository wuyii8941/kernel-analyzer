#!/usr/bin/env python3
"""Natural Gemma-3 vision-convolution probe without provenance side channels.

The candidate is the native BF16 patch convolution.  The reference changes
only the selected convolution's accumulation to FP32 and casts its output
back to the native dtype.  The result is scoped to the real image/text bank,
the first vision convolution, and the selected parameter-write endpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import types
from typing import Any

import torch
import torch.nn.functional as F
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


def install_fp32_conv(module: torch.nn.Conv2d) -> None:
    def forward(self: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        value = F.conv2d(
            x.float(), self.weight.float(),
            None if self.bias is None else self.bias.float(),
            self.stride, self.padding, self.dilation, self.groups,
        )
        return value.to(x.dtype)

    module.forward = types.MethodType(forward, module)


def adamw_write(gradient: torch.Tensor, lr: float, eps: float) -> torch.Tensor:
    grad = gradient.float()
    return -lr * grad / (grad.abs() + eps)


def ratio(effect: torch.Tensor, reference: torch.Tensor) -> float:
    denom = float(torch.linalg.vector_norm(reference).item())
    return float(torch.linalg.vector_norm(effect).item()) / max(denom, 1e-30)


def normal_interval(values: list[float]) -> list[float]:
    x = torch.tensor(values, dtype=torch.float64)
    mean = float(x.mean().item())
    if len(values) < 2:
        return [mean, mean]
    half = 1.96 * float(x.std(unbiased=True).item()) / (len(values) ** 0.5)
    return [mean - half, mean + half]


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    bank = json.loads(args.input_bank.read_text())
    states = bank["states"][: args.states]
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        str(args.model), dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True,
    ).to(device).train()
    model.gradient_checkpointing_enable()
    candidate_convs = [(name, module) for name, module in model.named_modules()
                       if isinstance(module, torch.nn.Conv2d)]
    if len(candidate_convs) != 1:
        raise RuntimeError(f"expected one vision convolution, got {len(candidate_convs)}")
    conv_name, candidate_conv = candidate_convs[0]
    candidate_weight = candidate_conv.weight
    original_forward = candidate_conv.forward
    rows = []
    effects = []
    write_effects = []
    write_references = []
    for state in states:
        values = prepare_values(state, processor, device)
        model.zero_grad(set_to_none=True)
        candidate_conv.forward = original_forward
        candidate_loss = model(
            input_ids=values[0], pixel_values=values[1], attention_mask=values[2],
            token_type_ids=values[3], labels=values[4], use_cache=False,
        ).loss
        candidate_loss.backward()
        candidate_gradient = candidate_weight.grad.detach().float().cpu().clone()
        candidate_write = adamw_write(candidate_gradient, args.learning_rate, args.eps)

        model.zero_grad(set_to_none=True)
        install_fp32_conv(candidate_conv)
        reference_loss = model(
            input_ids=values[0], pixel_values=values[1], attention_mask=values[2],
            token_type_ids=values[3], labels=values[4], use_cache=False,
        ).loss
        reference_loss.backward()
        reference_gradient = candidate_weight.grad.detach().float().cpu().clone()
        reference_write = adamw_write(reference_gradient, args.learning_rate, args.eps)

        gradient_effect = candidate_gradient - reference_gradient
        write_effect = candidate_write - reference_write
        effects.append(gradient_effect)
        write_effects.append(write_effect)
        write_references.append(reference_write)
        rows.append({
            "state_id": state["state_id"],
            "loss_difference": float((candidate_loss - reference_loss).detach().cpu().item()),
            "gradient_effect_rms_over_reference": ratio(gradient_effect, reference_gradient),
            "write_effect_rms_over_reference": ratio(write_effect, reference_write),
            "gradient_signed_mean": float(gradient_effect.mean().item()),
            "write_signed_mean": float(write_effect.mean().item()),
        })
        candidate_conv.forward = original_forward
        del values, candidate_loss, reference_loss
        torch.cuda.empty_cache()

    split = len(effects) // 2
    direction = torch.stack([x.double() for x in effects[:split]]).mean(dim=0)
    direction_norm = float(torch.linalg.vector_norm(direction).item())
    projections: list[float] = []
    if direction_norm:
        direction = direction / direction_norm
        projections = [float(torch.sum(effect.double() * direction).item()) for effect in effects[split:]]
    aligned = []
    for effect, repair in zip(write_effects[split:], write_references[split:]):
        effect_flat = effect.double().reshape(-1)
        repair_flat = repair.double().reshape(-1)
        den = float(torch.dot(repair_flat, repair_flat).item())
        aligned.append(float(torch.dot(effect_flat, repair_flat).item()) / max(den, 1e-30))

    result = {
        "schema": "kernel-analyzer-gemma3-convolution-clean-natural-probe-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "operator": "Gemma-3 vision patch convolution",
        "module": conv_name,
        "candidate": "native BF16 convolution",
        "reference": "same convolution with FP32 accumulation and native-dtype write-back",
        "input_source": "real image/text input bank",
        "optimizer": {"name": "AdamW", "zero_moments": True, "learning_rate": args.learning_rate},
        "claim_boundary": "One real Gemma-3 checkpoint, one vision convolution, and the declared image/text bank; not a population or quality guarantee.",
        "rows": rows,
        "summary": {
            "state_count": len(rows),
            "calibration_count": split,
            "confirmation_count": len(rows) - split,
            "gradient_effect_rms_mean": sum(r["gradient_effect_rms_over_reference"] for r in rows) / len(rows),
            "write_effect_rms_mean": sum(r["write_effect_rms_over_reference"] for r in rows) / len(rows),
            "projection_mean": sum(projections) / len(projections) if projections else None,
            "projection_interval_normal_95": normal_interval(projections) if projections else None,
            "projection_positive": sum(x > 0 for x in projections),
            "projection_negative": sum(x < 0 for x in projections),
            "aligned_write_mean": sum(aligned) / len(aligned),
            "aligned_write_interval_normal_95": normal_interval(aligned),
            "loss_difference_interval_normal_95": normal_interval([r["loss_difference"] for r in rows]),
            "calibration_direction_norm": direction_norm,
        },
    }
    return result


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
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
