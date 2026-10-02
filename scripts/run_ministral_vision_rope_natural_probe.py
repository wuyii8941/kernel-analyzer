#!/usr/bin/env python3
"""Natural Ministral vision 2-D RoPE materialization probe.

The candidate is the released Pixtral vision RoPE path: cosine/sine values
are computed in FP32, cast to the model dtype, and applied in that dtype.
The reference keeps the same operands and model path, but performs the
rotation in FP32 before one cast back to the model dtype.  The probe measures
the resulting real multimodal loss gradient and a one-step zero-moment
parameter write on the vision patch convolution.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoTokenizer, Mistral3ForConditionalGeneration
import transformers.models.pixtral.modeling_pixtral as pixtral_impl


MEAN = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)[:, None, None]
STD = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)[:, None, None]


def image_tensor(path: Path, side: int, dtype: torch.dtype) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((side, side), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array.transpose(2, 0, 1) - MEAN) / STD
    return torch.tensor(array, dtype=dtype).unsqueeze(0)


def multimodal_batch(
    tokenizer: Any,
    config: Any,
    image: torch.Tensor,
    side: int,
) -> dict[str, torch.Tensor]:
    patch_grid = side // int(config.vision_config.patch_size)
    merge = int(config.spatial_merge_size)
    image_token_count = (patch_grid // merge) ** 2
    prompt = [
        int(config.text_config.bos_token_id),
        3,  # [INST]
        *([int(config.image_token_id)] * image_token_count),
        *tokenizer.encode("Describe the image.", add_special_tokens=False),
        4,  # [/INST]
    ]
    answer = tokenizer.encode("The image shows a scene.", add_special_tokens=False)
    answer.append(int(config.text_config.eos_token_id))
    ids = torch.tensor([prompt + answer], dtype=torch.long)
    labels = torch.full_like(ids, -100)
    labels[:, len(prompt) :] = ids[:, len(prompt) :]
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "labels": labels,
        "pixel_values": image,
        "image_sizes": torch.tensor([[side, side]], dtype=torch.long),
    }


def first_step_write(gradient: torch.Tensor, learning_rate: float) -> torch.Tensor:
    # Same zero-moment AdamW direction for both branches; this is a local
    # training-state comparison, not a claim about a full optimizer trajectory.
    return -learning_rate * gradient / (gradient.abs() + 1e-8)


def run_branch(
    model: torch.nn.Module,
    vision: torch.nn.Module,
    original_rope: Any,
    original_apply: Any,
    batch: dict[str, torch.Tensor],
    target: torch.nn.Parameter,
    *,
    fp32_rotation: bool,
) -> tuple[float, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    if fp32_rotation:
        def rope_forward(_: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            frequencies = vision.patch_positional_embedding.inv_freq[position_ids].float()
            return frequencies.cos(), frequencies.sin()

        def apply_rotation(
            query: torch.Tensor,
            key: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            unsqueeze_dim: int = 1,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            query_rot, key_rot = original_apply(
                query.float(), key.float(), cos, sin, unsqueeze_dim=unsqueeze_dim
            )
            return query_rot.to(query.dtype), key_rot.to(key.dtype)

        vision.patch_positional_embedding.forward = rope_forward
        pixtral_impl.apply_rotary_pos_emb = apply_rotation
    else:
        vision.patch_positional_embedding.forward = original_rope
        pixtral_impl.apply_rotary_pos_emb = original_apply
    try:
        output = model(**batch, use_cache=False, return_dict=True)
        loss = output.loss
        if loss is None or not bool(torch.isfinite(loss.detach())):
            raise RuntimeError("non-finite multimodal loss")
        loss.backward()
        if target.grad is None:
            raise RuntimeError("vision patch-convolution gradient is absent")
        return float(loss.detach()), target.grad.detach().float().clone()
    finally:
        vision.patch_positional_embedding.forward = original_rope
        pixtral_impl.apply_rotary_pos_emb = original_apply


def confidence_interval(values: list[float]) -> list[float] | None:
    if len(values) < 2:
        return None
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    return [mean - 2.145 * standard_error, mean + 2.145 * standard_error]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--side", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    if args.states < 4:
        raise ValueError("at least four states are required for calibration and confirmation")

    torch.set_num_threads(min(16, torch.get_num_threads()))
    dtype = torch.bfloat16
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, fix_mistral_regex=True
    )
    model = Mistral3ForConditionalGeneration.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()
    vision = model.model.vision_tower
    target = vision.patch_conv.weight
    original_rope = vision.patch_positional_embedding.forward
    original_apply = pixtral_impl.apply_rotary_pos_emb
    images = sorted(args.image_dir.glob("image_*.png"))[: args.states]
    if len(images) != args.states:
        raise RuntimeError(f"requested {args.states} images but found {len(images)}")

    rows: list[dict[str, Any]] = []
    update_deltas: list[torch.Tensor] = []
    for index, image_path in enumerate(images):
        batch = multimodal_batch(tokenizer, config, image_tensor(image_path, args.side, dtype), args.side)
        candidate_loss, candidate_gradient = run_branch(
            model, vision, original_rope, original_apply, batch, target, fp32_rotation=False
        )
        reference_loss, reference_gradient = run_branch(
            model, vision, original_rope, original_apply, batch, target, fp32_rotation=True
        )
        candidate_update = first_step_write(candidate_gradient, args.learning_rate)
        reference_update = first_step_write(reference_gradient, args.learning_rate)
        update_delta = candidate_update - reference_update
        update_deltas.append(update_delta)
        aligned = float(
            torch.sum(update_delta * reference_update)
            / (torch.sum(reference_update * reference_update) + 1e-30)
        )
        rows.append(
            {
                "state_id": image_path.stem,
                "candidate_loss": candidate_loss,
                "reference_loss": reference_loss,
                "gradient_effect_rms": float((candidate_gradient - reference_gradient).pow(2).mean().sqrt()),
                "write_effect_rms": float(update_delta.pow(2).mean().sqrt()),
                "reference_write_rms": float(reference_update.pow(2).mean().sqrt()),
                "relative_write_rms": float(update_delta.norm() / (reference_update.norm() + 1e-30)),
                "aligned_write": aligned,
            }
        )
        print(index + 1, image_path.name, aligned, flush=True)

    calibration = torch.stack(update_deltas[: args.states // 2])
    confirmation = torch.stack(update_deltas[args.states // 2 :])
    direction = calibration.mean(dim=0)
    direction_norm = float(direction.norm())
    direction = direction / max(direction_norm, 1e-30)
    confirmation_projection = [
        float(torch.sum(delta * direction) / (delta.norm() + 1e-30)) for delta in confirmation
    ]
    aligned_confirmation = [row["aligned_write"] for row in rows[args.states // 2 :]]
    result = {
        "schema": "kernel-analyzer-ministral-vision-rope-natural-probe-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "operator": "Ministral Pixtral vision two-dimensional rotary position embedding",
        "candidate": "native BF16 cosine/sine rotation",
        "reference": "FP32 cosine/sine rotation with one BF16 write-back",
        "parameter": "model.vision_tower.patch_conv.weight",
        "input_source": "real CIFAR image bank with real multimodal language loss",
        "comparison_scope": {
            "same_model_weights": True,
            "same_image_and_text_inputs": True,
            "single_changed_boundary": "vision RoPE intermediate rotation precision",
            "optimizer_proxy": "zero-moment AdamW first-step write",
        },
        "state_count": len(rows),
        "calibration_count": args.states // 2,
        "confirmation_count": args.states - args.states // 2,
        "rows": rows,
        "summary": {
            "write_effect_rms_mean": float(np.mean([row["write_effect_rms"] for row in rows])),
            "relative_write_rms_mean": float(np.mean([row["relative_write_rms"] for row in rows])),
            "aligned_write_mean_confirmation": float(np.mean(aligned_confirmation)),
            "aligned_write_interval_normal_95": confidence_interval(aligned_confirmation),
            "heldout_additive_direction_norm_calibration": direction_norm,
            "heldout_additive_projection_confirmation": confirmation_projection,
            "heldout_additive_projection_interval_normal_95": confidence_interval(confirmation_projection),
            "aligned_sign_counts_confirmation": {
                "positive": sum(value > 0 for value in aligned_confirmation),
                "negative": sum(value < 0 for value in aligned_confirmation),
            },
        },
        "claim_boundary": "One Ministral-3 checkpoint, one vision patch-convolution carrier, and the declared image/text bank; no universal vision or language-model quality guarantee.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
