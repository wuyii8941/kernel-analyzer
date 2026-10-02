#!/usr/bin/env python3
"""Natural source-factorial probe for the Ministral vision RoPE boundary.

Each branch keeps the same model, image/text input and downstream path.  Only
one arithmetic choice in the reviewed vision RoPE boundary changes at a time:
native BF16 trigonometry/rotation, higher-precision trigonometry with the
native BF16 rotation, FP32 multiply/add with native BF16 trigonometry, or both.
This is a source-isolation experiment, not a population or quality claim.
"""

from __future__ import annotations

import argparse
import json
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


def multimodal_batch(tokenizer: Any, config: Any, image: torch.Tensor, side: int) -> dict[str, torch.Tensor]:
    patch_grid = side // int(config.vision_config.patch_size)
    merge = int(config.spatial_merge_size)
    image_token_count = (patch_grid // merge) ** 2
    prompt = [
        int(config.text_config.bos_token_id), 3,
        *([int(config.image_token_id)] * image_token_count),
        *tokenizer.encode("Describe the image.", add_special_tokens=False), 4,
    ]
    answer = tokenizer.encode("The image shows a scene.", add_special_tokens=False)
    answer.append(int(config.text_config.eos_token_id))
    ids = torch.tensor([prompt + answer], dtype=torch.long)
    labels = torch.full_like(ids, -100)
    labels[:, len(prompt):] = ids[:, len(prompt):]
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "labels": labels,
        "pixel_values": image,
        "image_sizes": torch.tensor([[side, side]], dtype=torch.long),
    }


def first_step_write(gradient: torch.Tensor, learning_rate: float) -> torch.Tensor:
    return -learning_rate * gradient / (gradient.abs() + 1e-8)


def run_branch(
    model: torch.nn.Module,
    vision: torch.nn.Module,
    original_rope: Any,
    original_apply: Any,
    batch: dict[str, torch.Tensor],
    target: torch.nn.Parameter,
    variant: str,
    learning_rate: float,
) -> tuple[float, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    batch = {name: value.to(next(model.parameters()).device) for name, value in batch.items()}
    if variant == "native":
        vision.patch_positional_embedding.forward = original_rope
        pixtral_impl.apply_rotary_pos_emb = original_apply
    else:
        # Keep the released positional embedding shape and BF16 output contract.
        # Only trig precision changes for trig_fp64; rotation precision changes
        # for muladd_fp32; both changes for full_fp32.
        def rope_forward(_: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            frequencies = vision.patch_positional_embedding.inv_freq[position_ids]
            if variant in ("trig_fp64", "full_fp32"):
                return frequencies.double().cos().to(dtype=torch.bfloat16), frequencies.double().sin().to(dtype=torch.bfloat16)
            return frequencies.cos().to(dtype=torch.bfloat16), frequencies.sin().to(dtype=torch.bfloat16)

        def apply_rotation(
            query: torch.Tensor,
            key: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            position_ids: torch.Tensor | None = None,
            unsqueeze_dim: int = 1,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            cos_u = cos.unsqueeze(unsqueeze_dim)
            sin_u = sin.unsqueeze(unsqueeze_dim)
            if variant in ("muladd_fp32", "full_fp32"):
                q = query.float()
                k = key.float()
                c = cos_u.float()
                s = sin_u.float()
                qrot = (q * c) + (pixtral_impl.rotate_half(q) * s)
                krot = (k * c) + (pixtral_impl.rotate_half(k) * s)
                return qrot.to(query.dtype), krot.to(key.dtype)
            # Transformers releases disagree on the optional position_ids
            # keyword; the reviewed Pixtral implementation only needs the
            # four tensors and unsqueeze dimension.
            try:
                return original_apply(query, key, cos, sin, position_ids=position_ids, unsqueeze_dim=unsqueeze_dim)
            except TypeError:
                return original_apply(query, key, cos, sin, unsqueeze_dim=unsqueeze_dim)

        vision.patch_positional_embedding.forward = rope_forward
        pixtral_impl.apply_rotary_pos_emb = apply_rotation
    try:
        output = model(**batch, use_cache=False, return_dict=True)
        loss = output.loss
        if loss is None or not bool(torch.isfinite(loss.detach())):
            raise RuntimeError("non-finite multimodal loss")
        loss.backward()
        if target.grad is None:
            raise RuntimeError("vision patch-convolution gradient is absent")
        return float(loss.detach()), first_step_write(target.grad.detach().float(), learning_rate)
    finally:
        vision.patch_positional_embedding.forward = original_rope
        pixtral_impl.apply_rotary_pos_emb = original_apply


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--side", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.states < 4:
        raise ValueError("at least four states are required")
    device = torch.device(args.device)
    dtype = torch.bfloat16
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, fix_mistral_regex=True)
    model = Mistral3ForConditionalGeneration.from_pretrained(
        args.model, local_files_only=True, dtype=dtype, attn_implementation="eager", low_cpu_mem_usage=True,
    ).to(device).eval()
    vision = model.model.vision_tower
    target = vision.patch_conv.weight
    original_rope = vision.patch_positional_embedding.forward
    original_apply = pixtral_impl.apply_rotary_pos_emb
    images = sorted(args.image_dir.glob("image_*.png"))[: args.states]
    if len(images) != args.states:
        raise RuntimeError(f"requested {args.states} images but found {len(images)}")

    variants = ("native", "trig_fp64", "muladd_fp32", "full_fp32")
    rows: list[dict[str, Any]] = []
    for index, image_path in enumerate(images):
        batch = multimodal_batch(tokenizer, config, image_tensor(image_path, args.side, dtype), args.side)
        writes: dict[str, torch.Tensor] = {}
        losses: dict[str, float] = {}
        for variant in variants:
            losses[variant], writes[variant] = run_branch(
                model, vision, original_rope, original_apply, batch, target, variant, args.learning_rate,
            )
        native_write = writes["native"]
        native_norm = float(native_write.norm().item())
        row = {"state_id": image_path.stem, "losses": losses}
        for variant in variants[1:]:
            delta = native_write - writes[variant]
            ref = writes[variant]
            row[variant] = {
                "write_effect_rms": float(delta.pow(2).mean().sqrt().item()),
                "relative_write_rms": float(delta.norm().item() / max(float(ref.norm().item()), 1e-30)),
                "aligned_write": float((delta * ref).sum().item() / max(float((ref * ref).sum().item()), 1e-30)),
                "signed_write_mean": float(delta.mean().item()),
                "native_write_rms": native_norm,
            }
        rows.append(row)
        print(index + 1, image_path.name, {v: row[v]["aligned_write"] for v in variants[1:]}, flush=True)

    result: dict[str, Any] = {
        "schema": "kernel-analyzer-ministral-vision-rope-source-factorial-v1",
        "status": "COMPLETE_SOURCE_FACTORIAL_NATURAL_OPERANDS",
        "model": str(args.model),
        "operator": "Ministral Pixtral vision rotary position embedding",
        "variants": {
            "native": "released BF16 cosine/sine and BF16 rotation",
            "trig_fp64": "FP64 cosine/sine then BF16 cast; native BF16 rotation",
            "muladd_fp32": "native BF16 cosine/sine; FP32 multiply/add then BF16 cast",
            "full_fp32": "FP64 cosine/sine plus FP32 multiply/add then BF16 cast",
        },
        "state_count": len(rows),
        "input_source": "real CIFAR images with real multimodal language loss",
        "rows": rows,
        "claim_boundary": "Natural source-factorial evidence for one vision checkpoint and patch-convolution carrier; no universal or population claim.",
    }
    for variant in variants[1:]:
        values = [row[variant]["aligned_write"] for row in rows]
        result.setdefault("summary", {})[variant] = {
            "write_effect_rms_mean": float(np.mean([row[variant]["write_effect_rms"] for row in rows])),
            "relative_write_rms_mean": float(np.mean([row[variant]["relative_write_rms"] for row in rows])),
            "aligned_write_mean": float(np.mean(values)),
            "aligned_write_sign_counts": {"positive": sum(x > 0 for x in values), "negative": sum(x < 0 for x in values)},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
