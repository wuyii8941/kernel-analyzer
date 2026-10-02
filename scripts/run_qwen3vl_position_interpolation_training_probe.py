#!/usr/bin/env python3
"""Natural multimodal loss probe for Qwen3-VL position interpolation.

The only changed operation is the weighted sum of the learned 2-D position
embedding taps.  The same trained model, patch inputs, language inputs,
vision blocks, and loss are used for the native-dtype and FP32-accumulation
branches.  This is a bounded source/gradient/write screen, not a population
certificate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl import modeling_qwen3_vl as qvl
from transformers.models.qwen3_vl.modeling_qwen3_vl import BaseModelOutputWithDeepstackFeatures


def patchify(path: Path, side: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((side, side), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = array.transpose(2, 0, 1)
    mean = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)[:, None, None]
    std = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)[:, None, None]
    frame = torch.tensor((array - mean) / std, device=device, dtype=dtype)
    frames = frame.unsqueeze(0).repeat(2, 1, 1, 1)
    patch = 16
    temporal = 2
    h = side // patch
    w = side // patch
    frames = frames.view(1, temporal, 3, h, patch, w, patch)
    return frames.permute(0, 3, 5, 1, 2, 4, 6).reshape(-1, 3 * temporal * patch * patch)


def run_visual(visual, patch_values, grid_thw, *, fp32_position_sum: bool):
    indices, weights = qvl.get_vision_interpolation_indices_and_weights(
        grid_thw,
        num_grid_per_side=visual.num_grid_per_side,
        mode=visual.interpolation_mode,
        align_corners=visual.interpolation_align_corners,
        spatial_merge_size=visual.spatial_merge_size,
    )
    hidden = visual.patch_embed(patch_values)
    taps = visual.pos_embed(indices)
    if fp32_position_sum:
        pos = (taps.float() * weights[:, :, None].float()).sum(1).to(hidden.dtype)
    else:
        pos = (taps * weights[:, :, None].to(taps.dtype)).sum(1)
    hidden = hidden + pos
    position_ids = qvl.get_vision_position_ids(grid_thw, visual.spatial_merge_size)
    cu_seqlens, max_seqlen = qvl.get_vision_attention_seqlens(grid_thw, visual.config)
    rotary = visual.rotary_pos_emb(position_ids).reshape(position_ids.shape[0], -1)
    embedding = torch.cat((rotary, rotary), dim=-1)
    position_embeddings = (embedding.cos(), embedding.sin())
    deepstack = []
    for layer_num, block in enumerate(visual.blocks):
        hidden = block(
            hidden,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_embeddings=position_embeddings,
        )
        if layer_num in visual.deepstack_visual_indexes:
            index = visual.deepstack_visual_indexes.index(layer_num)
            deepstack.append(visual.deepstack_merger_list[index](hidden))
    merged = visual.merger(hidden)
    return BaseModelOutputWithDeepstackFeatures(
        last_hidden_state=hidden,
        pooler_output=merged,
        deepstack_features=deepstack,
    )


def make_inputs(tokenizer, model, device, side):
    text = tokenizer(" describe this image", add_special_tokens=False)["input_ids"][:4]
    image_count = (side // 16) * (side // 16) // 4
    ids = [model.config.vision_start_token_id]
    ids.extend([model.config.image_token_id] * image_count)
    ids.append(model.config.vision_end_token_id)
    ids.extend(text)
    input_ids = torch.tensor([ids], device=device, dtype=torch.long)
    image_start = 1
    image_end = image_start + image_count
    token_types = torch.zeros_like(input_ids, dtype=torch.int32)
    token_types[:, image_start:image_end] = 1
    labels = input_ids.clone()
    labels[:, :image_end + 1] = -100
    attention = torch.ones_like(input_ids)
    grid = torch.tensor([[1, side // 16, side // 16]], device=device, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "mm_token_type_ids": token_types,
        "labels": labels,
        "image_grid_thw": grid,
    }


def run_branch(model, patch_values, inputs, *, fp32_position_sum: bool):
    original = model.model.get_image_features

    def get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        del kwargs
        output = run_visual(
            self.visual,
            pixel_values,
            image_grid_thw,
            fp32_position_sum=fp32_position_sum,
        )
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        output.pooler_output = torch.split(output.pooler_output, split_sizes)
        return output

    model.model.get_image_features = MethodType(get_image_features, model.model)
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        output = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            mm_token_type_ids=inputs["mm_token_type_ids"],
            pixel_values=patch_values,
            image_grid_thw=inputs["image_grid_thw"],
            labels=inputs["labels"],
            use_cache=False,
        )
        output.loss.backward()
    grad = model.model.visual.pos_embed.weight.grad.detach().float().clone()
    model.model.get_image_features = original
    return float(output.loss.detach().item()), grad


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--side", type=int, default=224)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    bank = [Path(line.strip()) for line in args.image_bank.read_text().splitlines() if line.strip()]
    bank = bank[: args.limit]
    if len(bank) < 4:
        raise ValueError("need at least four real image paths")
    device = torch.device(args.device)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    inputs = make_inputs(tokenizer, model, device, args.side)
    target = model.model.visual.pos_embed.weight
    rows = []
    deltas = []
    references = []
    for image_path in bank:
        patches = patchify(image_path, args.side, device, torch.bfloat16)
        native_loss, native_grad = run_branch(model, patches, inputs, fp32_position_sum=False)
        fp32_loss, fp32_grad = run_branch(model, patches, inputs, fp32_position_sum=True)
        delta = native_grad - fp32_grad
        # One zero-moment AdamW step; weight decay cancels in the branch delta.
        ref_step = -1e-3 * fp32_grad / (fp32_grad.abs() + 1e-8)
        cand_step = -1e-3 * native_grad / (native_grad.abs() + 1e-8)
        write_delta = cand_step - ref_step
        rows.append({
            "image": str(image_path),
            "native_loss": native_loss,
            "fp32_loss": fp32_loss,
            "loss_delta": native_loss - fp32_loss,
            "gradient_rms": float(delta.norm().item() / max(delta.numel(), 1) ** 0.5),
            "write_rms": float(write_delta.norm().item() / max(write_delta.numel(), 1) ** 0.5),
            "reference_write_rms": float(ref_step.norm().item() / max(ref_step.numel(), 1) ** 0.5),
            "aligned_ratio": float(torch.sum(write_delta * ref_step).item() / (torch.sum(ref_step * ref_step).item() + 1e-30)),
        })
        deltas.append(write_delta.detach().cpu())
        references.append(ref_step.detach().cpu())
        del model.model.visual.pos_embed.weight.grad
        torch.cuda.empty_cache()

    cal = torch.stack(deltas[: len(deltas) // 2])
    conf = torch.stack(deltas[len(deltas) // 2 :])
    direction = cal.sum(0)
    direction_norm = direction.norm().item()
    if direction_norm:
        direction = direction / direction_norm
        projections = [float(torch.sum(row * direction).item()) for row in conf]
    else:
        projections = []
    result = {
        "status": "COMPLETE_NATURAL_POSITION_INTERPOLATION_TRAINING_PROBE",
        "model": str(args.model),
        "image_count": len(bank),
        "calibration_count": len(cal),
        "confirmation_count": len(conf),
        "target_parameter": "model.visual.pos_embed.weight",
        "candidate": "native_dtype_weighted_position_sum",
        "reference": "fp32_weighted_position_sum_then_original_dtype_write",
        "rows": rows,
        "confirmation_projections": projections,
        "confirmation_positive_count": sum(v > 0 for v in projections),
        "confirmation_negative_count": sum(v < 0 for v in projections),
        "fixed_suite_mean_write_rms": float(
            torch.stack([row.norm() for row in deltas]).mean().item()
            / max(deltas[0].numel(), 1) ** 0.5
        ),
        "interpretation": "This is a natural-image, real-loss source probe. Promote only after an independent image bank and a predeclared population rule confirm a signed write effect; otherwise retain as a candidate source boundary.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
