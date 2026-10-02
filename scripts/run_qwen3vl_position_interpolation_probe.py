#!/usr/bin/env python3
"""Probe Qwen3-VL learned 2-D position interpolation on a real image.

The probe keeps the trained vision tower, patch inputs, rotary path, blocks,
and merger fixed.  It changes only the arithmetic used to combine the four
learned-position interpolation taps: native dtype versus FP32 accumulation
followed by the original dtype write.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl import modeling_qwen3_vl as qvl


def patchify(path: Path, side: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((side, side), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = array.transpose(2, 0, 1)
    mean = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)[:, None, None]
    std = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)[:, None, None]
    frames = torch.tensor((array - mean) / std, device=device, dtype=dtype).unsqueeze(0).repeat(2, 1, 1, 1)
    patch = 16
    temporal = 2
    if side % patch:
        raise ValueError("side must be divisible by the Qwen3-VL patch size")
    h = side // patch
    w = side // patch
    frames = frames.view(1, temporal, 3, h, patch, w, patch)
    frames = frames.permute(0, 3, 5, 1, 2, 4, 6).reshape(-1, 3 * temporal * patch * patch)
    return frames


def run_visual(visual, patch_values, grid_thw, position_values):
    hidden = visual.patch_embed(patch_values).detach() + position_values.to(visual.dtype)
    position_ids = qvl.get_vision_position_ids(grid_thw, visual.spatial_merge_size)
    cu_seqlens, max_seqlen = qvl.get_vision_attention_seqlens(grid_thw, visual.config)
    rotary = visual.rotary_pos_emb(position_ids)
    rotary = rotary.reshape(rotary.shape[0], -1)
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
    return visual.merger(hidden)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--side", type=int, default=224)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    ).to(device)
    model.eval()
    visual = model.model.visual
    patch_values = patchify(args.image, args.side, device, torch.bfloat16)
    # The temporal patch of two frames has already been folded into each
    # patch row; the model's grid therefore uses one temporal token per image.
    grid_thw = torch.tensor([[1, args.side // 16, args.side // 16]], device=device, dtype=torch.long)
    indices, weights = qvl.get_vision_interpolation_indices_and_weights(
        grid_thw,
        num_grid_per_side=visual.num_grid_per_side,
        mode=visual.interpolation_mode,
        align_corners=visual.interpolation_align_corners,
        spatial_merge_size=visual.spatial_merge_size,
    )
    with torch.enable_grad():
        taps = visual.pos_embed(indices)
        native = (taps * weights[:, :, None].to(taps.dtype)).sum(1)
        fp32 = (taps.float() * weights[:, :, None].float()).sum(1).to(taps.dtype)
        candidate_output = run_visual(visual, patch_values, grid_thw, native)
        candidate_loss = candidate_output.float().square().mean()
        candidate_loss.backward(retain_graph=True)
        candidate_grad = visual.pos_embed.weight.grad.detach().float().clone()
        visual.pos_embed.weight.grad = None
        reference_output = run_visual(visual, patch_values, grid_thw, fp32)
        reference_loss = reference_output.float().square().mean()
        reference_loss.backward()
        reference_grad = visual.pos_embed.weight.grad.detach().float().clone()

    result = {
        "status": "COMPLETE_POSITION_INTERPOLATION_SOURCE_PROBE",
        "model": str(args.model),
        "image": str(args.image),
        "grid_thw": grid_thw.detach().cpu().tolist(),
        "candidate": "native_dtype_weighted_position_sum",
        "reference": "fp32_weighted_position_sum_then_original_dtype_write",
        "same_patch_inputs": True,
        "position_delta_max_abs": float((native - fp32).float().abs().max().item()),
        "position_delta_l2": float((native - fp32).float().norm().item()),
        "visual_output_delta_max_abs": float((candidate_output - reference_output).float().abs().max().item()),
        "visual_output_delta_l2": float((candidate_output - reference_output).float().norm().item()),
        "pos_embedding_gradient_delta_max_abs": float((candidate_grad - reference_grad).abs().max().item()),
        "pos_embedding_gradient_delta_l2": float((candidate_grad - reference_grad).norm().item()),
        "candidate_loss": float(candidate_loss.item()),
        "reference_loss": float(reference_loss.item()),
        "interpretation": "Nonzero arithmetic deltas are only a source screen; promotion requires multiple natural images, real downstream parameter reach, and a signed effect analysis.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
