#!/usr/bin/env python3
"""Probe the Ministral-3 vision patch-merger boundary without digest metadata.

The candidate is the released ``F.unfold`` patch grouping.  The reference uses
an explicit view/permute/unfold expression on the *same* vision features and
the same trained merger weights.  This is a source probe, not a claim that a
different semantic patch merger is a valid repair.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Mistral3ForConditionalGeneration


def image_tensor(path: Path, size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = array.transpose(2, 0, 1)
    mean = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)[:, None, None]
    std = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)[:, None, None]
    array = (array - mean) / std
    return torch.tensor(array, device=device, dtype=dtype).unsqueeze(0)


def explicit_patch_group(image_features: torch.Tensor, image_sizes: torch.Tensor, patch_size: int, merge: int) -> torch.Tensor:
    image_sizes_list = [
        (int(size[0]) // patch_size, int(size[1]) // patch_size) for size in image_sizes
    ]
    token_counts = [h * w for h, w in image_sizes_list]
    depth = image_features.shape[-1]
    groups = []
    for image_index, image_tokens in enumerate(image_features.split(token_counts)):
        h, w = image_sizes_list[image_index]
        grid = image_tokens.view(h, w, depth).permute(2, 0, 1).unsqueeze(0)
        # Keep the same patch contents and ordering as F.unfold, but expose
        # the spatial-to-feature permutation explicitly for auditability.
        blocks = grid.unfold(2, merge, merge).unfold(3, merge, merge)
        blocks = blocks.permute(0, 2, 3, 1, 4, 5).reshape(-1, depth * merge * merge)
        groups.append(blocks)
    return torch.cat(groups, dim=0)


def released_patch_group(image_features: torch.Tensor, image_sizes: torch.Tensor, patch_size: int, merge: int) -> torch.Tensor:
    image_sizes_list = [
        (int(size[0]) // patch_size, int(size[1]) // patch_size) for size in image_sizes
    ]
    token_counts = [h * w for h, w in image_sizes_list]
    depth = image_features.shape[-1]
    groups = []
    for image_index, image_tokens in enumerate(image_features.split(token_counts)):
        h, w = image_sizes_list[image_index]
        grid = image_tokens.view(h, w, depth).permute(2, 0, 1).unsqueeze(0)
        blocks = torch.nn.functional.unfold(grid, kernel_size=merge, stride=merge)
        blocks = blocks.view(depth * merge * merge, -1).t()
        groups.append(blocks)
    return torch.cat(groups, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model = Mistral3ForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    ).to(device)
    model.eval()
    pixel_values = image_tensor(args.image, args.size, device, torch.bfloat16)
    image_sizes = torch.tensor([[args.size, args.size]], device=device, dtype=torch.long)

    with torch.enable_grad():
        vision = model.model.vision_tower(
            pixel_values,
            image_sizes=image_sizes,
            output_hidden_states=True,
            return_dict=True,
        )
        layer = model.config.vision_feature_layer
        selected = vision.hidden_states[layer].squeeze(0)
        projector = model.model.multi_modal_projector
        normalized = projector.norm(selected)
        candidate_blocks = released_patch_group(
            normalized, image_sizes, model.config.vision_config.patch_size, model.config.spatial_merge_size
        )
        reference_blocks = explicit_patch_group(
            normalized, image_sizes, model.config.vision_config.patch_size, model.config.spatial_merge_size
        )
        candidate_features = projector.patch_merger.merging_layer(candidate_blocks)
        reference_features = projector.patch_merger.merging_layer(reference_blocks)
        candidate_loss = candidate_features.float().square().mean()
        reference_loss = reference_features.float().square().mean()
        candidate_loss.backward(retain_graph=True)
        candidate_grad = projector.patch_merger.merging_layer.weight.grad.detach().float().clone()
        projector.patch_merger.merging_layer.weight.grad = None
        reference_loss.backward()
        reference_grad = projector.patch_merger.merging_layer.weight.grad.detach().float().clone()

    block_delta = (candidate_blocks - reference_blocks).float()
    grad_delta = candidate_grad - reference_grad
    result = {
        "status": "COMPLETE_ZERO_OR_NONZERO_PATCH_MERGER_SOURCE_PROBE",
        "model": str(args.model),
        "image": str(args.image),
        "image_size": [args.size, args.size],
        "candidate": "released_f_unfold_patch_group",
        "reference": "explicit_spatial_unfold_patch_group",
        "same_input_features": True,
        "same_merging_weights": True,
        "block_max_abs": float(block_delta.abs().max().item()),
        "block_l2": float(block_delta.norm().item()),
        "output_max_abs": float((candidate_features - reference_features).float().abs().max().item()),
        "output_l2": float((candidate_features - reference_features).float().norm().item()),
        "gradient_max_abs": float(grad_delta.abs().max().item()),
        "gradient_l2": float(grad_delta.norm().item()),
        "candidate_loss": float(candidate_loss.item()),
        "reference_loss": float(reference_loss.item()),
        "interpretation": "No new problem group if all deltas are zero; a nonzero delta is a patch-layout/materialization source probe and requires multi-state confirmation before promotion.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
