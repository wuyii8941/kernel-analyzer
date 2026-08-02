"""Exact fixed-grid normalization for fullgraph Qwen3-VL capture.

The upstream implementation converts ``grid_thw`` tensor values to Python
lists and scalar loop bounds. This module evaluates those integer-only control
decisions once, then retains the same tensor arithmetic and trainable weights.
The caller must prove bitwise equality of loss and every parameter gradient.
"""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as qvl


def _fixed_attention_forward(
    self: Any,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: torch.Tensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    del cu_seqlens, rotary_pos_emb
    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        self.qkv(hidden_states)
        .reshape(seq_length, 3, self.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    if position_embeddings is None:
        raise RuntimeError("fixed visual attention requires rotary embeddings")
    cos, sin = position_embeddings
    query_states, key_states = qvl.apply_rotary_pos_emb_vision(
        query_states,
        key_states,
        cos,
        sin,
    )
    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)
    attention, _ = qvl.eager_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask=None,
        scaling=self.scaling,
        dropout=0.0 if not self.training else self.attention_dropout,
        is_causal=False,
        **kwargs,
    )
    return self.proj(attention.reshape(seq_length, -1).contiguous())


def specialize_fixed_grid(
    model: Any,
    grid_thw: torch.Tensor,
    input_ids: torch.Tensor,
) -> dict[str, Any]:
    """Specialize one-image visual control flow without changing its math."""

    rows = grid_thw.detach().cpu().tolist()
    if len(rows) != 1:
        raise ValueError("round-2 static capture currently requires one image")
    temporal, height, width = (int(value) for value in rows[0])
    if temporal != 1:
        raise ValueError("round-2 static capture currently requires one frame")

    subject = model.model
    visual = subject.visual
    language = subject.language_model
    merge = int(visual.spatial_merge_size)
    if height % merge or width % merge:
        raise ValueError("visual grid is not divisible by merge size")

    h_idxs = torch.linspace(0, visual.num_grid_per_side - 1, height)
    w_idxs = torch.linspace(0, visual.num_grid_per_side - 1, width)
    h_floor = h_idxs.int()
    w_floor = w_idxs.int()
    h_ceil = (h_floor + 1).clip(max=visual.num_grid_per_side - 1)
    w_ceil = (w_floor + 1).clip(max=visual.num_grid_per_side - 1)
    dh = h_idxs - h_floor
    dw = w_idxs - w_floor
    base_h = h_floor * visual.num_grid_per_side
    base_h_ceil = h_ceil * visual.num_grid_per_side
    indices = [
        (base_h[None].T + w_floor[None]).flatten(),
        (base_h[None].T + w_ceil[None]).flatten(),
        (base_h_ceil[None].T + w_floor[None]).flatten(),
        (base_h_ceil[None].T + w_ceil[None]).flatten(),
    ]
    weights = [
        ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
        ((1 - dh)[None].T * dw[None]).flatten(),
        (dh[None].T * (1 - dw)[None]).flatten(),
        (dh[None].T * dw[None]).flatten(),
    ]
    visual.register_buffer(
        "_round2_pos_indices",
        torch.stack(indices).to(device=visual.pos_embed.weight.device),
        persistent=False,
    )
    visual_positions = torch.nonzero(
        input_ids[0] == subject.config.image_token_id,
        as_tuple=False,
    ).flatten()
    language.register_buffer(
        "_round2_visual_positions",
        visual_positions.to(device=visual.pos_embed.weight.device),
        persistent=False,
    )
    visual.register_buffer(
        "_round2_pos_weights",
        torch.stack(weights).to(
            device=visual.pos_embed.weight.device,
            dtype=visual.pos_embed.weight.dtype,
        ),
        persistent=False,
    )

    block_rows = torch.arange(height // merge)
    block_cols = torch.arange(width // merge)
    intra_row = torch.arange(merge)
    intra_col = torch.arange(merge)
    row_idx = (
        block_rows[:, None, None, None] * merge
        + intra_row[None, None, :, None]
    )
    col_idx = (
        block_cols[None, :, None, None] * merge
        + intra_col[None, None, None, :]
    )
    row_idx = row_idx.expand(
        height // merge, width // merge, merge, merge
    ).reshape(-1)
    col_idx = col_idx.expand(
        height // merge, width // merge, merge, merge
    ).reshape(-1)
    visual.register_buffer(
        "_round2_rotary_pos_ids",
        torch.stack((row_idx, col_idx), dim=-1).to(
            device=visual.pos_embed.weight.device
        ),
        persistent=False,
    )
    visual.register_buffer(
        "_round2_cu_seqlens",
        torch.tensor(
            [0, temporal * height * width],
            dtype=torch.int32,
            device=visual.pos_embed.weight.device,
        ),
        persistent=False,
    )

    def fixed_pos(self: Any, ignored_grid: torch.Tensor) -> torch.Tensor:
        del ignored_grid
        values = self.pos_embed(self._round2_pos_indices)
        values = values * self._round2_pos_weights[:, :, None]
        values = values[0] + values[1] + values[2] + values[3]
        return (
            values.view(
                temporal,
                height // merge,
                merge,
                width // merge,
                merge,
                -1,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )

    def fixed_rotary(self: Any, ignored_grid: torch.Tensor) -> torch.Tensor:
        del ignored_grid
        frequencies = self.rotary_pos_emb(max(height, width))
        return frequencies[self._round2_rotary_pos_ids].flatten(1)

    def fixed_visual_forward(
        self: Any,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states + self.fast_pos_embed_interpolate(grid_thw)
        rotary = self.rot_pos_emb(grid_thw).reshape(hidden_states.shape[0], -1)
        embedding = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = (embedding.cos(), embedding.sin())
        deepstack = []
        for layer_num, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=self._round2_cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_num in self.deepstack_visual_indexes:
                index = self.deepstack_visual_indexes.index(layer_num)
                deepstack.append(self.deepstack_merger_list[index](hidden_states))
        return self.merger(hidden_states), deepstack

    def fixed_image_features(
        self: Any,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor], list[torch.Tensor]]:
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds, deepstack = self.visual(
            pixel_values,
            grid_thw=image_grid_thw,
        )
        return (image_embeds,), deepstack

    def fixed_placeholder_mask(
        self: Any,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_features: torch.Tensor | None = None,
        video_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del image_features, video_features
        image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
        video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1)
        return image_mask.expand_as(inputs_embeds), video_mask.expand_as(inputs_embeds)

    def fixed_deepstack(
        self: Any,
        hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        visual_embeds: torch.Tensor,
    ) -> torch.Tensor:
        del visual_pos_masks
        visual_embeds = visual_embeds.to(
            hidden_states.device,
            hidden_states.dtype,
        )
        selected = hidden_states[0, self._round2_visual_positions, :].clone()
        hidden_states[0, self._round2_visual_positions, :] = (
            selected + visual_embeds
        )
        return hidden_states

    visual.fast_pos_embed_interpolate = MethodType(fixed_pos, visual)
    visual.rot_pos_emb = MethodType(fixed_rotary, visual)
    visual.forward = MethodType(fixed_visual_forward, visual)
    subject.get_image_features = MethodType(fixed_image_features, subject)
    subject.get_placeholder_mask = MethodType(fixed_placeholder_mask, subject)
    language._deepstack_process = MethodType(fixed_deepstack, language)
    for block in visual.blocks:
        block.attn.forward = MethodType(_fixed_attention_forward, block.attn)

    return {
        "grid_thw": rows,
        "visual_patch_tokens": temporal * height * width,
        "merged_visual_tokens": temporal * height * width // (merge * merge),
        "specializations": [
            "position interpolation integer control",
            "vision rotary index integer control",
            "single-image attention split",
            "single-image embedding split",
            "valid placeholder cardinality check",
            "fixed visual-token deepstack indexing",
        ],
    }
