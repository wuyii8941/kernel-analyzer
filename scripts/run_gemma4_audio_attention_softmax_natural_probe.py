#!/usr/bin/env python3
"""Isolate the softmax materialisation in a real Gemma-4 audio attention layer.

This is deliberately separate from the existing softcap probe.  The attention
logits, softcap arithmetic, waveform, text path and target parameter are held
fixed; only the softmax evaluation dtype changes before the original BF16
attention-weight write-back.  A nonzero write effect is promoted only after
the result is checked against the existing audio problem-group ledger.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def interval(values: list[float]) -> list[float]:
    x = np.asarray(values, dtype=np.float64)
    mean = float(x.mean())
    if len(values) < 2:
        return [mean, mean]
    half = 1.96 * float(x.std(ddof=1)) / math.sqrt(len(values))
    return [mean - half, mean + half]


def write(gradient: torch.Tensor, learning_rate: float) -> torch.Tensor:
    return -learning_rate * gradient.float()


def make_forward(softmax_mode: str):
    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_length, self.num_heads, self.head_dim)
        query_states = self.q_proj(hidden_states).float().view(hidden_shape)
        key_states = self.k_proj(hidden_states).float().view(hidden_shape)
        value_states = self.v_proj(hidden_states).float().view(hidden_shape)
        query_states = query_states * self.q_scale * F.softplus(self.per_dim_scale)
        key_states = key_states * self.k_scale
        query_states = self._convert_to_block(query_states)
        key_states = self._extract_block_context(key_states)
        value_states = self._extract_block_context(value_states)
        num_blocks = query_states.shape[1]
        relative_key_states = self.relative_k_proj(position_embeddings)
        relative_key_states = relative_key_states.view(-1, self.num_heads, self.head_dim)
        relative_key_states = relative_key_states.to(dtype=query_states.dtype)
        queries = query_states.permute(0, 3, 1, 2, 4)
        matrix_ac = queries @ key_states.permute(0, 3, 1, 4, 2)
        queries_flat = queries.reshape(batch_size, self.num_heads, -1, self.head_dim)
        matrix_bd = queries_flat @ relative_key_states.permute(1, 2, 0)
        matrix_bd = matrix_bd.reshape(batch_size, self.num_heads, num_blocks, self.chunk_size, -1)
        matrix_bd = self._rel_shift(matrix_bd)
        attn_weights = matrix_ac + matrix_bd
        # Keep the native softcap path exactly the same in both arms.
        attn_weights = torch.tanh(attn_weights / self.softcap) * self.softcap
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(
                attention_mask.logical_not(), self.config.attention_invalid_logits_value
            )
        keep_fp32 = False
        if softmax_mode == "native":
            weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        elif softmax_mode == "fp64":
            weights = F.softmax(attn_weights.double(), dim=-1, dtype=torch.float64).to(torch.float32)
        elif softmax_mode == "fp32_keep":
            # Same native FP32 softmax arithmetic, but do not materialize the
            # attention probabilities to BF16 before the value contraction.
            weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
            keep_fp32 = True
        else:
            raise ValueError(softmax_mode)
        attn_weights = weights if keep_fp32 else weights.to(value_states.dtype)
        attn_output = attn_weights @ value_states.permute(0, 3, 1, 2, 4)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(batch_size, num_blocks * self.chunk_size, -1)
        attn_output = attn_output[:, :seq_length].contiguous()
        attn_output = self.post(attn_output.to(hidden_states.dtype))
        return attn_output, attn_weights

    return forward


def run(args: argparse.Namespace) -> dict[str, Any]:
    import sys

    transformers_site = "/data1/tzh/envs/pt_nightly_transformers5/lib/python3.11/site-packages"
    if transformers_site not in sys.path:
        sys.path.insert(0, transformers_site)
    from transformers import AutoConfig, AutoFeatureExtractor, AutoTokenizer
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    model_dir = args.model.resolve()
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    extractor = AutoFeatureExtractor.from_pretrained(model_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    waves = np.load(args.waveforms, allow_pickle=True)
    if len(waves) < args.states:
        raise ValueError("waveform bank is shorter than requested states")
    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_dir, config=config, torch_dtype=torch.bfloat16,
        local_files_only=True, low_cpu_mem_usage=True,
    ).to(device).eval()
    model.config.use_cache = False
    target_layer = model.model.audio_tower.layers[args.layer].self_attn
    target_parameter = target_layer.q_proj.linear.weight
    native_forward = target_layer.forward
    text_ids = tokenizer(args.text, return_tensors="pt").input_ids
    rows: list[dict[str, Any]] = []
    effects: list[torch.Tensor] = []
    for state_id, waveform in enumerate(waves[: args.states]):
        features = extractor([waveform], sampling_rate=16_000, return_tensors="pt")
        with torch.no_grad():
            encoded = model.model.audio_tower(
                features["input_features"].to(device=device, dtype=torch.bfloat16),
                features["input_features_mask"].to(device), return_dict=True,
            )
            audio_tokens = int(encoded.attention_mask.sum().item())
        input_ids = torch.cat([
            torch.full((1, audio_tokens), config.audio_token_id, dtype=torch.long),
            text_ids,
        ], dim=1).to(device)
        labels = input_ids.clone(); labels[:, :audio_tokens] = -100
        attention_mask = torch.ones_like(input_ids)
        input_features = features["input_features"].to(device=device, dtype=torch.bfloat16)
        input_features_mask = features["input_features_mask"].to(device)

        model.zero_grad(set_to_none=True)
        target_layer.forward = native_forward
        native_loss = model(
            input_ids=input_ids, input_features=input_features,
            input_features_mask=input_features_mask, attention_mask=attention_mask,
            labels=labels, use_cache=False,
        ).loss
        native_loss.backward()
        native_grad = target_parameter.grad.detach().float().cpu().clone()
        native_write = write(native_grad, args.learning_rate)

        model.zero_grad(set_to_none=True)
        target_layer.forward = MethodType(make_forward(args.reference_mode), target_layer)
        altered_loss = model(
            input_ids=input_ids, input_features=input_features,
            input_features_mask=input_features_mask, attention_mask=attention_mask,
            labels=labels, use_cache=False,
        ).loss
        altered_loss.backward()
        altered_grad = target_parameter.grad.detach().float().cpu().clone()
        altered_write = write(altered_grad, args.learning_rate)
        effect = native_write - altered_write
        effects.append(effect.reshape(-1))
        rows.append({
            "state_id": state_id,
            "audio_tokens": audio_tokens,
            "loss_difference": float((native_loss - altered_loss).detach().cpu()),
            "gradient_effect_rms_over_reference": float((native_grad - altered_grad).norm()) / max(float(altered_grad.norm()), 1e-30),
            "write_effect_rms_over_reference": float(effect.norm()) / max(float(altered_write.norm()), 1e-30),
            "write_aligned": float((effect * altered_write).sum()) / max(float((altered_write * altered_write).sum()), 1e-30),
        })
        del input_ids, labels, input_features, input_features_mask, native_loss, altered_loss
        del native_grad, altered_grad, native_write, altered_write
        torch.cuda.empty_cache()
    target_layer.forward = native_forward
    split = args.calibration_states
    direction = torch.stack([x.double() for x in effects[:split]]).mean(dim=0)
    norm = float(direction.norm())
    projections: list[float] = []
    if norm:
        direction /= norm
        projections = [float(torch.dot(x.double(), direction)) for x in effects[split:]]
    return {
        "schema": "kernel-analyzer-gemma4-audio-attention-softmax-natural-v1",
        "status": "COMPLETE_NATURAL_AUDIO_ATTENTION_SOFTMAX_BOUNDARY",
        "model": str(model_dir),
        "operator_family": "audio_attention_softmax_materialization",
        "candidate": "native Gemma4 audio attention softmax in FP32 before BF16 write",
        "reference": (
            "same logits and softcap with FP64 softmax before original BF16 write"
            if args.reference_mode == "fp64"
            else "same logits and softcap with native FP32 softmax retained in FP32 before value contraction"
        ),
        "layer": args.layer,
        "target_parameter": f"model.audio_tower.layers.{args.layer}.self_attn.q_proj.linear.weight",
        "input_source": "real waveform bank and fixed text target",
        "comparison_scope": {"same_audio": True, "same_text": True, "same_layer": True, "same_softcap": True, "single_changed_boundary": "audio attention softmax evaluation/materialization"},
        "rows": rows,
        "summary": {
            "state_count": len(rows), "calibration_count": split, "confirmation_count": len(rows) - split,
            "write_effect_rms_mean": float(np.mean([r["write_effect_rms_over_reference"] for r in rows])),
            "write_aligned_interval_normal_95": interval([r["write_aligned"] for r in rows]),
            "loss_difference_interval_normal_95": interval([r["loss_difference"] for r in rows]),
            "heldout_projection_interval_normal_95": interval(projections) if projections else None,
            "heldout_projection_positive": sum(x > 0 for x in projections),
            "heldout_projection_negative": sum(x < 0 for x in projections),
            "calibration_direction_norm": norm,
        },
        "claim_boundary": "One Gemma-4 checkpoint, one audio-attention layer and declared real waveform bank; fixed-suite source/write result only, no population or quality claim.",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("/data1/tzh/models/google/gemma-4-E2B"))
    p.add_argument("--waveforms", type=Path, required=True)
    p.add_argument("--states", type=int, default=8)
    p.add_argument("--calibration-states", type=int, default=4)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--text", default="THE QUICK BROWN FOX")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--reference-mode", choices=("fp64", "fp32_keep"), default="fp64")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.states < 4 or args.calibration_states <= 0 or args.calibration_states >= args.states:
        raise ValueError("states must exceed calibration-states and be at least 4")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
