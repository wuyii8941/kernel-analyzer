#!/usr/bin/env python3
"""Natural Gemma-4 audio-attention softcap materialisation probe.

Only the softcap arithmetic in one audio-attention layer is changed.  The
audio encoder, language model, inputs, labels, and all other attention
operations are kept on the native path.  This is a source-isolating boundary
probe, not a claim about the whole audio model or a training population.
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


def make_softcap_forward(mode: str):
    """Return a Gemma4AudioAttention.forward with one softcap precision change."""

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
        if mode == "bf16":
            capped = torch.tanh((attn_weights / self.softcap).to(torch.bfloat16)).to(torch.float32)
            attn_weights = (capped * self.softcap).to(torch.float32)
        elif mode == "fp64":
            capped = torch.tanh(attn_weights.double() / self.softcap.double())
            attn_weights = (capped * self.softcap.double()).to(torch.float32)
        else:
            attn_weights = torch.tanh(attn_weights / self.softcap) * self.softcap
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(
                attention_mask.logical_not(), self.config.attention_invalid_logits_value
            )
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
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
    references: list[torch.Tensor] = []
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

        row: dict[str, Any] = {"state_id": state_id, "audio_tokens": audio_tokens,
                               "loss_native": float(native_loss.detach().cpu())}
        for mode in ("bf16", "fp64"):
            model.zero_grad(set_to_none=True)
            target_layer.forward = MethodType(make_softcap_forward(mode), target_layer)
            altered_loss = model(
                input_ids=input_ids, input_features=input_features,
                input_features_mask=input_features_mask, attention_mask=attention_mask,
                labels=labels, use_cache=False,
            ).loss
            altered_loss.backward()
            altered_grad = target_parameter.grad.detach().float().cpu().clone()
            altered_write = write(altered_grad, args.learning_rate)
            effect = native_write - altered_write
            reference_norm = float(altered_write.norm())
            rows.append({}) if False else None
            row[f"loss_{mode}"] = float(altered_loss.detach().cpu())
            row[f"loss_difference_{mode}"] = float((native_loss - altered_loss).detach().cpu())
            row[f"gradient_effect_rms_over_{mode}"] = float((native_grad - altered_grad).norm()) / max(float(altered_grad.norm()), 1e-30)
            row[f"write_effect_rms_over_{mode}"] = float(effect.norm()) / max(reference_norm, 1e-30)
            row[f"write_aligned_{mode}"] = float((effect * altered_write).sum()) / max(float((altered_write * altered_write).sum()), 1e-30)
            if mode == "bf16":
                effects.append(effect.reshape(-1)); references.append(altered_write.reshape(-1))
            del altered_loss, altered_grad, altered_write
            torch.cuda.empty_cache()
        rows.append(row)
        print(json.dumps({"event": "GEMMA4_AUDIO_ATTN_SOFTCAP_STATE", **row}), flush=True)
        del input_ids, labels, input_features, input_features_mask, native_loss, native_grad, native_write
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
        "schema": "kernel-analyzer-gemma4-audio-attention-softcap-natural-v1",
        "status": "COMPLETE_NATURAL_AUDIO_ATTENTION_SOFTCAP_BOUNDARY",
        "model": str(model_dir), "operator_family": "audio_attention_logit_softcap_materialization",
        "candidate": "native Gemma4 audio attention FP32 logit softcap",
        "reference_variants": {
            "bf16": "same layer with only softcap tanh evaluation rounded to BF16 before writeback",
            "fp64": "same layer with only softcap division/tanh/multiply evaluated in FP64 then cast to FP32",
        },
        "layer": args.layer, "target_parameter": f"model.audio_tower.layers.{args.layer}.self_attn.q_proj.linear.weight",
        "input_source": "real waveform bank and fixed text target",
        "comparison_scope": {"same_audio": True, "same_text": True, "same_layer": True, "single_changed_boundary": "audio attention logit softcap arithmetic"},
        "rows": rows,
        "summary": {
            "state_count": len(rows), "calibration_count": split, "confirmation_count": len(rows) - split,
            "bf16_write_effect_rms_mean": float(np.mean([r["write_effect_rms_over_bf16"] for r in rows])),
            "bf16_aligned_interval": interval([r["write_aligned_bf16"] for r in rows]),
            "bf16_loss_difference_interval": interval([r["loss_difference_bf16"] for r in rows]),
            "bf16_heldout_projection_interval": interval(projections) if projections else None,
            "bf16_heldout_projection_positive": sum(x > 0 for x in projections),
            "bf16_heldout_projection_negative": sum(x < 0 for x in projections),
            "fp64_write_effect_rms_mean": float(np.mean([r["write_effect_rms_over_fp64"] for r in rows])),
            "fp64_aligned_interval": interval([r["write_aligned_fp64"] for r in rows]),
        },
        "claim_boundary": "One checkpoint, one audio layer and declared real waveform bank; fixed-suite source/write result only, no population or quality claim.",
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
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
