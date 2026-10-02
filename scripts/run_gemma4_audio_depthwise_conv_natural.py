#!/usr/bin/env python3
"""Natural Gemma-4 audio LightConv depthwise-convolution probe.

The candidate uses the model's native depthwise causal Conv1d.  The reference
uses the same padded input and kernel, but evaluates the depthwise windowed
multiply-accumulate in FP32 before writing back to the original activation
dtype.  All surrounding audio-language computation is unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch


def normal_interval(values: list[float]) -> list[float]:
    x = np.asarray(values, dtype=np.float64)
    mean = float(x.mean())
    half = 0.0 if len(x) < 2 else 1.96 * float(x.std(ddof=1)) / np.sqrt(len(x))
    return [mean - half, mean + half]


def make_reference_forward():
    def reference_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.pre_layer_norm(hidden_states)
        hidden_states = self.linear_start(hidden_states)
        hidden_states = torch.nn.functional.glu(hidden_states, dim=-1)

        conv = self.depthwise_conv1d
        x = torch.nn.functional.pad(hidden_states.transpose(1, 2), (conv.left_pad, 0))
        windows = x.unfold(dimension=2, size=conv.kernel_size[0], step=conv.stride[0])
        # windows: [batch, channels, time, kernel]; depthwise weight: [channels, 1, kernel]
        output = (
            windows.float()
            * conv.weight[:, 0, :].float().view(1, -1, 1, conv.kernel_size[0])
        ).sum(dim=-1).to(dtype=hidden_states.dtype)
        hidden_states = output.transpose(1, 2)

        gradient_clipping = min(self.gradient_clipping, torch.finfo(hidden_states.dtype).max)
        hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.conv_norm(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_end(hidden_states)
        hidden_states += residual
        return hidden_states

    return reference_forward


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
        raise ValueError(f"waveform bank has {len(waves)} states, requested {args.states}")

    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    audio = model.model.audio_tower
    target_layer = audio.layers[args.layer].lconv1d
    original_forward = target_layer.forward
    reference_forward = make_reference_forward()
    target_parameter = target_layer.depthwise_conv1d.weight

    rows: list[dict[str, Any]] = []
    effects: list[torch.Tensor] = []
    aligned_numerators: list[float] = []
    aligned_denominators: list[float] = []
    for state_id, waveform in enumerate(waves[: args.states]):
        features = extractor([waveform], sampling_rate=16_000, return_tensors="pt")
        with torch.no_grad():
            encoded = audio(
                features["input_features"].to(device=device, dtype=torch.bfloat16),
                features["input_features_mask"].to(device),
                return_dict=True,
            )
            audio_tokens = int(encoded.attention_mask.sum().item())
        text_ids = tokenizer(args.text, return_tensors="pt").input_ids
        input_ids = torch.cat(
            [
                torch.full((1, audio_tokens), config.audio_token_id, dtype=torch.long),
                text_ids,
            ],
            dim=1,
        ).to(device)
        labels = input_ids.clone()
        labels[:, :audio_tokens] = -100
        attention_mask = torch.ones_like(input_ids)
        input_features = features["input_features"].to(device=device, dtype=torch.bfloat16)
        input_features_mask = features["input_features_mask"].to(device)

        model.zero_grad(set_to_none=True)
        target_layer.forward = original_forward
        candidate_loss = model(
            input_ids=input_ids,
            input_features=input_features,
            input_features_mask=input_features_mask,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        ).loss
        candidate_loss.backward()
        candidate_gradient = target_parameter.grad.detach().float().cpu().clone()

        model.zero_grad(set_to_none=True)
        target_layer.forward = MethodType(reference_forward, target_layer)
        reference_loss = model(
            input_ids=input_ids,
            input_features=input_features,
            input_features_mask=input_features_mask,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        ).loss
        reference_loss.backward()
        reference_gradient = target_parameter.grad.detach().float().cpu().clone()

        target_layer.forward = original_forward
        effect = candidate_gradient - reference_gradient
        candidate_write = -args.learning_rate * candidate_gradient
        reference_write = -args.learning_rate * reference_gradient
        aligned_numerator = float(((candidate_write - reference_write) * reference_write).sum())
        aligned_denominator = float(reference_write.square().sum())
        aligned_numerators.append(aligned_numerator)
        aligned_denominators.append(aligned_denominator)
        effects.append(effect.reshape(-1))
        rows.append(
            {
                "state_id": state_id,
                "audio_tokens": audio_tokens,
                "loss_candidate": float(candidate_loss.detach().cpu()),
                "loss_reference": float(reference_loss.detach().cpu()),
                "loss_difference": float((candidate_loss - reference_loss).detach().cpu()),
                "gradient_effect_rms_over_reference": float(
                    effect.norm() / (reference_gradient.norm() + 1e-30)
                ),
                "write_effect_rms_over_reference": float(
                    (candidate_write - reference_write).norm() / (reference_write.norm() + 1e-30)
                ),
                "aligned_write_scaling": float(
                    aligned_numerator / (aligned_denominator + 1e-30)
                ),
                "aligned_write_numerator": aligned_numerator,
                "aligned_write_denominator": aligned_denominator,
            }
        )
        print(json.dumps({"event": "GEMMA4_AUDIO_DEPTHWISE_STATE", **rows[-1]}), flush=True)
        del candidate_loss, reference_loss, input_ids, labels, input_features, input_features_mask
        torch.cuda.empty_cache()

    effect_matrix = torch.stack(effects).double()
    calibration = effect_matrix[: args.calibration_states]
    confirmation = effect_matrix[args.calibration_states :]
    direction = calibration.mean(dim=0)
    direction_norm = float(direction.norm())
    if direction_norm:
        direction = direction / direction_norm
    projections = [float(torch.sum(value * direction)) for value in confirmation]
    aligned = [row["aligned_write_scaling"] for row in rows[args.calibration_states :]]
    return {
        "schema": "kernel-analyzer-gemma4-audio-depthwise-conv-natural-v1",
        "status": "COMPLETE_NATURAL_AUDIO_DEPTHWISE_CONV_BOUNDARY",
        "model": str(model_dir),
        "operator_family": "audio_lightconv_depthwise_causal_conv_materialization",
        "candidate": f"native Gemma-4 audio LightConv1d layer{args.layer} depthwise causal Conv1d",
        "reference": "same padded depthwise window evaluated with FP32 multiply-accumulate and original-dtype writeback",
        "input_source": "real LibriSpeech waveforms and a fixed text target",
        "parameter": f"model.audio_tower.layers.{args.layer}.lconv1d.depthwise_conv1d.weight",
        "comparison_scope": {
            "same_checkpoint": True,
            "same_audio": True,
            "same_text_labels": True,
            "same_audio_path_except_boundary": True,
            "single_changed_boundary": f"audio LightConv1d layer{args.layer} depthwise causal Conv1d",
        },
        "rows": rows,
        "summary": {
            "state_count": len(rows),
            "calibration_count": args.calibration_states,
            "confirmation_count": len(rows) - args.calibration_states,
            "loss_difference_interval_normal_95": normal_interval([row["loss_difference"] for row in rows]),
            "write_effect_rms_mean": float(np.mean([row["write_effect_rms_over_reference"] for row in rows])),
            "confirmation_projection_interval_normal_95": normal_interval(projections),
            "confirmation_projection_positive": sum(value > 0 for value in projections),
            "confirmation_projection_negative": sum(value < 0 for value in projections),
            "confirmation_aligned_write_interval_normal_95": normal_interval(aligned),
            "confirmation_aligned_write_mean": float(np.mean(aligned)),
            "confirmation_aligned_write_ratio_of_sums": float(
                sum(aligned_numerators[args.calibration_states :])
                / max(sum(aligned_denominators[args.calibration_states :]), 1e-30)
            ),
            "calibration_direction_norm": direction_norm,
        },
        "claim_boundary": (
            "One Gemma-4 E2B checkpoint, real speech inputs and one fixed text target; this is "
            "a fixed-suite natural audio-language boundary result, not a population or long-run "
            "training claim and not a kernel-identity claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/google/gemma-4-E2B"))
    parser.add_argument("--waveforms", type=Path, required=True)
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--calibration-states", type=int, default=4)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--text", default="THE QUICK BROWN FOX")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
