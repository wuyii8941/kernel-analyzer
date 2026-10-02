#!/usr/bin/env python3
"""Natural Gemma-4 audio-language boundary probe.

The script compares the native BF16 audio output projection with an otherwise
identical FP32-accumulate/BF16-writeback projection.  It uses real speech
waveforms, real checkpoint weights, and a real causal language-model loss.  The
probe is intentionally a fixed-suite boundary result: it does not claim a
population mean or a long-run training-quality effect.

The Gemma-4 implementation is supplied by the local Transformers 5 environment
used for the checkpoint.  Run with that environment's site-packages appended
after importing the CUDA PyTorch environment, for example:

  PYTHONPATH=/data1/tzh/envs/pt_nightly_transformers5/lib/python3.11/site-packages \
  python scripts/run_gemma4_audio_language_output_projection_natural.py ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FP32Projection(nn.Module):
    def __init__(self, original: nn.Linear) -> None:
        super().__init__()
        self.weight = original.weight
        self.bias = original.bias

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.linear(values.float(), self.weight.float(), self.bias.float()).to(values.dtype)


def normal_interval(values: list[float]) -> list[float]:
    x = np.asarray(values, dtype=np.float64)
    mean = float(x.mean())
    half = 0.0 if len(x) < 2 else 1.96 * float(x.std(ddof=1)) / np.sqrt(len(x))
    return [mean - half, mean + half]


def run(args: argparse.Namespace) -> dict[str, Any]:
    import sys

    # Keep the CUDA torch package already imported by the selected environment;
    # only the local Transformers implementation is added for Gemma-4 support.
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
    native_projection = model.model.audio_tower.output_proj
    if not isinstance(native_projection, nn.Linear):
        raise TypeError(f"unexpected output projection: {type(native_projection)!r}")

    rows: list[dict[str, Any]] = []
    effects: list[torch.Tensor] = []
    writes: list[torch.Tensor] = []
    references: list[torch.Tensor] = []
    for state_id, waveform in enumerate(waves[: args.states]):
        features = extractor([waveform], sampling_rate=16_000, return_tensors="pt")
        with torch.no_grad():
            encoded = model.model.audio_tower(
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
        model.model.audio_tower.output_proj = native_projection
        candidate_loss = model(
            input_ids=input_ids,
            input_features=input_features,
            input_features_mask=input_features_mask,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        ).loss
        candidate_loss.backward()
        candidate_gradient = native_projection.weight.grad.detach().float().cpu().clone()

        model.zero_grad(set_to_none=True)
        model.model.audio_tower.output_proj = FP32Projection(native_projection)
        reference_loss = model(
            input_ids=input_ids,
            input_features=input_features,
            input_features_mask=input_features_mask,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        ).loss
        reference_loss.backward()
        reference_gradient = native_projection.weight.grad.detach().float().cpu().clone()

        effect = candidate_gradient - reference_gradient
        candidate_write = -args.learning_rate * candidate_gradient
        reference_write = -args.learning_rate * reference_gradient
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
                    ((candidate_write - reference_write) * reference_write).sum()
                    / (reference_write.square().sum() + 1e-30)
                ),
            }
        )
        effects.append(effect)
        writes.append(candidate_write)
        references.append(reference_write)
        print(json.dumps({"event": "GEMMA4_AUDIO_STATE", **rows[-1]}), flush=True)
        del candidate_loss, reference_loss, input_ids, labels, input_features, input_features_mask
        torch.cuda.empty_cache()

    model.model.audio_tower.output_proj = native_projection
    calibration = torch.stack([x.double() for x in effects[: args.calibration_states]])
    confirmation = torch.stack([x.double() for x in effects[args.calibration_states :]])
    direction = calibration.mean(dim=0)
    direction_norm = float(direction.norm())
    if direction_norm:
        direction = direction / direction_norm
    projections = [float(torch.sum(x.double() * direction)) for x in effects[args.calibration_states :]]
    aligned = [row["aligned_write_scaling"] for row in rows[args.calibration_states :]]
    return {
        "schema": "kernel-analyzer-gemma4-audio-language-output-projection-natural-v1",
        "status": "COMPLETE_NATURAL_AUDIO_LANGUAGE_BOUNDARY",
        "model": str(model_dir),
        "operator_family": "audio_encoder_output_projection_materialization",
        "candidate": "native BF16 audio output projection in full Gemma-4 audio-language forward",
        "reference": "same audio-language forward with FP32 projection accumulation followed by BF16 writeback",
        "input_source": "real LibriSpeech waveforms and a fixed text target",
        "parameter": "model.audio_tower.output_proj.weight",
        "optimizer_proxy": {"name": "SGD", "learning_rate": args.learning_rate},
        "comparison_scope": {
            "same_checkpoint": True,
            "same_audio": True,
            "same_text_labels": True,
            "same_audio_encoder": True,
            "single_changed_boundary": "audio_tower.output_proj accumulation/materialization",
        },
        "rows": rows,
        "summary": {
            "state_count": len(rows),
            "calibration_count": args.calibration_states,
            "confirmation_count": len(rows) - args.calibration_states,
            "loss_difference_mean": float(np.mean([row["loss_difference"] for row in rows])),
            "loss_difference_interval_normal_95": normal_interval([row["loss_difference"] for row in rows]),
            "write_effect_rms_mean": float(np.mean([row["write_effect_rms_over_reference"] for row in rows])),
            "confirmation_projection_mean": float(np.mean(projections)),
            "confirmation_projection_interval_normal_95": normal_interval(projections),
            "confirmation_projection_positive": sum(value > 0 for value in projections),
            "confirmation_projection_negative": sum(value < 0 for value in projections),
            "confirmation_aligned_write_mean": float(np.mean(aligned)),
            "confirmation_aligned_write_interval_normal_95": normal_interval(aligned),
            "calibration_direction_norm": direction_norm,
        },
        "claim_boundary": (
            "One Gemma-4 E2B checkpoint, eight real speech inputs and one fixed text target; "
            "this is a fixed-suite natural audio-language boundary result, not a population "
            "or long-run training claim and not a kernel-identity claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/google/gemma-4-E2B"))
    parser.add_argument("--waveforms", type=Path, required=True)
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--calibration-states", type=int, default=4)
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
