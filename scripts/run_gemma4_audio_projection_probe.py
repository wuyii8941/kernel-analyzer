#!/usr/bin/env python3
"""Screen the Gemma-4 audio encoder for a new numerical problem family.

This is deliberately a component screen, not a natural training claim.  The
checkpoint weights are real, but the local environment has no audio corpus, so
the waveform bank is declared synthetic.  The probe compares the native BF16
output projection with the same projection evaluated in FP32 and written back
to BF16, then measures output, gradient, and one-step update differences.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoFeatureExtractor
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioModel


def _load_audio_tower(model_dir: Path) -> Gemma4AudioModel:
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True).audio_config
    model = Gemma4AudioModel(config).to(dtype=torch.bfloat16).eval()
    state: dict[str, torch.Tensor] = {}
    with safe_open(str(model_dir / "model.safetensors"), framework="pt", device="cpu") as handle:
        prefix = "model.audio_tower."
        for key in handle.keys():
            if key.startswith(prefix):
                state[key[len(prefix) :]] = handle.get_tensor(key)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"audio checkpoint binding mismatch: missing={missing}, unexpected={unexpected}")
    return model


def _wave_bank(states: int, samples: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    time = np.arange(samples, dtype=np.float32) / 16_000.0
    bank: list[np.ndarray] = []
    for idx in range(states):
        kind = idx % 4
        if kind == 0:
            wave = np.sin(2 * np.pi * (220.0 + 37.0 * idx) * time)
        elif kind == 1:
            wave = np.sin(2 * np.pi * (150.0 + 650.0 * time) * time)
        elif kind == 2:
            wave = rng.normal(0.0, 0.1, samples)
        else:
            wave = np.zeros(samples, dtype=np.float32)
        bank.append(np.asarray(wave, dtype=np.float32))
    return bank


def run(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = args.model.resolve()
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir, local_files_only=True)
    features = feature_extractor(
        _wave_bank(args.states, args.samples, args.seed),
        sampling_rate=16_000,
        return_tensors="pt",
    )
    model = _load_audio_tower(model_dir)

    # Extract the real audio-path representation once.  The output projection
    # is the only changed boundary in the comparison below.
    output_projection = model.output_proj
    model.output_proj = torch.nn.Identity()
    rows: list[dict[str, Any]] = []
    gradients: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []
    for state_id in range(args.states):
        with torch.no_grad():
            encoded = model(
                features["input_features"][state_id : state_id + 1].to(torch.bfloat16),
                features["input_features_mask"][state_id : state_id + 1],
                return_dict=True,
            ).last_hidden_state.detach()
        weight = output_projection.weight.detach()
        bias = output_projection.bias.detach()
        branches: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]] = {}
        for name in ("native_bf16", "fp32_accumulate"):
            branch_weight = weight.clone().requires_grad_(True)
            branch_bias = bias.clone().requires_grad_(True)
            if name == "native_bf16":
                projected = torch.nn.functional.linear(encoded, branch_weight, branch_bias)
            else:
                projected = torch.nn.functional.linear(
                    encoded.float(), branch_weight.float(), branch_bias.float()
                ).to(encoded.dtype)
            loss = projected.float().square().mean()
            loss.backward()
            branches[name] = (
                projected.detach().float(),
                branch_weight.grad.detach().float(),
                branch_bias.grad.detach().float(),
                float(loss.detach()),
            )
        candidate = branches["native_bf16"]
        reference = branches["fp32_accumulate"]
        delta_output = candidate[0] - reference[0]
        delta_gradient = candidate[1] - reference[1]
        delta_update = -args.learning_rate * delta_gradient
        outputs.append(delta_output.reshape(-1))
        gradients.append(delta_gradient.reshape(-1))
        rows.append(
            {
                "state_id": state_id,
                "loss_difference": candidate[3] - reference[3],
                "output_rms": float(delta_output.square().mean().sqrt()),
                "output_relative_rms": float(
                    delta_output.square().mean().sqrt() / (reference[0].square().mean().sqrt() + 1e-30)
                ),
                "weight_gradient_rms": float(delta_gradient.square().mean().sqrt()),
                "weight_gradient_relative_rms": float(
                    delta_gradient.square().mean().sqrt() / (reference[1].square().mean().sqrt() + 1e-30)
                ),
                "weight_update_relative_rms": float(
                    delta_update.norm() / (args.learning_rate * reference[1].norm() + 1e-30)
                ),
            }
        )

    output_matrix = torch.stack(outputs)
    gradient_matrix = torch.stack(gradients)
    mean_output = output_matrix.mean(dim=0)
    mean_gradient = gradient_matrix.mean(dim=0)
    return {
        "schema": "kernel-analyzer-gemma4-audio-projection-probe-v1",
        "status": "COMPONENT_CANDIDATE_REQUIRES_REAL_AUDIO_CONFIRMATION",
        "model": str(model_dir),
        "operator_family": "audio_encoder_output_projection",
        "source": "real Gemma-4 E2B checkpoint with a declared synthetic waveform bank",
        "candidate": "native BF16 linear output",
        "reference": "FP32 linear accumulation followed by BF16 writeback",
        "states": args.states,
        "rows": rows,
        "direction_diagnostics": {
            "output_mean_norm_over_rms": float(mean_output.norm() / (output_matrix.square().mean().sqrt() + 1e-30)),
            "gradient_mean_norm_over_rms": float(mean_gradient.norm() / (gradient_matrix.square().mean().sqrt() + 1e-30)),
            "output_mean_coordinate_sign": int(torch.sign(mean_output).sum()),
            "gradient_mean_coordinate_sign": int(torch.sign(mean_gradient).sum()),
        },
        "claim_boundary": (
            "This is a real-weight audio-encoder component screen only.  The input bank is synthetic, "
            "there is no full audio-language training loss or GPU Triton endpoint, and it is therefore "
            "not counted as an additional natural training problem group."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/google/gemma-4-E2B"))
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--samples", type=int, default=16_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["direction_diagnostics"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
