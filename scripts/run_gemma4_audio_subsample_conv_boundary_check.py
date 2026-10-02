#!/usr/bin/env python3
"""Check the isolated Gemma-4 audio subsampling Conv2d boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/google/gemma-4-E2B"))
    parser.add_argument("--waveforms", type=Path, required=True)
    parser.add_argument("--states", type=int, default=3)
    parser.add_argument("--layer", type=int, choices=[0, 1], default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import sys
    transformers_site = "/data1/tzh/envs/pt_nightly_transformers5/lib/python3.11/site-packages"
    if transformers_site not in sys.path:
        sys.path.insert(0, transformers_site)
    from transformers import AutoConfig, AutoFeatureExtractor, AutoTokenizer
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

    device = torch.device(args.device)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    extractor = AutoFeatureExtractor.from_pretrained(args.model, local_files_only=True)
    waves = np.load(args.waveforms, allow_pickle=True)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        args.model, config=config, torch_dtype=torch.bfloat16,
        local_files_only=True, low_cpu_mem_usage=True,
    ).to(device).eval()
    audio = model.model.audio_tower
    layer = audio.subsample_conv_projection.layer0 if args.layer == 0 else audio.subsample_conv_projection.layer1
    conv = layer.conv
    captured: dict[str, torch.Tensor] = {}

    def pre_hook(_module, inputs):
        captured["input"] = inputs[0].detach()

    def post_hook(_module, _inputs, output):
        captured["native_output"] = output.detach()

    h1 = conv.register_forward_pre_hook(pre_hook)
    h2 = conv.register_forward_hook(post_hook)
    rows = []
    try:
        for state_id, waveform in enumerate(waves[: args.states]):
            features = extractor([waveform], sampling_rate=16_000, return_tensors="pt")
            captured.clear()
            with torch.no_grad():
                audio(
                    features["input_features"].to(device=device, dtype=torch.bfloat16),
                    features["input_features_mask"].to(device), return_dict=True,
                )
            x = captured["input"]
            native = captured["native_output"]
            fp32 = F.conv2d(
                x.float(), conv.weight.float(),
                None if conv.bias is None else conv.bias.float(),
                conv.stride, conv.padding, conv.dilation, conv.groups,
            ).to(dtype=x.dtype)
            effect = native.float() - fp32.float()
            denom = fp32.float().norm().item()
            rows.append({
                "state_id": state_id,
                "input_shape": list(x.shape),
                "output_shape": list(native.shape),
                "native_vs_explicit_fp32_relative_rms": float(effect.norm().item() / max(denom, 1e-30)),
                "native_vs_explicit_fp32_max_abs": float(effect.abs().max().item()),
                "native_output_rms": float(native.float().norm().item()),
                "explicit_fp32_output_rms": float(fp32.float().norm().item()),
            })
            torch.cuda.empty_cache()
    finally:
        h1.remove()
        h2.remove()
    output = {
        "schema": "kernel-analyzer-gemma4-audio-subsample-conv-boundary-check-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "operator": f"Gemma-4 audio subsampling Conv2d layer{args.layer}",
        "candidate": "native checkpoint Conv2d",
        "reference": "same captured input with FP32 convolution and original dtype writeback",
        "input_source": "real LibriSpeech waveforms",
        "rows": rows,
        "claim_boundary": "Forward boundary only; does not by itself establish a training-quality or population claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "max_relative_rms": max(r["native_vs_explicit_fp32_relative_rms"] for r in rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
