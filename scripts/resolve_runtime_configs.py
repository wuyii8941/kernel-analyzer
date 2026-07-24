#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from forkcert.config import load_config


def resolve_value(value: Any, use_bf16: bool) -> Any:
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            resolved_key = key
            if not use_bf16 and key == "allow_bf16_reduced_precision_reduction":
                resolved_key = "allow_fp16_reduced_precision_reduction"
            output[resolved_key] = resolve_value(item, use_bf16)
        return output
    if isinstance(value, list):
        return [resolve_value(item, use_bf16) for item in value]
    if not use_bf16 and isinstance(value, str):
        lowered = value.lower()
        if lowered == "bfloat16":
            return "float16"
        if lowered == "bf16":
            return "fp16"
        if lowered in {"flash", "flash_attention"}:
            return "efficient"
        return value.replace("bf16", "fp16").replace("Flash", "Efficient").replace("flash", "efficient")
    return value


def resolve_config(config_path: str, config: dict[str, Any], use_bf16: bool) -> dict[str, Any]:
    resolved = resolve_value(config, use_bf16)
    if Path(config_path).name == "hf_materialization.example.yaml":
        alt = resolved["path_alt"]
        alt["materialization_dtype"] = "fp16" if use_bf16 else "bf16"
        alt["name"] = (
            "hf-eager-bf16-fp16-roundtrip-sensitivity"
            if use_bf16
            else "hf-eager-fp16-bf16-roundtrip-sensitivity"
        )
    if use_bf16 or Path(config_path).name != "hf_sdpa_math_flash.example.yaml":
        return resolved

    # Turing GPUs cannot execute PyTorch Flash, efficient, or cuDNN SDPA kernels.
    # Keep this a same-dtype, single-variable attention comparison by switching
    # between Transformers' eager attention implementation and SDPA-MATH.
    ref = resolved["path_ref"]
    alt = resolved["path_alt"]
    ref.update(
        {
            "name": "hf-eager-attention-fp16",
            "attn_implementation": "eager",
            "attention_backend": None,
        }
    )
    alt.update(
        {
            "name": "hf-sdpa-math-fp16",
            "attn_implementation": "sdpa",
            "attention_backend": "math",
        }
    )
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve ForkCert precision/backend configs for the selected GPU.")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--out-dir", default="results/runtime_configs")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("runtime config resolution requires a CUDA-visible shell")
    capability = torch.cuda.get_device_capability(0)
    use_bf16 = capability[0] >= 8
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for config_path in args.config:
        source = Path(config_path)
        resolved = resolve_config(str(source), load_config(str(source)), use_bf16)
        out = out_dir / source.name
        out.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
        written.append(str(out))
    payload = {
        "precision": "bf16" if use_bf16 else "fp16",
        "attention_alt": "sdpa_math_vs_flash" if use_bf16 else "eager_attention_vs_sdpa_math",
        "selected_device": torch.cuda.get_device_name(0),
        "capability": list(capability),
        "configs": written,
    }
    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
