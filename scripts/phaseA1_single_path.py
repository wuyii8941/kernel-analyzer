#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from forkcert.config import load_config
from forkcert.io import read_jsonl, write_jsonl
from forkcert.logprob_runner import (
    PathConfig,
    configure_determinism,
    load_hf_path,
    model_artifact_fingerprint,
    response_logprobs_for_sample,
    tokenization_fingerprint_for_sample,
)


def path_config(data: dict, key: str) -> PathConfig:
    item = data[key]
    return PathConfig(
        name=item["name"],
        model_name_or_path=item["model_name_or_path"],
        dtype=item.get("dtype", "bf16"),
        autocast_dtype=item.get("autocast_dtype"),
        device=item.get("device", "cuda"),
        compile_model=item.get("compile_model", False),
        attn_implementation=item.get("attn_implementation"),
        attention_backend=item.get("attention_backend"),
        logits_upcast_fp32=item.get("logits_upcast_fp32", True),
        rmsnorm_reference=item.get("rmsnorm_reference", False),
        rmsnorm_no_upcast=item.get("rmsnorm_no_upcast", False),
        rmsnorm_compile=item.get("rmsnorm_compile", False),
        materialize_bf16_outputs=item.get("materialize_bf16_outputs", False),
        materialization_dtype=item.get("materialization_dtype"),
        allow_bf16_reduced_precision_reduction=item.get("allow_bf16_reduced_precision_reduction"),
        allow_fp16_reduced_precision_reduction=item.get("allow_fp16_reduced_precision_reduction"),
        model_training_mode=item.get("model_training_mode", False),
        gradient_checkpointing=item.get("gradient_checkpointing", False),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one ForkCert path once in an isolated process/CUDA context.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--path-key", choices=["path_ref", "path_alt"], required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--warmup-passes", type=int, default=0)
    args = parser.parse_args()

    import torch

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 0))
    configure_determinism(seed=seed)
    path = path_config(cfg, args.path_key)
    samples = read_jsonl(args.samples)
    tokenizer, model = load_hf_path(path)
    for _ in range(args.warmup_passes):
        for sample in samples:
            response_logprobs_for_sample(tokenizer, model, path, sample)
    rows = []
    for sample in samples:
        fingerprint = tokenization_fingerprint_for_sample(tokenizer, sample, path.device)
        for row in response_logprobs_for_sample(tokenizer, model, path, sample):
            rows.append({"case_id": sample["case_id"], **fingerprint, **row})
    write_jsonl(args.out_jsonl, rows)

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    metadata = {
        "pid": os.getpid(),
        "config": args.config,
        "path_key": args.path_key,
        "path": path.__dict__,
        "samples": len(samples),
        "tokens": len(rows),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "cuda_device_uuid": str(getattr(props, "uuid", "unknown")),
        "torchinductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        "model_artifact_fingerprint": model_artifact_fingerprint(path.model_name_or_path),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "discarded_full_warmup_passes": args.warmup_passes,
    }
    out = Path(args.metadata)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: metadata[key] for key in ["pid", "path_key", "samples", "tokens", "cuda_visible_devices", "cuda_device_name", "cuda_device_uuid"]}, indent=2))


if __name__ == "__main__":
    main()
