#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from forkcert.config import load_config
from forkcert.io import read_jsonl
from forkcert.logprob_runner import cleanup_memory, configure_determinism, load_hf_path, model_artifact_fingerprint
from phase7_sampling_scan import decisions_for_sample, path_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture one isolated R1 sampling-decision path run.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--path-key", choices=["path_ref", "path_alt"], required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--warmup-passes", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    config = path_config(cfg, args.path_key)
    configure_determinism(int(cfg.get("seed", 0)))
    samples = read_jsonl(args.samples)
    tokenizer, model = load_hf_path(config)
    try:
        for _ in range(args.warmup_passes):
            for sample in samples:
                decisions_for_sample(
                    tokenizer, model, config, sample, args.top_k, args.top_p, args.temperature, args.draws
                )
        rows = []
        for sample in samples:
            for row in decisions_for_sample(
                tokenizer, model, config, sample, args.top_k, args.top_p, args.temperature, args.draws
            ):
                row["path"] = config.name
                row.pop("top_p_ids", None)
                rows.append(row)
    finally:
        del model, tokenizer
        cleanup_memory()
    payload = {
        "metadata": {
            "schema_version": "forkcert.r1.sampling-path.v1",
            "pid": os.getpid(),
            "path_key": args.path_key,
            "path": config.name,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torchinductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
            "model_artifact_fingerprint": model_artifact_fingerprint(config.model_name_or_path),
            "samples": len(samples),
            "tokens": len(rows),
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "draws": args.draws,
            "discarded_full_warmup_passes": args.warmup_passes,
        },
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload["metadata"][key] for key in ["pid", "path_key", "path", "samples", "tokens"]}, indent=2))
    expected = sum(len(sample["response_ids"]) for sample in samples)
    if len(rows) != expected:
        raise SystemExit(f"unexpected R1 sampling coverage: {len(rows)} != {expected}")


if __name__ == "__main__":
    main()
