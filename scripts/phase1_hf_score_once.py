#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

from forkcert.config import load_config
from forkcert.io import read_jsonl
from forkcert.logprob_runner import run_path_twice
from phase6_twin_training import path_config


def main() -> None:
    parser = argparse.ArgumentParser(description="HF path scorer with two in-process deterministic self runs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--path-key", choices=["path_ref", "path_alt"], default="path_ref")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    config = path_config(cfg, args.path_key)
    samples = read_jsonl(args.samples)[: args.max_samples]
    runs = run_path_twice(config, samples, seed=int(cfg.get("seed", 0)))
    rows = []
    for first, second in zip(runs[0], runs[1], strict=True):
        key = (str(first["case_id"]), int(first["token_index"]), int(first["token_id"]))
        other = (str(second["case_id"]), int(second["token_index"]), int(second["token_id"]))
        if key != other:
            raise ValueError(f"HF self token alignment mismatch: {key} != {other}")
        rows.append(
            {
                "case_id": key[0],
                "token_index": key[1],
                "token_id": key[2],
                "logp": float(first["logp"]),
                "delta_self": abs(float(second["logp"]) - float(first["logp"])),
            }
        )
    payload = {
        "metadata": {
            "schema_version": "forkcert.hf_score.v1",
            "path": config.name,
            "model": config.model_name_or_path,
            "dtype": config.dtype,
            "autocast_dtype": config.autocast_dtype,
            "torch_version": importlib.metadata.version("torch"),
            "transformers_version": importlib.metadata.version("transformers"),
            "requests": len(samples),
            "tokens": len(rows),
        },
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["metadata"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
