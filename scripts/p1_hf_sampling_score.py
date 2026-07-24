#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

from forkcert.config import load_config
from forkcert.io import read_jsonl
from forkcert.logprob_runner import cleanup_memory, configure_determinism, load_hf_path
from phase6_twin_training import path_config
from phase7_sampling_scan import decisions_for_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture one independent HF sampling-decision run.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--state-jsonl", required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    config = path_config(cfg, "path_ref")
    configure_determinism(int(cfg.get("seed", 0)))
    case_ids = {row["case_id"] for row in read_jsonl(args.state_jsonl)}
    samples = [row for row in read_jsonl(args.samples) if row["case_id"] in case_ids]
    tokenizer, model = load_hf_path(config)
    try:
        rows = []
        for sample in samples:
            for row in decisions_for_sample(
                tokenizer,
                model,
                config,
                sample,
                args.top_k,
                args.top_p,
                args.temperature,
                args.draws,
            ):
                row["path"] = config.name
                row.pop("top_p_ids", None)
                rows.append(row)
    finally:
        del model, tokenizer
        cleanup_memory()
    payload = {
        "metadata": {
            "schema_version": "forkcert.p1.hf-sampling.v1",
            "path": config.name,
            "torch_version": importlib.metadata.version("torch"),
            "transformers_version": importlib.metadata.version("transformers"),
            "samples": len(samples),
            "tokens": len(rows),
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "draws": args.draws,
        },
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["metadata"], indent=2, sort_keys=True))
    if len(rows) != len(samples) * 128:
        raise SystemExit(f"unexpected HF sampling coverage: {len(rows)}")


if __name__ == "__main__":
    main()
