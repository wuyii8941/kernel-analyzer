#!/usr/bin/env python3
"""Freeze tokenizer-specific natural text windows for any local LM."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--validation-arrow", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--states", type=int, default=32)
    parser.add_argument("--stride-multiplier", type=int, default=17)
    args = parser.parse_args()
    for path in (args.model.resolve(), args.validation_arrow.resolve(), args.output.resolve()):
        if not path.is_relative_to(Path("/data1/tzh")):
            raise RuntimeError(f"path outside /data1/tzh: {path}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    dataset = Dataset.from_file(str(args.validation_arrow))
    text = "\n".join(str(row["text"]) for row in dataset if str(row["text"]).strip())
    tokens = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    stride = args.seq_len * args.stride_multiplier
    required = (args.states - 1) * stride + args.seq_len
    if tokens.numel() < required:
        raise RuntimeError(f"token stream too short: {tokens.numel()} < {required}")
    windows = [tokens[i * stride:i * stride + args.seq_len].clone() for i in range(args.states)]
    payload = {
        "schema": "kernel-analyzer-natural-lm-input-bank-v2",
        "model": str(args.model.resolve()),
        "model_config_sha256": hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
        "source": str(args.validation_arrow.resolve()),
        "source_sha256": hashlib.sha256(args.validation_arrow.read_bytes()).hexdigest(),
        "tokenizer_class": type(tokenizer).__name__,
        "seq_len": args.seq_len,
        "stride_multiplier": args.stride_multiplier,
        "states": [{
            "state_id": index,
            "token_ids": window.tolist(),
            "token_sha256": hashlib.sha256(window.numpy().tobytes()).hexdigest(),
        } for index, window in enumerate(windows)],
    }
    payload["result_sha256"] = digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "states": len(windows), "sha256": payload["result_sha256"]}))


if __name__ == "__main__":
    main()
