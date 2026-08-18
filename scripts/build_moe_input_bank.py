#!/usr/bin/env python3
"""Freeze natural text windows for the Granite MoE campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/ibm-granite/granite-3.1-1b-a400m-base"))
    parser.add_argument(
        "--validation-arrow",
        type=Path,
        default=Path(
            "/data1/tzh/cache/huggingface/datasets/Salesforce___wikitext/"
            "wikitext-103-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/"
            "wikitext-validation.arrow"
        ),
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results/moe/input_bank.json")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--states", type=int, default=24)
    parser.add_argument("--stride-multiplier", type=int, default=17)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    dataset = Dataset.from_file(str(args.validation_arrow))
    text = "\n".join(row["text"] for row in dataset if row["text"].strip())
    tokens = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    stride = args.seq_len * args.stride_multiplier
    required = (args.states - 1) * stride + args.seq_len
    if tokens.numel() < required:
        raise RuntimeError(f"validation token bank too short: {tokens.numel()} < {required}")
    windows = [tokens[i * stride : i * stride + args.seq_len].clone() for i in range(args.states)]
    output = {
        "schema": "kernel-analyzer-natural-lm-input-bank-v1",
        "model": str(args.model),
        "source": str(args.validation_arrow),
        "source_sha256": hashlib.sha256(args.validation_arrow.read_bytes()).hexdigest(),
        "tokenizer_class": type(tokenizer).__name__,
        "seq_len": args.seq_len,
        "stride_multiplier": args.stride_multiplier,
        "states": [
            {
                "state_id": i,
                "token_ids": window.tolist(),
                "token_sha256": hashlib.sha256(window.numpy().tobytes()).hexdigest(),
            }
            for i, window in enumerate(windows)
        ],
    }
    output["result_sha256"] = canonical_hash(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "states": len(windows)}, sort_keys=True))


if __name__ == "__main__":
    main()
