#!/usr/bin/env python3
"""Freeze natural Mamba token windows for host and container campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from mamba_scan_screen import canonical_hash, natural_windows


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/state-spaces/mamba-130m-hf"))
    parser.add_argument(
        "--validation-arrow",
        type=Path,
        default=Path(
            "/data1/tzh/cache/huggingface/datasets/Salesforce___wikitext/"
            "wikitext-103-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/"
            "wikitext-validation.arrow"
        ),
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results/mamba_scan/input_bank.json")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--states", type=int, default=32)
    parser.add_argument("--stride-multiplier", type=int, default=17)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    windows = natural_windows(
        tokenizer, args.validation_arrow, args.seq_len, args.states, args.stride_multiplier
    )
    output = {
        "schema": "kernel-analyzer-mamba-natural-input-bank-v1",
        "model": str(args.model),
        "source": str(args.validation_arrow),
        "source_sha256": hashlib.sha256(args.validation_arrow.read_bytes()).hexdigest(),
        "tokenizer_class": type(tokenizer).__name__,
        "seq_len": args.seq_len,
        "stride_multiplier": args.stride_multiplier,
        "states": [
            {
                "state_id": index,
                "token_ids": window.tolist(),
                "token_sha256": hashlib.sha256(window.numpy().tobytes()).hexdigest(),
            }
            for index, window in enumerate(windows)
        ],
    }
    output["result_sha256"] = canonical_hash(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "states": len(windows)}, sort_keys=True))


if __name__ == "__main__":
    main()
