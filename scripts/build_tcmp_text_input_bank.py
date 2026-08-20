#!/usr/bin/env python3
"""Build frozen 2+8+16 natural-text state banks for TCMP model cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sequence_length not in {128, 256, 512}:
        raise ValueError("TCMP v1 admits only seq128/256/512")
    for path in (args.model, args.output.parent):
        resolved = path.resolve()
        if Path("/data1/tzh") not in (resolved, *resolved.parents):
            raise ValueError("all model/data paths must remain under /data1/tzh")

    os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
    os.environ.setdefault("HF_DATASETS_CACHE", "/data1/tzh/cache/huggingface/datasets")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="train",
        download_mode="reuse_dataset_if_exists",
    )
    # Add one token between blocks so adjacent states do not share a boundary.
    required = 26 * (args.sequence_length + 1)
    stream: list[int] = []
    documents = 0
    for row in dataset:
        text = str(row["text"]).strip()
        if not text:
            continue
        stream.extend(tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"])
        documents += 1
        if len(stream) >= required:
            break
    if len(stream) < required:
        raise RuntimeError("cached natural-text stream is too short")

    roles = ["ENGINEERING"] * 2 + ["SCREENING"] * 8 + ["CONFIRMATION"] * 16
    rows = []
    for index, role in enumerate(roles):
        start = index * (args.sequence_length + 1)
        values = stream[start:start + args.sequence_length]
        import numpy as np
        encoded = np.asarray(values, dtype=np.int64).tobytes()
        rows.append({
            "state_id": f"{args.cell_id}-{role.lower()}-{index:02d}",
            "role": role,
            "order_within_role": sum(previous == role for previous in roles[:index]),
            "token_ids": values,
            "token_sha256": hashlib.sha256(encoded).hexdigest(),
        })
    payload = {
        "schema": "kernel-analyzer-tcmp-input-bank-v1",
        "cell_id": args.cell_id,
        "model_path": str(args.model.resolve()),
        "sequence_length": args.sequence_length,
        "dataset": "Salesforce/wikitext:wikitext-103-raw-v1:train",
        "nonempty_documents_consumed": documents,
        "states": rows,
        "splits": {"ENGINEERING": 2, "SCREENING": 8, "CONFIRMATION": 16},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "states": len(rows), "sequence_length": args.sequence_length}))


if __name__ == "__main__":
    main()
