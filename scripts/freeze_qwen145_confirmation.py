#!/usr/bin/env python3
"""Freeze a candidate-blind continuation bank for backward:145 confirmation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("/data1/tzh/models/Qwen/Qwen3-1.7B")
ORIGINAL = ROOT / "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/state_design.json"
OUTPUT = ROOT / "results/coverage/qwen145_confirmation_design.json"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()).hexdigest()


def main() -> None:
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from datasets import load_dataset
    from transformers import AutoTokenizer

    original = json.loads(ORIGINAL.read_text())
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
    dataset = load_dataset(
        original["dataset"]["name"], original["dataset"]["config"],
        split=original["dataset"]["split"], revision=original["dataset"]["revision"],
    )
    documents = [str(row["text"]).strip() for row in dataset if str(row["text"]).strip()]
    stream = tokenizer(
        "\n\n".join(documents), add_special_tokens=False, return_attention_mask=False
    )["input_ids"]
    cursor = sum(int(row["length"]) + 1 for row in original["records"])
    records = []
    for index in range(64):
        input_ids = stream[cursor:cursor + 64]
        if len(input_ids) != 64:
            raise RuntimeError("token stream is too short for confirmation bank")
        cursor += 65
        row = {
            "sequence_id": f"qwen145-confirm-{index:04d}",
            "cluster_id": f"qwen145-confirm-natural-text-{index:04d}",
            "split": "confirmation",
            "length": 64,
            "length_bucket": "seq64",
            "layer_role": "not_used_for_selection",
            "input_ids": input_ids,
        }
        row["record_sha256"] = digest(row)
        records.append(row)
    payload = {
        "schema": "kernel-analyzer-qwen145-confirmation-design-v1",
        "status": "FROZEN_BEFORE_CONFIRMATION_CANDIDATE_EXECUTION",
        "parent_design_sha256": original["design_sha256"],
        "dataset": original["dataset"],
        "continuation_cursor_start": sum(
            int(row["length"]) + 1 for row in original["records"]
        ),
        "candidate_data_used": False,
        "hypothesis": {
            "region_id": "backward:145",
            "endpoint": "out_ptr0",
            "reference_role": "BF16_EAGER_SEMANTIC_BOUNDARY",
            "carrier_parameter": "model.layers.23.post_attention_layernorm.weight",
            "carrier_coordinates": "ALL_2048_COORDINATES",
            "primary_statistic": "DISTINCT_STATE_CROSS_INNER_PRODUCT_U",
            "success_gate": "TWO_SIDED_CLUSTER_BOOTSTRAP_95_LOWER_BOUND_GT_ZERO",
            "states": 64,
            "repeats": 2,
        },
        "records": records,
    }
    payload["design_sha256"] = digest(payload)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUTPUT), "states": 64, "sha256": payload["design_sha256"]}))


if __name__ == "__main__":
    main()
