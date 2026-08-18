#!/usr/bin/env python3
"""Freeze the protocol-v2 full-carrier reconfirmation for seq128 lm_head dX."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("/data1/tzh/models/Qwen/Qwen3-1.7B")
ORIGINAL = ROOT / "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/state_design.json"
Q145 = ROOT / "results/coverage/qwen145_confirmation_design.json"
OUTPUT = ROOT / "results/coverage/lmhead_t3_confirmation_design.json"


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
    q145 = json.loads(Q145.read_text())
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
    dataset = load_dataset(
        original["dataset"]["name"], original["dataset"]["config"],
        split=original["dataset"]["split"], revision=original["dataset"]["revision"],
    )
    documents = [str(row["text"]).strip() for row in dataset if str(row["text"]).strip()]
    stream = tokenizer("\n\n".join(documents), add_special_tokens=False, return_attention_mask=False)["input_ids"]
    # Skip the original mixed-shape bank and the later 64-state seq64 q145 bank.
    cursor = sum(int(row["length"]) + 1 for row in original["records"]) + sum(
        int(row["length"]) + 1 for row in q145["records"]
    )
    start = cursor
    records = []
    for index in range(32):
        ids = stream[cursor:cursor + 128]
        if len(ids) != 128:
            raise RuntimeError("token stream too short")
        cursor += 129
        row = {
            "sequence_id": f"lmhead-t3-confirm-{index:04d}",
            "cluster_id": f"lmhead-t3-natural-text-{index:04d}",
            "split": "confirmation",
            "length": 128,
            "length_bucket": "seq128",
            "input_ids": ids,
        }
        row["record_sha256"] = digest(row)
        records.append(row)
    payload = {
        "schema": "kernel-analyzer-lmhead-t3-confirmation-design-v1",
        "status": "FROZEN_BEFORE_CANDIDATE_EXECUTION",
        "candidate_data_used": False,
        "parent_design_sha256": original["design_sha256"],
        "excluded_q145_design_sha256": q145["design_sha256"],
        "continuation_cursor_start": start,
        "dataset": original["dataset"],
        "hypothesis": {
            "proof_unit_id": "eager-semantic-region::forward:3454",
            "forward": "Y=X W^T",
            "actual_vjp": "dX=G W",
            "runtime_autograd_node": "MmBackward0",
            "autograd_sequence_nr": 6673,
            "tuple_index": 0,
            "carrier": "ALL_COORDINATES_OF_ALL_310_REACHABLE_PARAMETER_GRADIENTS",
            "states": 32,
            "passes": 2,
            "primary_statistic": "DISTINCT_STATE_FULL_VECTOR_U",
            "uncertainty": "LEAVE_ONE_STATE_U_PSEUDOVALUE_CLUSTER_BOOTSTRAP",
            "success_gate": "BOOTSTRAP_95_LOWER_BOUND_GT_ZERO",
            "multiplicity_family": ["seq128_lm_head_input_vjp_mm"],
            "optional_stopping": False,
        },
        "records": records,
    }
    payload["design_sha256"] = digest(payload)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUTPUT), "sha256": payload["design_sha256"], "states": 32}))


if __name__ == "__main__":
    main()
