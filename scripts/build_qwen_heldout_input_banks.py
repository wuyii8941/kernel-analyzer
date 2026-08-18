#!/usr/bin/env python3
"""Materialize the three frozen Qwen held-out shape banks for generic runners."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/state_design.json"


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    design = json.loads(SOURCE.read_text())
    for length in (64, 128, 256):
        bucket = f"seq{length}"
        rows = [
            row for row in design["records"]
            if row["split"] == "heldout" and row["length_bucket"] == bucket
        ]
        if len(rows) != 32 or any(len(row["input_ids"]) != length for row in rows):
            raise RuntimeError(f"frozen Qwen held-out denominator changed: {bucket}")
        payload = {
            "schema": "kernel-analyzer-qwen-heldout-input-bank-v1",
            "model": design["model_path"], "split": "heldout",
            "length_bucket": bucket, "source_design_sha256": design["design_sha256"],
            "states": rows,
        }
        payload["result_sha256"] = digest(payload)
        output = ROOT / f"results/coverage/qwen_seq{length}_input_bank.json"
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        temporary.replace(output)
        print(json.dumps({"output": str(output.relative_to(ROOT)), "states": len(rows)}))


if __name__ == "__main__":
    main()
