#!/usr/bin/env python3
"""Refresh embedded result hashes after compact metadata edits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "results" / "final"


def main() -> None:
    names = (
        "checkpoint_matrix.json", "checkpoint_carrier.json", "checkpoint_inductor.json",
        "checkpoint_matrix_seq64.json", "checkpoint_matrix_seq256.json",
        "checkpoint_carrier_seq64.json", "checkpoint_carrier_seq256.json",
        "checkpoint_inductor_bf16_seq64.json", "checkpoint_inductor_bf16_seq256.json",
        "checkpoint_inductor_fp16.json", "checkpoint_inductor_fp16_seq64.json", "checkpoint_inductor_fp16_seq256.json",
        "checkpoint_inductor_fp32.json", "checkpoint_inductor_fp32_seq64.json", "checkpoint_inductor_fp32_seq256.json",
        "checkpoint_inductor_tf32.json", "checkpoint_inductor_tf32_seq64.json", "checkpoint_inductor_tf32_seq256.json",
    )
    for name in names:
        path = ROOT / name
        data = json.loads(path.read_text())
        if "bank_manifest" in data:
            data["bank_manifest"] = "results/final/natural_bank.json"
        data.pop("result_sha256", None)
        data["result_sha256"] = hashlib.sha256(
            json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"refreshed": list(names)}))


if __name__ == "__main__":
    main()
