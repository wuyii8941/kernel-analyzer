#!/usr/bin/env python
"""Shape-canonicalized v0.6 wrapper for the frozen grad-branch repair."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import theory_oracle.qwen3_grpo_grad_branch_repair_v0_5 as base


_v05_tensor_sha256 = base.tensor_sha256


def native_batch_tensor_sha256(tensor: Any) -> str:
    """Hash the complete scorer in the frozen Trainer-native [4, 128] shape."""
    if int(tensor.numel()) != 512:
        raise RuntimeError(f"unexpected scorer size for frozen witness: {tensor.numel()}")
    return _v05_tensor_sha256(tensor.reshape(4, 128))


def main() -> None:
    base.tensor_sha256 = native_batch_tensor_sha256
    out_index = sys.argv.index("--out") + 1
    out = Path(sys.argv[out_index])
    try:
        base.main()
    finally:
        if out.is_file():
            payload = json.loads(out.read_text(encoding="utf-8"))
            payload["schema_version"] = "forkcert.qwen3-grpo-grad-branch-repair.v0.6"
            payload["hash_canonical_shape"] = [4, 128]
            payload["v05_invalid_preserved"] = True
            payload["contract"] = str(
                Path(__file__).with_name(
                    "QWEN3_GRPO_GRAD_BRANCH_REPAIR_CONTRACT_V0_6_2026-07-17.md"
                ).resolve()
            )
            out.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )


if __name__ == "__main__":
    main()
