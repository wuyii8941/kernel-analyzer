#!/usr/bin/env python
"""Resource-bounded wrapper around the frozen Accelerate-native v0.8 query."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import Any

import theory_oracle.qwen3_grpo_grad_branch_repair_v0_5 as base
import theory_oracle.qwen3_grpo_grad_branch_repair_v0_8 as v08


_original_run_arm = base.run_arm


def resource_bounded_run_arm(*args: Any, **kwargs: Any):
    import torch

    try:
        return _original_run_arm(*args, **kwargs)
    finally:
        v08._accelerators.clear()
        gc.collect()
        torch._dynamo.reset()
        torch.cuda.empty_cache()


def main() -> None:
    base.run_arm = resource_bounded_run_arm
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    try:
        v08.main()
    finally:
        if out.is_file():
            payload = json.loads(out.read_text(encoding="utf-8"))
            payload["schema_version"] = "forkcert.qwen3-grpo-grad-branch-repair.v0.9"
            payload["resource_lifetime_isolated_per_arm"] = True
            payload["prior_invalid_versions_preserved"] = [
                "v0.5", "v0.6", "v0.7", "v0.8"
            ]
            payload["contract"] = str(
                Path(__file__).with_name(
                    "QWEN3_GRPO_GRAD_BRANCH_REPAIR_CONTRACT_V0_9_2026-07-17.md"
                ).resolve()
            )
            out.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )


if __name__ == "__main__":
    main()
