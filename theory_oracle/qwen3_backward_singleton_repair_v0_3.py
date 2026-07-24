#!/usr/bin/env python
"""Revision 3: eager-boundary repair of the singleton attention safe-softmax."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import theory_oracle.qwen3_backward_singleton_repair_v0_1 as v1


TREATMENT = "attention_safe_softmax"
FAMILY = "triton_red_fused__safe_softmax_add_prepare_softmax_online_view_12"


def repair_attention_safe_softmax(torch: Any, values: tuple[Any, ...]) -> None:
    destination, attention_scores, attention_mask = values[:3]
    scores = attention_scores.reshape_as(destination)
    logits = scores + attention_mask.to(scores.dtype)
    destination.copy_(torch.ops.aten._safe_softmax.default(logits, -1, torch.float32))


def main() -> None:
    if "--out-dir" not in sys.argv:
        raise ValueError("--out-dir is required")
    out_dir = Path(sys.argv[sys.argv.index("--out-dir") + 1]).resolve()
    v1.TREATMENTS[TREATMENT] = {
        "family": FAMILY,
        "repair": repair_attention_safe_softmax,
        "semantic_boundary": "FP32 score-plus-mask followed by eager aten._safe_softmax over the last axis",
    }
    v1.main()
    result_path = out_dir / "result.json"
    result = json.loads(result_path.read_text())
    result["schema_version"] = (
        "forkcert.qwen3-natural-transition-with-backward-singleton-repair.v0.3"
    )
    result["backward_singleton_repair"]["repair_revision"] = "v0.3-safe-softmax"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
