#!/usr/bin/env python
"""History-realized v0.7 wrapper for the frozen Qwen grad-branch repair."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import theory_oracle.qwen3_grpo_grad_branch_repair_v0_5 as base
from forkcert.io import read_jsonl


_original_hash = base.tensor_sha256
_original_scorer = base.trainer_response_logps_with_grad


def native_batch_hash(tensor: Any) -> str:
    if int(tensor.numel()) != 512:
        raise RuntimeError(f"unexpected scorer size: {tensor.numel()}")
    return _original_hash(tensor.reshape(4, 128))


def main() -> None:
    samples_path = Path(sys.argv[sys.argv.index("--samples") + 1])
    all_samples = read_jsonl(samples_path)
    history_samples = [
        row for row in all_samples if int(row["metadata"]["rollout_batch"]) == 0
    ]
    if len(history_samples) != 4 or any(
        len(row["prompt_ids"]) != 40 or len(row["response_ids"]) != 128
        for row in history_samples
    ):
        raise RuntimeError("frozen [4,168] specialization history batch is unavailable")

    def history_realized_scorer(
        tokenizer: Any, model: Any, config: Any, selected_samples: list[dict[str, Any]]
    ):
        if hasattr(model, "_orig_mod") and not getattr(
            model, "_forkcert_history_warmed", False
        ):
            historical, _ = _original_scorer(
                tokenizer, model, config, history_samples
            )
            del historical
            model._forkcert_history_warmed = True
        return _original_scorer(tokenizer, model, config, selected_samples)

    base.tensor_sha256 = native_batch_hash
    base.trainer_response_logps_with_grad = history_realized_scorer
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    expected_graphs = [
        "75f04d2a0756df28ece303756af05663c5aaf76a770c10c437000affbf9e3863",
        "ee4053cb35f6351f6303e6b9922ccf0fa2189246fc5bcbee31d4793241164e5b",
    ]
    try:
        base.main()
        payload = json.loads(out.read_text(encoding="utf-8"))
        arms = {row["arm"]: row for row in payload["arms"]}
        for name in ("B_candidate", "C_branch_repair"):
            graph = arms[name]["compile_audit"]
            if graph["graph_hashes"] != expected_graphs or graph["graph_nodes"] != [455, 457]:
                raise RuntimeError(f"{name}: specialization-history graph gate failed")
        payload["specialization_history_exact"] = True
        payload["specialization_history_input_shape"] = [4, 168]
        payload["target_compiled_graph_sequence"] = expected_graphs
    except Exception as error:
        if out.is_file():
            existing = json.loads(out.read_text(encoding="utf-8"))
        else:
            existing = {}
        payload = {
            **existing,
            "status": "INVALID",
            "failure_type": type(error).__name__,
            "failure_message": str(error),
        }
        raise
    finally:
        if "payload" in locals():
            payload["schema_version"] = "forkcert.qwen3-grpo-grad-branch-repair.v0.7"
            payload["hash_canonical_shape"] = [4, 128]
            payload["prior_invalid_versions_preserved"] = ["v0.5", "v0.6"]
            payload["contract"] = str(
                Path(__file__).with_name(
                    "QWEN3_GRPO_GRAD_BRANCH_REPAIR_CONTRACT_V0_7_2026-07-17.md"
                ).resolve()
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )


if __name__ == "__main__":
    main()
