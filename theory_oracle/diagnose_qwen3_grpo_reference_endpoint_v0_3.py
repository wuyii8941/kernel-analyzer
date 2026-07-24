#!/usr/bin/env python
"""Diagnose A_reference endpoint realization; this script makes no effect claim."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

from forkcert.config import load_config
from forkcert.io import read_jsonl
from forkcert.logprob_runner import configure_determinism, load_hf_path
from scripts.phase6_twin_training import path_config
from theory_oracle.qwen3_grpo_branch_repair_oracle import select_batch
from theory_oracle.qwen3_grpo_branch_repair_oracle_v0_2 import (
    trainer_response_logps_with_grad,
)


def tensor_hash(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--states", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import torch

    configure_determinism(20260720)
    evaluation = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
    event = evaluation["first_event_for_one_step_followup"]
    samples, states, target = select_batch(
        read_jsonl(args.samples), read_jsonl(args.states), event
    )
    cfg = path_config(load_config(args.config), "path_ref")
    tokenizer, model = load_hf_path(replace(cfg, compile_model=False))

    rows: list[dict[str, Any]] = []
    tensors: dict[str, Any] = {}
    for checkpointing in (True, False):
        if checkpointing:
            model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_disable()
        for grad_enabled in (False, True):
            mode = f"checkpointing_{str(checkpointing).lower()}__grad_{str(grad_enabled).lower()}"
            repeats = []
            for repeat in range(2):
                context = nullcontext() if grad_enabled else torch.no_grad()
                with context:
                    logps, token_ids = trainer_response_logps_with_grad(
                        tokenizer, model, cfg, samples
                    )
                detached = logps.detach().cpu()
                key = f"{mode}__repeat_{repeat}"
                tensors[key] = detached
                repeats.append(
                    {
                        "repeat": repeat,
                        "target_logp": float(detached[target]),
                        "tensor_sha256": tensor_hash(detached),
                        "dtype": str(detached.dtype),
                        "shape": list(detached.shape),
                        "token_alignment": token_ids
                        == [int(row["token_id"]) for row in states],
                    }
                )
                del logps
            rows.append(
                {
                    "mode": mode,
                    "accelerate_output_fp32_wrapper": False,
                    "gradient_checkpointing": checkpointing,
                    "grad_enabled": grad_enabled,
                    "repeats": repeats,
                    "self_exact": repeats[0]["tensor_sha256"]
                    == repeats[1]["tensor_sha256"],
                    "target_matches_parent_exactly": all(
                        row["target_logp"] == float(event["logp_ref"])
                        for row in repeats
                    ),
                    "target_minus_parent": [
                        row["target_logp"] - float(event["logp_ref"])
                        for row in repeats
                    ],
                }
            )

    # Accelerate's FP16 preparation converts the model output structure to
    # FP32 before TRL slices logits and calls selective_log_softmax.  The raw
    # reloaded model above has no such wrapper, so test that execution context
    # explicitly rather than treating it as an approximate implementation.
    from accelerate.utils.operations import convert_outputs_to_fp32

    model.forward = convert_outputs_to_fp32(model.forward)
    model.gradient_checkpointing_enable()
    for grad_enabled in (False, True):
        mode = f"accelerate_output_fp32__grad_{str(grad_enabled).lower()}"
        repeats = []
        for repeat in range(2):
            context = nullcontext() if grad_enabled else torch.no_grad()
            with context:
                logps, token_ids = trainer_response_logps_with_grad(
                    tokenizer, model, cfg, samples
                )
            detached = logps.detach().cpu()
            key = f"{mode}__repeat_{repeat}"
            tensors[key] = detached
            repeats.append(
                {
                    "repeat": repeat,
                    "target_logp": float(detached[target]),
                    "tensor_sha256": tensor_hash(detached),
                    "dtype": str(detached.dtype),
                    "shape": list(detached.shape),
                    "token_alignment": token_ids
                    == [int(row["token_id"]) for row in states],
                }
            )
            del logps
        rows.append(
            {
                "mode": mode,
                "accelerate_output_fp32_wrapper": True,
                "gradient_checkpointing": True,
                "grad_enabled": grad_enabled,
                "repeats": repeats,
                "self_exact": repeats[0]["tensor_sha256"]
                == repeats[1]["tensor_sha256"],
                "target_matches_parent_exactly": all(
                    row["target_logp"] == float(event["logp_ref"])
                    for row in repeats
                ),
                "target_minus_parent": [
                    row["target_logp"] - float(event["logp_ref"])
                    for row in repeats
                ],
            }
        )

    first_keys = [f"{row['mode']}__repeat_0" for row in rows]
    comparisons = []
    for left_index, left in enumerate(first_keys):
        for right in first_keys[left_index + 1 :]:
            delta = (tensors[left].float() - tensors[right].float()).abs()
            comparisons.append(
                {
                    "left": left.rsplit("__repeat_", 1)[0],
                    "right": right.rsplit("__repeat_", 1)[0],
                    "max_abs": float(delta.max()),
                    "nonzero_count": int(torch.count_nonzero(delta)),
                }
            )

    payload = {
        "schema_version": "forkcert.qwen3-grpo-reference-endpoint-diagnostic.v0.3",
        "claim_scope": "diagnostic only; no candidate, repair, attribution, correctness or population claim",
        "parent_target_logp": float(event["logp_ref"]),
        "target_flat_index": target,
        "modes": rows,
        "cross_mode_comparisons": comparisons,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
