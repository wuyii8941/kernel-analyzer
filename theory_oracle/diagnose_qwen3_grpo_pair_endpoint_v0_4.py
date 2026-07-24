#!/usr/bin/env python
"""Preflight eager/compiled endpoint identity; no update or effect claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

from accelerate.utils.operations import convert_outputs_to_fp32

from forkcert.config import load_config
from forkcert.io import read_jsonl
from forkcert.logprob_runner import configure_determinism, load_hf_path
from scripts.phase6_twin_training import path_config
from theory_oracle.qwen3_grpo_branch_repair_oracle import Audit, select_batch, tracking_backend
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


def measure(
    *, tokenizer: Any, model: Any, cfg: Any, samples: list[dict[str, Any]],
    states: list[dict[str, Any]], target: int, grad_enabled: bool,
) -> dict[str, Any]:
    import torch

    context = nullcontext() if grad_enabled else torch.no_grad()
    with context:
        logps, token_ids = trainer_response_logps_with_grad(tokenizer, model, cfg, samples)
    value = logps.detach().cpu()
    result = {
        "grad_enabled": grad_enabled,
        "target_logp": float(value[target]),
        "tensor_sha256": tensor_hash(value),
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "token_alignment": token_ids == [int(row["token_id"]) for row in states],
        "_values": value.tolist(),
    }
    del logps
    return result


def clipping_summary(
    reference: dict[str, Any], candidate: dict[str, Any], states: list[dict[str, Any]]
) -> dict[str, Any]:
    lower, upper = math.log(0.8), math.log(1.2)
    events = []
    applicable = 0
    for index, (ref, alt, state) in enumerate(
        zip(reference["_values"], candidate["_values"], states, strict=True)
    ):
        advantage = float(state["advantage"])
        if advantage == 0.0:
            continue
        applicable += 1
        old = float(state["old_logp"])
        ref_delta, alt_delta = float(ref) - old, float(alt) - old
        if advantage > 0.0:
            ref_clip, alt_clip = ref_delta > upper, alt_delta > upper
        else:
            ref_clip, alt_clip = ref_delta < lower, alt_delta < lower
        if ref_clip != alt_clip:
            events.append(
                {
                    "flat_index": index,
                    "case_id": state["case_id"],
                    "token_index": int(state["token_index"]),
                    "token_id": int(state["token_id"]),
                    "advantage_sign": 1 if advantage > 0.0 else -1,
                    "reference_logp": float(ref),
                    "candidate_logp": float(alt),
                    "old_logp": old,
                    "reference_clip": ref_clip,
                    "candidate_clip": alt_clip,
                    "direction": f"{int(ref_clip)}->{int(alt_clip)}",
                }
            )
    return {
        "applicable_tokens": applicable,
        "disagreement_count": len(events),
        "direction_0_to_1_count": sum(e["direction"] == "0->1" for e in events),
        "direction_1_to_0_count": sum(e["direction"] == "1->0" for e in events),
        "selection_rule": "earliest flat_index disagreement, frozen before this full-state scan",
        "selected_event": events[0] if events else None,
        "events": events,
    }


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
    event = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))[
        "first_event_for_one_step_followup"
    ]
    samples, states, target = select_batch(
        read_jsonl(args.samples), read_jsonl(args.states), event
    )
    root_cfg = load_config(args.config)
    cfg = path_config(root_cfg, "path_ref")
    tokenizer, model = load_hf_path(replace(cfg, compile_model=False))
    model.forward = convert_outputs_to_fp32(model.forward)

    eager = []
    for grad_enabled in (False, True):
        eager.append(
            measure(
                tokenizer=tokenizer, model=model, cfg=cfg, samples=samples,
                states=states, target=target, grad_enabled=grad_enabled,
            )
        )

    audit = Audit()
    compiled = torch.compile(model, backend=tracking_backend(audit))
    compiled_rows: list[dict[str, Any]] = []
    for grad_enabled in (False, True):
        # Discard one warm-up for each autograd specialization, then retain two
        # exact-repeat measurements.
        measure(
            tokenizer=tokenizer, model=compiled, cfg=cfg, samples=samples,
            states=states, target=target, grad_enabled=grad_enabled,
        )
        repeats = [
            measure(
                tokenizer=tokenizer, model=compiled, cfg=cfg, samples=samples,
                states=states, target=target, grad_enabled=grad_enabled,
            )
            for _ in range(2)
        ]
        compiled_rows.append(
            {
                "grad_enabled": grad_enabled,
                "repeats": repeats,
                "self_exact": repeats[0]["tensor_sha256"] == repeats[1]["tensor_sha256"],
                "matches_parent_compiled_target_exactly": all(
                    row["target_logp"] == float(event["logp_alt"]) for row in repeats
                ),
            }
        )

    measurement_context = clipping_summary(
        eager[0], compiled_rows[0]["repeats"][0], states
    )
    transition_context = clipping_summary(
        eager[1], compiled_rows[1]["repeats"][0], states
    )
    for row in eager:
        row.pop("_values")
    for context_row in compiled_rows:
        for row in context_row["repeats"]:
            row.pop("_values")

    payload = {
        "schema_version": "forkcert.qwen3-grpo-pair-endpoint-diagnostic.v0.4",
        "claim_scope": "preflight diagnostic only; no update, repair, attribution, correctness or population claim",
        "target_flat_index": target,
        "parent_reference_target_logp": float(event["logp_ref"]),
        "parent_compiled_target_logp": float(event["logp_alt"]),
        "eager": eager,
        "eager_matches_parent_reference_exactly": all(
            row["target_logp"] == float(event["logp_ref"]) for row in eager
        ),
        "compiled": compiled_rows,
        "compiled_matches_parent_in_all_contexts": all(
            row["matches_parent_compiled_target_exactly"] for row in compiled_rows
        ),
        "clipping_semantics": {
            "measurement_no_grad": measurement_context,
            "transition_grad_enabled": transition_context,
        },
        "compile_audit": {
            "backend_compiles": audit.compiles,
            "runtime_invocations": audit.invocations,
            "graph_hashes": audit.hashes,
            "graph_nodes": audit.nodes,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
