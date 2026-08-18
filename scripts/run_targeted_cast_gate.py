#!/usr/bin/env python3
"""Batch the declared-dtype cast gate for exact generated candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch._inductor.codecache import PyCodeCache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "archive/round1_code/src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_frozen_candidate_fp32_screen import wrapper_modules  # noqa: E402
from scripts.run_generated_fp32_screen import (  # noqa: E402
    file_digest, gradient_digest, load_model, tensor_digest,
)
from scripts.run_targeted_full_coordinate import validate_release  # noqa: E402
from scripts.targeted_external_intervention import TargetedExternalIntervention  # noqa: E402


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", action="append", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument(
        "--queue", type=Path,
        default=ROOT / "results/coverage/bias_candidate_queue.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if len(set(args.candidate_id)) != len(args.candidate_id):
        raise ValueError("candidate IDs must be unique")

    queue = json.loads(args.queue.read_text())
    by_id = {row["candidate_id"]: row for row in queue["candidates"]}
    missing = sorted(set(args.candidate_id) - set(by_id))
    if missing:
        raise RuntimeError(f"candidate IDs absent from queue: {missing}")
    selected = [by_id[candidate_id] for candidate_id in args.candidate_id]
    for row in selected:
        if not row["gates"].get("full_coordinate_t1"):
            raise RuntimeError(f"full-coordinate T1 not passed: {row['candidate_id']}")
        if row["exact_generated_call"]["implementation_kind"] != "EXTERN":
            raise RuntimeError("cast gate currently supports exact EXTERN calls")

    capture = json.loads((args.release_dir / "capture.json").read_text())
    bank = json.loads(args.input_bank.read_text())
    states = bank.get("states", bank.get("records"))
    if len(states) < args.states:
        raise RuntimeError("input bank is shorter than requested cast gate")
    if file_digest(args.input_bank) != capture["input"]["input_bank_sha256"]:
        raise RuntimeError("input bank does not match frozen release")

    device = torch.device(args.device)
    configure_candidate_runtime(24000)
    model = load_model("mamba", args.model, device)
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=True, dynamic=False)
    warm_tokens = states[0].get("token_ids", states[0].get("input_ids"))
    warm = torch.tensor([warm_tokens], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    candidate(warm).backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    validate_release(wrapper_modules(modules), capture)

    candidate_rows: dict[str, list[dict[str, Any]]] = {
        row["candidate_id"]: [] for row in selected
    }
    if args.output.exists():
        prior = json.loads(args.output.read_text())
        if prior.get("status") == "RUNNING" and prior.get("candidate_ids") == args.candidate_id:
            candidate_rows = {
                candidate_id: list(prior["rows"].get(candidate_id, []))
                for candidate_id in args.candidate_id
            }
    completed_states = set.intersection(*(
        {row["state_id"] for row in rows} for rows in candidate_rows.values()
    )) if candidate_rows else set()
    for state_index, state in enumerate(states[: args.states]):
        state_id = str(state.get("sequence_id", state.get("state_id", state_index)))
        if state_id in completed_states:
            continue
        tokens = state.get("token_ids", state.get("input_ids"))
        values = torch.tensor([tokens], dtype=torch.long, device=device)
        seed = 28000 + state_index
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True)
        baseline_loss = candidate(values)
        baseline_loss.backward()
        torch.cuda.synchronize(device)
        baseline = {
            "loss": tensor_digest(baseline_loss), "gradients": gradient_digest(model),
        }
        for selected_row in selected:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model.zero_grad(set_to_none=True)
            observer = TargetedExternalIntervention(
                modules=modules, target=selected_row["exact_generated_call"], mode="OBSERVE",
            )
            with observer:
                loss = candidate(values)
                loss.backward()
            torch.cuda.synchronize(device)
            identity = {"loss": tensor_digest(loss), "gradients": gradient_digest(model)}
            if identity != baseline:
                raise RuntimeError(
                    f"observer perturbed {selected_row['candidate_id']} state {state_id}"
                )
            summary = observer.summary()
            candidate_rows[selected_row["candidate_id"]].append({
                "state_id": state_id,
                "token_ids_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                "reference_cast_changed_coordinates": summary[
                    "reference_cast_changed_coordinates"
                ],
                "reference_cast_max_abs_change": summary["reference_cast_max_abs_change"],
                "full_coordinate_count": summary["full_coordinate_count"],
            })
        del values
        torch.cuda.empty_cache()
        write(args.output, {
            "schema": "kernel-analyzer-targeted-declared-dtype-cast-gate-v1",
            "status": "RUNNING",
            "candidate_ids": args.candidate_id,
            "states_complete": state_index + 1,
            "rows": candidate_rows,
        })
        print(json.dumps({"event": "STATE_COMPLETE", "state": state_id}), flush=True)

    results = []
    for selected_row in selected:
        rows = candidate_rows[selected_row["candidate_id"]]
        nonnull = sum(row["reference_cast_changed_coordinates"] > 0 for row in rows)
        results.append({
            "candidate_id": selected_row["candidate_id"],
            "states": rows,
            "nonnull_states": nonnull,
            "total_changed_coordinates": sum(
                row["reference_cast_changed_coordinates"] for row in rows
            ),
            "promote_to_causal_t2": nonnull > 0,
        })
    output = {
        "schema": "kernel-analyzer-targeted-declared-dtype-cast-gate-v1",
        "status": "COMPLETE_BATCH_CAST_GATE",
        "queue_sha256": queue["result_sha256"],
        "release_capture_sha256": capture["result_sha256"],
        "input_bank_sha256": file_digest(args.input_bank),
        "state_count": args.states,
        "candidate_count": len(results),
        "results": results,
        "claim_boundary": (
            "This gate only tests whether the FP32 same-operation reference changes the "
            "candidate at its declared output dtype. It does not establish causal propagation."
        ),
    }
    output["result_sha256"] = canonical_hash(output)
    write(args.output, output)
    print(json.dumps({
        "event": "CAST_GATE_COMPLETE",
        "promoted": [row["candidate_id"] for row in results if row["promote_to_causal_t2"]],
    }), flush=True)


if __name__ == "__main__":
    main()
