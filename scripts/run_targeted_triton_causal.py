#!/usr/bin/env python3
"""Test whether one exact Triton endpoint causes a real backward carrier."""

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
from scripts.run_targeted_causal_repair import (  # noqa: E402
    clone_gradients, gradient_delta_summary,
)
from scripts.run_targeted_full_coordinate import validate_release  # noqa: E402
from scripts.targeted_triton_intervention import TargetedTritonIntervention  # noqa: E402


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
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument(
        "--queue", type=Path,
        default=ROOT / "results/coverage/triton_scalar_candidate_queue.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.repeat != 2:
        raise ValueError("causal Triton pilot requires exactly two repeats")
    queue = json.loads(args.queue.read_text())
    matches = [row for row in queue["candidates"] if row["candidate_id"] == args.candidate_id]
    if len(matches) != 1:
        raise RuntimeError("candidate ID is absent or non-unique")
    selected = matches[0]
    target = selected["exact_generated_call"]
    capture = json.loads((args.release_dir / "capture.json").read_text())
    bank = json.loads(args.input_bank.read_text())
    states = bank.get("states", bank.get("records"))
    if file_digest(args.input_bank) != capture["input"]["input_bank_sha256"]:
        raise RuntimeError("input bank does not match frozen release")

    device = torch.device(args.device)
    configure_candidate_runtime(24000)
    model = load_model(selected["architecture"], args.model, device)
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=True, dynamic=False)
    warm_tokens = states[0].get("token_ids", states[0].get("input_ids"))
    warm = torch.tensor([warm_tokens], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    candidate(warm).backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    validate_release(wrapper_modules(modules), capture)

    rows = []
    for state_index, state in enumerate(states[: args.states]):
        state_id = str(state.get("sequence_id", state.get("state_id", state_index)))
        tokens = state.get("token_ids", state.get("input_ids"))
        values = torch.tensor([tokens], dtype=torch.long, device=device)
        seed = 30000 + state_index
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True)
        baseline_loss = candidate(values)
        baseline_loss.backward()
        torch.cuda.synchronize(device)
        baseline_identity = {
            "loss": tensor_digest(baseline_loss), "gradients": gradient_digest(model),
        }
        baseline_gradients = clone_gradients(model)

        repeats = []
        frozen = None
        for repeat in range(args.repeat):
            mode_rows = {}
            for mode in ("SHAM", "REPAIR"):
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                model.zero_grad(set_to_none=True)
                observer = TargetedTritonIntervention(
                    modules=modules, target=target, mode=mode,
                )
                with observer:
                    loss = candidate(values)
                    loss.backward()
                torch.cuda.synchronize(device)
                identity = {"loss": tensor_digest(loss), "gradients": gradient_digest(model)}
                target_summary = observer.summary()
                delta = gradient_delta_summary(model, baseline_gradients)
                mode_rows[mode] = {
                    "identity": identity, "target": target_summary,
                    "gradient_delta": delta,
                }
            if mode_rows["SHAM"]["identity"] != baseline_identity:
                raise RuntimeError(f"Triton restoration sham perturbed state {state_id}")
            if mode_rows["SHAM"]["target"]["signed_error_sha256"] != mode_rows["REPAIR"]["target"]["signed_error_sha256"]:
                raise RuntimeError(f"target error changed between sham and repair: {state_id}")
            repeat_summary = {
                "target": mode_rows["REPAIR"]["target"],
                "repair_identity": mode_rows["REPAIR"]["identity"],
                "repair_gradient_delta": mode_rows["REPAIR"]["gradient_delta"],
            }
            if frozen is None:
                frozen = repeat_summary
            elif frozen != repeat_summary:
                raise RuntimeError(f"Triton causal result changed across repeats: {state_id}")
            repeats.append({"repeat": repeat, **repeat_summary})
        rows.append({
            "state_id": state_id,
            "baseline_identity": baseline_identity,
            "repeats": repeats,
            "repair_changes_endpoint": bool(
                frozen["target"]["reference_cast_changed_coordinates"] > 0
            ),
            "repair_changes_loss": frozen["repair_identity"]["loss"] != baseline_identity["loss"],
            "repair_reaches_parameter_gradients": bool(
                frozen["repair_gradient_delta"]["changed_parameter_count"] > 0
            ),
        })
        del baseline_gradients, values
        torch.cuda.empty_cache()
        write(args.output, {
            "schema": "kernel-analyzer-targeted-triton-causal-v1",
            "status": "RUNNING", "candidate_id": args.candidate_id,
            "states_complete": len(rows), "rows": rows,
        })
        print(json.dumps({
            "event": "STATE_COMPLETE", "state": state_id,
            "loss_changed": rows[-1]["repair_changes_loss"],
            "gradient_changed": rows[-1]["repair_reaches_parameter_gradients"],
        }), flush=True)

    endpoint_states = sum(row["repair_changes_endpoint"] for row in rows)
    gradient_states = sum(row["repair_reaches_parameter_gradients"] for row in rows)
    output = {
        "schema": "kernel-analyzer-targeted-triton-causal-v1",
        "status": "COMPLETE_TRITON_CAUSAL_PILOT",
        "candidate_id": args.candidate_id,
        "queue_sha256": queue["result_sha256"],
        "release_capture_sha256": capture["result_sha256"],
        "states": args.states,
        "repeats": args.repeat,
        "restoration_sham_exact": True,
        "endpoint_nonnull_states": endpoint_states,
        "parameter_gradient_changed_states": gradient_states,
        "complete_fb_candidate": endpoint_states == args.states and gradient_states == args.states,
        "rows": rows,
        "claim_boundary": (
            "A repaired loss scalar with unchanged parameter gradients is a forward-only "
            "numerical difference, not a complete training-bias F+B case."
        ),
    }
    output["result_sha256"] = canonical_hash(output)
    write(args.output, output)
    print(json.dumps({
        "event": "TRITON_CAUSAL_COMPLETE",
        "endpoint_states": endpoint_states, "gradient_states": gradient_states,
    }), flush=True)


if __name__ == "__main__":
    main()
