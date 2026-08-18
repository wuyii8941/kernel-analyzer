#!/usr/bin/env python3
"""Run a restoration-sham controlled FP32 repair at one generated callsite."""

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


def clone_gradients(model: torch.nn.Module) -> dict[str, torch.Tensor | None]:
    return {
        name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
        for name, parameter in sorted(model.named_parameters())
    }


def gradient_delta_summary(
    model: torch.nn.Module, baseline: dict[str, torch.Tensor | None],
) -> dict[str, Any]:
    rows = []
    squared_l2 = 0.0
    changed_elements = 0
    for name, parameter in sorted(model.named_parameters()):
        left = baseline[name]
        right = None if parameter.grad is None else parameter.grad.detach().cpu()
        if left is None or right is None:
            if left is not None or right is not None:
                rows.append({"parameter": name, "status": "PRESENCE_CHANGED"})
            continue
        if torch.equal(left, right):
            continue
        delta = right.float() - left.float()
        count = int(torch.count_nonzero(delta))
        norm2 = float(torch.sum(delta.double().square()))
        squared_l2 += norm2
        changed_elements += count
        rows.append({
            "parameter": name,
            "parameter_numel": int(parameter.numel()),
            "parameter_shape": list(parameter.shape),
            "parameter_dtype": str(parameter.dtype),
            "changed_elements": count,
            "l2": norm2 ** 0.5,
            "max_abs": float(delta.abs().max()),
            "signed_sum": float(delta.double().sum()),
        })
    return {
        "changed_parameter_count": len(rows),
        "changed_elements": changed_elements,
        "global_l2": squared_l2 ** 0.5,
        "parameters": rows,
    }


def run_step(
    *, model: torch.nn.Module, candidate: Any, values: torch.Tensor,
    modules: list[Any], target: dict[str, Any], mode: str, seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model.zero_grad(set_to_none=True)
    observer = TargetedExternalIntervention(modules=modules, target=target, mode=mode)
    with observer:
        loss = candidate(values)
        loss.backward()
    torch.cuda.synchronize(values.device)
    return loss, observer.summary()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument(
        "--queue", type=Path,
        default=ROOT / "results/coverage/bias_candidate_queue.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.states < 1 or args.repeat != 2:
        raise ValueError("causal pilot requires at least one state and exactly two repeats")

    queue = json.loads(args.queue.read_text())
    matches = [row for row in queue["candidates"] if row["candidate_id"] == args.candidate_id]
    if len(matches) != 1:
        raise RuntimeError("candidate ID is absent or non-unique")
    selected = matches[0]
    if not selected["gates"].get("full_coordinate_t1"):
        raise RuntimeError("causal repair requires a passed full-coordinate T1 gate")
    target = selected["exact_generated_call"]
    capture = json.loads((args.release_dir / "capture.json").read_text())
    bank = json.loads(args.input_bank.read_text())
    states = bank.get("states", bank.get("records"))
    if len(states) < args.states:
        raise RuntimeError("input bank is shorter than requested pilot")
    if file_digest(args.input_bank) != capture["input"]["input_bank_sha256"]:
        raise RuntimeError("input bank does not match the frozen runtime release")

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

    rows = []
    for state_index, state in enumerate(states[: args.states]):
        state_id = str(state.get("sequence_id", state.get("state_id", state_index)))
        tokens = state.get("token_ids", state.get("input_ids"))
        values = torch.tensor([tokens], dtype=torch.long, device=device)
        seed = 26000 + state_index

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
        frozen_repair_delta = None
        frozen_target_error = None
        for repeat in range(args.repeat):
            sham_loss, sham_target = run_step(
                model=model, candidate=candidate, values=values, modules=modules,
                target=target, mode="SHAM", seed=seed,
            )
            sham_identity = {
                "loss": tensor_digest(sham_loss), "gradients": gradient_digest(model),
            }
            if sham_identity != baseline_identity:
                raise RuntimeError(f"restoration sham perturbed state {state_id}")

            repair_loss, repair_target = run_step(
                model=model, candidate=candidate, values=values, modules=modules,
                target=target, mode="REPAIR", seed=seed,
            )
            repair_identity = {
                "loss": tensor_digest(repair_loss), "gradients": gradient_digest(model),
            }
            if repair_identity["loss"] != baseline_identity["loss"]:
                raise RuntimeError(f"backward-only repair changed loss in state {state_id}")
            delta = gradient_delta_summary(model, baseline_gradients)
            if not repair_target["delivered_matches_reference_cast"]:
                raise RuntimeError(f"repair was not delivered in state {state_id}")
            target_error = repair_target["signed_error"]
            if sham_target["signed_error"] != target_error:
                raise RuntimeError(f"target inputs changed between sham and repair: {state_id}")
            if frozen_target_error is None:
                frozen_target_error = target_error
                frozen_repair_delta = delta
            elif target_error != frozen_target_error or delta != frozen_repair_delta:
                raise RuntimeError(f"causal result changed across repeats: {state_id}")
            repeats.append({
                "repeat": repeat,
                "sham_identity": sham_identity,
                "repair_identity": repair_identity,
                "target": repair_target,
                "gradient_delta": delta,
            })

        rows.append({
            "state_id": state_id,
            "token_ids_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
            "baseline_identity": baseline_identity,
            "repeats": repeats,
            "repair_reaches_parameter_gradients": bool(
                frozen_repair_delta and frozen_repair_delta["changed_parameter_count"] > 0
            ),
            "repair_changes_declared_dtype_output": bool(
                repeats[0]["target"]["reference_cast_changed_coordinates"] > 0
            ),
        })
        del baseline_gradients, values
        torch.cuda.empty_cache()
        write(args.output, {
            "schema": "kernel-analyzer-targeted-causal-repair-v1",
            "status": "RUNNING", "candidate_id": args.candidate_id,
            "states_complete": len(rows), "rows": rows,
        })
        print(json.dumps({
            "event": "STATE_COMPLETE", "state": state_id,
            "changed_parameters": frozen_repair_delta["changed_parameter_count"],
        }), flush=True)

    reached = sum(row["repair_reaches_parameter_gradients"] for row in rows)
    nonnull = sum(row["repair_changes_declared_dtype_output"] for row in rows)
    output = {
        "schema": "kernel-analyzer-targeted-causal-repair-v1",
        "status": "COMPLETE_CAUSAL_REPAIR_PILOT",
        "candidate_id": args.candidate_id,
        "queue_sha256": queue["result_sha256"],
        "release_capture_sha256": capture["result_sha256"],
        "input_bank_sha256": file_digest(args.input_bank),
        "states": args.states,
        "repeats": args.repeat,
        "restoration_sham_exact": True,
        "repair_reached_parameter_gradients_in_states": reached,
        "repair_nonnull_at_declared_dtype_in_states": nonnull,
        "causal_t2_positive": reached == args.states and nonnull == args.states,
        "rows": rows,
        "claim_boundary": (
            "This intervention establishes or rejects causality for bar_C only. A complete "
            "forward+backward case additionally requires the forward y and bar_S VJP edge."
        ),
    }
    output["result_sha256"] = canonical_hash(output)
    write(args.output, output)
    print(json.dumps({
        "event": "CAUSAL_PILOT_COMPLETE", "states": args.states,
        "states_reaching_parameter_gradients": reached,
    }), flush=True)


if __name__ == "__main__":
    main()
