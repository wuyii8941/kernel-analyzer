#!/usr/bin/env python3
"""Run every generated external/direct compute call against FP32 storage."""

from __future__ import annotations

import argparse
import gzip
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
PROTOCOL = ROOT / "results/coverage/generated_fp32_protocol.json"
sys.path.insert(0, str(ROOT / "archive/round1_code/src"))
sys.path.insert(0, str(ROOT))

from qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.generated_nontriton_fp32_observer import (  # noqa: E402
    GeneratedNonTritonFP32Observer,
)
from scripts.run_generated_fp32_screen import (  # noqa: E402
    file_digest,
    gradient_digest,
    load_model,
    tensor_digest,
)


def write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=3) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    with gzip.open(temporary, "rt", encoding="utf-8") as handle:
        if json.load(handle)["inventory_sha256"] != payload["inventory_sha256"]:
            raise RuntimeError("screen post-write validation failed")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture",
        choices=("qwen", "mamba", "phi", "deepseek8", "generic", "gemma3"),
        required=True,
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--expected-states", type=int, default=32)
    parser.add_argument(
        "--state-role",
        choices=("ENGINEERING", "SCREENING", "CONFIRMATION"),
        help="Run only the frozen role while retaining the complete bank binding.",
    )
    parser.add_argument(
        "--state-indices",
        help="Comma-separated original bank indices; used for state-specific dynamic schedules.",
    )
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--metric-chunk-elements", type=int, default=1_048_576)
    parser.add_argument("--allow-graph-breaks", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    protocol_sha256 = protocol.get("protocol_sha256") or hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if not 0 <= args.shard_index < args.shard_count or args.repeat < 1:
        raise ValueError("invalid shard or repeat count")
    bank = json.loads(args.input_bank.read_text())
    states = bank.get("states", bank.get("records"))
    if len(states) != args.expected_states:
        raise RuntimeError("frozen input population changed")
    requested_indices = (
        {int(value) for value in args.state_indices.split(",") if value}
        if args.state_indices else None
    )
    if requested_indices is not None and (
        min(requested_indices, default=0) < 0
        or max(requested_indices, default=-1) >= len(states)
    ):
        raise ValueError("state index outside frozen input bank")
    eligible = [
        row for index, row in enumerate(states)
        if (requested_indices is None or index in requested_indices)
        and (args.state_role is None or row.get("role") == args.state_role)
    ]
    if args.state_role is not None and not eligible:
        raise RuntimeError(f"input bank has no {args.state_role} states")
    selected = [
        row for index, row in enumerate(eligible)
        if index % args.shard_count == args.shard_index
    ]
    with gzip.open(args.inventory, "rt", encoding="utf-8") as handle:
        inventory = json.load(handle)
    if inventory.get("status") not in {
        "COMPLETE_GENERATED_SCHEDULE_AND_POINTER_DATAFLOW",
        "COMPLETE_GENERATED_SCHEDULE_PARTIAL_POINTER_DATAFLOW",
    }:
        raise RuntimeError("generated runtime-call inventory is incomplete")
    rows = inventory["runtime_call_audit"]["rows"]
    kinds = {"EXTERN", "DIRECT_ATEN", "DIRECT_TORCH_OP", "DIRECT_TENSOR_METHOD"}
    expected = sum(
        row.get("category") == "COMPUTE"
        and row.get("implementation_kind_or_helper_role") in kinds
        for row in rows
    )
    if expected < 1:
        raise RuntimeError("inventory contains no non-Triton compute denominator")

    device = torch.device(args.device)
    configure_candidate_runtime(24000)
    model = load_model(args.architecture, args.model, device)
    module_start = len(PyCodeCache.modules)
    candidate = torch.compile(
        LossStep(model), backend="inductor", fullgraph=not args.allow_graph_breaks, dynamic=False
    )
    # Compile every shard from the same frozen state used to capture the exact
    # generated inventory.  A shard-local warm state can select a different
    # Inductor schedule even when tensor shapes are unchanged.
    warm_state = eligible[0]
    warm_tokens = warm_state.get("token_ids", warm_state.get("input_ids"))
    warm = torch.tensor([warm_tokens], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    candidate(warm).backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[module_start:])
    if len(modules) < 2:
        raise RuntimeError("candidate did not compile complete F+B modules")

    if args.output.exists():
        with gzip.open(args.output, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload["inventory_sha256"] != inventory["result_sha256"]:
            raise RuntimeError("existing shard binds another inventory")
        if payload.get("protocol_sha256") != protocol_sha256:
            raise RuntimeError("existing shard lacks the frozen FP32 protocol binding")
    else:
        payload = {
            "schema": "kernel-analyzer-generated-nontriton-fp32-screen-v1",
            "status": "RUNNING",
            "architecture": args.architecture,
            "model": str(args.model.resolve()),
            "input_bank_sha256": file_digest(args.input_bank),
            "inventory_sha256": inventory["result_sha256"],
            "protocol_sha256": protocol_sha256,
            "state_role": args.state_role,
            "state_indices": sorted(requested_indices) if requested_indices is not None else None,
            "inventory": str(args.inventory.resolve().relative_to(ROOT)),
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "repeat": args.repeat,
            "denominator": {
                "frozen_states": len(states), "eligible_states": len(eligible),
                "shard_states": len(selected),
                "static_generated_nontriton_calls": expected,
                "actual_invocations_per_measured_step": "FROZEN_BY_REPEAT_STABLE_RUNTIME_CENSUS",
                "static_upper_bound_records": len(selected) * args.repeat * expected,
            },
            "states": {},
            "claim_boundary": (
                "Precision-only comparison of every generated external/direct compute call "
                "to the same declared operation over FP32 floating storages."
            ),
        }

    for local_index, state in enumerate(selected):
        state_id = str(state.get("sequence_id", state.get("state_id", local_index)))
        if len(payload["states"].get(state_id, {}).get("repeats", [])) == args.repeat:
            continue
        tokens = state.get("token_ids", state.get("input_ids"))
        values = torch.tensor([tokens], dtype=torch.long, device=device)
        seed = 24000 + local_index * args.shard_count + args.shard_index
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True)
        baseline_loss = candidate(values)
        baseline_loss.backward()
        torch.cuda.synchronize(device)
        baseline = {"loss": tensor_digest(baseline_loss), "gradients": gradient_digest(model)}
        state_row = {
            "token_ids_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
            "repeats": [],
        }
        frozen_runtime_identity = None
        frozen_missing_rows = None
        for repeat in range(args.repeat):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            observer = GeneratedNonTritonFP32Observer(
                modules=modules, inventory_rows=rows, sample_size=args.sample_size,
                metric_chunk_elements=args.metric_chunk_elements,
            )
            model.zero_grad(set_to_none=True)
            with observer:
                loss = candidate(values)
                loss.backward()
            torch.cuda.synchronize(device)
            observed = {"loss": tensor_digest(loss), "gradients": gradient_digest(model)}
            summary = observer.summary()
            if observed != baseline:
                raise RuntimeError(f"observer perturbed full step: {state_id}")
            if summary["status"] != "COMPLETE_RUNTIME_NONTRITON_FP32_REPLAY_WITH_STATIC_DISPOSITION":
                raise RuntimeError(
                    f"non-Triton census incomplete: {state_id}: "
                    f"{json.dumps({key: summary[key] for key in ('denominator', 'missing_rows', 'unmatched_generated_copy_calls')}, sort_keys=True)}"
                )
            identity = summary["runtime_identity"]
            missing = summary["missing_rows"]
            if frozen_runtime_identity is None:
                frozen_runtime_identity = identity
                frozen_missing_rows = missing
            elif identity != frozen_runtime_identity or missing != frozen_missing_rows:
                raise RuntimeError(f"non-Triton runtime identity changed across repeats: {state_id}")
            state_row["repeats"].append({"repeat": repeat, "summary": summary})
        state_row["runtime_denominator"] = {
            "actual_invocations_per_repeat": len(frozen_runtime_identity or []),
            "static_not_executed_per_repeat": len(frozen_missing_rows or []),
            "repeat_stable": True if args.repeat > 1 else "SMOKE_SINGLE_REPEAT_ONLY",
        }
        payload["states"][state_id] = state_row
        write(args.output, payload)
        print(json.dumps({
            "event": "STATE_COMPLETE", "state": state_id,
            "actual_records": state_row["runtime_denominator"]["actual_invocations_per_repeat"],
            "static_not_executed": state_row["runtime_denominator"]["static_not_executed_per_repeat"],
        }), flush=True)
    payload["status"] = "COMPLETE_SHARD_ALL_NONTRITON_FP32_REPLAY"
    write(args.output, payload)
    print(json.dumps({"event": "SHARD_COMPLETE", "states": len(selected), "output": str(args.output)}))


if __name__ == "__main__":
    main()
