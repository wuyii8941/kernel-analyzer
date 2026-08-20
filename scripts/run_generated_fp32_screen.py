#!/usr/bin/env python3
"""Run every Triton boundary against the identical FP32-storage program."""

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
from transformers import AutoModelForCausalLM, MambaForCausalLM
from transformers.models.mamba import modeling_mamba

from qwen_candidate_step import LossStep, configure_candidate_runtime


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "results/coverage/generated_fp32_protocol.json"
sys.path.insert(0, str(ROOT / "archive/round1_code/src"))
sys.path.insert(0, str(ROOT))

from scripts.generated_fp32_observer import GeneratedFP32Observer  # noqa: E402
from scripts.inductor_buffer_origins import InductorBufferOriginRecorder  # noqa: E402


def tensor_digest(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    result = hashlib.sha256()
    result.update(str(tensor.dtype).encode())
    result.update(repr(tuple(tensor.shape)).encode())
    result.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def gradient_digest(model: torch.nn.Module) -> str:
    result = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        result.update(name.encode())
        result.update(b"NONE" if parameter.grad is None else tensor_digest(parameter.grad).encode())
    return result.hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(architecture: str, path: Path, device: torch.device) -> torch.nn.Module:
    if architecture == "mamba":
        modeling_mamba.selective_scan_fn = None
        modeling_mamba.mamba_inner_fn = None
        modeling_mamba.selective_state_update = None
        model = MambaForCausalLM.from_pretrained(path, dtype=torch.bfloat16, local_files_only=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True
        )
    model = model.to(device).train()
    model.config.use_cache = False
    return model


def write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    # These state-resumable artifacts can contain tens of thousands of sketches.
    # Level 3 keeps them compact without repeatedly spending minutes at level 9.
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=3) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    with gzip.open(temporary, "rt", encoding="utf-8") as handle:
        if json.load(handle)["campaign_sha256"] != payload["campaign_sha256"]:
            raise RuntimeError("screen post-write validation failed")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture",
        choices=("qwen", "mamba", "phi", "deepseek8", "generic"),
        required=True,
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--expected-states", type=int, default=32)
    parser.add_argument(
        "--state-role",
        choices=("ENGINEERING", "SCREENING", "CONFIRMATION"),
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
    records = bank.get("states", bank.get("records"))
    if len(records) != args.expected_states:
        raise RuntimeError("frozen input population changed")
    requested_indices = (
        {int(value) for value in args.state_indices.split(",") if value}
        if args.state_indices else None
    )
    if requested_indices is not None and (
        min(requested_indices, default=0) < 0
        or max(requested_indices, default=-1) >= len(records)
    ):
        raise ValueError("state index outside frozen input bank")
    eligible = [
        row for index, row in enumerate(records)
        if (requested_indices is None or index in requested_indices)
        and (args.state_role is None or row.get("role") == args.state_role)
    ]
    if args.state_role is not None and not eligible:
        raise RuntimeError(f"input bank has no {args.state_role} states")
    selected = [
        row for index, row in enumerate(eligible)
        if index % args.shard_count == args.shard_index
    ]
    with gzip.open(args.campaign, "rt", encoding="utf-8") as handle:
        campaign = json.load(handle)
    if campaign["status"] != "COMPLETE_ALL_TRITON_FP32_REPLAY_PLAN":
        raise RuntimeError("FP32 replay campaign is incomplete")
    if campaign["architecture"] != args.architecture:
        raise RuntimeError("campaign architecture mismatch")
    rows = campaign["rows"]
    device = torch.device(args.device)
    configure_candidate_runtime(24000)
    model = load_model(args.architecture, args.model, device)
    module_start = len(PyCodeCache.modules)
    candidate = torch.compile(
        LossStep(model), backend="inductor", fullgraph=not args.allow_graph_breaks, dynamic=False
    )
    # Every shard must instantiate the same frozen candidate artifact.  Using
    # the first state *inside the shard* can change Inductor's generated
    # schedule for value-sensitive model code (observed on DeepSeek), making a
    # shard silently target a different denominator from the state-0 capture.
    warm_state = eligible[0]
    warm_tokens = warm_state.get("token_ids", warm_state.get("input_ids"))
    warm = torch.tensor([warm_tokens], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    with InductorBufferOriginRecorder() as origin_recorder:
        warm_loss = candidate(warm)
        warm_loss.backward()
    torch.cuda.synchronize(device)
    buffer_origin_certificate = origin_recorder.certificate()
    modules = list(PyCodeCache.modules[module_start:])
    if len(modules) < 2:
        raise RuntimeError("candidate did not compile complete F+B modules")

    if args.output.exists():
        with gzip.open(args.output, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload["campaign_sha256"] != campaign["result_sha256"]:
            raise RuntimeError("existing shard binds another campaign")
        if payload.get("protocol_sha256") != protocol_sha256:
            raise RuntimeError("existing shard lacks the frozen FP32 protocol binding")
    else:
        payload = {
            "schema": "kernel-analyzer-generated-typed-fp32-screen-v2",
            "status": "RUNNING",
            "architecture": args.architecture,
            "model": str(args.model.resolve()),
            "input_bank_sha256": file_digest(args.input_bank),
            "campaign_sha256": campaign["result_sha256"],
            "protocol_sha256": protocol_sha256,
            "state_role": args.state_role,
            "state_indices": sorted(requested_indices) if requested_indices is not None else None,
            "campaign": str(args.campaign.resolve().relative_to(ROOT)),
            "inductor_buffer_origins": buffer_origin_certificate,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "repeat": args.repeat,
            "denominator": {
                "frozen_states": len(records),
                "eligible_states": len(eligible),
                "shard_states": len(selected),
                "triton_invocations_per_state": len(rows),
                "planned_records": len(selected) * args.repeat * len(rows),
            },
            "states": {},
            "claim_boundary": (
                "Precision-only comparison of each BF16 Triton invocation to an independently "
                "recompiled copy of the identical generated program with physical FP32 floating "
                "pointer ABIs. Eager semantic equivalence is separate."
            ),
        }
    if payload.get("inductor_buffer_origins", {}).get("result_sha256") != (
        buffer_origin_certificate["result_sha256"]
    ):
        raise RuntimeError("Inductor IR-buffer origins changed across resumed shard")

    for state_index, state in enumerate(selected):
        state_id = str(state.get("sequence_id", state.get("state_id", state_index)))
        if len(payload["states"].get(state_id, {}).get("repeats", [])) == args.repeat:
            continue
        token_ids = state.get("token_ids", state.get("input_ids"))
        values = torch.tensor([token_ids], dtype=torch.long, device=device)
        seed = 24000 + state_index * args.shard_count + args.shard_index
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True)
        baseline_loss = candidate(values)
        baseline_loss.backward()
        torch.cuda.synchronize(device)
        baseline = {"loss": tensor_digest(baseline_loss), "gradients": gradient_digest(model)}
        state_row = {"token_ids_sha256": hashlib.sha256(bytes(json.dumps(token_ids), "utf-8")).hexdigest(), "repeats": []}
        frozen_runtime_identity = None
        frozen_missing_regions = None
        for repeat in range(args.repeat):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            observer = GeneratedFP32Observer(
                modules=modules, campaign_rows=rows, sample_size=args.sample_size,
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
            if summary["status"] != "COMPLETE_ALL_TRITON_FP32_REPLAY":
                raise RuntimeError(
                    f"Triton replay census incomplete: {state_id}: "
                    f"{summary['denominator']}"
                )
            identity = [
                (
                    row["region_id"], row["symbol"],
                    row.get("runtime_invocation_ordinal"),
                    row.get("callsite_execution_ordinal"),
                    tuple(sorted(row["endpoint_metrics"])),
                )
                for row in summary["records"]
            ]
            missing = summary.get("missing_region_ids", [])
            if frozen_runtime_identity is None:
                frozen_runtime_identity, frozen_missing_regions = identity, missing
            elif identity != frozen_runtime_identity or missing != frozen_missing_regions:
                raise RuntimeError(f"Triton runtime identity changed across repeats: {state_id}")
            state_row["repeats"].append({"repeat": repeat, "summary": summary})
        state_row["runtime_denominator"] = {
            "actual_invocations_per_repeat": len(frozen_runtime_identity or []),
            "static_not_executed_per_repeat": len(frozen_missing_regions or []),
            "repeat_stable": True if args.repeat > 1 else "SMOKE_SINGLE_REPEAT_ONLY",
        }
        payload["states"][state_id] = state_row
        write(args.output, payload)
        print(json.dumps({
            "event": "STATE_COMPLETE", "state": state_id,
            "records": state_row["runtime_denominator"]["actual_invocations_per_repeat"],
        }, sort_keys=True), flush=True)
    payload["status"] = "COMPLETE_SHARD_ALL_TRITON_FP32_REPLAY"
    write(args.output, payload)
    print(json.dumps({"event": "SHARD_COMPLETE", "states": len(selected), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
