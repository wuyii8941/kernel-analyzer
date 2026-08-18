#!/usr/bin/env python3
"""Run compact heldout eager/candidate full-step endpoint observations."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch._inductor.compile_fx import compile_fx
from transformers import AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/state_design.json"
MODEL = Path("/data1/tzh/models/Qwen/Qwen3-1.7B")
PROTOCOL = ROOT / "results/coverage/qwen_oracle_protocol.json"


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sample_indices(name: str, count: int, size: int) -> list[int]:
    if size <= count:
        return list(range(size))
    seed = int(hashlib.sha256(name.encode()).hexdigest()[:16], 16)
    step = (seed | 1) % size
    while step == 0 or math.gcd(step, size) != 1:
        step = (step + 2) % size
    values = []
    seen = set()
    current = seed % size
    while len(values) < count:
        if current not in seen:
            seen.add(current)
            values.append(current)
        current = (current + step) % size
        if len(seen) == size:
            break
    return sorted(values)


def snapshot(
    model: torch.nn.Module, indices: dict[str, torch.Tensor]
) -> dict[str, Any]:
    rows = {}
    for name, parameter in sorted(model.named_parameters()):
        gradient = parameter.grad
        if gradient is None:
            rows[name] = {"status": "NONE", "values": []}
            continue
        flat = gradient.detach().reshape(-1)
        selected = flat[indices[name]]
        rows[name] = {
            "status": "SAMPLED",
            "values": selected.float().cpu().tolist(),
            "finite": bool(torch.isfinite(selected).all()),
        }
    return rows


def write_progress(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    temporary.replace(path)


class LossStep(torch.nn.Module):
    def __init__(self, subject: torch.nn.Module) -> None:
        super().__init__()
        self.subject = subject

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.subject(
            input_ids=values, labels=values, use_cache=False, return_dict=False
        )[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    design = json.loads(DESIGN.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    all_states = [row for row in design["records"] if row["split"] == "heldout"]
    states = [
        row for index, row in enumerate(all_states)
        if index % args.shard_count == args.shard_index
    ]
    device = torch.device(args.device)
    torch.manual_seed(24000)
    torch.cuda.manual_seed_all(24000)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False

    payload: dict[str, Any]
    if args.output.exists():
        with gzip.open(args.output, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload["protocol_sha256"] != protocol["protocol_sha256"]:
            raise RuntimeError("existing output has a different protocol")
    else:
        payload = {
            "schema": "kernel-analyzer-qwen-heldout-endpoints-v1",
            "status": "RUNNING",
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "protocol_sha256": protocol["protocol_sha256"],
            "sample_size": args.sample_size,
            "parameter_coordinates": {},
            "states": {},
            "claim_boundary": (
                "Complete-step loss and fixed parameter-gradient coordinates; this is a training-risk "
                "endpoint screen, not yet every internal fused-region endpoint."
            ),
        }

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True
    ).to(device).train()
    model.config.use_cache = False
    indices = {
        name: sample_indices(name, args.sample_size, parameter.numel())
        for name, parameter in model.named_parameters()
    }
    if payload["parameter_coordinates"] and payload["parameter_coordinates"] != indices:
        raise RuntimeError("parameter coordinates changed")
    payload["parameter_coordinates"] = indices
    device_indices = {
        name: torch.tensor(values, dtype=torch.long, device=device)
        for name, values in indices.items()
    }

    def backend(graph_module: torch.fx.GraphModule, example_inputs: list[Any]):
        return compile_fx(graph_module, example_inputs, decompositions={})

    candidate_preserve = torch.compile(
        LossStep(model), backend=backend, fullgraph=True, dynamic=False
    )
    candidate_standard = torch.compile(
        LossStep(model), backend="inductor", fullgraph=True, dynamic=False
    )

    for state in states:
        state_id = state["sequence_id"]
        row = payload["states"].setdefault(state_id, {
            "record_sha256": state["record_sha256"],
            "stratum": state["length_bucket"],
            "bf16_eager": [],
            "bf16_inductor_standard": [],
            "bf16_inductor_preserve_aot_aten": [],
            "fp32_eager_strict": [],
        })
        row.setdefault("fp32_eager_strict", [])
        row.setdefault("bf16_inductor_standard", [])
        inputs = torch.tensor([state["input_ids"]], dtype=torch.long, device=device)
        for configuration, function in (
            ("bf16_eager", LossStep(model)),
            ("bf16_inductor_standard", candidate_standard),
            ("bf16_inductor_preserve_aot_aten", candidate_preserve),
        ):
            while len(row[configuration]) < 2:
                repeat = len(row[configuration])
                model.zero_grad(set_to_none=True)
                loss = function(inputs)
                loss.backward()
                torch.cuda.synchronize(device)
                observation = {
                    "repeat": repeat,
                    "loss": float(loss.detach()),
                    "parameter_gradients": snapshot(model, device_indices),
                }
                observation["observation_sha256"] = digest(observation)
                row[configuration].append(observation)
                print(json.dumps({
                    "event": "COMPLETE_REPEAT",
                    "shard": args.shard_index,
                    "state": state_id,
                    "configuration": configuration,
                    "repeat": repeat,
                }), flush=True)
            write_progress(args.output, payload)

    del candidate_preserve, candidate_standard, model
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    fp32_model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager", local_files_only=True
    ).to(device).train()
    fp32_model.config.use_cache = False
    fp32_indices = {
        name: sample_indices(name, args.sample_size, parameter.numel())
        for name, parameter in fp32_model.named_parameters()
    }
    if fp32_indices != indices:
        raise RuntimeError("BF16 and FP32 parameter coordinates differ")
    fp32_step = LossStep(fp32_model)
    for state in states:
        state_id = state["sequence_id"]
        row = payload["states"][state_id]
        while len(row["fp32_eager_strict"]) < 1:
            inputs = torch.tensor([state["input_ids"]], dtype=torch.long, device=device)
            fp32_model.zero_grad(set_to_none=True)
            loss = fp32_step(inputs)
            loss.backward()
            torch.cuda.synchronize(device)
            observation = {
                "repeat": 0,
                "loss": float(loss.detach()),
                "parameter_gradients": snapshot(fp32_model, device_indices),
            }
            observation["observation_sha256"] = digest(observation)
            row["fp32_eager_strict"].append(observation)
            write_progress(args.output, payload)
            print(json.dumps({
                "event": "COMPLETE_REPEAT",
                "shard": args.shard_index,
                "state": state_id,
                "configuration": "fp32_eager_strict",
                "repeat": 0,
            }), flush=True)

    payload["status"] = "COMPLETE_HELDOUT_ENDPOINT_SHARD"
    payload["completed_states"] = len(payload["states"])
    payload["result_sha256"] = digest({k: v for k, v in payload.items() if k != "result_sha256"})
    write_progress(args.output, payload)
    print(json.dumps({
        "event": "SHARD_COMPLETE",
        "shard": args.shard_index,
        "states": len(payload["states"]),
        "output": str(args.output),
        "result_sha256": payload["result_sha256"],
    }), flush=True)


if __name__ == "__main__":
    main()
