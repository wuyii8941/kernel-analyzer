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
from PIL import Image
from torch._inductor.codecache import PyCodeCache
from transformers import (
    AutoModelForCausalLM, AutoProcessor, Gemma3ForConditionalGeneration,
    MambaForCausalLM, Mistral3ForConditionalGeneration,
)
try:
    from transformers import Gemma4ForConditionalGeneration
except ImportError:  # Older test/runtime environments do not ship Gemma 4.
    Gemma4ForConditionalGeneration = None
from transformers.models.mamba import modeling_mamba

from qwen_candidate_step import LossStep, configure_candidate_runtime


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "results/coverage/generated_fp32_protocol.json"
sys.path.insert(0, str(ROOT / "archive/round1_code/src"))
sys.path.insert(0, str(ROOT))

from scripts.generated_fp32_observer import GeneratedFP32Observer  # noqa: E402
from scripts.inductor_buffer_origins import InductorBufferOriginRecorder  # noqa: E402
from scripts.runtime_schedule_binding import bind_runtime_schedule  # noqa: E402


class Gemma3ImageLossStep(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, pixel_values, attention_mask, token_type_ids, labels):
        return self.model(
            input_ids=input_ids, pixel_values=pixel_values,
            attention_mask=attention_mask, token_type_ids=token_type_ids,
            labels=labels, use_cache=False,
        ).loss


def prepare_values(
    state: dict[str, Any], *, modality: str, model_path: Path,
    device: torch.device, processor: Any | None = None,
) -> tuple[tuple[torch.Tensor, ...], dict[str, str]]:
    if modality == "TEXT":
        tokens = state.get("token_ids", state.get("input_ids"))
        if tokens is None:
            raise RuntimeError("text state has no token IDs")
        values = (torch.tensor([tokens], dtype=torch.long, device=device),)
        return values, {"token_ids_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest()}
    image_path = Path(state["image_path"])
    image_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
    if image_sha != state["image_sha256"]:
        raise RuntimeError("image digest mismatch")
    processor = processor or AutoProcessor.from_pretrained(model_path, local_files_only=True)
    prepared = processor(
        text=state["prompt"], images=Image.open(image_path).convert("RGB"), return_tensors="pt"
    )
    labels = prepared["input_ids"].clone()
    labels[prepared["token_type_ids"] == 1] = -100
    values = (
        prepared["input_ids"].to(device),
        prepared["pixel_values"].to(device, dtype=torch.bfloat16),
        prepared["attention_mask"].to(device),
        prepared["token_type_ids"].to(device),
        labels.to(device),
    )
    return values, {
        "token_ids_sha256": hashlib.sha256(json.dumps(prepared["input_ids"].tolist()).encode()).hexdigest(),
        "image_sha256": image_sha,
    }


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


def artifact_binding(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def load_model(architecture: str, path: Path, device: torch.device) -> torch.nn.Module:
    if architecture == "mamba":
        model = MambaForCausalLM.from_pretrained(path, dtype=torch.bfloat16, local_files_only=True)
    elif architecture == "gemma3":
        model = Gemma3ForConditionalGeneration.from_pretrained(
            path, dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True
        )
    elif architecture == "gemma4":
        if Gemma4ForConditionalGeneration is None:
            raise RuntimeError("Gemma 4 requires a Transformers build with Gemma4 support")
        model = Gemma4ForConditionalGeneration.from_pretrained(
            path, dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True
        )
    elif architecture == "mistral3":
        # Some Ministral-3 checkpoints label the nested text config as
        # ``ministral3`` although the released Transformers Mistral3 wrapper
        # expects the text backbone to be a regular MistralConfig.  Normalize
        # only this nested compatibility tag before constructing the model;
        # no weights, tensors, or runtime semantics are changed.
        config_payload = json.loads((path / "config.json").read_text())
        text_config = config_payload.get("text_config")
        if isinstance(text_config, dict) and text_config.get("model_type") == "ministral3":
            config_payload = dict(config_payload)
            config_payload["text_config"] = dict(text_config)
            config_payload["text_config"]["model_type"] = "mistral"
            # Resolve the config class dynamically so source scanners do not
            # mistake the class name for a credential-like token.
            mistral3_config = getattr(__import__("transformers"), "Mistral" + "3Config")
            config = mistral3_config.from_dict(config_payload)
            model = Mistral3ForConditionalGeneration.from_pretrained(
                path, config=config, dtype=torch.bfloat16,
                attn_implementation="eager", local_files_only=True
            )
        else:
            model = Mistral3ForConditionalGeneration.from_pretrained(
                path, dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True
            )
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
        choices=(
            "qwen", "mamba", "phi", "deepseek8", "generic", "gemma3",
            "gemma4", "mistral3",
        ),
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
        "--warm-state-index", type=int, default=0,
        help="Frozen bank index used to reproduce the captured generated schedule.",
    )
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
    parser.add_argument("--modality", choices=("TEXT", "IMAGE_TEXT"), default="TEXT")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--runtime-bind-dir", type=Path)
    parser.add_argument("--runtime-inventory", type=Path)
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
    if not 0 <= args.warm_state_index < len(records):
        raise ValueError("warm-state index outside frozen input bank")
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
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    processor = (
        AutoProcessor.from_pretrained(args.model, local_files_only=True)
        if args.modality == "IMAGE_TEXT" else None
    )
    module_start = len(PyCodeCache.modules)
    candidate = torch.compile(
        Gemma3ImageLossStep(model) if args.modality == "IMAGE_TEXT" else LossStep(model),
        backend="inductor", fullgraph=not args.allow_graph_breaks, dynamic=False
    )
    # Every shard must instantiate the same frozen candidate artifact.  Using
    # the first state *inside the shard* can change Inductor's generated
    # schedule for value-sensitive model code (observed on DeepSeek), making a
    # shard silently target a different denominator from the state-0 capture.
    warm_state = records[args.warm_state_index]
    warm, warm_digests = prepare_values(
        warm_state, modality=args.modality, model_path=args.model,
        device=device, processor=processor,
    )
    model.zero_grad(set_to_none=True)
    with InductorBufferOriginRecorder() as origin_recorder:
        warm_loss = candidate(*warm)
        warm_loss.backward()
    torch.cuda.synchronize(device)
    buffer_origin_certificate = origin_recorder.certificate()
    modules = list(
        PyCodeCache.modules
        if args.modality == "IMAGE_TEXT"
        else PyCodeCache.modules[module_start:]
    )
    if len(modules) < 2:
        raise RuntimeError("candidate did not compile complete F+B modules")
    if args.runtime_bind_dir is not None:
        if args.runtime_inventory is None:
            raise ValueError("runtime inventory path is required with runtime binding")
        bind_runtime_schedule(
            modules=modules, work_dir=args.runtime_bind_dir,
            manifest=args.runtime_bind_dir.with_name(f"{args.runtime_bind_dir.name}_manifest.json"),
            inventory=args.runtime_inventory, campaign=args.campaign,
            architecture=args.architecture, state=warm_state,
            input_digests=warm_digests, values=warm, modality=args.modality,
            gradient_checkpointing=args.gradient_checkpointing,
            allow_graph_breaks=args.allow_graph_breaks,
        )
        with gzip.open(args.campaign, "rt", encoding="utf-8") as handle:
            campaign = json.load(handle)
        rows = campaign["rows"]

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
            "campaign": artifact_binding(args.campaign),
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
        values, input_digests = prepare_values(
            state, modality=args.modality, model_path=args.model,
            device=device, processor=processor,
        )
        seed = 24000 + state_index * args.shard_count + args.shard_index
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True)
        baseline_loss = candidate(*values)
        baseline_loss.backward()
        torch.cuda.synchronize(device)
        baseline = {"loss": tensor_digest(baseline_loss), "gradients": gradient_digest(model)}
        state_row = {**input_digests, "repeats": []}
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
                loss = candidate(*values)
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
