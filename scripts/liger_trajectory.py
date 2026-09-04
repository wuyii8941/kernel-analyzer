#!/usr/bin/env python3
"""Run and certify the paired Liger dW-accumulator weight trajectory."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/data1/tzh").resolve()
MODEL = DATA_ROOT / "models/Qwen/Qwen3-1.7B"
DEFAULT_DESIGN = (
    ROOT
    / "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/supplementary_state_design_v1.json"
)
DEFAULT_LOCAL_PROTOCOL = (
    ROOT / "archive/nonprecision_v1/runs/liger.fused_ce.protocol.json"
)
DEFAULT_LOCAL_CERTIFICATE = (
    ROOT / "archive/nonprecision_v1/runs/liger.fused_ce.certificate.json"
)
DEFAULT_MECHANISM_CERTIFICATE = (
    ROOT / "archive/nonprecision_v1/runs/liger.fused_ce.mechanism.json"
)
DEFAULT_CARRIER = (
    ROOT / "archive/nonprecision_v1/runs/liger.fused_ce.parameter.carrier.json.gz"
)
DEFAULT_PROTOCOL = ROOT / "results/trajectory/liger_protocol.json"
DEFAULT_RESULTS = ROOT / "results/trajectory/liger_steps"
DEFAULT_CHECKPOINTS = DATA_ROOT / "cache/kernel-analyzer/liger_trajectory"
DEFAULT_OUTPUT = ROOT / "results/trajectory/liger_trajectory.json"

PROTOCOL_SCHEMA = "kernel-analyzer.liger-trajectory-protocol.v1"
WORKER_SCHEMA = "kernel-analyzer.liger-trajectory-step.v1"
PAIR_SCHEMA = "kernel-analyzer.liger-trajectory-pair.v1"
CAMPAIGN_SCHEMA = "kernel-analyzer.liger-trajectory-campaign.v1"
ARMS = ("DEFAULT_BF16_ACCUM", "FP32_ACCUM_REPAIR")
SKETCH_SIZE = 8192
SKETCH_SEED = 3407
CHUNK = 50_000_000


def checked(path: Path) -> Path:
    result = path.resolve()
    if DATA_ROOT not in (result, *result.parents):
        raise RuntimeError(f"path must stay under /data1/tzh: {result}")
    return result


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def tensor_digest(value: Any) -> str:
    import torch

    tensor = value.detach().contiguous().cpu()
    return hashlib.sha256(
        tensor.view(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def validate_protocol(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != PROTOCOL_SCHEMA:
        raise RuntimeError("trajectory protocol schema differs")
    if value.get("status") != "FROZEN_BEFORE_TRAJECTORY_VALUES":
        raise RuntimeError("trajectory protocol is not frozen")
    if tuple(value["trajectory"]["arms"]) != ARMS:
        raise RuntimeError("trajectory arms differ")
    if value["trajectory"]["steps"] != len(value["trajectory"]["state_order"]):
        raise RuntimeError("trajectory step denominator differs")
    unsigned = dict(value)
    observed = unsigned.pop("artifact_sha256", None)
    if observed != digest(unsigned):
        raise RuntimeError("trajectory protocol digest differs")


def validate_worker(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != WORKER_SCHEMA:
        raise RuntimeError("trajectory worker schema differs")
    if value.get("status") != "COMPLETE":
        raise RuntimeError("trajectory worker is incomplete")
    if value["step"]["arm"] not in ARMS:
        raise RuntimeError("trajectory worker arm differs")
    unsigned = dict(value)
    observed = unsigned.pop("artifact_sha256", None)
    if observed != digest(unsigned):
        raise RuntimeError("trajectory worker digest differs")
    if not all(value["gates"].values()):
        raise RuntimeError("trajectory worker failed a gate")


def validate_pair(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != PAIR_SCHEMA or value.get("status") != "COMPLETE":
        raise RuntimeError("trajectory pair is incomplete")
    unsigned = dict(value)
    observed = unsigned.pop("artifact_sha256", None)
    if observed != digest(unsigned):
        raise RuntimeError("trajectory pair digest differs")
    if not all(value["gates"].values()):
        raise RuntimeError("trajectory pair failed a gate")


def freeze(args: argparse.Namespace) -> None:
    output = checked(args.output)
    if output.exists():
        raise FileExistsError(output)
    design_path = checked(args.design)
    local_protocol_path = checked(args.local_protocol)
    local_certificate_path = checked(args.local_certificate)
    mechanism_path = checked(args.mechanism_certificate)
    carrier_path = checked(args.carrier)
    design = load_json(design_path)
    local_protocol = load_json(local_protocol_path)
    local_certificate = load_json(local_certificate_path)
    mechanism = load_json(mechanism_path)
    carrier = load_gzip_json(carrier_path)
    if local_certificate.get("verdict") != (
        "NATURAL_LIGER_FUSED_CE_BF16_DW_ACCUMULATION_BIAS_CONFIRMED"
    ):
        raise RuntimeError("local Liger certificate differs")
    if mechanism.get("verdict") != (
        "COMPLETE_FLASHATTENTION_STYLE_LIGER_FUSED_CE_ACCUMULATION_MECHANISM"
    ):
        raise RuntimeError("full-step Liger mechanism certificate differs")
    if carrier.get("status") != "FROZEN_BEFORE_24_CONFIRMATION_VALUES":
        raise RuntimeError("Liger carrier freeze differs")

    allocations = []
    for phase in ("pilot", "discovery", "confirmation"):
        allocations.extend(local_protocol["state_allocations"][phase])
    if len(allocations) != 32 or len({row["state_id"] for row in allocations}) != 32:
        raise RuntimeError("expected 32 unique frozen seq128 states")
    records = {row["sequence_id"]: row for row in design["records"]}
    for row in allocations:
        record = records.get(row["state_id"])
        if record is None or record["record_sha256"] != row["record_sha256"]:
            raise RuntimeError(f"state binding differs: {row['state_id']}")

    payload: dict[str, Any] = {
        "schema_version": PROTOCOL_SCHEMA,
        "status": "FROZEN_BEFORE_TRAJECTORY_VALUES",
        "purpose": (
            "Test whether the already-established BF16 dW-accumulator bias "
            "produces live-weight divergence and whether the FP32-accumulator "
            "repair removes that local cause at every evolving arm state."
        ),
        "trajectory": {
            "arms": list(ARMS),
            "steps": len(allocations),
            "state_order": [row["state_id"] for row in allocations],
            "state_record_sha256s": {
                row["state_id"]: row["record_sha256"] for row in allocations
            },
            "optimizer": "STATELESS_SGD_WITH_FP32_MASTER",
            "learning_rate": 1.0e-4,
            "initial_master": "SOURCE_BF16_PARAMETERS_EXACTLY_PROMOTED_TO_FP32",
            "model_materialization_each_step": "CURRENT_FP32_MASTER_CAST_TO_BF16",
            "update_equation": "W_master[t+1]=W_master[t]-lr*g_arm[t].float()",
        },
        "per_step_control": {
            "same_weight_counterfactuals": (
                "Both accumulator implementations execute at each arm's current weights "
                "before that arm is updated."
            ),
            "fixed": (
                "forward hidden computation, labels, loss reduction, external BF16 dtype, "
                "64-chunk schedule, TF32=false, state order and optimizer"
            ),
            "treatment": "only Liger dW accum_dtype: None(BF16 storage) versus torch.float32",
            "required_zero_controls": ["loss", "hidden", "labels", "terminal_dH", "309 untied gradients"],
            "only_allowed_gradient_change": "model.embed_tokens.weight",
        },
        "readout": {
            "same_weight": [
                "default-minus-repair full-gradient L2",
                "projection onto the previously frozen tied-weight carrier",
            ],
            "live_weight": [
                "default-minus-repair FP32-master L2",
                "default-minus-repair materialized-BF16 L2",
            ],
            "success": (
                "all local controls pass at every evolving state; local accumulator delta "
                "remains nonzero; and the two live-weight arms diverge after the paired updates"
            ),
        },
        "claim_boundary": {
            "supported": "32-step causal live-weight consequence of the established Liger accumulator repair",
            "not_supported": [
                "AdamW or optimizer-state behavior",
                "catastrophic loss instability",
                "cross-model generalization",
                "an additional independent operator case",
            ],
        },
        "bindings": {
            "state_design": {"path": str(design_path), "sha256": file_sha256(design_path)},
            "local_protocol": {"path": str(local_protocol_path), "sha256": file_sha256(local_protocol_path)},
            "local_certificate": {"path": str(local_certificate_path), "sha256": file_sha256(local_certificate_path)},
            "mechanism_certificate": {"path": str(mechanism_path), "sha256": file_sha256(mechanism_path)},
            "frozen_carrier": {"path": str(carrier_path), "sha256": file_sha256(carrier_path)},
        },
    }
    payload["artifact_sha256"] = digest(payload)
    validate_protocol(payload)
    atomic_json(output, payload)
    print(json.dumps({"output": str(output), "steps": len(allocations), "artifact_sha256": payload["artifact_sha256"]}, sort_keys=True))


def add_sketch(accumulator: Any, delta: Any, *, name: str) -> None:
    import torch

    flat = delta.detach().reshape(-1)
    name_seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
    seed = (SKETCH_SEED + name_seed) % (2**31 - 1)
    position = torch.arange(SKETCH_SIZE, device=flat.device, dtype=torch.int64)
    signs = (
        torch.remainder(position * 1_103_515_245 + seed, 2)
        .to(torch.float32)
        .mul_(2)
        .sub_(1)
    )
    buckets = torch.remainder(position * 2_654_435_761 + seed, SKETCH_SIZE)
    columns = torch.zeros(SKETCH_SIZE, device=flat.device, dtype=torch.float64)
    block_count = math.ceil(flat.numel() / SKETCH_SIZE)
    for block_start in range(0, block_count, 1024):
        start = block_start * SKETCH_SIZE
        stop = min(flat.numel(), (block_start + 1024) * SKETCH_SIZE)
        chunk = flat[start:stop].to(torch.float32)
        if chunk.numel() % SKETCH_SIZE:
            chunk = torch.nn.functional.pad(
                chunk, (0, SKETCH_SIZE - chunk.numel() % SKETCH_SIZE)
            )
        columns += (chunk.reshape(-1, SKETCH_SIZE) * signs).sum(
            dim=0, dtype=torch.float64
        )
    accumulator[buckets] += columns


def full_step(model: Any, loss_module: Any, input_ids: Any) -> dict[str, Any]:
    import torch

    model.zero_grad(set_to_none=True)
    outputs = model.model(input_ids=input_ids, use_cache=False, return_dict=True)
    hidden = outputs.last_hidden_state
    observed: list[Any] = []
    hidden.register_hook(lambda gradient: observed.append(gradient.detach().clone()))
    labels = torch.nn.functional.pad(input_ids, (0, 1), value=-100)[..., 1:]
    labels = labels.contiguous().reshape(-1)
    loss = loss_module(
        model.lm_head.weight,
        hidden.reshape(-1, hidden.shape[-1]),
        labels,
    )
    loss_value = loss.detach().clone()
    loss.backward()
    if len(observed) != 1:
        raise RuntimeError("terminal hidden VJP was not observed exactly once")
    return {
        "loss": loss_value,
        "hidden_digest": tensor_digest(hidden),
        "labels_digest": tensor_digest(labels),
        "dH_digest": tensor_digest(observed[0]),
        "gradients": {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
        },
    }


def gradient_contrast(default: Mapping[str, Any], repair: Mapping[str, Any], device: Any) -> dict[str, Any]:
    import torch

    if set(default) != set(repair):
        raise RuntimeError("gradient populations differ")
    sketch = torch.zeros(SKETCH_SIZE, device=device, dtype=torch.float64)
    square = torch.zeros((), device=device, dtype=torch.float64)
    max_abs = 0.0
    nonzero_names = []
    for name in default:
        left, right = default[name], repair[name]
        if left is None or right is None:
            if left is not right:
                raise RuntimeError(f"gradient reach differs: {name}")
            continue
        delta = left.float().sub(right.float())
        local_max = float(delta.abs().max().item())
        if local_max:
            nonzero_names.append(name)
        square += torch.sum(delta.to(torch.float64).square())
        max_abs = max(max_abs, local_max)
        add_sketch(sketch, delta, name=name)
    return {
        "orientation": "default_BF16_accum_minus_FP32_accum_repair",
        "l2_norm": float(torch.sqrt(square).item()),
        "max_abs": max_abs,
        "nonzero_parameter_names": nonzero_names,
        "countsketch8192": sketch.cpu().tolist(),
    }


def save_checkpoint(path: Path, masters: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(masters, temporary)
    os.replace(temporary, path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def worker(args: argparse.Namespace) -> None:
    protocol_path = checked(args.protocol)
    design_path = checked(args.design)
    carrier_path = checked(args.carrier)
    model_path = checked(args.model)
    output_checkpoint = checked(args.output_checkpoint)
    output = checked(args.output)
    protocol = load_json(protocol_path)
    validate_protocol(protocol)
    design = load_json(design_path)
    if file_sha256(design_path) != protocol["bindings"]["state_design"]["sha256"]:
        raise RuntimeError("state design binding differs")
    if file_sha256(carrier_path) != protocol["bindings"]["frozen_carrier"]["sha256"]:
        raise RuntimeError("carrier binding differs")
    carrier = load_gzip_json(carrier_path)
    direction = carrier["normalized_values"]
    if len(direction) != SKETCH_SIZE:
        raise RuntimeError("frozen carrier dimension differs")
    step = int(args.step_index)
    if not 0 <= step < protocol["trajectory"]["steps"]:
        raise RuntimeError("step outside frozen trajectory")
    state_id = protocol["trajectory"]["state_order"][step]
    records = {row["sequence_id"]: row for row in design["records"]}
    record = records[state_id]
    if record["record_sha256"] != protocol["trajectory"]["state_record_sha256s"][state_id]:
        raise RuntimeError("state record binding differs")

    previous_binding = None
    if step == 0:
        if args.input_checkpoint is not None or args.input_worker is not None:
            raise RuntimeError("step zero must start from source weights")
    else:
        if args.input_checkpoint is None or args.input_worker is None:
            raise RuntimeError("nonzero step requires predecessor files")
        previous = load_json(checked(args.input_worker))
        validate_worker(previous)
        input_checkpoint = checked(args.input_checkpoint)
        if previous["step"]["arm"] != args.arm or previous["step"]["step_index"] != step - 1:
            raise RuntimeError("predecessor worker differs")
        expected = previous["output_master_checkpoint"]
        if str(input_checkpoint) != expected["path"] or input_checkpoint.stat().st_size != expected["bytes"]:
            raise RuntimeError("predecessor checkpoint binding differs")
        if file_sha256(input_checkpoint) != expected["sha256"]:
            raise RuntimeError("predecessor checkpoint digest differs")
        previous_binding = {
            "worker_sha256": previous["artifact_sha256"],
            "checkpoint_sha256": expected["sha256"],
        }

    import torch
    import transformers
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    from transformers import AutoModelForCausalLM

    started = time.monotonic()
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    ).to(device)
    model.config.use_cache = False
    model.eval()
    if len(list(model.named_parameters())) != 310:
        raise RuntimeError("model parameter denominator differs")
    tied = (
        model.lm_head.weight.untyped_storage().data_ptr()
        == model.model.embed_tokens.weight.untyped_storage().data_ptr()
    )
    if not tied:
        raise RuntimeError("model weight is not tied")
    if step == 0:
        masters = {
            name: parameter.detach().cpu().float().clone()
            for name, parameter in model.named_parameters()
        }
    else:
        masters = torch.load(checked(args.input_checkpoint), map_location="cpu", weights_only=True)
        if set(masters) != {name for name, _ in model.named_parameters()}:
            raise RuntimeError("master parameter population differs")
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(masters[name].to(device=device, dtype=parameter.dtype))

    input_ids = torch.tensor([record["input_ids"]], device=device, dtype=torch.long)
    modules = {
        "default": LigerFusedLinearCrossEntropyLoss(
            ignore_index=-100, reduction="mean", accum_dtype=None
        ).to(device),
        "repair": LigerFusedLinearCrossEntropyLoss(
            ignore_index=-100, reduction="mean", accum_dtype=torch.float32
        ).to(device),
    }
    torch.cuda.reset_peak_memory_stats(device)
    default = full_step(model, modules["default"], input_ids)
    repair = full_step(model, modules["repair"], input_ids)
    common = {
        key: default[key] == repair[key]
        for key in ("hidden_digest", "labels_digest", "dH_digest")
    }
    common["loss"] = bool(torch.equal(default["loss"], repair["loss"]))
    contrast = gradient_contrast(default["gradients"], repair["gradients"], device)
    direction_tensor = torch.tensor(direction, device=device, dtype=torch.float64)
    sketch_tensor = torch.tensor(contrast.pop("countsketch8192"), device=device, dtype=torch.float64)
    contrast["frozen_carrier_projection"] = float(torch.dot(sketch_tensor, direction_tensor).item())
    contrast["frozen_carrier_cosine"] = float(
        torch.dot(sketch_tensor, direction_tensor).item()
        / torch.linalg.vector_norm(sketch_tensor).item()
    )

    selected = default["gradients"] if args.arm == ARMS[0] else repair["gradients"]
    learning_rate = float(protocol["trajectory"]["learning_rate"])
    with torch.no_grad():
        for name, gradient in selected.items():
            if gradient is not None:
                masters[name].add_(gradient.cpu().float(), alpha=-learning_rate)
    checkpoint = save_checkpoint(output_checkpoint, masters)
    gates = {
        "model_weight_is_tied": tied,
        "all_same_weight_forward_and_dH_controls_exact": all(common.values()),
        "only_tied_embedding_gradient_differs": contrast["nonzero_parameter_names"] == ["model.embed_tokens.weight"],
        "same_weight_accumulator_delta_nonzero": contrast["l2_norm"] > 0.0,
        "finite_loss_delta_and_projection": all(
            math.isfinite(value)
            for value in (
                float(default["loss"].item()),
                float(repair["loss"].item()),
                contrast["l2_norm"],
                contrast["frozen_carrier_projection"],
            )
        ),
        "all_310_parameters_updated_or_retained": len(masters) == 310,
    }
    payload: dict[str, Any] = {
        "schema_version": WORKER_SCHEMA,
        "status": "COMPLETE" if all(gates.values()) else "FAILED_GATE",
        "step": {
            "step_index": step,
            "step_number": step + 1,
            "state_id": state_id,
            "arm": args.arm,
            "learning_rate": learning_rate,
        },
        "bindings": {
            "protocol_sha256": protocol["artifact_sha256"],
            "state_record_sha256": record["record_sha256"],
            "predecessor": previous_binding,
        },
        "same_weight_control": {
            "common": common,
            "loss": {
                "default": float(default["loss"].item()),
                "repair": float(repair["loss"].item()),
            },
            "gradient_contrast": contrast,
        },
        "gates": gates,
        "output_master_checkpoint": checkpoint,
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "triton": __import__("triton").__version__,
            "gpu": torch.cuda.get_device_name(device),
            "dtype": "torch.bfloat16",
            "tf32": False,
        },
        "resource_usage": {
            "wall_time_seconds": time.monotonic() - started,
            "peak_cuda_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
    }
    payload["artifact_sha256"] = digest(payload)
    validate_worker(payload)
    atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"],
        "step": payload["step"],
        "same_weight_l2": contrast["l2_norm"],
        "carrier_projection": contrast["frozen_carrier_projection"],
        "wall_time_seconds": payload["resource_usage"]["wall_time_seconds"],
        "peak_cuda_memory_reserved_bytes": payload["resource_usage"]["peak_cuda_memory_reserved_bytes"],
    }, sort_keys=True))


def compare(args: argparse.Namespace) -> None:
    import torch

    protocol = load_json(checked(args.protocol))
    validate_protocol(protocol)
    default_worker = load_json(checked(args.default_worker))
    repair_worker = load_json(checked(args.repair_worker))
    validate_worker(default_worker)
    validate_worker(repair_worker)
    if default_worker["step"]["arm"] != ARMS[0] or repair_worker["step"]["arm"] != ARMS[1]:
        raise RuntimeError("trajectory pair arm differs")
    for key in ("step_index", "step_number", "state_id"):
        if default_worker["step"][key] != repair_worker["step"][key]:
            raise RuntimeError("trajectory pair step binding differs")
    paths = {
        "default": checked(Path(default_worker["output_master_checkpoint"]["path"])),
        "repair": checked(Path(repair_worker["output_master_checkpoint"]["path"])),
    }
    for role, worker_value in (("default", default_worker), ("repair", repair_worker)):
        expected = worker_value["output_master_checkpoint"]
        if paths[role].stat().st_size != expected["bytes"]:
            raise RuntimeError(f"{role} checkpoint size differs")
    default = torch.load(paths["default"], map_location="cpu", weights_only=True)
    repair = torch.load(paths["repair"], map_location="cpu", weights_only=True)
    if set(default) != set(repair):
        raise RuntimeError("trajectory master populations differ")
    device = torch.device(args.device)
    stats = {
        "fp32_master": {"square": 0.0, "nonzero": 0, "max_abs": 0.0, "parameters": 0},
        "bf16_materialized": {"square": 0.0, "nonzero": 0, "max_abs": 0.0, "parameters": 0},
    }
    elements = 0
    for name in sorted(default):
        left = default[name].reshape(-1)
        right = repair[name].reshape(-1)
        elements += left.numel()
        parameter_nonzero = {key: False for key in stats}
        for start in range(0, left.numel(), CHUNK):
            a = left[start : start + CHUNK].to(device)
            b = right[start : start + CHUNK].to(device)
            values = {
                "fp32_master": a - b,
                "bf16_materialized": a.to(torch.bfloat16).float() - b.to(torch.bfloat16).float(),
            }
            for key, delta in values.items():
                stats[key]["square"] += float(torch.sum(delta.square(), dtype=torch.float64).item())
                count = int(torch.count_nonzero(delta).item())
                stats[key]["nonzero"] += count
                parameter_nonzero[key] |= count > 0
                if delta.numel():
                    stats[key]["max_abs"] = max(stats[key]["max_abs"], float(delta.abs().max().item()))
        for key in stats:
            stats[key]["parameters"] += int(parameter_nonzero[key])
    readout = {
        key: {
            "orientation": "default_minus_FP32_accumulator_repair",
            "l2_norm": math.sqrt(row["square"]),
            "max_abs": row["max_abs"],
            "nonzero_count": row["nonzero"],
            "parameters_with_nonzero": row["parameters"],
        }
        for key, row in stats.items()
    }
    gates = {
        "fp32_master_divergence_nonzero": readout["fp32_master"]["l2_norm"] > 0.0,
        "all_parameters_and_elements_retained": len(default) == 310 and elements > 0,
        "finite_readouts": all(math.isfinite(row["l2_norm"]) for row in readout.values()),
    }
    payload: dict[str, Any] = {
        "schema_version": PAIR_SCHEMA,
        "status": "COMPLETE" if all(gates.values()) else "FAILED_GATE",
        "step": {key: default_worker["step"][key] for key in ("step_index", "step_number", "state_id")},
        "bindings": {
            "protocol_sha256": protocol["artifact_sha256"],
            "default_worker_sha256": default_worker["artifact_sha256"],
            "repair_worker_sha256": repair_worker["artifact_sha256"],
            "default_checkpoint_sha256": default_worker["output_master_checkpoint"]["sha256"],
            "repair_checkpoint_sha256": repair_worker["output_master_checkpoint"]["sha256"],
        },
        "denominators": {"parameters": len(default), "elements": elements},
        "readout": readout,
        "gates": gates,
    }
    payload["artifact_sha256"] = digest(payload)
    validate_pair(payload)
    atomic_json(checked(args.output), payload)
    print(json.dumps({"step": payload["step"], "readout": readout}, sort_keys=True))


def aggregate(args: argparse.Namespace) -> None:
    protocol = load_json(checked(args.protocol))
    validate_protocol(protocol)
    input_dir = checked(args.input_dir)
    workers: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    pairs = []
    for step in range(protocol["trajectory"]["steps"]):
        tag = f"{step:02d}"
        for arm, prefix in zip(ARMS, ("default", "repair"), strict=True):
            value = load_json(input_dir / f"{prefix}_step{tag}.json")
            validate_worker(value)
            workers[arm].append(value)
        pair = load_json(input_dir / f"pair_step{tag}.json")
        validate_pair(pair)
        pairs.append(pair)
    expected_order = protocol["trajectory"]["state_order"]
    for step, state_id in enumerate(expected_order):
        if pairs[step]["step"]["state_id"] != state_id:
            raise RuntimeError("trajectory state order differs")

    rows = []
    for step, pair in enumerate(pairs):
        default = workers[ARMS[0]][step]
        repair = workers[ARMS[1]][step]
        rows.append({
            "step_index": step,
            "state_id": pair["step"]["state_id"],
            "default_same_weight_gradient_l2": default["same_weight_control"]["gradient_contrast"]["l2_norm"],
            "repair_same_weight_gradient_l2": repair["same_weight_control"]["gradient_contrast"]["l2_norm"],
            "default_carrier_projection": default["same_weight_control"]["gradient_contrast"]["frozen_carrier_projection"],
            "repair_carrier_projection": repair["same_weight_control"]["gradient_contrast"]["frozen_carrier_projection"],
            "fp32_master_pair_l2": pair["readout"]["fp32_master"]["l2_norm"],
            "bf16_materialized_pair_l2": pair["readout"]["bf16_materialized"]["l2_norm"],
            "bf16_materialized_pair_nonzero": pair["readout"]["bf16_materialized"]["nonzero_count"],
        })
    projections = [
        row[key]
        for row in rows
        for key in ("default_carrier_projection", "repair_carrier_projection")
    ]
    gates = {
        "all_32_steps_and_both_arms_complete": len(rows) == 32,
        "all_64_same_weight_local_controls_pass": all(
            all(worker_value["gates"].values())
            for arm in ARMS
            for worker_value in workers[arm]
        ),
        "all_64_same_weight_accumulator_deltas_nonzero": all(
            row[key] > 0.0
            for row in rows
            for key in ("default_same_weight_gradient_l2", "repair_same_weight_gradient_l2")
        ),
        "all_64_frozen_carrier_projections_positive": all(value > 0.0 for value in projections),
        "fp32_master_divergence_after_every_step": all(row["fp32_master_pair_l2"] > 0.0 for row in rows),
        "bf16_live_weight_feedback_observed": any(row["bf16_materialized_pair_nonzero"] > 0 for row in rows),
        "final_master_divergence_exceeds_first": rows[-1]["fp32_master_pair_l2"] > rows[0]["fp32_master_pair_l2"],
    }
    payload: dict[str, Any] = {
        "schema_version": CAMPAIGN_SCHEMA,
        "status": "COMPLETE" if all(gates.values()) else "COMPLETE_WITH_FAILED_CAUSAL_GATE",
        "verdict": (
            "COMPLETE_LIGER_ACCUMULATOR_REPAIR_LIVE_WEIGHT_CAUSAL_CHAIN"
            if all(gates.values())
            else "LIGER_TRAJECTORY_MEASURED_BUT_STRICT_CAUSAL_GATE_FAILED"
        ),
        "bindings": {
            "protocol": {"path": str(checked(args.protocol)), "sha256": file_sha256(checked(args.protocol)), "artifact_sha256": protocol["artifact_sha256"]},
            "local_certificate_sha256": protocol["bindings"]["local_certificate"]["sha256"],
            "mechanism_certificate_sha256": protocol["bindings"]["mechanism_certificate"]["sha256"],
            "frozen_carrier_sha256": protocol["bindings"]["frozen_carrier"]["sha256"],
        },
        "denominators": {"steps": len(rows), "arms": 2, "same_weight_counterfactuals": len(rows) * 2},
        "aggregate": {
            "first_fp32_master_pair_l2": rows[0]["fp32_master_pair_l2"],
            "final_fp32_master_pair_l2": rows[-1]["fp32_master_pair_l2"],
            "max_fp32_master_pair_l2": max(row["fp32_master_pair_l2"] for row in rows),
            "first_bf16_materialized_pair_l2": rows[0]["bf16_materialized_pair_l2"],
            "final_bf16_materialized_pair_l2": rows[-1]["bf16_materialized_pair_l2"],
            "positive_frozen_carrier_projections": sum(value > 0.0 for value in projections),
            "carrier_projection_denominator": len(projections),
        },
        "gates": gates,
        "step_rows": rows,
        "claim_boundary": protocol["claim_boundary"],
    }
    payload["artifact_sha256"] = digest(payload)
    atomic_json(checked(args.output), payload)
    print(json.dumps({"status": payload["status"], "verdict": payload["verdict"], "aggregate": payload["aggregate"], "gates": gates}, indent=2, sort_keys=True))


def drive(args: argparse.Namespace) -> None:
    protocol_path = checked(args.protocol)
    protocol = load_json(protocol_path)
    validate_protocol(protocol)
    results = checked(args.results_dir)
    checkpoints = checked(args.checkpoint_dir)
    results.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    logs = checkpoints / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({
        "HF_HOME": "/data1/tzh/cache/huggingface",
        "HUGGINGFACE_HUB_CACHE": "/data1/tzh/cache/huggingface/hub",
        "TRANSFORMERS_CACHE": "/data1/tzh/cache/huggingface/transformers",
        "XDG_CACHE_HOME": "/data1/tzh/cache/xdg",
    })
    script = Path(__file__).resolve()
    worker_gpus = [item.strip() for item in args.worker_gpus.split(",")]
    if len(worker_gpus) != len(ARMS) or any(not item for item in worker_gpus):
        raise RuntimeError("--worker-gpus must name one physical GPU for each training configuration")
    compare_gpu = str(args.compare_gpu)
    stop_step = (
        protocol["trajectory"]["steps"]
        if args.stop_step is None
        else min(args.stop_step, protocol["trajectory"]["steps"])
    )
    if not 0 <= args.start_step < stop_step:
        raise RuntimeError("driver step interval is empty or outside the protocol")
    for step in range(args.start_step, stop_step):
        tag = f"{step:02d}"
        worker_paths = {
            ARMS[0]: results / f"default_step{tag}.json",
            ARMS[1]: results / f"repair_step{tag}.json",
        }
        checkpoint_paths = {
            ARMS[0]: checkpoints / f"default_step{tag}.pt",
            ARMS[1]: checkpoints / f"repair_step{tag}.pt",
        }
        pair_path = results / f"pair_step{tag}.json"
        processes = []
        for gpu, arm in zip(worker_gpus, ARMS):
            prefix = "default" if arm == ARMS[0] else "repair"
            command = [
                sys.executable, str(script), "worker",
                "--protocol", str(protocol_path),
                "--design", str(checked(args.design)),
                "--carrier", str(checked(args.carrier)),
                "--model", str(checked(args.model)),
                "--arm", arm,
                "--step-index", str(step),
                "--output-checkpoint", str(checkpoint_paths[arm]),
                "--output", str(worker_paths[arm]),
            ]
            if step > 0:
                previous_tag = f"{step - 1:02d}"
                command.extend([
                    "--input-worker", str(results / f"{prefix}_step{previous_tag}.json"),
                    "--input-checkpoint", str(checkpoints / f"{prefix}_step{previous_tag}.pt"),
                ])
            arm_env = dict(environment)
            arm_env["CUDA_VISIBLE_DEVICES"] = gpu
            log_path = logs / f"{prefix}_step{tag}.log"
            handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(command, cwd=ROOT, env=arm_env, stdout=handle, stderr=subprocess.STDOUT)
            processes.append((arm, process, handle, log_path))
        failures = []
        for arm, process, handle, log_path in processes:
            returncode = process.wait()
            handle.close()
            if returncode:
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-10000:]
                failures.append((arm, returncode, tail))
        if failures:
            raise RuntimeError(f"trajectory workers failed at step {step}: {failures}")
        pair_env = dict(environment)
        pair_env["CUDA_VISIBLE_DEVICES"] = compare_gpu
        pair_log = logs / f"pair_step{tag}.log"
        command = [
            sys.executable, str(script), "compare",
            "--protocol", str(protocol_path),
            "--default-worker", str(worker_paths[ARMS[0]]),
            "--repair-worker", str(worker_paths[ARMS[1]]),
            "--device", "cuda",
            "--output", str(pair_path),
        ]
        with pair_log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(command, cwd=ROOT, env=pair_env, stdout=handle, stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            tail = pair_log.read_text(encoding="utf-8", errors="replace")[-10000:]
            raise RuntimeError(f"trajectory pair failed at step {step}: {tail}")
        pair = load_json(pair_path)
        validate_pair(pair)
        for _, _, _, log_path in processes:
            log_path.unlink(missing_ok=True)
        pair_log.unlink(missing_ok=True)
        if step > 0:
            previous_tag = f"{step - 1:02d}"
            for prefix in ("default", "repair"):
                previous = checkpoints / f"{prefix}_step{previous_tag}.pt"
                if previous.exists():
                    previous.unlink()
        print(json.dumps({
            "event": "PAIR_COMPLETE",
            "step_index": step,
            "state_id": pair["step"]["state_id"],
            "fp32_master_l2": pair["readout"]["fp32_master"]["l2_norm"],
            "bf16_materialized_l2": pair["readout"]["bf16_materialized"]["l2_norm"],
        }, sort_keys=True), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    freeze_parser.add_argument("--local-protocol", type=Path, default=DEFAULT_LOCAL_PROTOCOL)
    freeze_parser.add_argument("--local-certificate", type=Path, default=DEFAULT_LOCAL_CERTIFICATE)
    freeze_parser.add_argument("--mechanism-certificate", type=Path, default=DEFAULT_MECHANISM_CERTIFICATE)
    freeze_parser.add_argument("--carrier", type=Path, default=DEFAULT_CARRIER)
    freeze_parser.add_argument("--output", type=Path, default=DEFAULT_PROTOCOL)
    freeze_parser.set_defaults(function=freeze)

    worker_parser = commands.add_parser("worker")
    worker_parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    worker_parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    worker_parser.add_argument("--carrier", type=Path, default=DEFAULT_CARRIER)
    worker_parser.add_argument("--model", type=Path, default=MODEL)
    worker_parser.add_argument("--arm", choices=ARMS, required=True)
    worker_parser.add_argument("--step-index", type=int, required=True)
    worker_parser.add_argument("--input-worker", type=Path)
    worker_parser.add_argument("--input-checkpoint", type=Path)
    worker_parser.add_argument("--output-checkpoint", type=Path, required=True)
    worker_parser.add_argument("--output", type=Path, required=True)
    worker_parser.add_argument("--device", default="cuda")
    worker_parser.set_defaults(function=worker)

    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    compare_parser.add_argument("--default-worker", type=Path, required=True)
    compare_parser.add_argument("--repair-worker", type=Path, required=True)
    compare_parser.add_argument("--device", default="cuda")
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.set_defaults(function=compare)

    aggregate_parser = commands.add_parser("aggregate")
    aggregate_parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    aggregate_parser.add_argument("--input-dir", type=Path, default=DEFAULT_RESULTS)
    aggregate_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    aggregate_parser.set_defaults(function=aggregate)

    drive_parser = commands.add_parser("drive")
    drive_parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    drive_parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    drive_parser.add_argument("--carrier", type=Path, default=DEFAULT_CARRIER)
    drive_parser.add_argument("--model", type=Path, default=MODEL)
    drive_parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    drive_parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINTS)
    drive_parser.add_argument("--start-step", type=int, default=0)
    drive_parser.add_argument("--stop-step", type=int)
    drive_parser.add_argument(
        "--worker-gpus", default="0,1",
        help="Comma-separated physical GPUs for candidate and repair training.",
    )
    drive_parser.add_argument(
        "--compare-gpu", default="2",
        help="Physical GPU used to compare the two saved training states.",
    )
    drive_parser.set_defaults(function=drive)
    return result


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
