#!/usr/bin/env python3
"""Run full-model Liger candidate/repair and update-injection experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data1/tzh/cache/huggingface/transformers")
os.environ.setdefault("XDG_CACHE_HOME", "/data1/tzh/cache/xdg")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kernel_analyzer.single_boundary_collapse import (  # noqa: E402
    INJECTION_MODES,
    balanced_sign,
)


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def load_protocol(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("schema") != "kernel-analyzer-liger-single-boundary-collapse-protocol-v1":
        raise RuntimeError("unexpected protocol schema")
    if value.get("status") != "FROZEN_BEFORE_EMPIRICAL_RESULTS":
        raise RuntimeError("protocol is not frozen")
    data = Path(value["data"]["path"])
    if not data.is_file() or file_sha256(data) != value["data"]["sha256"]:
        raise RuntimeError("training text does not match the frozen protocol")
    return value


def encode_text(protocol: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    text = Path(protocol["data"]["path"]).read_text(encoding="utf-8")
    symbols = sorted(set(text))
    mapping = {symbol: index for index, symbol in enumerate(symbols)}
    encoded = np.fromiter((mapping[symbol] for symbol in text), dtype=np.int64)
    split = int(len(encoded) * float(protocol["data"]["train_fraction"]))
    return encoded[:split], encoded[split:], mapping


def batch_from_stream(
    tokens: np.ndarray,
    *,
    batch_size: int,
    sequence_length: int,
    stream: int,
    step: int,
    seed: int,
    repeat_within_batch: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    key = np.random.SeedSequence([seed, stream, step])
    rng = np.random.default_rng(key)
    count = 1 if repeat_within_batch else batch_size
    offsets = rng.integers(0, len(tokens) - sequence_length - 1, size=count)
    if repeat_within_batch:
        offsets = np.repeat(offsets, batch_size)
    inputs = np.stack([tokens[offset : offset + sequence_length] for offset in offsets])
    labels = np.stack([tokens[offset + 1 : offset + sequence_length + 1] for offset in offsets])
    return (
        torch.from_numpy(inputs).to(device=device, dtype=torch.long),
        torch.from_numpy(labels).to(device=device, dtype=torch.long),
        [int(value) for value in offsets],
    )


def build_model(protocol: dict[str, Any], device: torch.device):
    from transformers import GPT2Config, GPT2LMHeadModel

    config = protocol["model"]
    torch.manual_seed(int(config["initialization_seed"]))
    torch.cuda.manual_seed_all(int(config["initialization_seed"]))
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=int(config["vocab_size"]),
        n_positions=int(config["sequence_length"]),
        n_ctx=int(config["sequence_length"]),
        n_embd=int(config["embedding_dimension"]),
        n_layer=int(config["layers"]),
        n_head=int(config["heads"]),
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
    ))
    model.tie_weights()
    model.to(device=device, dtype=torch.bfloat16)
    model.train()
    names = dict(model.named_parameters())
    target_name = str(protocol["implementation"]["target_parameter"])
    if target_name not in names:
        raise RuntimeError(f"target parameter is absent: {target_name}")
    if names[target_name].untyped_storage().data_ptr() != model.lm_head.weight.untyped_storage().data_ptr():
        raise RuntimeError("target embedding is not tied to the language-model head")
    return model, target_name


def make_loss_modules(protocol: dict[str, Any], device: torch.device) -> dict[str, Any]:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

    common = dict(ignore_index=-100, reduction="mean")
    return {
        "CANDIDATE": LigerFusedLinearCrossEntropyLoss(
            **common, accum_dtype=None,
        ).to(device),
        "REPAIR": LigerFusedLinearCrossEntropyLoss(
            **common, accum_dtype=torch.float32,
        ).to(device),
    }


def initialize_optimizer_state(model: torch.nn.Module) -> tuple[dict[str, torch.Tensor], ...]:
    master = {name: parameter.detach().float().clone() for name, parameter in model.named_parameters()}
    first = {name: torch.zeros_like(value) for name, value in master.items()}
    second = {name: torch.zeros_like(value) for name, value in master.items()}
    return master, first, second


@torch.no_grad()
def materialize(model: torch.nn.Module, master: dict[str, torch.Tensor]) -> None:
    for name, parameter in model.named_parameters():
        parameter.copy_(master[name].to(dtype=parameter.dtype))


def learning_rate(protocol: dict[str, Any], step: int, total_steps: int) -> float:
    config = protocol["optimizer"]
    peak = float(config["peak_learning_rate"])
    warmup = int(config["warmup_steps"])
    minimum = float(config["minimum_learning_rate"])
    if step <= warmup:
        return peak * step / max(warmup, 1)
    progress = min(1.0, (step - warmup) / max(total_steps - warmup, 1))
    return minimum + 0.5 * (1.0 + math.cos(math.pi * progress)) * (peak - minimum)


def proposed_delta(
    master: torch.Tensor,
    gradient: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    step: int,
    lr: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    weight_decay: float,
) -> torch.Tensor:
    next_first = beta1 * first + (1.0 - beta1) * gradient
    next_second = beta2 * second + (1.0 - beta2) * gradient.square()
    numerator = next_first / (1.0 - beta1**step)
    denominator = torch.sqrt(next_second / (1.0 - beta2**step)) + epsilon
    return -lr * numerator / denominator - lr * weight_decay * master


def backward_pass(
    model: torch.nn.Module,
    module: Any,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    target_name: str,
) -> tuple[float, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    hidden = model.transformer(input_ids=inputs, use_cache=False, return_dict=True).last_hidden_state
    loss = module(
        model.lm_head.weight,
        hidden.reshape(-1, hidden.shape[-1]),
        labels.reshape(-1).contiguous(),
    )
    loss.backward()
    target = dict(model.named_parameters())[target_name]
    if target.grad is None:
        raise RuntimeError("target gradient is absent")
    return float(loss.detach().float().item()), target.grad.detach().float().clone()


@torch.no_grad()
def apply_adamw(
    model: torch.nn.Module,
    master: dict[str, torch.Tensor],
    first: dict[str, torch.Tensor],
    second: dict[str, torch.Tensor],
    *,
    step: int,
    lr: float,
    optimizer: dict[str, Any],
) -> tuple[float, float, float]:
    beta1, beta2 = (float(value) for value in optimizer["betas"])
    epsilon = float(optimizer["epsilon"])
    weight_decay = float(optimizer["weight_decay"])
    update_energy = 0.0
    gradient_energy = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"gradient is absent: {name}")
        gradient = parameter.grad.detach().float()
        gradient_energy += float(torch.sum(gradient.square()).item())
        first[name].mul_(beta1).add_(gradient, alpha=1.0 - beta1)
        second[name].mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
        numerator = first[name] / (1.0 - beta1**step)
        denominator = torch.sqrt(second[name] / (1.0 - beta2**step)) + epsilon
        delta = -lr * numerator / denominator - lr * weight_decay * master[name]
        update_energy += float(torch.sum(delta.square()).item())
        master[name].add_(delta)
    moment1_energy = math.fsum(float(torch.sum(value.square()).item()) for value in first.values())
    moment2_energy = math.fsum(float(torch.sum(value.square()).item()) for value in second.values())
    return math.sqrt(gradient_energy), math.sqrt(update_energy), math.sqrt(moment1_energy + moment2_energy)


def candidate_repair_target_delta(
    protocol: dict[str, Any],
    *,
    master: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    candidate_gradient: torch.Tensor,
    repair_gradient: torch.Tensor,
    step: int,
    lr: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    optimizer = protocol["optimizer"]
    beta1, beta2 = (float(value) for value in optimizer["betas"])
    common = dict(
        step=step,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        epsilon=float(optimizer["epsilon"]),
        weight_decay=float(optimizer["weight_decay"]),
    )
    candidate = proposed_delta(master, candidate_gradient, first, second, **common)
    repair = proposed_delta(master, repair_gradient, first, second, **common)
    return candidate, repair, candidate - repair


def calibration(args: argparse.Namespace, protocol: dict[str, Any]) -> None:
    device = torch.device(args.device)
    train, _, mapping = encode_text(protocol)
    model, target_name = build_model(protocol, device)
    modules = make_loss_modules(protocol, device)
    master, first, second = initialize_optimizer_state(model)
    model_initial_digest = tensor_sha256(master[target_name])
    config = protocol["calibration"]
    vectors: list[torch.Tensor] = []
    repair_update_energy = 0.0
    rows = []
    started = time.monotonic()
    for index in range(int(config["state_count"])):
        inputs, labels, offsets = batch_from_stream(
            train,
            batch_size=int(protocol["model"]["batch_size"]),
            sequence_length=int(protocol["model"]["sequence_length"]),
            stream=int(config["stream"]),
            step=index,
            seed=int(protocol["data"]["sampling_seed"]),
            repeat_within_batch=bool(protocol["data"].get("repeat_within_batch", False)),
            device=device,
        )
        materialize(model, master)
        candidate_loss, candidate_gradient = backward_pass(
            model, modules["CANDIDATE"], inputs, labels, target_name=target_name,
        )
        materialize(model, master)
        repair_loss, repair_gradient = backward_pass(
            model, modules["REPAIR"], inputs, labels, target_name=target_name,
        )
        lr = learning_rate(protocol, 1, int(protocol["discovery"]["steps"]))
        candidate_delta, repair_delta, effect = candidate_repair_target_delta(
            protocol,
            master=master[target_name],
            first=first[target_name],
            second=second[target_name],
            candidate_gradient=candidate_gradient,
            repair_gradient=repair_gradient,
            step=1,
            lr=lr,
        )
        vectors.append(effect.cpu())
        repair_update_energy += float(torch.sum(repair_delta.square()).item())
        rows.append({
            "state": index,
            "offsets": offsets,
            "candidate_loss": candidate_loss,
            "repair_loss": repair_loss,
            "loss_difference": candidate_loss - repair_loss,
            "candidate_update_norm": float(torch.linalg.vector_norm(candidate_delta).item()),
            "repair_update_norm": float(torch.linalg.vector_norm(repair_delta).item()),
            "effect_norm": float(torch.linalg.vector_norm(effect).item()),
        })
        print(json.dumps({"event": "CALIBRATION_STATE", **rows[-1]}), flush=True)
    split = int(config["calibration_count"])
    calibration_mean = torch.stack(vectors[:split]).mean(dim=0)
    norm = float(torch.linalg.vector_norm(calibration_mean).item())
    if norm == 0.0:
        raise RuntimeError("calibration direction is not identifiable")
    direction = calibration_mean / norm
    confirmation = [float(torch.dot(vector.reshape(-1), direction.reshape(-1)).item()) for vector in vectors[split:]]
    all_sum = torch.stack(vectors).sum(dim=0)
    effect_energy = math.fsum(float(torch.sum(vector.square()).item()) for vector in vectors)
    result = {
        "schema": "kernel-analyzer-liger-natural-effect-calibration-v1",
        "status": "COMPLETE",
        "protocol": str(args.protocol.resolve()),
        "model_initial_target_digest": model_initial_digest,
        "target_parameter": target_name,
        "character_count": len(mapping),
        "state_count": len(vectors),
        "calibration_count": split,
        "confirmation_count": len(vectors) - split,
        "direction_confirmation": {
            "mean_projection": float(np.mean(confirmation)),
            "positive_count": sum(value > 0.0 for value in confirmation),
            "negative_count": sum(value < 0.0 for value in confirmation),
            "confirmed": float(np.mean(confirmation)) > 0.0,
        },
        "natural_effect": {
            "relative_rms": math.sqrt(effect_energy / repair_update_energy),
            "relative_mean": float(torch.linalg.vector_norm(all_sum).item())
            / math.sqrt(len(vectors) * repair_update_energy),
        },
        "rows": rows,
        "elapsed_seconds": time.monotonic() - started,
        "claim_boundary": (
            "Matched update effects at one randomly initialized small decoder and zero AdamW "
            "history. This freezes an injection direction; it is not a collapse result."
        ),
    }
    args.direction.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"direction": direction, "target_parameter": target_name}, args.direction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], **result["direction_confirmation"], **result["natural_effect"]}))


def train(args: argparse.Namespace, protocol: dict[str, Any]) -> None:
    device = torch.device(args.device)
    train_tokens, validation_tokens, mapping = encode_text(protocol)
    model, target_name = build_model(protocol, device)
    modules = make_loss_modules(protocol, device)
    master, first, second = initialize_optimizer_state(model)
    initial_target_digest = tensor_sha256(master[target_name])
    direction = None
    if args.mode in {"COHERENT_DIRECTION", "BALANCED_DIRECTION"}:
        saved = torch.load(args.direction, map_location=device, weights_only=True)
        if saved["target_parameter"] != target_name:
            raise RuntimeError("frozen direction targets another parameter")
        direction = saved["direction"].to(device=device, dtype=torch.float32)
        direction /= torch.linalg.vector_norm(direction)
    if args.mode == "NATURAL_CANDIDATE":
        implementation = "CANDIDATE"
    else:
        implementation = "REPAIR"
    if args.mode not in {*INJECTION_MODES, "NATURAL_CANDIDATE", "NATURAL_REPAIR"}:
        raise ValueError(f"unsupported mode: {args.mode}")

    injection_sum = torch.zeros_like(master[target_name])
    injected_energy = 0.0
    repair_update_energy = 0.0
    losses: list[float] = []
    rows = []
    first_nonfinite_step = None
    started = time.monotonic()
    for index in range(args.steps):
        step = index + 1
        inputs, labels, offsets = batch_from_stream(
            train_tokens,
            batch_size=int(protocol["model"]["batch_size"]),
            sequence_length=int(protocol["model"]["sequence_length"]),
            stream=args.stream,
            step=index,
            seed=int(protocol["data"]["sampling_seed"]),
            repeat_within_batch=bool(protocol["data"].get("repeat_within_batch", False)),
            device=device,
        )
        lr = learning_rate(protocol, step, args.steps)
        candidate_gradient = None
        if args.mode in INJECTION_MODES and args.mode != "ZERO":
            materialize(model, master)
            _, candidate_gradient = backward_pass(
                model, modules["CANDIDATE"], inputs, labels, target_name=target_name,
            )
        materialize(model, master)
        loss, repair_or_natural_gradient = backward_pass(
            model, modules[implementation], inputs, labels, target_name=target_name,
        )
        injection = torch.zeros_like(master[target_name])
        natural_effect_norm = 0.0
        if args.mode in INJECTION_MODES and args.mode != "ZERO":
            _, repair_delta, effect = candidate_repair_target_delta(
                protocol,
                master=master[target_name],
                first=first[target_name],
                second=second[target_name],
                candidate_gradient=candidate_gradient,
                repair_gradient=repair_or_natural_gradient,
                step=step,
                lr=lr,
            )
            natural_effect_norm = float(torch.linalg.vector_norm(effect).item())
            sign = balanced_sign(
                index,
                block_size=int(protocol["injection"]["balance_block_steps"]),
                seed=int(protocol["injection"]["sign_seed"]),
            ) if args.mode.startswith("BALANCED_") else 1
            if args.mode in {"COHERENT_DIRECTION", "BALANCED_DIRECTION"}:
                injection = direction * (args.multiplier * natural_effect_norm * sign)
            else:
                injection = effect * (args.multiplier * sign)
        gradient_norm, update_norm, moment_norm = apply_adamw(
            model,
            master,
            first,
            second,
            step=step,
            lr=lr,
            optimizer=protocol["optimizer"],
        )
        if args.mode in INJECTION_MODES:
            master[target_name].add_(injection)
        injection_sum.add_(injection)
        injection_norm = float(torch.linalg.vector_norm(injection).item())
        injected_energy += injection_norm**2
        repair_update_energy += update_norm**2
        losses.append(loss)
        finite = bool(
            math.isfinite(loss)
            and math.isfinite(gradient_norm)
            and math.isfinite(update_norm)
            and all(torch.isfinite(value).all().item() for value in master.values())
        )
        row = {
            "step": step,
            "learning_rate": lr,
            "training_loss": loss,
            "gradient_norm": gradient_norm,
            "repair_path_update_norm": update_norm,
            "moment_combined_norm": moment_norm,
            "natural_effect_norm": natural_effect_norm,
            "injection_norm": injection_norm,
            "relative_injection_energy": injected_energy / max(repair_update_energy, 1e-30),
            "relative_mean_direction_energy": float(torch.sum(injection_sum.square()).item())
            / max(step * repair_update_energy, 1e-30),
            "finite": finite,
            "offset_digest": hashlib.sha256(json.dumps(offsets).encode()).hexdigest(),
        }
        rows.append(row)
        if step == 1 or step % args.log_every == 0 or not finite:
            print(json.dumps({"event": "TRAIN_STEP", "mode": args.mode, **row}), flush=True)
        if not finite:
            first_nonfinite_step = step
            break

    validation_losses = []
    if first_nonfinite_step is None:
        for index in range(int(protocol["validation"]["batches"])):
            inputs, labels, _ = batch_from_stream(
                validation_tokens,
                batch_size=int(protocol["model"]["batch_size"]),
                sequence_length=int(protocol["model"]["sequence_length"]),
                stream=args.stream,
                step=index,
                seed=int(protocol["validation"]["sampling_seed"]),
                repeat_within_batch=bool(protocol["data"].get("repeat_within_batch", False)),
                device=device,
            )
            materialize(model, master)
            loss, _ = backward_pass(
                model, modules[implementation], inputs, labels, target_name=target_name,
            )
            validation_losses.append(loss)

    checkpoint_path = None if args.no_checkpoint else Path(args.checkpoint)
    checkpoint_digest = None
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "master": {name: value.cpu() for name, value in master.items()},
            "first": {name: value.cpu() for name, value in first.items()},
            "second": {name: value.cpu() for name, value in second.items()},
            "steps_completed": len(rows),
            "mode": args.mode,
            "multiplier": args.multiplier,
            "stream": args.stream,
        }, checkpoint_path)
        checkpoint_digest = file_sha256(checkpoint_path)
    payload = {
        "schema": "kernel-analyzer-liger-single-boundary-training-run-v1",
        "status": "NUMERICAL_COLLAPSE" if first_nonfinite_step is not None else "COMPLETE_FINITE",
        "protocol": str(args.protocol.resolve()),
        "mode": args.mode,
        "multiplier": args.multiplier,
        "stream": args.stream,
        "steps_requested": args.steps,
        "steps_completed": len(rows),
        "first_nonfinite_step": first_nonfinite_step,
        "target_parameter": target_name,
        "initial_target_digest": initial_target_digest,
        "character_count": len(mapping),
        "training_loss": {
            "first": losses[0],
            "last": losses[-1],
            "minimum": min(losses),
            "maximum": max(losses),
        },
        "validation_loss_mean": None if not validation_losses else float(np.mean(validation_losses)),
        "final_relative_injection_energy": rows[-1]["relative_injection_energy"],
        "final_relative_mean_direction_energy": rows[-1]["relative_mean_direction_energy"],
        "checkpoint": None if checkpoint_path is None else str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_digest,
        "elapsed_seconds": time.monotonic() - started,
        "rows": rows,
        "claim_boundary": (
            "Full-parameter training of a small decoder-only language model. Injection modes "
            "measure an amplified implementation-shaped boundary; only NATURAL_CANDIDATE versus "
            "NATURAL_REPAIR can establish an unamplified candidate-specific collapse."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "mode": args.mode,
        "multiplier": args.multiplier,
        "steps_completed": payload["steps_completed"],
        "last_loss": payload["training_loss"]["last"],
        "validation_loss_mean": payload["validation_loss_mean"],
        "elapsed_seconds": payload["elapsed_seconds"],
    }))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("calibrate", "train"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--direction", type=Path, required=True)
    parser.add_argument("--mode", default="ZERO")
    parser.add_argument("--multiplier", type=float, default=0.0)
    parser.add_argument("--stream", type=int, default=0)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=16)
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "/data1/tzh/cache/kernel-analyzer/single_point_collapse_v1/checkpoint.pt"
    ))
    parser.add_argument("--no-checkpoint", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    if args.command == "calibrate":
        calibration(args, protocol)
    else:
        if args.multiplier < 0.0:
            raise ValueError("multiplier must be nonnegative")
        train(args, protocol)


if __name__ == "__main__":
    main()
