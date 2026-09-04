#!/usr/bin/env python3
"""Measure loss sensitivity along one saved candidate--repair displacement.

This is a controlled local stress test.  It does not claim that future
training will follow any of the evaluated straight-line parameter paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tensor(path: Path, parameter: str | None = None) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        if parameter is not None:
            if parameter not in value:
                raise KeyError(f"{parameter!r} is absent from {path}")
            value = value[parameter]
        elif len(value) == 1:
            value = next(iter(value.values()))
        else:
            raise ValueError(f"{path} contains a state dictionary; select one tensor before use")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{path} does not contain a tensor")
    return value.detach().float().contiguous()


def evaluate_loss(model, states: list[dict], device: torch.device) -> tuple[float, list[float]]:
    values: list[float] = []
    with torch.no_grad():
        for state in states:
            token_ids = state.get("token_ids", state.get("input_ids"))
            if token_ids is None:
                raise KeyError("each evaluation state must contain token_ids or input_ids")
            ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            loss = model(input_ids=ids, labels=ids, use_cache=False).loss
            values.append(float(loss.detach().float().cpu()))
    return sum(values) / len(values), values


def normalized_like(value: torch.Tensor, target_norm: float) -> torch.Tensor:
    norm = float(torch.linalg.vector_norm(value.double()).item())
    if norm == 0.0:
        raise ValueError("comparison direction has zero norm")
    return value * (target_norm / norm)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--parameter", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--eval-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scales", default="1,10,100,1000")
    parser.add_argument("--random-seeds", default="1701,1702,1703,1704")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("host GPU required")
    scales = [float(item) for item in args.scales.split(",")]
    if any(scale <= 0.0 for scale in scales):
        raise ValueError("all scales must be positive")
    seeds = [int(item) for item in args.random_seeds.split(",")]
    if not seeds:
        raise ValueError("at least one random seed is required")

    bank = json.loads(args.eval_bank.read_text())
    states = bank.get("states", bank.get("records"))
    if not states:
        raise ValueError("evaluation bank is empty")
    candidate = load_tensor(args.candidate, args.parameter)
    repair = load_tensor(args.repair, args.parameter)
    if candidate.shape != repair.shape:
        raise ValueError("candidate and repair tensors have different shapes")

    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
    ).to(device).eval()
    parameters = dict(model.named_parameters())
    if args.parameter not in parameters:
        raise KeyError(f"unknown parameter: {args.parameter}")
    parameter = parameters[args.parameter]
    initial = parameter.detach().cpu().float().clone()
    if initial.shape != repair.shape:
        raise ValueError("saved tensor shape does not match model parameter")

    displacement = candidate - repair
    displacement_norm = float(torch.linalg.vector_norm(displacement.double()).item())
    if displacement_norm == 0.0:
        raise ValueError("candidate and repair tensors are identical")
    normal_training_direction = normalized_like(repair - initial, displacement_norm)

    directions: dict[str, torch.Tensor] = {
        "measured_direction": displacement,
        "opposite_direction": -displacement,
        "normal_training_direction": normal_training_direction,
    }
    for seed in seeds:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        random_direction = torch.randn(displacement.shape, generator=generator, dtype=torch.float32)
        directions[f"random_direction_seed_{seed}"] = normalized_like(
            random_direction, displacement_norm
        )

    with torch.no_grad():
        parameter.copy_(repair.to(device=device, dtype=parameter.dtype))
    baseline_mean, baseline_values = evaluate_loss(model, states, device)
    rows = []
    for name, direction in directions.items():
        for scale in scales:
            value = repair + scale * direction
            with torch.no_grad():
                parameter.copy_(value.to(device=device, dtype=parameter.dtype))
            mean_loss, per_state = evaluate_loss(model, states, device)
            paired = [loss - base for loss, base in zip(per_state, baseline_values)]
            rows.append({
                "direction": name,
                "scale": scale,
                "mean_loss": mean_loss,
                "mean_loss_change": mean_loss - baseline_mean,
                "mean_absolute_paired_loss_change": sum(abs(x) for x in paired) / len(paired),
                "maximum_absolute_paired_loss_change": max(abs(x) for x in paired),
                "positive_loss_change_count": sum(x > 0.0 for x in paired),
                "negative_loss_change_count": sum(x < 0.0 for x in paired),
                "zero_loss_change_count": sum(x == 0.0 for x in paired),
            })

    payload = {
        "schema": "kernel-analyzer-loss-direction-stress-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "parameter": args.parameter,
        "candidate": {"path": str(args.candidate), "sha256": file_sha256(args.candidate)},
        "repair": {"path": str(args.repair), "sha256": file_sha256(args.repair)},
        "evaluation_bank": {"path": str(args.eval_bank), "sha256": file_sha256(args.eval_bank)},
        "evaluation_state_count": len(states),
        "baseline_repair_loss": baseline_mean,
        "candidate_minus_repair_l2": displacement_norm,
        "scales": scales,
        "random_seeds": seeds,
        "rows": rows,
        "claim_boundary": (
            "This is a controlled loss sensitivity test around the saved repair parameter. "
            "It does not claim that future training follows a straight parameter-space line."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "rows": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
