#!/usr/bin/env python3
"""Reconstruct stored-moment and parameter-write differences step by step."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def difference_rms(left, right) -> float:
    numerator = float((left - right).detach().double().square().sum().item())
    denominator = float(right.detach().double().square().sum().item())
    return math.sqrt(numerator / max(denominator, 1e-300))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT):
        parser.error("choose a new output inside kernel-analyzer")

    import torch
    from torchao.optim.subclass_8bit import OptimState8bit
    from kernel_analyzer.compensation_control import TensorScalarCompensationControl
    from kernel_analyzer.moment_reconstruction import (
        reconstruction_terms, update_component_terms,
    )

    source_paths = [
        Path(__file__).resolve(), Path(inspect.getfile(TensorScalarCompensationControl)),
        Path(inspect.getfile(OptimState8bit)),
        ROOT / "src/kernel_analyzer/moment_reconstruction.py",
    ]
    settings = dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    protocol = {
        "schema": "adamw8bit-moment-reconstruction-v1",
        "data_use": "CONTROLLED_FIXED_GRADIENT_MECHANISM_VALIDATION",
        "sizes": [4096, 32768], "seeds": [11, 29], "steps": 8,
        "conditions": ["COMPENSATION_OFF", "COMPENSATION_ON"],
        "settings": settings, "device": args.device,
        "question": (
            "Do measured storage errors reconstruct later first/second moment "
            "differences, and how do both moments enter the actual write?"
        ),
        "claim_boundary": [
            "FIXED_SYNTHETIC_GRADIENT_HISTORIES", "NO_NONZERO_MEAN_CLAIM",
            "NO_TRAINING_LOSS_CLAIM",
        ],
        "source_sha256": {str(path): sha(path) for path in source_paths},
    }
    output.mkdir(parents=True)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    def read_state(optimizer, parameter, key):
        state = optimizer.state[parameter]
        compensation = state[key + "_compensation"]
        return optimizer._read(state[key], compensation)

    def quantized_only(optimizer, parameter, key):
        value = optimizer.state[parameter][key]
        return value.dequantize() if isinstance(value, OptimState8bit) else value.float()

    def initialize(optimizer, parameter):
        state = optimizer.state[parameter]
        if state:
            return
        state["step"] = torch.tensor(0.0)
        state["exp_avg"], state["exp_avg_compensation"] = optimizer._new_state(
            parameter, signed=True,
        )
        state["exp_avg_sq"], state["exp_avg_sq_compensation"] = optimizer._new_state(
            parameter, signed=False,
        )

    def proposed_write(base, first, second, *, step, group):
        beta1, beta2 = group["betas"]
        updated = base.float()
        if group["weight_decay"]:
            updated = updated - group["lr"] * group["weight_decay"] * updated
        bias1 = 1.0 - beta1 ** step
        bias2 = 1.0 - beta2 ** step
        denominator = second.sqrt() / bias2 ** 0.5 + group["eps"]
        updated = updated - group["lr"] * (first / bias1) / denominator
        return updated - base

    rows = []
    for size in protocol["sizes"]:
        for seed in protocol["seeds"]:
            generator = torch.Generator(device=args.device).manual_seed(seed)
            base = torch.randn(size, generator=generator, device=args.device)
            parameters = {
                mode: torch.nn.Parameter(base.clone()) for mode in ("OFF", "ON")
            }
            optimizers = {
                mode: TensorScalarCompensationControl(
                    [parameters[mode]], compensation_enabled=mode == "ON", **settings,
                ) for mode in parameters
            }
            reference_first = torch.zeros_like(base)
            reference_second = torch.zeros_like(base)
            for step in range(1, protocol["steps"] + 1):
                gradient = torch.randn(
                    size, generator=generator, device=args.device,
                ) * torch.logspace(-4, 0, size, device=args.device)
                reference_first_previous = reference_first
                reference_second_previous = reference_second
                reference_first = reference_first_previous.lerp(
                    gradient, 1.0 - settings["betas"][0],
                )
                reference_second = reference_second_previous.lerp(
                    gradient.square(), 1.0 - settings["betas"][1],
                )
                mode_rows = {}
                for mode, parameter in parameters.items():
                    optimizer = optimizers[mode]
                    initialize(optimizer, parameter)
                    previous_first = read_state(optimizer, parameter, "exp_avg").clone()
                    previous_second = read_state(optimizer, parameter, "exp_avg_sq").clone()
                    candidate_first = previous_first.lerp(
                        gradient, 1.0 - settings["betas"][0],
                    )
                    candidate_second = previous_second.lerp(
                        gradient.square(), 1.0 - settings["betas"][1],
                    )
                    with torch.no_grad():
                        parameter.copy_(base)
                    parameter.grad = gradient.clone()
                    optimizer.step()
                    actual_write = parameter.detach() - base
                    current_first = read_state(optimizer, parameter, "exp_avg")
                    current_second = read_state(optimizer, parameter, "exp_avg_sq")
                    group = optimizer.param_groups[0]
                    predicted_write = proposed_write(
                        base, candidate_first, candidate_second, step=step, group=group,
                    )
                    reference_write = proposed_write(
                        base, reference_first, reference_second, step=step, group=group,
                    )
                    first_only = proposed_write(
                        base, candidate_first, reference_second, step=step, group=group,
                    )
                    second_only = proposed_write(
                        base, reference_first, candidate_second, step=step, group=group,
                    )
                    mode_rows[mode] = {
                        "first_moment": reconstruction_terms(
                            reference_previous=reference_first_previous,
                            reference_current=reference_first,
                            candidate_previous_read=previous_first,
                            candidate_current_unstored=candidate_first,
                            candidate_current_read=current_first,
                            beta=settings["betas"][0],
                        ),
                        "second_moment": reconstruction_terms(
                            reference_previous=reference_second_previous,
                            reference_current=reference_second,
                            candidate_previous_read=previous_second,
                            candidate_current_unstored=candidate_second,
                            candidate_current_read=current_second,
                            beta=settings["betas"][1],
                        ),
                        "write_components": update_component_terms(
                            reference=reference_write, candidate=predicted_write,
                            first_only=first_only, second_only=second_only,
                        ),
                        "predicted_write_matches_actual_relative_rms": difference_rms(
                            predicted_write, actual_write,
                        ),
                        "quantized_first_storage_energy": float((
                            quantized_only(optimizer, parameter, "exp_avg")
                            - candidate_first
                        ).double().square().sum().item()),
                        "quantized_second_storage_energy": float((
                            quantized_only(optimizer, parameter, "exp_avg_sq")
                            - candidate_second
                        ).double().square().sum().item()),
                    }
                rows.append({"size": size, "seed": seed, "step": step, "conditions": mode_rows})

    summary = {}
    for mode in ("OFF", "ON"):
        summary[mode] = {}
        for moment in ("first_moment", "second_moment"):
            values = [row["conditions"][mode][moment] for row in rows]
            summary[mode][moment] = {
                "maximum_relative_reconstruction_residual": max(
                    value["relative_reconstruction_residual"] for value in values
                ),
                "mean_storage_error_energy": math.fsum(
                    value["storage_error_energy"] for value in values
                ) / len(values),
                "mean_actual_error_energy": math.fsum(
                    value["actual_error_energy"] for value in values
                ) / len(values),
            }
        components = [row["conditions"][mode]["write_components"] for row in rows]
        summary[mode]["mean_write_component_relative_rms"] = {
            key: math.fsum(value[key] for value in components) / len(components)
            for key in (
                "total_relative_rms", "first_moment_only_relative_rms",
                "second_moment_only_relative_rms", "joint_nonlinear_remainder_relative_rms",
            )
        }
        summary[mode]["maximum_predicted_write_mismatch_relative_rms"] = max(
            row["conditions"][mode]["predicted_write_matches_actual_relative_rms"]
            for row in rows
        )
    result = {
        "schema": protocol["schema"], "status": "COMPLETE",
        "row_count": len(rows), "rows": rows, "summary": summary,
        "interpretation": (
            "The algebraic recurrence is reconstructed including the measured "
            "floating remainder; no expectation, independence, or loss claim is made."
        ),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "summary": summary}))


if __name__ == "__main__":
    main()
