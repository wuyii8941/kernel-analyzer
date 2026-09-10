#!/usr/bin/env python3
"""Diagnose blockwise AdamW8bit state residuals on saved real gradients.

This is a result-aware development analysis, not a prospective repair result.
It reproduces TorchAO's documented block scaling, nearest-code selection and
AdamW equations on CPU, then measures how much quantization error is a shared
offset within each block.  A low-memory block-mean correction is simulated to
decide whether it merits an actual optimizer implementation and new training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torchao.optim.quant_utils import (
    quantize_8bit_with_qmap,
    scale_tensor,
)
from torchao.optim.subclass_8bit import get_qmap_signed, get_qmap_unsigned


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/property/numerical_coverage_v1/torchao_adamw8bit_mamba_v2"
GRADIENTS = SOURCE / "real_gradient_inputs.pt"
SOURCE_PROTOCOL = SOURCE / "protocol.json"
TORCHAO_QUANT = Path(
    "/data1/tzh/envs/liger/lib/python3.10/site-packages/torchao/optim/quant_utils.py"
)
TORCHAO_STATE = Path(
    "/data1/tzh/envs/liger/lib/python3.10/site-packages/torchao/optim/subclass_8bit.py"
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantize_dequantize(value: torch.Tensor, qmap: torch.Tensor, block_size: int) -> torch.Tensor:
    scaled, scale = scale_tensor(value, block_size)
    codes = quantize_8bit_with_qmap(scaled, qmap)
    return (qmap[codes.int()].view(-1, block_size) * scale[:, None]).view_as(value)


def _block_mean(value: torch.Tensor, block_size: int) -> torch.Tensor:
    return value.view(-1, block_size).mean(dim=1, keepdim=True).expand(-1, block_size).reshape_as(value)


def _block_scale_corrected(
    exact: torch.Tensor, quantized: torch.Tensor, block_size: int
) -> torch.Tensor:
    exact_blocks = exact.view(-1, block_size).double()
    quantized_blocks = quantized.view(-1, block_size).double()
    denominator = torch.sum(quantized_blocks.square(), dim=1, keepdim=True)
    gain = torch.where(
        denominator > 0.0,
        torch.sum(exact_blocks * quantized_blocks, dim=1, keepdim=True) / denominator,
        torch.ones_like(denominator),
    )
    return (quantized_blocks * gain).reshape_as(exact_blocks).to(exact.dtype).reshape_as(exact)


def _energy(value: torch.Tensor) -> float:
    return float(torch.sum(value.double() * value.double()).item())


def _inner(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.sum(left.double() * right.double()).item())


def _state_metrics(exact: torch.Tensor, stored: torch.Tensor, block_size: int) -> dict:
    residual = stored - exact
    common = _block_mean(residual, block_size)
    remainder = residual - common
    scale_corrected = _block_scale_corrected(exact, stored, block_size)
    residual_energy = _energy(residual)
    exact_energy = _energy(exact)
    return {
        "relative_rms": math.sqrt(residual_energy / exact_energy) if exact_energy > 0.0 else 0.0,
        "aligned_ratio": _inner(residual, exact) / exact_energy if exact_energy > 0.0 else 0.0,
        "block_common_energy_fraction": (
            _energy(common) / residual_energy if residual_energy > 0.0 else 0.0
        ),
        "block_scale_correction_error_reduction_fraction": (
            1.0 - _energy(scale_corrected - exact) / residual_energy
            if residual_energy > 0.0
            else 0.0
        ),
        "signed_residual_mean": float(residual.double().mean().item()),
        "positive_residual_fraction": float((residual > 0).double().mean().item()),
        "residual_energy": residual_energy,
        "block_remainder_energy": _energy(remainder),
    }


def _simulate(gradients: list[torch.Tensor], base: torch.Tensor, block_size: int) -> dict:
    beta1, beta2 = 0.9, 0.999
    learning_rate, epsilon, weight_decay = 1e-3, 1e-8, 0.01
    signed_qmap = torch.tensor(get_qmap_signed(), dtype=torch.float32)
    unsigned_qmap = torch.tensor(get_qmap_unsigned(), dtype=torch.float32)

    reference_m = torch.zeros_like(base)
    reference_v = torch.zeros_like(base)
    default_m = torch.zeros_like(base)
    default_v = torch.zeros_like(base)
    corrected_m = torch.zeros_like(base)
    corrected_v = torch.zeros_like(base)
    scale_corrected_m = torch.zeros_like(base)
    rows = []

    for index, gradient in enumerate(gradients, start=1):
        gradient = gradient.float()
        reference_m = reference_m.lerp(gradient, 1.0 - beta1)
        reference_v = reference_v.lerp(gradient.square(), 1.0 - beta2)
        default_high_m = default_m.lerp(gradient, 1.0 - beta1)
        default_high_v = default_v.lerp(gradient.square(), 1.0 - beta2)
        corrected_high_m = corrected_m.lerp(gradient, 1.0 - beta1)
        corrected_high_v = corrected_v.lerp(gradient.square(), 1.0 - beta2)
        scale_corrected_high_m = scale_corrected_m.lerp(gradient, 1.0 - beta1)

        default_stored_m = _quantize_dequantize(default_high_m, signed_qmap, block_size)
        default_stored_v = _quantize_dequantize(default_high_v, unsigned_qmap, block_size)
        corrected_quant_m = _quantize_dequantize(corrected_high_m, signed_qmap, block_size)
        scale_corrected_quant_m = _quantize_dequantize(
            scale_corrected_high_m, signed_qmap, block_size
        )
        corrected_stored_m = corrected_quant_m + _block_mean(
            corrected_high_m - corrected_quant_m, block_size
        )
        # Only the signed first moment is corrected.  Adding a shared residual
        # to the nonnegative second moment can make reconstructed values
        # negative and is not a legal Adam denominator without another rule.
        corrected_stored_v = _quantize_dequantize(
            corrected_high_v, unsigned_qmap, block_size
        )
        scale_corrected_stored_m = _block_scale_corrected(
            scale_corrected_high_m, scale_corrected_quant_m, block_size
        )

        correction1 = 1.0 - beta1**index
        correction2 = 1.0 - beta2**index

        def parameter_write(moment1: torch.Tensor, moment2: torch.Tensor) -> torch.Tensor:
            decayed = base - learning_rate * weight_decay * base
            after = decayed - learning_rate * (moment1 / correction1) / (
                moment2.sqrt() / math.sqrt(correction2) + epsilon
            )
            return after - base

        reference_write = parameter_write(reference_m, reference_v)
        default_write = parameter_write(default_high_m, default_high_v)
        corrected_write = parameter_write(corrected_high_m, corrected_high_v)
        scale_corrected_write = parameter_write(
            scale_corrected_high_m, corrected_high_v
        )
        repair_energy = _energy(reference_write)
        rows.append({
            "step": index,
            "default_first_moment_quantization": _state_metrics(
                default_high_m, default_stored_m, block_size
            ),
            "default_second_moment_quantization": _state_metrics(
                default_high_v, default_stored_v, block_size
            ),
            "parameter_write_relative_rms": {
                "default": math.sqrt(_energy(default_write - reference_write) / repair_energy),
                "first_moment_block_mean_corrected": math.sqrt(
                    _energy(corrected_write - reference_write) / repair_energy
                ),
                "first_moment_block_scale_corrected": math.sqrt(
                    _energy(scale_corrected_write - reference_write) / repair_energy
                ),
            },
        })
        default_m, default_v = default_stored_m, default_stored_v
        corrected_m, corrected_v = corrected_stored_m, corrected_stored_v
        scale_corrected_m = scale_corrected_stored_m

    return {"rows": rows}


def _ratio_of_sums(rows: list[dict], key: str) -> float:
    numerator = math.fsum(row[key]["residual_energy"] - row[key]["block_remainder_energy"] for row in rows)
    denominator = math.fsum(row[key]["residual_energy"] for row in rows)
    return numerator / denominator if denominator > 0.0 else 0.0


def _scale_reduction_ratio_of_sums(rows: list[dict], key: str) -> float:
    denominator = math.fsum(row[key]["residual_energy"] for row in rows)
    numerator = math.fsum(
        row[key]["residual_energy"]
        * row[key]["block_scale_correction_error_reduction_fraction"]
        for row in rows
    )
    return numerator / denominator if denominator > 0.0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=256)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.output.exists() or not output.is_relative_to(Path("/data1/tzh")):
        parser.error("use a new output below /data1/tzh")
    if args.block_size <= 0:
        parser.error("block size must be positive")

    saved = torch.load(GRADIENTS, map_location="cpu", weights_only=True)
    base = saved["base_parameter"].float().contiguous()
    gradients = [value.float().contiguous() for value in saved["gradients"]]
    if base.numel() % args.block_size:
        parser.error("the target parameter must be divisible by block size")
    simulation = _simulate(gradients, base, args.block_size)
    rows = simulation["rows"]
    confirmation = rows[len(rows) // 2 :]
    default_rms = math.sqrt(math.fsum(
        row["parameter_write_relative_rms"]["default"] ** 2 for row in confirmation
    ) / len(confirmation))
    corrected_rms = math.sqrt(math.fsum(
        row["parameter_write_relative_rms"]["first_moment_block_mean_corrected"] ** 2
        for row in confirmation
    ) / len(confirmation))
    scale_corrected_rms = math.sqrt(math.fsum(
        row["parameter_write_relative_rms"]["first_moment_block_scale_corrected"] ** 2
        for row in confirmation
    ) / len(confirmation))
    payload = {
        "schema": "adamw8bit-block-residual-development-v1",
        "status": "COMPLETE",
        "data_use": "RESULT_AWARE_DEVELOPMENT_DIAGNOSTIC_NOT_CONFIRMATION",
        "candidate_math": (
            "CPU replay of TorchAO block scaling, nearest qmap code, dequantization, "
            "and AdamW equations on 32 saved real gradients"
        ),
        "actual_triton_execution_in_this_artifact": False,
        "block_size": args.block_size,
        "state_count": len(rows),
        "confirmation_steps": [row["step"] for row in confirmation],
        "aggregate": {
            "first_moment_error_energy_in_block_common_offset": _ratio_of_sums(
                confirmation, "default_first_moment_quantization"
            ),
            "second_moment_error_energy_in_block_common_offset": _ratio_of_sums(
                confirmation, "default_second_moment_quantization"
            ),
            "first_moment_local_error_reduction_from_block_scale": _scale_reduction_ratio_of_sums(
                confirmation, "default_first_moment_quantization"
            ),
            "second_moment_local_error_reduction_from_block_scale": _scale_reduction_ratio_of_sums(
                confirmation, "default_second_moment_quantization"
            ),
            "parameter_write_relative_rms_default": default_rms,
            "parameter_write_relative_rms_first_moment_block_mean_corrected": corrected_rms,
            "block_mean_relative_rms_reduction_fraction": (
                1.0 - corrected_rms / default_rms if default_rms > 0.0 else 0.0
            ),
            "parameter_write_relative_rms_first_moment_block_scale_corrected": scale_corrected_rms,
            "block_scale_relative_rms_reduction_fraction": (
                1.0 - scale_corrected_rms / default_rms if default_rms > 0.0 else 0.0
            ),
            "extra_fp32_values_for_first_moment": base.numel() // args.block_size,
            "extra_bytes_for_first_moment": (base.numel() // args.block_size) * 4,
        },
        "rows": rows,
        "decision_boundary": (
            "A favorable simulation only nominates a modification. It cannot establish "
            "actual optimizer behavior or training improvement."
        ),
        "source_sha256": {
            str(path): _sha(path)
            for path in (
                Path(__file__), GRADIENTS, SOURCE_PROTOCOL, TORCHAO_QUANT, TORCHAO_STATE
            )
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(payload["aggregate"], indent=2))


if __name__ == "__main__":
    main()
