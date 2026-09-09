#!/usr/bin/env python3
"""Test which quantized AdamW moment drives the recorded parameter-write effect."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/property/numerical_coverage_v1/torchao_adamw8bit_mamba_v2"
GRADIENTS = SOURCE / "real_gradient_inputs.pt"
SOURCE_PROTOCOL = SOURCE / "protocol.json"
VARIANTS = ("DEFAULT_8BIT", "FP32_FIRST_MOMENT", "FP32_SECOND_MOMENT")
STAGES = ("MOMENT1_STATE", "MOMENT2_STATE", "PARAMETER_WRITE")


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def save_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def freeze(output: Path) -> None:
    if output.exists():
        raise ValueError("freeze requires a new output directory")
    prior = load(SOURCE_PROTOCOL)
    protocol = {
        "schema": "optimizer-state-component-mechanism-v1",
        "status": "FROZEN_BEFORE_COMPONENT_VARIANTS",
        "case_id": "mamba_x_proj_adamw8bit_moment_components",
        "variants": list(VARIANTS),
        "block_size": 256,
        "reference": "torch.optim.AdamW with FP32 moments",
        "target_parameter": prior["target_parameter"],
        "state_ids": prior["state_ids"],
        "calibration_state_ids": prior["calibration_state_ids"],
        "confirmation_state_ids": prior["confirmation_state_ids"],
        "optimizer": prior["optimizer"],
        "gradient_source": str(GRADIENTS),
        "prediction_fixed_before_component_variants": {
            "identity_checks": (
                "FP32_FIRST_MOMENT has zero MOMENT1_STATE difference; "
                "FP32_SECOND_MOMENT has zero MOMENT2_STATE difference"
            ),
            "primary": (
                "FP32_FIRST_MOMENT produces smaller confirmation PARAMETER_WRITE RMS "
                "difference than DEFAULT_8BIT"
            ),
            "reason": (
                "The prior fixed-gradient decomposition measured substantially larger "
                "first-moment than second-moment distortion"
            ),
            "success_rule": (
                "both identity checks hold and PARAMETER_WRITE RMS(FP32_FIRST_MOMENT) "
                "is below PARAMETER_WRITE RMS(DEFAULT_8BIT)"
            ),
        },
        "data_use": "RESULT_AWARE_SECOND_MECHANISM_FOLLOWUP_AFTER_BLOCK64_TRAINING_NOT_CONFIRMED",
        "claim_scope": "FIXED_RECORDED_GRADIENT_SEQUENCE",
        "source_sha256": {
            str(path.resolve()): sha(path)
            for path in (
                Path(__file__).resolve(), GRADIENTS, SOURCE_PROTOCOL,
                ROOT / "src/kernel_analyzer/optimizer_state_repairs.py",
                ROOT / "src/kernel_analyzer/optimizer_implementation_capture.py",
            )
        },
    }
    save_new(output / "protocol.json", protocol)


def verify(output: Path) -> dict[str, Any]:
    protocol = load(output / "protocol.json")
    if protocol.get("schema") != "optimizer-state-component-mechanism-v1":
        raise ValueError("unexpected protocol")
    if tuple(protocol.get("variants", ())) != VARIANTS:
        raise ValueError("component variants changed")
    for name, expected in protocol["source_sha256"].items():
        path = Path(name)
        if not path.is_file() or sha(path) != expected:
            raise ValueError("frozen dependency changed: " + name)
    return protocol


def dequantized(value):
    if hasattr(value, "dequantize"):
        return value.dequantize().detach().float().clone()
    return value.detach().float().clone()


def optimizer_for(variant: str, parameter, settings):
    from torchao.optim import AdamW8bit
    from kernel_analyzer.optimizer_state_repairs import (
        AdamWFirstMomentFP32,
        AdamWSecondMomentFP32,
    )

    classes = {
        "DEFAULT_8BIT": AdamW8bit,
        "FP32_FIRST_MOMENT": AdamWFirstMomentFP32,
        "FP32_SECOND_MOMENT": AdamWSecondMomentFP32,
    }
    return classes[variant]([parameter], block_size=256, **settings)


def transition_sequence(base, gradients, variant: str, device: str):
    import torch

    settings = dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                    weight_decay=0.01, amsgrad=False)
    candidate_parameter = torch.nn.Parameter(base.to(device).clone())
    reference_parameter = torch.nn.Parameter(base.to(device).clone())
    candidate_optimizer = optimizer_for(variant, candidate_parameter, settings)
    reference_optimizer = torch.optim.AdamW(
        [reference_parameter], foreach=False, fused=False, **settings
    )
    rows = []
    base_device = base.to(device)
    for index, gradient_cpu in enumerate(gradients):
        gradient = gradient_cpu.to(device)
        with torch.no_grad():
            candidate_parameter.copy_(base_device)
            reference_parameter.copy_(base_device)
        candidate_parameter.grad = gradient.clone()
        reference_parameter.grad = gradient.clone()
        candidate_optimizer.step()
        reference_optimizer.step()
        torch.cuda.synchronize()
        candidate_state = candidate_optimizer.state[candidate_parameter]
        reference_state = reference_optimizer.state[reference_parameter]
        rows.append({
            "GRADIENT_INPUT": (gradient.detach().cpu(), gradient.detach().cpu()),
            "MOMENT1_STATE": (
                dequantized(candidate_state["exp_avg"]).cpu(),
                dequantized(reference_state["exp_avg"]).cpu(),
            ),
            "MOMENT2_STATE": (
                dequantized(candidate_state["exp_avg_sq"]).cpu(),
                dequantized(reference_state["exp_avg_sq"]).cpu(),
            ),
            "PARAMETER_WRITE": (
                (candidate_parameter.detach() - base_device).cpu(),
                (reference_parameter.detach() - base_device).cpu(),
            ),
        })
        print(json.dumps({"event": "MOMENT_COMPONENT_TRANSITION", "variant": variant,
                          "state": index + 1}), flush=True)
    return rows


def digest_sequence(rows):
    from kernel_analyzer.optimizer_implementation_capture import tensor_digest

    return [{stage + ":candidate": tensor_digest(pair[0]) for stage, pair in row.items()} |
            {stage + ":reference": tensor_digest(pair[1]) for stage, pair in row.items()}
            for row in rows]


def summarize(protocol: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    if set(raw.get("variants", {})) != set(VARIANTS):
        raise ValueError("component variants are incomplete")
    values = {
        name: {stage: float(value) for stage, value in row["confirmation_relative_rms"].items()}
        for name, row in raw["variants"].items()
    }
    identities = {
        "FP32_FIRST_MOMENT_MOMENT1_ZERO": values["FP32_FIRST_MOMENT"]["MOMENT1_STATE"] == 0,
        "FP32_SECOND_MOMENT_MOMENT2_ZERO": values["FP32_SECOND_MOMENT"]["MOMENT2_STATE"] == 0,
    }
    primary = (values["FP32_FIRST_MOMENT"]["PARAMETER_WRITE"]
               < values["DEFAULT_8BIT"]["PARAMETER_WRITE"])
    success = all(identities.values()) and primary
    return {
        "schema": "optimizer-state-component-mechanism-summary-v1",
        "case_id": protocol["case_id"],
        "frozen_prediction": protocol["prediction_fixed_before_component_variants"],
        "confirmation_relative_rms": values,
        "identity_checks": identities,
        "primary_parameter_write_reduction": primary,
        "prediction_result": "CONFIRMED" if success else "NOT_CONFIRMED",
        "recommended_training_variant": "FP32_FIRST_MOMENT" if success else None,
        "scope": "Fixed recorded Mamba gradient sequence; no new training claim",
    }


def capture(output: Path, device: str) -> None:
    import torch
    from kernel_analyzer.optimizer_implementation_capture import (
        OptimizerImplementationCapture,
        fixed_suite_total_rms,
    )

    protocol = verify(output)
    saved = torch.load(GRADIENTS, map_location="cpu", weights_only=True)
    base = saved["base_parameter"].float()
    gradients = [value.float() for value in saved["gradients"]]
    if [str(value) for value in saved["state_ids"]] != protocol["state_ids"]:
        raise ValueError("gradient state IDs differ from protocol")
    variants = {}
    for variant in VARIANTS:
        first = transition_sequence(base, gradients, variant, device)
        second = transition_sequence(base, gradients, variant, device)
        first_digest = digest_sequence(first)
        if first_digest != digest_sequence(second):
            raise RuntimeError("determinism replay failed for " + variant)
        collector = OptimizerImplementationCapture(protocol["state_ids"])
        for row in first:
            collector.append(row)
        statistics, profiles = collector.finish()
        variants[variant] = {
            "confirmation_relative_rms": {
                stage: fixed_suite_total_rms(statistics[stage]) for stage in STAGES
            },
            "profiles": profiles,
            "transition_digests": first_digest,
        }
    raw = {
        "schema": "optimizer-state-component-mechanism-raw-v1",
        "status": "COMPLETE",
        "protocol_sha256": sha(output / "protocol.json"),
        "variants": variants,
    }
    save_new(output / "raw.json", raw)
    save_new(output / "summary.json", summarize(protocol, raw))


def report(output: Path) -> None:
    save_new(output / "recomputed_summary.json", summarize(verify(output), load(output / "raw.json")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "capture", "report"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = args.output_root.resolve()
    if not output.is_relative_to(Path("/data1/tzh")):
        raise ValueError("output must remain under /data1/tzh")
    if args.command == "freeze":
        freeze(output)
    elif args.command == "capture":
        capture(output, args.device)
    else:
        report(output)


if __name__ == "__main__":
    main()
