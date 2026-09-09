#!/usr/bin/env python3
"""Test a frozen block-size prediction for TorchAO AdamW8bit.

The experiment reuses the exact real-gradient sequence from the completed
optimizer-family measurement.  It changes only the reviewed moment
quantization block size and compares every variant with torch.optim.AdamW.
This is a result-aware mechanism follow-up, not a new blind case or a training
quality experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/property/numerical_coverage_v1/torchao_adamw8bit_mamba_v2"
GRADIENTS = SOURCE / "real_gradient_inputs.pt"
SOURCE_PROTOCOL = SOURCE / "protocol.json"
BLOCK_SIZES = (64, 256, 1024)
STAGES = ("MOMENT1_STATE", "MOMENT2_STATE", "PARAMETER_WRITE")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    if prior.get("schema") != "optimizer-implementation-analysis-v1":
        raise ValueError("completed source protocol is unavailable")
    protocol = {
        "schema": "optimizer-quantization-mechanism-v1",
        "status": "FROZEN_BEFORE_BLOCK_SIZE_VARIANTS",
        "case_id": "mamba_x_proj_adamw8bit_block_size",
        "operator_family": "OPTIMIZER_PARAMETER_UPDATE",
        "candidate_implementation": "torchao.optim.AdamW8bit",
        "reference_implementation": "torch.optim.AdamW",
        "block_sizes": list(BLOCK_SIZES),
        "baseline_block_size": 256,
        "target_parameter": prior["target_parameter"],
        "state_ids": prior["state_ids"],
        "calibration_state_ids": prior["calibration_state_ids"],
        "confirmation_state_ids": prior["confirmation_state_ids"],
        "optimizer": prior["optimizer"],
        "gradient_source": str(GRADIENTS),
        "comparison": (
            "same recorded real gradients and same FP32 parameter; each optimizer "
            "keeps its own moment history; the parameter is restored before every step"
        ),
        "prediction_fixed_before_variants": {
            "primary": (
                "On the confirmation states, finer moment-quantization blocks produce "
                "smaller MOMENT1_STATE and PARAMETER_WRITE RMS differences from FP32 AdamW"
            ),
            "success_rule": (
                "RMS(block=64) < RMS(block=256) < RMS(block=1024) for both "
                "MOMENT1_STATE and PARAMETER_WRITE"
            ),
            "reason": (
                "Each block shares one quantization scale; smaller blocks restrict the "
                "value range represented by a scale and are predicted to reduce distortion"
            ),
        },
        "data_use": "RESULT_AWARE_MECHANISM_FOLLOWUP_NOT_BLIND_DISCOVERY",
        "claim_scope": "FIXED_RECORDED_GRADIENT_SEQUENCE",
        "training_outcome_established": False,
        "source_sha256": {
            str(path.resolve()): sha(path)
            for path in (
                Path(__file__).resolve(), GRADIENTS, SOURCE_PROTOCOL,
                ROOT / "src/kernel_analyzer/optimizer_implementation_capture.py",
            )
        },
    }
    save_new(output / "protocol.json", protocol)


def verify(output: Path) -> dict[str, Any]:
    protocol = load(output / "protocol.json")
    if protocol.get("schema") != "optimizer-quantization-mechanism-v1":
        raise ValueError("unexpected protocol")
    if tuple(protocol.get("block_sizes", ())) != BLOCK_SIZES:
        raise ValueError("block-size plan changed")
    for name, expected in protocol.get("source_sha256", {}).items():
        path = Path(name)
        if not path.is_file() or sha(path) != expected:
            raise ValueError("frozen source changed: " + name)
    return protocol


def dequantized(value):
    if hasattr(value, "dequantize"):
        return value.dequantize().detach().float().clone()
    return value.detach().float().clone()


def transition_sequence(base, gradients, block_size: int, device: str):
    import torch
    from torchao.optim import AdamW8bit

    settings = dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                    weight_decay=0.01, amsgrad=False)
    candidate_parameter = torch.nn.Parameter(base.to(device).clone())
    reference_parameter = torch.nn.Parameter(base.to(device).clone())
    candidate_optimizer = AdamW8bit(
        [candidate_parameter], block_size=block_size, **settings
    )
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
        print(json.dumps({"event": "BLOCK_SIZE_TRANSITION", "block_size": block_size,
                          "state": index + 1}), flush=True)
    return rows


def digest_sequence(rows) -> list[dict[str, str]]:
    from kernel_analyzer.optimizer_implementation_capture import tensor_digest
    return [{stage + ":candidate": tensor_digest(pair[0]) for stage, pair in row.items()} |
            {stage + ":reference": tensor_digest(pair[1]) for stage, pair in row.items()}
            for row in rows]


def capture(output: Path, device: str) -> None:
    import torch
    from kernel_analyzer.optimizer_implementation_capture import (
        OptimizerImplementationCapture, fixed_suite_total_rms,
    )

    protocol = verify(output)
    if (output / "raw.json").exists():
        raise ValueError("capture refuses to overwrite results")
    saved = torch.load(GRADIENTS, map_location="cpu", weights_only=True)
    base = saved["base_parameter"].float()
    gradients = [value.float() for value in saved["gradients"]]
    if [str(value) for value in saved["state_ids"]] != protocol["state_ids"]:
        raise ValueError("gradient state IDs differ from the frozen protocol")
    if len(gradients) != 32 or base.numel() % max(BLOCK_SIZES):
        raise ValueError("recorded parameter is incompatible with the frozen variants")

    variants = {}
    for block_size in BLOCK_SIZES:
        first = transition_sequence(base, gradients, block_size, device)
        second = transition_sequence(base, gradients, block_size, device)
        first_digest = digest_sequence(first)
        if first_digest != digest_sequence(second):
            raise RuntimeError(f"determinism replay failed for block size {block_size}")
        collector = OptimizerImplementationCapture(protocol["state_ids"])
        for row in first:
            collector.append(row)
        statistics, profiles = collector.finish()
        variants[str(block_size)] = {
            "original_coordinate_statistics": statistics,
            "profiles": profiles,
            "confirmation_relative_rms": {
                stage: fixed_suite_total_rms(statistics[stage]) for stage in STAGES
            },
            "transition_digests": first_digest,
        }
    raw = {
        "schema": "optimizer-quantization-mechanism-raw-v1",
        "status": "COMPLETE",
        "protocol_sha256": sha(output / "protocol.json"),
        "state_ids": protocol["state_ids"],
        "calibration_state_ids": protocol["calibration_state_ids"],
        "confirmation_state_ids": protocol["confirmation_state_ids"],
        "variants": variants,
        "training_outcome_established": False,
    }
    save_new(output / "raw.json", raw)
    save_new(output / "summary.json", summarize(protocol, raw))


def summarize(protocol: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    if raw.get("schema") != "optimizer-quantization-mechanism-raw-v1":
        raise ValueError("unexpected raw mechanism record")
    values = {
        int(block): record["confirmation_relative_rms"]
        for block, record in raw["variants"].items()
    }
    required = set(BLOCK_SIZES)
    if set(values) != required:
        raise ValueError("mechanism variants are incomplete")
    stage_results = {}
    for stage in STAGES:
        ordered = [float(values[size][stage]) for size in BLOCK_SIZES]
        stage_results[stage] = {
            "block_sizes": list(BLOCK_SIZES),
            "relative_rms": ordered,
            "strictly_increases_with_block_size": ordered[0] < ordered[1] < ordered[2],
        }
    success = all(stage_results[stage]["strictly_increases_with_block_size"]
                  for stage in ("MOMENT1_STATE", "PARAMETER_WRITE"))
    return {
        "schema": "optimizer-quantization-mechanism-summary-v1",
        "case_id": protocol["case_id"],
        "frozen_prediction": protocol["prediction_fixed_before_variants"],
        "stage_results": stage_results,
        "prediction_result": "CONFIRMED" if success else "NOT_CONFIRMED",
        "recommended_training_variant": 64 if success else None,
        "training_selection_status": (
            "READY_FOR_HUMAN_REVIEW" if success else "NOT_READY_PREDICTION_FAILED"
        ),
        "scope": (
            "Fixed recorded Mamba gradient sequence; optimizer quantization mechanism, "
            "not a natural training or quality result"
        ),
    }


def report(output: Path) -> None:
    protocol = verify(output)
    raw = load(output / "raw.json")
    save_new(output / "recomputed_summary.json", summarize(protocol, raw))


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
    os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
    os.environ.setdefault("XDG_CACHE_HOME", "/data1/tzh/cache/xdg")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data1/tzh/cache/torchinductor")
    main()
