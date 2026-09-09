#!/usr/bin/env python3
"""Freeze, run, and recompute one real-gradient optimizer implementation study.

The reviewed comparison is TorchAO AdamW8bit against torch.optim.AdamW.  Both
receive the same real Mamba parameter gradients and the same FP32 parameter at
every transition.  Their moment histories evolve naturally and separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("/data1/tzh/models/state-spaces/mamba-130m-hf")
BANK = ROOT / "results/coverage/mamba_seq64_input_bank.json"
TARGET = "backbone.layers.0.mixer.x_proj.weight"
CACHE = Path("/data1/tzh/cache/torchao_adamw8bit_mamba_v2")
STATE_COUNT = 32


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha(value) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def save_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def torchao_sources() -> list[Path]:
    import inspect
    import torchao.optim.adam as adam
    import torchao.optim.quant_utils as quant_utils
    import torchao.optim.subclass_8bit as subclass
    return [Path(inspect.getfile(module)).resolve() for module in (adam, quant_utils, subclass)]


def source_manifest() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        ROOT / "src/kernel_analyzer/optimizer_implementation_capture.py",
        ROOT / "src/kernel_analyzer/training_numerical_analysis.py",
        ROOT / "src/kernel_analyzer/training_bias_profile.py",
        ROOT / "src/kernel_analyzer/training_equivalence.py",
        ROOT / "src/kernel_analyzer/analysis_result.py",
        BANK,
        MODEL / "config.json",
        MODEL / "model.safetensors",
        *torchao_sources(),
    ]
    return {str(path): sha(path) for path in paths}


def freeze(output: Path, gradient_input: Path | None = None) -> None:
    if output.exists():
        raise ValueError("freeze requires a new output directory")
    bank = load(BANK)
    states = bank.get("states", [])
    if len(states) < STATE_COUNT:
        raise ValueError("input bank has fewer than 32 states")
    state_ids = [str(row["state_id"]) for row in states[:STATE_COUNT]]
    if gradient_input is not None:
        gradient_input = gradient_input.resolve()
        if not gradient_input.is_relative_to(Path("/data1/tzh")) or not gradient_input.is_file():
            raise ValueError("reused gradient input must be an existing file under /data1/tzh")
    sources = source_manifest()
    if gradient_input is not None:
        sources[str(gradient_input)] = sha(gradient_input)
    protocol = {
        "schema": "optimizer-implementation-analysis-v1",
        "status": "FROZEN_BEFORE_REAL_GRADIENT_CAPTURE",
        "case_id": "mamba_x_proj_torchao_adamw8bit_vs_torch_adamw",
        "operator_family": "OPTIMIZER_PARAMETER_UPDATE",
        "semantic_operation": "ADAMW_PARAMETER_AND_MOMENT_TRANSITION",
        "candidate": {
            "implementation": "torchao.optim.AdamW8bit",
            "moment_representation": "BLOCKWISE_8BIT",
            "block_size": 256,
            "actual_backend": "TORCH_COMPILE_GENERATED_TRITON",
        },
        "reference": {
            "implementation": "torch.optim.AdamW",
            "moment_representation": "FP32",
            "reference_is_absolute_truth": False,
        },
        "model": str(MODEL),
        "target_parameter": TARGET,
        "input_bank": str(BANK),
        "state_ids": state_ids,
        "calibration_state_ids": state_ids[:16],
        "confirmation_state_ids": state_ids[16:],
        "gradient_source": (
            "REUSED_REAL_MAMBA_LM_GRADIENTS_FROM_RECORDED_FAILED_CAPTURE"
            if gradient_input is not None else
            "REAL_MAMBA_LM_LOSS_AT_ONE_FIXED_CHECKPOINT"
        ),
        "reused_gradient_input": str(gradient_input) if gradient_input is not None else None,
        "transition_design": (
            "same frozen gradient and same FP32 parameter for each implementation; "
            "candidate and reference moment histories evolve separately"
        ),
        "optimizer": {
            "lr": 1e-3, "betas": [0.9, 0.999], "eps": 1e-8,
            "weight_decay": 0.01, "amsgrad": False,
        },
        "primary_stage": "PARAMETER_WRITE",
        "claim_scope": "FIXED_SUITE_UPDATE",
        "fixed_suite_margins": {"full_update_rms": 0.01},
        "contrast_id": "OPTIMIZER_IMPLEMENTATION_COMPARISON",
        "data_use": "DEVELOPMENT_REAL_GRADIENT_CONFIRMATION_NOT_BLIND_DISCOVERY",
        "prior_information": (
            "A random-vector feasibility probe was observed before freeze; it is not "
            "a formal training result and is not included in this suite."
        ),
        "automation_boundary": {
            "human_review": [
                "candidate/reference semantic comparison",
                "target parameter and training-state selection",
            ],
            "automatic_after_freeze": [
                "real gradient generation", "matched optimizer transitions",
                "determinism check", "statistics", "report recomputation",
            ],
        },
        "torchinductor_cache_dir": str(CACHE),
        "source_sha256": sources,
    }
    save_new(output / "protocol.json", protocol)


def verify_protocol(output: Path) -> dict[str, Any]:
    protocol = load(output / "protocol.json")
    if protocol.get("schema") != "optimizer-implementation-analysis-v1":
        raise ValueError("unexpected protocol")
    for name, expected in protocol.get("source_sha256", {}).items():
        path = Path(name)
        if not path.is_file() or sha(path) != expected:
            raise ValueError("frozen source or input changed: " + name)
    if Path(os.environ.get("TORCHINDUCTOR_CACHE_DIR", "")).resolve() != CACHE:
        raise ValueError("dedicated TORCHINDUCTOR_CACHE_DIR is required")
    return protocol


def collect_gradients(protocol: dict[str, Any], device: str):
    import torch
    from transformers import AutoModelForCausalLM

    torch.manual_seed(271828)
    torch.cuda.manual_seed_all(271828)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, local_files_only=True,
    ).to(device).eval()
    model.config.use_cache = False
    parameters = dict(model.named_parameters())
    if TARGET not in parameters:
        raise ValueError("declared target parameter is absent")
    target = parameters[TARGET]
    for name, parameter in parameters.items():
        parameter.requires_grad_(name == TARGET)
    base = target.detach().float().clone()
    bank = load(BANK)
    rows = bank["states"][:STATE_COUNT]
    gradients = []
    records = []
    for index, row in enumerate(rows):
        model.zero_grad(set_to_none=True)
        tokens = torch.tensor([row["token_ids"]], dtype=torch.long, device=device)
        if tensor_sha(tokens) != row["token_sha256"]:
            raise ValueError(f"token digest mismatch at state {index}")
        loss = model(input_ids=tokens, labels=tokens).loss
        loss.backward()
        if target.grad is None:
            raise RuntimeError("target gradient is absent")
        gradient = target.grad.detach().float().cpu().clone()
        gradients.append(gradient)
        records.append({
            "state_id": str(row["state_id"]),
            "loss": float(loss.detach().cpu()),
            "gradient_sha256": tensor_sha(gradient),
            "gradient_nonzero_coordinates": int(torch.count_nonzero(gradient).item()),
        })
        print(json.dumps({"event": "REAL_GRADIENT", "state": index + 1}), flush=True)
    del model
    torch.cuda.empty_cache()
    return base.cpu(), gradients, records


def dequantized(value):
    if hasattr(value, "dequantize"):
        return value.dequantize().detach().float().clone()
    return value.detach().float().clone()


def transition_sequence(base, gradients, device: str):
    import torch
    from torchao.optim import AdamW8bit

    settings = dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                    weight_decay=0.01, amsgrad=False)
    candidate_parameter = torch.nn.Parameter(base.to(device).clone())
    reference_parameter = torch.nn.Parameter(base.to(device).clone())
    candidate_optimizer = AdamW8bit([candidate_parameter], block_size=256, **settings)
    reference_optimizer = torch.optim.AdamW(
        [reference_parameter], foreach=False, fused=False, **settings
    )
    output = []
    for index, gradient_cpu in enumerate(gradients):
        gradient = gradient_cpu.to(device)
        with torch.no_grad():
            candidate_parameter.copy_(base.to(device))
            reference_parameter.copy_(base.to(device))
        candidate_parameter.grad = gradient.clone()
        reference_parameter.grad = gradient.clone()
        candidate_optimizer.step(); reference_optimizer.step()
        torch.cuda.synchronize()
        candidate_state = candidate_optimizer.state[candidate_parameter]
        reference_state = reference_optimizer.state[reference_parameter]
        output.append({
            "GRADIENT_INPUT": (gradient.detach().cpu().clone(), gradient.detach().cpu().clone()),
            "MOMENT1_STATE": (
                dequantized(candidate_state["exp_avg"]).cpu(),
                dequantized(reference_state["exp_avg"]).cpu(),
            ),
            "MOMENT2_STATE": (
                dequantized(candidate_state["exp_avg_sq"]).cpu(),
                dequantized(reference_state["exp_avg_sq"]).cpu(),
            ),
            "PARAMETER_WRITE": (
                (candidate_parameter.detach() - base.to(device)).cpu(),
                (reference_parameter.detach() - base.to(device)).cpu(),
            ),
        })
        print(json.dumps({"event": "OPTIMIZER_TRANSITION", "state": index + 1}), flush=True)
    return output


def transition_digests(sequence) -> list[dict[str, str]]:
    return [{stage + ":candidate": tensor_sha(pair[0]) for stage, pair in row.items()} |
            {stage + ":reference": tensor_sha(pair[1]) for stage, pair in row.items()}
            for row in sequence]


def preserve_generated_sources(output: Path) -> dict[str, Any]:
    sources = sorted(CACHE.rglob("*.py"))
    destination = output / "generated_triton_sources"
    destination.mkdir(exist_ok=False)
    rows = []
    for index, source in enumerate(sources):
        target = destination / f"{index:04d}_{source.name}"
        shutil.copy2(source, target)
        rows.append({"source": str(source), "copy": str(target.relative_to(output)),
                     "sha256": sha(target)})
    return {"count": len(rows), "files": rows, "dedicated_cache": str(CACHE)}


def capture(output: Path, device: str) -> None:
    import torch
    from kernel_analyzer.optimizer_implementation_capture import (
        OptimizerImplementationCapture, fixed_suite_total_rms,
    )
    from kernel_analyzer.training_numerical_analysis import analyze_artifact

    protocol = verify_protocol(output)
    if (output / "raw.json").exists():
        raise ValueError("capture refuses to overwrite raw results")
    reused = protocol.get("reused_gradient_input")
    if reused:
        saved = torch.load(reused, weights_only=True)
        base, gradients = saved["base_parameter"], saved["gradients"]
        if [str(value) for value in saved["state_ids"]] != protocol["state_ids"]:
            raise ValueError("reused gradients have different state IDs")
        gradient_records = [{
            "state_id": state_id,
            "loss": None,
            "gradient_sha256": tensor_sha(gradient),
            "gradient_nonzero_coordinates": int(torch.count_nonzero(gradient).item()),
            "provenance": "REUSED_FROM_RECORDED_FAILED_CAPTURE",
        } for state_id, gradient in zip(protocol["state_ids"], gradients)]
    else:
        base, gradients, gradient_records = collect_gradients(protocol, device)
    torch.save({"base_parameter": base, "gradients": gradients,
                "state_ids": protocol["state_ids"]}, output / "real_gradient_inputs.pt")
    first = transition_sequence(base, gradients, device)
    second = transition_sequence(base, gradients, device)
    first_digests = transition_digests(first)
    second_digests = transition_digests(second)
    if first_digests != second_digests:
        raise RuntimeError("optimizer transition determinism replay failed")
    collector = OptimizerImplementationCapture(protocol["state_ids"])
    for row in first:
        collector.append(row)
    statistics, stages = collector.finish()
    generated = preserve_generated_sources(output)
    if generated["count"] == 0:
        raise RuntimeError("dedicated cache contains no generated source")
    raw = {
        "schema": "kernel-analyzer-optimizer-implementation-raw-v1",
        "status": "COMPLETE",
        "case_id": protocol["case_id"],
        "contrast_id": protocol["contrast_id"],
        "state_ids": protocol["state_ids"],
        "calibration_state_ids": protocol["calibration_state_ids"],
        "confirmation_state_ids": protocol["confirmation_state_ids"],
        "runtime_boundary": {
            "kind": "OPTIMIZER_IMPLEMENTATION",
            "candidate": protocol["candidate"], "reference": protocol["reference"],
            "generated_sources": generated,
        },
        "carrier": TARGET,
        "parameter_write_protocol": {
            "version": "optimizer-implementation-readback-v1",
            "measurement": "parameter_after_step_minus_parameter_before_step",
            "parameter_representation": "FP32",
        },
        "determinism": {"all_exact": True, "transition_digests": first_digests},
        "original_coordinate_statistics": statistics,
        "stages": stages,
        "gradient_records": gradient_records,
        "protocol_sha256": sha(output / "protocol.json"),
        "primary_update_endpoint": "PARAMETER_WRITE",
        "claim_boundary": (
            "32 frozen Mamba LM gradients at one checkpoint; fixed suite only; "
            "not a random-state or training-quality result"
        ),
    }
    save_new(output / "raw.json", raw)
    analysis = analyze_artifact(raw, protocol)
    save_new(output / "analysis.json", analysis)
    save_new(output / "capture_summary.json", {
        "schema": "optimizer-implementation-capture-summary-v1",
        "status": "COMPLETE",
        "case_id": protocol["case_id"],
        "generated_triton_source_count": generated["count"],
        "confirmation_parameter_write_rms": fixed_suite_total_rms(
            statistics["PARAMETER_WRITE"]
        ),
        "equivalence_decision": analysis["equivalence_decision"],
        "measurement_status": analysis["measurement_status"],
    })


def report(output: Path) -> None:
    from kernel_analyzer.training_numerical_analysis import analyze_artifact
    protocol = load(output / "protocol.json")
    raw = load(output / "raw.json")
    analysis = analyze_artifact(raw, protocol)
    destination = output / "recomputed_analysis.json"
    save_new(destination, analysis)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "capture", "report"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gradient-input", type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if not output.is_relative_to(Path("/data1/tzh")):
        raise ValueError("output must be under /data1/tzh")
    if args.command == "freeze":
        freeze(output, args.gradient_input)
    elif args.command == "capture":
        capture(output, args.device)
    else:
        report(output)


if __name__ == "__main__":
    main()
