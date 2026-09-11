#!/usr/bin/env python3
"""Trace why lower direct AdamW8bit write RMS did not improve training loss.

This is a result-aware diagnostic, not a new blind training confirmation.  It
reruns the eight frozen token streams with FP32 AdamW, block64, and block256 in
the same process, recording full-model parameter distance and common-evaluation
loss at fixed checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from kernel_analyzer.training_equivalence import exact_binomial_one_sided_bounds


ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("/data1/tzh/models/state-spaces/mamba-130m-hf")
EVAL_BANK = ROOT / "results/coverage/mamba_seq64_input_bank.json"
TRAIN_BANKS = tuple(ROOT / (
    "results/property/numerical_coverage_v1/"
    f"mamba_adamw8bit_confirm_bank_{index:02d}_v1.json"
) for index in range(8))
PRIOR_TRAINING = ROOT / (
    "results/property/numerical_coverage_v1/mamba_adamw8bit_training_confirmation_v1/verification.json"
)
FULL_MODEL_PROBE = ROOT / (
    "results/property/numerical_coverage_v1/adamw8bit_full_model_block_probe_v1/verification_v2.json"
)
CONDITIONS = ("FP32_ADAMW", "ADAMW8BIT_BLOCK64", "ADAMW8BIT_BLOCK256")
CHECKPOINTS = (0, 256, 512, 768, 1024)


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_new(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def freeze(output: Path) -> None:
    if output.exists():
        raise ValueError("freeze requires a new output directory")
    full_model = load(FULL_MODEL_PROBE)
    if full_model.get("prediction_result") != "CONFIRMED":
        raise ValueError("full-model direct-write premise is not verified")
    sources = [Path(__file__).resolve(), EVAL_BANK, PRIOR_TRAINING, FULL_MODEL_PROBE,
               MODEL / "config.json", MODEL / "model.safetensors", *TRAIN_BANKS]
    protocol = {
        "schema": "adamw8bit-trajectory-response-audit-v1",
        "status": "FROZEN_AFTER_PRIOR_LOSS_AND_DIRECT_WRITE_RESULTS",
        "data_use": "RESULT_AWARE_TRAJECTORY_DIAGNOSTIC_NOT_BLIND_CONFIRMATION",
        "question": (
            "Does block64 remain closer than block256 to the FP32 parameter trajectory, "
            "and does parameter proximity explain the already observed loss outcome?"
        ),
        "conditions": list(CONDITIONS),
        "streams": len(TRAIN_BANKS),
        "steps": 1024,
        "checkpoints": list(CHECKPOINTS),
        "training_banks": [str(path) for path in TRAIN_BANKS],
        "evaluation_bank": str(EVAL_BANK),
        "evaluation_states": 32,
        "pairing": "same checkpoint, tokens, and per-step random seed within each stream",
        "optimizer": {"lr": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8,
                      "weight_decay": 0.01},
        "primary_diagnostic": (
            "exact sign probability for final full-model parameter distance: "
            "block64-to-FP32 < block256-to-FP32"
        ),
        "interpretation_rules": {
            "block64_closer_but_loss_not_closer": (
                "direct-write and parameter L2 proximity are not sufficient to predict loss"
            ),
            "block64_not_closer": (
                "recursive feedback erased the direct-write RMS advantage"
            ),
        },
        "known_before_this_audit": {
            "block64_training_improvement": "NOT_CONFIRMED",
            "block64_full_model_direct_write_rms": "LOWER_IN_16_OF_16_NEW_HISTORIES",
        },
        "not_claimed": ["NEW_TRAINING_OUTCOME_CONFIRMATION", "UNIQUE_CAUSAL_MECHANISM"],
        "source_sha256": {str(path.resolve()): sha(path) for path in sources},
    }
    save_new(output / "protocol.json", protocol)


def verify_protocol(output: Path) -> dict[str, Any]:
    protocol = load(output / "protocol.json")
    if protocol.get("schema") != "adamw8bit-trajectory-response-audit-v1":
        raise ValueError("unexpected protocol")
    for name, expected in protocol["source_sha256"].items():
        path = Path(name)
        if not path.is_file() or sha(path) != expected:
            raise ValueError("frozen source or input changed: " + name)
    return protocol


def make_optimizer(condition: str, parameters, settings):
    import torch
    if condition == "FP32_ADAMW":
        return torch.optim.AdamW(parameters, foreach=False, fused=False, **settings)
    from torchao.optim import AdamW8bit
    return AdamW8bit(parameters, block_size=64 if condition.endswith("64") else 256,
                     **settings)


def evaluate(model, rows, device: str) -> float:
    import torch
    model.eval()
    values = []
    with torch.no_grad():
        for row in rows:
            tokens = torch.tensor([row.get("input_ids", row.get("token_ids"))],
                                  dtype=torch.long, device=device)
            values.append(float(model(input_ids=tokens, labels=tokens).loss.detach().cpu()))
    model.train()
    return math.fsum(values) / len(values)


def parameter_distance(candidate, reference) -> dict[str, float]:
    difference_energy = 0.0
    reference_energy = 0.0
    for left, right in zip(candidate.parameters(), reference.parameters()):
        difference_energy += float(((left.detach() - right.detach()).double().square().sum()).item())
        reference_energy += float((right.detach().double().square().sum()).item())
    return {
        "l2": math.sqrt(difference_energy),
        "relative_l2": math.sqrt(difference_energy / reference_energy),
    }


def run_stream(protocol: dict[str, Any], stream_index: int, device: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM
    models = {}
    optimizers = {}
    settings = dict(lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    for condition in CONDITIONS:
        torch.manual_seed(92000 + stream_index)
        torch.cuda.manual_seed_all(92000 + stream_index)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float32, local_files_only=True,
        ).to(device)
        model.config.use_cache = False
        model.train()
        models[condition] = model
        optimizers[condition] = make_optimizer(condition, model.parameters(), settings)
    train_rows = load(TRAIN_BANKS[stream_index])["states"]
    eval_rows = load(EVAL_BANK)["states"][:protocol["evaluation_states"]]
    records = {}

    def checkpoint(step: int) -> None:
        losses = {condition: evaluate(models[condition], eval_rows, device)
                  for condition in CONDITIONS}
        distances = {
            condition: parameter_distance(models[condition], models["FP32_ADAMW"])
            for condition in CONDITIONS if condition != "FP32_ADAMW"
        }
        records[str(step)] = {"evaluation_loss": losses,
                              "parameter_distance_to_fp32": distances}

    checkpoint(0)
    training_loss = {condition: [] for condition in CONDITIONS}
    for row_index, row in enumerate(train_rows):
        step = row_index + 1
        order = CONDITIONS[stream_index % len(CONDITIONS):] + CONDITIONS[:stream_index % len(CONDITIONS)]
        for condition in order:
            seed = 1_000_000 * stream_index + 81_000 + row_index
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            tokens = torch.tensor([row["token_ids"]], dtype=torch.long, device=device)
            optimizer = optimizers[condition]
            optimizer.zero_grad(set_to_none=True)
            loss = models[condition](input_ids=tokens, labels=tokens).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss for {condition} at step {step}")
            loss.backward()
            optimizer.step()
            training_loss[condition].append(float(loss.detach().cpu()))
        if step in CHECKPOINTS:
            checkpoint(step)
        if step % 128 == 0:
            print(json.dumps({"event": "TRAJECTORY_RESPONSE_STEP", "stream": stream_index,
                              "step": step}), flush=True)
    final = records[str(CHECKPOINTS[-1])]
    distance = final["parameter_distance_to_fp32"]
    losses = final["evaluation_loss"]
    return {
        "schema": "adamw8bit-trajectory-response-stream-v1",
        "status": "COMPLETE",
        "stream_index": stream_index,
        "checkpoints": records,
        "final_block64_parameter_closer": (
            distance["ADAMW8BIT_BLOCK64"]["relative_l2"]
            < distance["ADAMW8BIT_BLOCK256"]["relative_l2"]
        ),
        "final_block64_loss_closer": (
            abs(losses["ADAMW8BIT_BLOCK64"] - losses["FP32_ADAMW"])
            < abs(losses["ADAMW8BIT_BLOCK256"] - losses["FP32_ADAMW"])
        ),
        "training_loss": training_loss,
    }


def summarize(output: Path, protocol: dict[str, Any]) -> dict:
    rows = [load(output / "streams" / f"stream-{index:02d}.json")
            for index in range(protocol["streams"])]
    parameter_successes = sum(row["final_block64_parameter_closer"] for row in rows)
    loss_successes = sum(row["final_block64_loss_closer"] for row in rows)
    parameter_bounds = exact_binomial_one_sided_bounds(
        parameter_successes, len(rows), alpha=0.05
    )
    return {
        "schema": "adamw8bit-trajectory-response-summary-v1",
        "status": "COMPLETE_RESULT_AWARE_DIAGNOSTIC",
        "stream_count": len(rows),
        "block64_final_parameter_closer_count": parameter_successes,
        "block64_final_parameter_closer_probability_bounds": list(parameter_bounds),
        "block64_parameter_proximity_result": (
            "CONFIRMED_MORE_OFTEN_THAN_HALF" if parameter_bounds[0] > 0.5
            else "NOT_CONFIRMED"
        ),
        "block64_final_loss_closer_count": loss_successes,
        "mean_final_relative_parameter_distance": {
            condition: math.fsum(
                row["checkpoints"]["1024"]["parameter_distance_to_fp32"][condition]["relative_l2"]
                for row in rows
            ) / len(rows)
            for condition in ("ADAMW8BIT_BLOCK64", "ADAMW8BIT_BLOCK256")
        },
        "mean_final_evaluation_loss": {
            condition: math.fsum(
                row["checkpoints"]["1024"]["evaluation_loss"][condition] for row in rows
            ) / len(rows)
            for condition in CONDITIONS
        },
        "interpretation": (
            "RESULT_AWARE: parameter proximity and evaluation loss are reported separately; "
            "this diagnostic cannot create a new blind training claim."
        ),
    }


def run(output: Path, device: str, start_index: int, limit: int | None) -> None:
    import torch
    protocol = verify_protocol(output)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    indices = list(range(start_index, protocol["streams"]))
    if limit is not None:
        indices = indices[:limit]
    stream_dir = output / "streams"
    stream_dir.mkdir(exist_ok=True)
    for index in indices:
        path = stream_dir / f"stream-{index:02d}.json"
        if path.exists():
            continue
        save_new(path, run_stream(protocol, index, device))
    if all((stream_dir / f"stream-{index:02d}.json").exists()
           for index in range(protocol["streams"])):
        result_path = output / "summary.json"
        if not result_path.exists():
            save_new(result_path, summarize(output, protocol))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "summarize"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("output must be under /data1/tzh")
    if args.action == "freeze":
        freeze(args.output)
    elif args.action == "run":
        run(args.output, args.device, args.start_index, args.limit)
    else:
        protocol = verify_protocol(args.output)
        path = args.output / "summary.json"
        if path.exists():
            raise ValueError("summary already exists")
        save_new(path, summarize(args.output, protocol))


if __name__ == "__main__":
    main()
