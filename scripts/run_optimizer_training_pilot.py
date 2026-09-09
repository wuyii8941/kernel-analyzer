#!/usr/bin/env python3
"""Run a frozen natural-training feasibility pilot for AdamW8bit variants.

This pilot checks that FP32 AdamW, AdamW8bit(block=256), and the mechanism-led
block=64 modification can train the same Mamba language model on the same token
stream.  It records validation loss and cost, but it is not an independent-run
training conclusion and cannot by itself promote a variant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("/data1/tzh/models/state-spaces/mamba-130m-hf")
TRAIN_BANK = ROOT / "results/property/numerical_coverage_v1/mamba_adamw8bit_training_pilot_bank_v1.json"
EVAL_BANK = ROOT / "results/coverage/mamba_seq64_input_bank.json"
MECHANISM = ROOT / "results/property/numerical_coverage_v1/torchao_adamw8bit_block_mechanism_v1/summary.json"
CONDITIONS = ("FP32_ADAMW", "ADAMW8BIT_BLOCK256", "ADAMW8BIT_BLOCK64")
STEPS = 128


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
    mechanism = load(MECHANISM)
    if (mechanism.get("prediction_result") != "CONFIRMED"
            or mechanism.get("recommended_training_variant") != 64):
        raise ValueError("the frozen mechanism result does not support this pilot")
    train = load(TRAIN_BANK).get("states", [])
    evaluation = load(EVAL_BANK).get("states", [])
    if len(train) != STEPS or len(evaluation) < 32:
        raise ValueError("pilot banks have unexpected sizes")
    paths = [Path(__file__).resolve(), TRAIN_BANK, EVAL_BANK, MECHANISM,
             MODEL / "config.json", MODEL / "model.safetensors"]
    protocol = {
        "schema": "optimizer-training-pilot-v1",
        "status": "FROZEN_BEFORE_PILOT",
        "model": str(MODEL),
        "conditions": list(CONDITIONS),
        "steps": STEPS,
        "train_bank": str(TRAIN_BANK),
        "evaluation_bank": str(EVAL_BANK),
        "evaluation_states": 32,
        "optimizer": {"lr": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8,
                      "weight_decay": 0.01},
        "parameter_dtype": "float32",
        "paired_design": "same checkpoint and token order for all three conditions",
        "purpose": "FEASIBILITY_AND_COST_ONLY",
        "scientific_training_outcome": "NOT_ASSESSED_BY_PILOT",
        "promotion_rule": (
            "A later confirmation may be designed only if all conditions complete with "
            "finite loss and the measured cost is affordable; favorable pilot loss is not "
            "a promotion criterion"
        ),
        "source_sha256": {str(path.resolve()): sha(path) for path in paths},
    }
    save_new(output / "protocol.json", protocol)


def verify(output: Path) -> dict[str, Any]:
    protocol = load(output / "protocol.json")
    if protocol.get("schema") != "optimizer-training-pilot-v1":
        raise ValueError("unexpected protocol")
    for name, expected in protocol.get("source_sha256", {}).items():
        path = Path(name)
        if not path.is_file() or sha(path) != expected:
            raise ValueError("frozen pilot dependency changed: " + name)
    return protocol


def make_optimizer(condition: str, parameters, settings):
    import torch
    if condition == "FP32_ADAMW":
        return torch.optim.AdamW(parameters, foreach=False, fused=False, **settings)
    from torchao.optim import AdamW8bit
    block_size = 64 if condition.endswith("BLOCK64") else 256
    return AdamW8bit(parameters, block_size=block_size, **settings)


def evaluate(model, rows, device: str) -> list[float]:
    import torch
    model.eval()
    values = []
    with torch.no_grad():
        for row in rows:
            tokens = torch.tensor([row.get("input_ids", row.get("token_ids"))],
                                  dtype=torch.long, device=device)
            loss = model(input_ids=tokens, labels=tokens).loss
            values.append(float(loss.detach().cpu()))
    return values


def run_condition(protocol: dict[str, Any], condition: str, device: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    torch.manual_seed(90125)
    torch.cuda.manual_seed_all(90125)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, local_files_only=True,
    ).to(device)
    model.config.use_cache = False
    settings = dict(lr=protocol["optimizer"]["lr"],
                    betas=tuple(protocol["optimizer"]["betas"]),
                    eps=protocol["optimizer"]["eps"],
                    weight_decay=protocol["optimizer"]["weight_decay"])
    optimizer = make_optimizer(condition, model.parameters(), settings)
    train_rows = load(TRAIN_BANK)["states"]
    eval_rows = load(EVAL_BANK)["states"][:protocol["evaluation_states"]]
    torch.cuda.reset_peak_memory_stats(device)
    initial_eval = evaluate(model, eval_rows, device)
    model.train()
    losses = []
    started = time.perf_counter()
    for index, row in enumerate(train_rows):
        seed = 81000 + index
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        tokens = torch.tensor([row["token_ids"]], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=tokens, labels=tokens).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite training loss at step {index + 1}")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if (index + 1) % 16 == 0:
            print(json.dumps({"event": "TRAINING_PILOT", "condition": condition,
                              "step": index + 1, "loss": losses[-1]}), flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    final_eval = evaluate(model, eval_rows, device)
    parameter_digest = hashlib.sha256()
    parameter_norm_sq = 0.0
    for parameter in model.parameters():
        value = parameter.detach().float().cpu().contiguous()
        parameter_digest.update(value.numpy().tobytes())
        parameter_norm_sq += float(torch.dot(value.reshape(-1), value.reshape(-1)))
    result = {
        "condition": condition,
        "status": "COMPLETE",
        "training_loss": losses,
        "initial_evaluation_loss": initial_eval,
        "final_evaluation_loss": final_eval,
        "elapsed_training_seconds": elapsed,
        "steps_per_second": len(losses) / elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "final_parameter_sha256": parameter_digest.hexdigest(),
        "final_parameter_norm_sq": parameter_norm_sq,
    }
    del optimizer, model
    torch.cuda.empty_cache()
    return result


def capture(output: Path, device: str) -> None:
    protocol = verify(output)
    if (output / "raw.json").exists():
        raise ValueError("capture refuses to overwrite results")
    records = [run_condition(protocol, condition, device) for condition in CONDITIONS]
    raw = {
        "schema": "optimizer-training-pilot-raw-v1",
        "status": "COMPLETE",
        "protocol_sha256": sha(output / "protocol.json"),
        "records": records,
    }
    save_new(output / "raw.json", raw)
    save_new(output / "summary.json", summarize(protocol, raw))


def summarize(protocol: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    records = {row["condition"]: row for row in raw.get("records", [])}
    if set(records) != set(CONDITIONS) or any(row.get("status") != "COMPLETE"
                                              for row in records.values()):
        raise ValueError("pilot conditions are incomplete")
    import math
    compact = []
    all_finite = True
    for condition in CONDITIONS:
        row = records[condition]
        initial = row["initial_evaluation_loss"]
        final = row["final_evaluation_loss"]
        finite = all(math.isfinite(float(value)) for value in
                     [*row["training_loss"], *initial, *final])
        all_finite &= finite
        compact.append({
            "condition": condition,
            "finite": finite,
            "initial_mean_evaluation_loss": sum(initial) / len(initial),
            "final_mean_evaluation_loss": sum(final) / len(final),
            "final_minus_initial_evaluation_loss": (
                sum(final) / len(final) - sum(initial) / len(initial)
            ),
            "steps_per_second": row["steps_per_second"],
            "peak_allocated_bytes": row["peak_allocated_bytes"],
            "final_parameter_sha256": row["final_parameter_sha256"],
        })
    return {
        "schema": "optimizer-training-pilot-summary-v1",
        "purpose": protocol["purpose"],
        "records": compact,
        "all_conditions_finite_and_complete": all_finite,
        "confirmation_design_status": (
            "MAY_FREEZE_WITHOUT_USING_PILOT_LOSS_FOR_SELECTION"
            if all_finite else "BLOCKED_BY_NONFINITE_OR_INCOMPLETE_PILOT"
        ),
        "scientific_training_outcome": "NOT_ASSESSED_BY_PILOT",
        "scope": "One shared token stream and checkpoint; feasibility and measured cost only",
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
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
    os.environ.setdefault("XDG_CACHE_HOME", "/data1/tzh/cache/xdg")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data1/tzh/cache/torchinductor")
    main()
