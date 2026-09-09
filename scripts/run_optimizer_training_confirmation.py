#!/usr/bin/env python3
"""Independent data-stream confirmation for the AdamW8bit mechanism case."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("/data1/tzh/models/state-spaces/mamba-130m-hf")
EVAL_BANK = ROOT / "results/coverage/mamba_seq64_input_bank.json"
MECHANISM = ROOT / "results/property/numerical_coverage_v1/torchao_adamw8bit_block_mechanism_v1/summary.json"
PILOT = ROOT / "results/property/numerical_coverage_v1/mamba_adamw8bit_training_pilot_v1/summary.json"
CONDITIONS = ("FP32_ADAMW", "ADAMW8BIT_BLOCK256", "ADAMW8BIT_BLOCK64")
STREAMS = 8
STEPS = 1024
EVALUATION_STEPS = (0, 256, 512, 768, 1024)
DESIGN_SEED = 20260909
START_RANGE = (10000, 90000)
MINIMUM_SEPARATION = 1200
MATERIAL_LOSS_MARGIN = 0.01


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


def selected_start_blocks() -> list[int]:
    generator = random.Random(DESIGN_SEED)
    chosen = []
    for value in generator.sample(range(*START_RANGE), START_RANGE[1] - START_RANGE[0]):
        if all(abs(value - prior) >= MINIMUM_SEPARATION for prior in chosen):
            chosen.append(value)
            if len(chosen) == STREAMS:
                return chosen
    raise RuntimeError("unable to select nonoverlapping streams")


def bank_paths() -> list[Path]:
    return [ROOT / ("results/property/numerical_coverage_v1/"
                   f"mamba_adamw8bit_confirm_bank_{index:02d}_v1.json")
            for index in range(STREAMS)]


def freeze(output: Path) -> None:
    if output.exists():
        raise ValueError("freeze requires a new output directory")
    mechanism = load(MECHANISM)
    pilot = load(PILOT)
    if (mechanism.get("prediction_result") != "CONFIRMED"
            or mechanism.get("recommended_training_variant") != 64):
        raise ValueError("mechanism-led modification is not confirmed")
    if not pilot.get("confirmation_design_status", "").startswith("MAY_FREEZE"):
        raise ValueError("training feasibility pilot did not complete")
    starts = selected_start_blocks()
    banks = bank_paths()
    for index, (path, start) in enumerate(zip(banks, starts)):
        document = load(path)
        states = document.get("states", [])
        if (document.get("start_block") != start or len(states) != STEPS):
            raise ValueError(f"stream bank {index} differs from the random design")
    paths = [Path(__file__).resolve(), EVAL_BANK, MECHANISM, PILOT,
             MODEL / "config.json", MODEL / "model.safetensors", *banks]
    protocol = {
        "schema": "optimizer-training-confirmation-v1",
        "status": "FROZEN_BEFORE_INDEPENDENT_STREAMS",
        "model": str(MODEL),
        "conditions": list(CONDITIONS),
        "stream_count": STREAMS,
        "steps": STEPS,
        "evaluation_steps": list(EVALUATION_STEPS),
        "evaluation_bank": str(EVAL_BANK),
        "evaluation_states": 32,
        "train_banks": [str(path) for path in banks],
        "stream_selection": {
            "method": "SEEDED_WITHOUT_REPLACEMENT_AND_MINIMUM_BLOCK_SEPARATION",
            "seed": DESIGN_SEED,
            "start_range": list(START_RANGE),
            "minimum_start_block_separation": MINIMUM_SEPARATION,
            "selected_start_blocks": starts,
        },
        "optimizer": {"lr": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8,
                      "weight_decay": 0.01},
        "parameter_dtype": "float32",
        "pairing": "same checkpoint, tokens, and per-step RNG within each stream",
        "independent_unit": "NONOVERLAPPING_RANDOMLY_SELECTED_WIKITEXT_TOKEN_STREAM",
        "primary_endpoint": "MEAN_FIXED_EVALUATION_LOSS_AT_STEP_1024",
        "primary_contrast": "ADAMW8BIT_BLOCK256_MINUS_FP32_ADAMW",
        "material_loss_margin": MATERIAL_LOSS_MARGIN,
        "primary_decision": (
            "MATERIAL if the two-sided 95% t interval is entirely above +0.01 or "
            "below -0.01; DIFFERENT_BUT_BELOW_MARGIN if it excludes zero but not the "
            "margin; otherwise INCONCLUSIVE"
        ),
        "modification_endpoint": (
            "per-stream abs(block256-reference) minus abs(block64-reference)"
        ),
        "modification_success": "two-sided 95% t interval entirely above zero",
        "collapse_definition": (
            "nonfinite training/evaluation loss, or final evaluation loss more than "
            "1.0 above that condition's initial evaluation loss"
        ),
        "multiplicity": (
            "one primary material-effect contrast; modification and intermediate "
            "checkpoints are secondary and cannot replace it"
        ),
        "data_use": "PILOT_INFORMED_BUT_STREAM_INDEPENDENT_CONFIRMATION",
        "pilot_loss_used_to_select_conditions": False,
        "source_sha256": {str(path.resolve()): sha(path) for path in paths},
    }
    save_new(output / "protocol.json", protocol)


def verify(output: Path) -> dict[str, Any]:
    protocol = load(output / "protocol.json")
    if protocol.get("schema") != "optimizer-training-confirmation-v1":
        raise ValueError("unexpected protocol")
    for name, expected in protocol.get("source_sha256", {}).items():
        path = Path(name)
        if not path.is_file() or sha(path) != expected:
            raise ValueError("frozen confirmation dependency changed: " + name)
    return protocol


def make_optimizer(condition: str, parameters, settings):
    import torch
    if condition == "FP32_ADAMW":
        return torch.optim.AdamW(parameters, foreach=False, fused=False, **settings)
    from torchao.optim import AdamW8bit
    return AdamW8bit(parameters,
                     block_size=64 if condition.endswith("BLOCK64") else 256,
                     **settings)


def evaluate(model, rows, device: str) -> list[float]:
    import torch
    model.eval()
    values = []
    with torch.no_grad():
        for row in rows:
            tokens = torch.tensor([row.get("input_ids", row.get("token_ids"))],
                                  dtype=torch.long, device=device)
            values.append(float(model(input_ids=tokens, labels=tokens).loss.detach().cpu()))
    return values


def condition_order(stream_index: int) -> tuple[str, ...]:
    shift = stream_index % len(CONDITIONS)
    return CONDITIONS[shift:] + CONDITIONS[:shift]


def run_condition(protocol: dict[str, Any], stream_index: int,
                  condition: str, device: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    seed = 92000 + stream_index
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
    train_rows = load(Path(protocol["train_banks"][stream_index]))["states"]
    eval_rows = load(EVAL_BANK)["states"][:protocol["evaluation_states"]]
    torch.cuda.reset_peak_memory_stats(device)
    evaluations = {"0": evaluate(model, eval_rows, device)}
    losses = []
    model.train()
    started = time.perf_counter()
    for index, row in enumerate(train_rows):
        step = index + 1
        step_seed = 1000000 * stream_index + 81000 + index
        torch.manual_seed(step_seed)
        torch.cuda.manual_seed_all(step_seed)
        tokens = torch.tensor([row["token_ids"]], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=tokens, labels=tokens).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite training loss at step {step}")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step in EVALUATION_STEPS:
            torch.cuda.synchronize()
            evaluations[str(step)] = evaluate(model, eval_rows, device)
            model.train()
        if step % 128 == 0:
            print(json.dumps({"event": "TRAINING_CONFIRMATION", "stream": stream_index,
                              "condition": condition, "step": step,
                              "loss": losses[-1]}), flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    result = {
        "condition": condition,
        "status": "COMPLETE",
        "training_loss": losses,
        "evaluation_loss_by_step": evaluations,
        "elapsed_seconds_including_intermediate_evaluation": elapsed,
        "training_steps_per_second_with_evaluation_overhead": len(losses) / elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "final_parameter_sha256": digest.hexdigest(),
    }
    del optimizer, model
    torch.cuda.empty_cache()
    return result


def capture_stream(output: Path, stream_index: int, device: str, cache_dir: Path) -> None:
    if not 0 <= stream_index < STREAMS:
        raise ValueError("stream index is out of range")
    protocol = verify(output)
    destination = output / "streams" / f"stream_{stream_index:02d}.json"
    failure = output / "streams" / f"stream_{stream_index:02d}_failure.json"
    if destination.exists() or failure.exists():
        raise ValueError("stream result already exists")
    resolved_cache = cache_dir.resolve()
    if not resolved_cache.is_relative_to(Path("/data1/tzh/cache")):
        raise ValueError("cache must remain under /data1/tzh/cache")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(resolved_cache)
    try:
        rows = [run_condition(protocol, stream_index, condition, device)
                for condition in condition_order(stream_index)]
        save_new(destination, {
            "schema": "optimizer-training-confirmation-stream-v1",
            "status": "COMPLETE",
            "stream_index": stream_index,
            "train_bank": protocol["train_banks"][stream_index],
            "condition_execution_order": list(condition_order(stream_index)),
            "protocol_sha256": sha(output / "protocol.json"),
            "records": rows,
        })
    except Exception as error:
        save_new(failure, {
            "schema": "optimizer-training-confirmation-failure-v1",
            "status": "FAILED",
            "stream_index": stream_index,
            "error_type": type(error).__name__,
            "error": str(error),
            "protocol_sha256": sha(output / "protocol.json"),
        })
        raise


def t_interval(values: list[float], confidence: float = 0.95) -> list[float]:
    import scipy.stats
    if len(values) < 2 or not all(math.isfinite(value) for value in values):
        raise ValueError("at least two finite independent values are required")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    if variance == 0:
        return [mean, mean]
    critical = float(scipy.stats.t.ppf((1 + confidence) / 2, len(values) - 1))
    half = critical * math.sqrt(variance / len(values))
    return [mean - half, mean + half]


def summarize(protocol: dict[str, Any], streams: list[dict[str, Any]]) -> dict[str, Any]:
    if len(streams) != STREAMS:
        raise ValueError("all frozen streams are required")
    primary = []
    modified = []
    improvement = []
    rows = []
    for expected, stream in enumerate(sorted(streams, key=lambda row: row["stream_index"])):
        if stream.get("stream_index") != expected or stream.get("status") != "COMPLETE":
            raise ValueError("stream records are incomplete or reordered")
        records = {row["condition"]: row for row in stream["records"]}
        if set(records) != set(CONDITIONS):
            raise ValueError("condition records are incomplete")
        endpoint = {condition: sum(records[condition]["evaluation_loss_by_step"][str(STEPS)]) /
                    len(records[condition]["evaluation_loss_by_step"][str(STEPS)])
                    for condition in CONDITIONS}
        d256 = endpoint["ADAMW8BIT_BLOCK256"] - endpoint["FP32_ADAMW"]
        d64 = endpoint["ADAMW8BIT_BLOCK64"] - endpoint["FP32_ADAMW"]
        gain = abs(d256) - abs(d64)
        primary.append(d256); modified.append(d64); improvement.append(gain)
        rows.append({"stream_index": expected, "final_mean_evaluation_loss": endpoint,
                     "block256_minus_reference": d256,
                     "block64_minus_reference": d64,
                     "absolute_gap_reduction": gain})
    primary_interval = t_interval(primary)
    modification_interval = t_interval(improvement)
    margin = float(protocol["material_loss_margin"])
    if primary_interval[0] > margin or primary_interval[1] < -margin:
        decision = "MATERIAL_EFFECT"
    elif primary_interval[0] > 0 or primary_interval[1] < 0:
        decision = "DIFFERENT_BUT_BELOW_MATERIAL_MARGIN"
    else:
        decision = "INCONCLUSIVE"
    return {
        "schema": "optimizer-training-confirmation-summary-v1",
        "streams": rows,
        "primary": {
            "contrast": protocol["primary_contrast"],
            "paired_values": primary,
            "mean": sum(primary) / len(primary),
            "interval_95": primary_interval,
            "material_margin": margin,
            "decision": decision,
        },
        "modified_variant": {
            "paired_values": modified,
            "mean": sum(modified) / len(modified),
            "absolute_gap_reduction_values": improvement,
            "absolute_gap_reduction_interval_95": modification_interval,
            "decision": ("CONFIRMED_CLOSER_TO_REFERENCE"
                         if modification_interval[0] > 0 else "NOT_CONFIRMED"),
        },
        "claim_scope": (
            "Eight predeclared nonoverlapping WikiText token streams at one fixed "
            "Mamba checkpoint and optimizer setting"
        ),
    }


def report(output: Path) -> None:
    protocol = verify(output)
    streams = []
    for index in range(STREAMS):
        path = output / "streams" / f"stream_{index:02d}.json"
        if not path.is_file():
            raise ValueError(f"missing stream {index}")
        streams.append(load(path))
    save_new(output / "summary.json", summarize(protocol, streams))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "capture-stream", "report"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stream-index", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if not output.is_relative_to(Path("/data1/tzh")):
        raise ValueError("output must remain under /data1/tzh")
    if args.command == "freeze":
        freeze(output)
    elif args.command == "capture-stream":
        if args.stream_index is None or args.cache_dir is None:
            raise ValueError("capture-stream requires stream index and cache directory")
        capture_stream(output, args.stream_index, args.device, args.cache_dir)
    else:
        report(output)


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
    os.environ.setdefault("XDG_CACHE_HOME", "/data1/tzh/cache/xdg")
    main()
