#!/usr/bin/env python3
"""Complete structured carrier screen for the frozen seq1024 campaign.

Every parameter coordinate belongs to exactly one 128-vector or 128x128 tile.
An overlapping whole-parameter direction is also measured.  Directions are
formed only from all step-0 calibration states; steps 64..4096 are held out.
The artifact stores compact certificates and triggered blocks, not gradients.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

if __package__:
    from scripts.long_horizon_trigger import (
        atomic_json,
        build_model,
        load_eval_states,
        load_milestone,
        run_backward,
        under_root,
    )
else:
    from long_horizon_trigger import (
        atomic_json,
        build_model,
        load_eval_states,
        load_milestone,
        run_backward,
        under_root,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("results/final/long_horizon_bank.json"))
    parser.add_argument("--atlas", type=Path, default=Path("results/final/invocation_atlas.json"))
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument("--output", type=Path, default=Path("results/final/structured_carrier_trigger.json"))
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tile_count(shape: tuple[int, ...], block_size: int) -> int:
    if len(shape) == 1:
        return math.ceil(shape[0] / block_size)
    if len(shape) == 2:
        return math.ceil(shape[0] / block_size) * math.ceil(shape[1] / block_size)
    raise ValueError(f"only vector and matrix parameters are supported: {shape}")


def partition_sums(value: Any, block_size: int, torch: Any) -> Any:
    """Sum each disjoint complete tile without dropping boundary coordinates."""
    import torch.nn.functional as functional

    if value.ndim == 1:
        padding = (-value.shape[0]) % block_size
        padded = functional.pad(value, (0, padding)) if padding else value
        return padded.reshape(-1, block_size).sum(dim=1)
    if value.ndim == 2:
        rows, columns = value.shape
        row_padding = (-rows) % block_size
        column_padding = (-columns) % block_size
        padded = (
            functional.pad(value, (0, column_padding, 0, row_padding))
            if row_padding or column_padding
            else value
        )
        row_blocks = padded.shape[0] // block_size
        column_blocks = padded.shape[1] // block_size
        return padded.reshape(row_blocks, block_size, column_blocks, block_size).sum(dim=(1, 3)).reshape(-1)
    raise ValueError(f"only vector and matrix parameters are supported: {tuple(value.shape)}")


def block_metadata(shape: tuple[int, ...], block_size: int, index: int) -> dict[str, Any]:
    if index == 0:
        return {"level": "PARAMETER", "coordinate_count": math.prod(shape)}
    tile = index - 1
    if len(shape) == 1:
        start = tile * block_size
        stop = min(start + block_size, shape[0])
        return {"level": "VECTOR_BLOCK", "start": start, "stop": stop, "coordinate_count": stop - start}
    column_blocks = math.ceil(shape[1] / block_size)
    row_block, column_block = divmod(tile, column_blocks)
    row_start, column_start = row_block * block_size, column_block * block_size
    row_stop = min(row_start + block_size, shape[0])
    column_stop = min(column_start + block_size, shape[1])
    return {
        "level": "MATRIX_TILE",
        "row_start": row_start,
        "row_stop": row_stop,
        "column_start": column_start,
        "column_stop": column_stop,
        "coordinate_count": (row_stop - row_start) * (column_stop - column_start),
    }


def stats(delta: Any, pilot: Any, pilot_norms: Any, block_size: int, torch: Any) -> tuple[Any, Any]:
    dot_tiles = partition_sums(delta * pilot, block_size, torch)
    current_norm_tiles = partition_sums(delta.square(), block_size, torch).sqrt()
    global_dot = torch.dot(delta.reshape(-1), pilot.reshape(-1)).reshape(1)
    global_current_norm = delta.norm().reshape(1)
    dots = torch.cat((global_dot, dot_tiles))
    current_norms = torch.cat((global_current_norm, current_norm_tiles))
    valid = (pilot_norms > 0.0) & (current_norms > 0.0)
    projection = torch.full_like(dots, float("nan"))
    cosine = torch.full_like(dots, float("nan"))
    projection[valid] = dots[valid] / pilot_norms[valid]
    cosine[valid] = dots[valid] / (pilot_norms[valid] * current_norms[valid])
    return projection, cosine


def longest_run(values: Any, torch: Any) -> Any:
    current = torch.zeros(values.shape[1], dtype=torch.int16)
    longest = torch.zeros_like(current)
    for row in values:
        current = torch.where(row, current + 1, torch.zeros_like(current))
        longest = torch.maximum(longest, current)
    return longest


def summarize_parameter(
    name: str,
    shape: tuple[int, ...],
    projections: Any,
    cosines: Any,
    pilot_norms: Any,
    *,
    block_size: int,
    bootstrap: int,
    torch: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # [checkpoint, state, repeat, block]
    finite = torch.isfinite(projections) & torch.isfinite(cosines)
    valid = finite.all(dim=(0, 1, 2)) & (pilot_norms > 0.0)
    state_means = projections.mean(dim=2)
    checkpoint_means = state_means.mean(dim=1)
    checkpoint_cosines = cosines.mean(dim=(1, 2))
    repeat_sign_agreement = (
        (projections[:, :, 0, :] > 0.0) == (projections[:, :, 1, :] > 0.0)
    ).float().mean(dim=(0, 1))

    positive_run = longest_run(checkpoint_means > 0.0, torch)
    negative_run = longest_run(checkpoint_means < 0.0, torch)
    candidates: list[tuple[int, int, int]] = []
    # tuple: block, sign, window start; prefer five checkpoints then four.
    for sign, runs in ((1, positive_run), (-1, negative_run)):
        signed_means = checkpoint_means * sign
        signed_cosines = checkpoint_cosines * sign
        for length in (5, 4):
            for start in range(0, 5 - length + 1):
                mask = (
                    valid
                    & (runs >= 4)
                    & (signed_means[start : start + length] > 0.0).all(dim=0)
                    & (signed_cosines[start : start + length].min(dim=0).values > 0.1)
                    & (repeat_sign_agreement >= 0.875)
                )
                for block in torch.nonzero(mask, as_tuple=False).reshape(-1).tolist():
                    candidates.append((block, sign, start * 10 + length))

    # Deduplicate multiple valid windows per block/direction.  Bootstrap and
    # retain the window with the strongest lower confidence bound.
    grouped: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for block, sign, encoded in candidates:
        grouped.setdefault((block, sign), set()).add((encoded // 10, encoded % 10))
    generator = torch.Generator(device="cpu")
    seed = int.from_bytes(hashlib.sha256(f"structured:{name}".encode()).digest()[:8], "little") % (2**31)
    generator.manual_seed(seed)
    resamples = torch.randint(0, state_means.shape[1], (bootstrap, state_means.shape[1]), generator=generator)
    triggers = []
    for (block, sign), windows in grouped.items():
        best = None
        for start, length in windows:
            per_state = state_means[start : start + length, :, block].mean(dim=0) * sign
            samples = per_state[resamples].mean(dim=1)
            lower = float(torch.quantile(samples, 0.025))
            candidate = {
                "steps": [64, 256, 1024, 2048, 4096][start : start + length],
                "signed_projection_mean": float(per_state.mean()),
                "signed_projection_cluster_bootstrap_lower_95": lower,
                "minimum_signed_checkpoint_mean_cosine": float(
                    (checkpoint_cosines[start : start + length, block] * sign).min()
                ),
            }
            if best is None or lower > best["signed_projection_cluster_bootstrap_lower_95"]:
                best = candidate
        if best is None or best["signed_projection_cluster_bootstrap_lower_95"] <= 0.0:
            continue
        triggers.append({
            "parameter": name,
            "block_index": block,
            "direction_relative_to_step0_pilot": "POSITIVE" if sign > 0 else "NEGATIVE",
            "repeat_sign_agreement_fraction": float(repeat_sign_agreement[block]),
            "longest_signed_checkpoint_run": int(positive_run[block] if sign > 0 else negative_run[block]),
            "pilot_l2": float(pilot_norms[block]),
            **block_metadata(shape, block_size, block),
            "best_window": best,
        })
    triggers.sort(key=lambda row: -row["best_window"]["signed_projection_cluster_bootstrap_lower_95"])
    partition_blocks = tile_count(shape, block_size)
    summary = {
        "name": name,
        "shape": list(shape),
        "numel": math.prod(shape),
        "partition_blocks": partition_blocks,
        "auxiliary_global_directions": 1,
        "defined_pilot_directions": int((pilot_norms > 0.0).sum()),
        "abstained_zero_pilot_directions": int((pilot_norms == 0.0).sum()),
        "triggered_directions": len(triggers),
        "triggered_partition_blocks": sum(row["level"] != "PARAMETER" for row in triggers),
        "global_direction_triggered": any(row["level"] == "PARAMETER" for row in triggers),
    }
    return summary, triggers


def main() -> None:
    args = parse_args()
    if args.seq_len != 1024 or args.states != 8 or args.repeats != 2 or args.block_size != 128:
        raise ValueError("frozen protocol requires seq1024, 8 states, 2 repeats and block size 128")
    bank_path = under_root(args.bank, "bank")
    atlas_path = under_root(args.atlas, "atlas")
    model_path = under_root(args.model, "model")
    output_path = under_root(args.output, "output")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
    os.environ.setdefault("HF_DATASETS_CACHE", "/data1/tzh/cache/huggingface/datasets")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/data1/tzh/cache/huggingface/transformers")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data1/tzh/cache/kernel_analyzer/structured_trigger_compile")

    import torch
    from torch._dynamo.backends.registry import lookup_backend
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    bank = json.loads(bank_path.read_text())
    atlas = json.loads(atlas_path.read_text())
    milestones = bank["milestones"]
    if bank["status"] != "COMPLETE" or [row["step"] for row in milestones] != [0, 64, 256, 1024, 2048, 4096]:
        raise RuntimeError("long-horizon bank is incomplete or has a different milestone grid")
    if atlas["denominator"]["real_changed_sites_without_exact_fbv_binding"] != 0:
        raise RuntimeError("unified invocation atlas is not closed")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    states, evaluation = load_eval_states(tokenizer, args.seq_len, args.states, device)
    model = build_model(model_path, device)
    shapes = {name: tuple(parameter.shape) for name, parameter in model.named_parameters()}
    if any(len(shape) not in (1, 2) for shape in shapes.values()):
        raise RuntimeError("unexpected non-vector/non-matrix parameter")

    class LossStep(torch.nn.Module):
        def __init__(self, subject: Any) -> None:
            super().__init__()
            self.subject = subject

        def forward(self, input_ids: Any, labels: Any) -> Any:
            return self.subject(input_ids=input_ids, labels=labels, use_cache=False, return_dict=False)[0]

    audit = {"backend_compiles": 0, "runtime_invocations": 0, "graph_sha256": []}
    inductor = lookup_backend("inductor")

    def backend(graph_module: Any, example_inputs: list[Any]) -> Any:
        audit["backend_compiles"] += 1
        audit["graph_sha256"].append(hashlib.sha256(graph_module.code.encode()).hexdigest())
        compiled = inductor(graph_module, example_inputs)

        def counted(*values: Any) -> Any:
            audit["runtime_invocations"] += 1
            return compiled(*values)
        return counted

    candidate = torch.compile(LossStep(model), backend=backend, fullgraph=True, dynamic=False)
    model.zero_grad(set_to_none=True)
    warm = candidate(states[0][0], states[0][1])
    warm.backward()
    torch.cuda.synchronize(device)
    del warm

    heldout_steps = [64, 256, 1024, 2048, 4096]
    pilot_sums: dict[str, Any] = {}
    pilot_norms: dict[str, Any] = {}
    projections: dict[str, Any] = {}
    cosines: dict[str, Any] = {}
    result: dict[str, Any] = {
        "schema": "kernel-analyzer-structured-carrier-trigger-v1",
        "status": "RUNNING",
        "subject": "Qwen3-1.7B full-step eager versus Inductor structured carrier screen",
        "sources": {
            "bank": str(bank_path), "bank_sha256": sha256(bank_path),
            "unified_invocation_atlas": str(atlas_path), "unified_invocation_atlas_sha256": sha256(atlas_path),
        },
        "candidate": {"backend": "Inductor", "fullgraph": True, "changed_f_b_units": atlas["denominator"]["changed_fbv_units"]},
        "reference": {"backend": "eager", "dtype": "bfloat16", "tf32": False},
        "evaluation": evaluation,
        "calibration": {"checkpoint_step": 0, "state_ids": list(range(args.states)), "repeats": args.repeats},
        "heldout_checkpoint_steps": heldout_steps,
        "partition": {
            "block_size": args.block_size,
            "vectors": "disjoint contiguous blocks of at most 128 coordinates",
            "matrices": "disjoint row-major tiles of at most 128x128 coordinates",
            "coordinate_omission_allowed": False,
            "overlapping_auxiliary_direction": "one whole-parameter direction",
        },
        "loss_rows": [], "reference_repeat_rows": [], "completed_milestones": [],
        "full_gradient_tensors_saved": False,
    }

    for milestone_index, milestone in enumerate(milestones):
        step = int(milestone["step"])
        print(f"milestone {step}: loading weights", flush=True)
        load_milestone(model, milestone, model_path)
        references = []
        for state_id, inputs in enumerate(states):
            loss0, reference = run_backward(model, inputs)
            loss1, repeated = run_backward(model, inputs)
            exact = True
            maximum = 0.0
            for name in reference:
                delta = repeated[name].float() - reference[name].float()
                exact = exact and not bool(torch.count_nonzero(delta))
                maximum = max(maximum, float(delta.abs().max()))
            result["reference_repeat_rows"].append({
                "checkpoint_step": step, "state_id": state_id,
                "loss_delta": loss1 - loss0, "all_parameter_repeat_exact": exact,
                "max_parameter_repeat_abs": maximum,
            })
            references.append((loss0, reference))
            del repeated

        for state_id, inputs in enumerate(states):
            reference_loss, reference = references[state_id]
            for repeat in range(args.repeats):
                candidate_loss, observed = run_backward(model, inputs, candidate)
                result["loss_rows"].append({
                    "checkpoint_step": step, "state_id": state_id, "repeat": repeat,
                    "reference_loss": reference_loss, "candidate_loss": candidate_loss,
                    "loss_delta": candidate_loss - reference_loss,
                })
                for name in sorted(shapes):
                    delta = observed[name].float() - reference[name].float()
                    if step == 0:
                        if name not in pilot_sums:
                            pilot_sums[name] = torch.zeros_like(delta)
                        pilot_sums[name].add_(delta)
                    else:
                        checkpoint_index = heldout_steps.index(step)
                        projection, cosine = stats(delta, pilot_sums[name], pilot_norms[name], args.block_size, torch)
                        projections[name][checkpoint_index, state_id, repeat] = projection
                        cosines[name][checkpoint_index, state_id, repeat] = cosine
                    del delta
                print(json.dumps({"step": step, "state": state_id, "repeat": repeat, "loss_delta": candidate_loss - reference_loss}), flush=True)
                del observed
        del references

        if step == 0:
            calibration_count = args.states * args.repeats
            for name, value in pilot_sums.items():
                value.div_(calibration_count)
                tile_norms = partition_sums(value.square(), args.block_size, torch).sqrt()
                pilot_norms[name] = torch.cat((value.norm().reshape(1), tile_norms))
                count = 1 + tile_count(shapes[name], args.block_size)
                projections[name] = torch.empty((5, args.states, args.repeats, count), dtype=torch.float32)
                cosines[name] = torch.empty_like(projections[name])

        gc.collect()
        result["completed_milestones"].append(step)
        result["compile_audit"] = audit
        atomic_json(output_path, result)

    parameter_summaries = []
    triggers = []
    for index, name in enumerate(sorted(shapes)):
        summary, parameter_triggers = summarize_parameter(
            name, shapes[name], projections[name], cosines[name], pilot_norms[name],
            block_size=args.block_size, bootstrap=args.bootstrap, torch=torch,
        )
        parameter_summaries.append(summary)
        triggers.extend(parameter_triggers)
        print(json.dumps({"summarized": index + 1, "parameter": name, "triggers": len(parameter_triggers)}), flush=True)

    triggers.sort(key=lambda row: -row["best_window"]["signed_projection_cluster_bootstrap_lower_95"])
    unique_parameter_coordinates = sum(math.prod(shape) for shape in shapes.values())
    partition_coordinates = sum(
        block_metadata(shape, args.block_size, index)["coordinate_count"]
        for shape in shapes.values()
        for index in range(1, 1 + tile_count(shape, args.block_size))
    )
    result["coverage"] = {
        "parameters": len(shapes),
        "unique_parameter_coordinates": unique_parameter_coordinates,
        "partition_coordinate_memberships": partition_coordinates,
        "partition_blocks": sum(tile_count(shape, args.block_size) for shape in shapes.values()),
        "auxiliary_global_directions": len(shapes),
        "sampled_coordinate_subset": False,
        "every_coordinate_in_exactly_one_partition_block": partition_coordinates == unique_parameter_coordinates,
    }
    result["gate"] = {
        "consecutive_heldout_checkpoints": 4,
        "cluster_bootstrap_signed_lower_bound_positive": True,
        "minimum_signed_mean_cosine": 0.1,
        "repeat_sign_agreement_fraction": 0.875,
        "causal_intervention_required_after_screen": True,
    }
    result["parameter_summaries"] = parameter_summaries
    result["triggers"] = triggers
    result["trigger_count"] = len(triggers)
    result["triggered_parameters"] = sorted({row["parameter"] for row in triggers})
    result["natural_case_added"] = False
    result["case_claim_boundary"] = "A structured trigger is not a natural case until exact region/group replacement, sham restoration and live-weight repair close the F+B mechanism."
    result["status"] = "COMPLETE"
    result["result_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    atomic_json(output_path, result)
    print(json.dumps({"status": result["status"], "trigger_count": len(triggers), "triggered_parameters": result["triggered_parameters"]}), flush=True)


if __name__ == "__main__":
    main()
