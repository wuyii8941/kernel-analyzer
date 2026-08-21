#!/usr/bin/env python3
"""Complete-coordinate 16-state formation screen for Gemma-4 PLE/RMSNorm."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import torch
from torch._inductor.codecache import PyCodeCache

from scripts.generated_fp32_observer import GeneratedFP32Observer
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime
from scripts.run_frozen_candidate_fp32_screen import wrapper_modules
from scripts.run_generated_fp32_screen import load_model


CARRIERS = (
    "model.language_model.embed_tokens.weight",
    "model.language_model.per_layer_model_projection.weight",
)


def chunk_square(value: torch.Tensor) -> float:
    result = 0.0
    flat = value.reshape(-1)
    for start in range(0, flat.numel(), 1 << 22):
        chunk = flat[start:start + (1 << 22)]
        norm = float(torch.linalg.vector_norm(chunk).item())
        result += norm * norm
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--runtime-release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--runtime-seed", type=int, default=20260821)
    args = parser.parse_args()

    bank = json.loads(args.input_bank.read_text())
    states = [row for row in bank["states"] if row["role"] == "CONFIRMATION"]
    if len(states) != 16:
        raise RuntimeError("formation requires the frozen 16 confirmation states")
    device = torch.device(args.device)
    configure_candidate_runtime(args.runtime_seed)
    model = load_model("gemma4", args.model, device)
    parameters = dict(model.named_parameters())
    if any(name not in parameters for name in CARRIERS):
        raise RuntimeError("declared carrier absent")
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=False, dynamic=False)
    # Reproduce the exact engineering-state wrapper used by the frozen release;
    # confirmation values are introduced only after identity validation.
    warm_state = bank["states"][0]
    warm = torch.tensor([warm_state["token_ids"]], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True); candidate(warm).backward(); torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    capture = json.loads((args.runtime_release / "capture.json").read_text())
    expected = [row["sha256"] for row in capture["modules"]]
    observed = [
        hashlib.sha256(Path(module.__file__).resolve().read_bytes()).hexdigest()
        for module, _ in wrapper_modules(modules)
    ]
    if observed != expected:
        raise RuntimeError("runtime wrapper bytes differ from frozen release")
    with gzip.open(args.runtime_release / "campaign.json.gz", "rt") as handle:
        campaign = json.load(handle)
    choices = sorted(
        (
            row for row in campaign["rows"]
            if row["phase"] == "FORWARD" and "embedding_mean_mul_pow_view" in row["symbol"]
        ),
        key=lambda row: (row["source_path"], row["source_line"], row["region_id"]),
    )
    if not choices:
        raise RuntimeError("frozen PLE/RMSNorm target absent")
    target = choices[0]
    repair_targets = {target["region_id"]: target["output_names"]}

    total = {name: torch.zeros(parameters[name].shape, dtype=torch.float32) for name in CARRIERS}
    odd = {name: torch.zeros_like(total[name]) for name in CARRIERS}
    even = {name: torch.zeros_like(total[name]) for name in CARRIERS}
    energy = 0.0
    records = []
    for index, state in enumerate(states):
        values = torch.tensor([state["token_ids"]], dtype=torch.long, device=device)
        seed = 20260821 + index
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True); candidate(values).backward(); torch.cuda.synchronize(device)
        baseline = {
            name: parameters[name].grad.detach().float().cpu().clone() for name in CARRIERS
        }
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True)
        observer = GeneratedFP32Observer(
            modules=modules, campaign_rows=[target], repair_targets=repair_targets,
            allow_unlisted_calls=True,
        )
        with observer: candidate(values).backward()
        torch.cuda.synchronize(device)
        state_energy = 0.0
        for name in CARRIERS:
            delta = parameters[name].grad.detach().float().cpu() - baseline[name]
            total[name].add_(delta)
            (odd if index % 2 else even)[name].add_(delta)
            state_energy += chunk_square(delta)
            del delta, baseline[name]
        energy += state_energy
        records.append({
            "state_id": state["state_id"], "complete_delta_l2": state_energy**0.5,
            "target_executions": sum(
                row["region_id"] == target["region_id"] for row in observer.summary()["records"]
            ),
        })
        print(json.dumps({"event": "GEMMA4_NORM_FORMATION_STATE", "step": index + 1}), flush=True)
        del values
        torch.cuda.empty_cache()

    total_energy = sum(chunk_square(value) for value in total.values())
    odd_energy = sum(chunk_square(value) for value in odd.values())
    even_energy = sum(chunk_square(value) for value in even.values())
    odd_even_inner = 0.0
    for name in CARRIERS:
        left, right = odd[name].reshape(-1), even[name].reshape(-1)
        for start in range(0, left.numel(), 1 << 22):
            odd_even_inner += float(torch.sum(
                left[start:start + (1 << 22)] * right[start:start + (1 << 22)]
            ).item())
    amplification = (total_energy / max(energy, 1e-30))**0.5
    payload = {
        "schema": "kernel-analyzer-gemma4-norm-formation-v1",
        "status": "COMPLETE_COMPLETE_COORDINATES",
        "target": {
            "region_id": target["region_id"], "symbol": target["symbol"],
            "endpoints": target["output_names"], "repeated_regions_in_denominator": len(choices),
        },
        "carriers": list(CARRIERS), "states": 16,
        "complete_coordinate_statistics": {
            "path_energy": energy, "resultant_energy": total_energy,
            "coherence_amplification": amplification,
            "odd_even_resultant_cosine": odd_even_inner / max((odd_energy * even_energy)**0.5, 1e-30),
        },
        "records": records,
        "claim_boundary": (
            "Complete declared parameter coordinates and exact F+B repair. This formation screen "
            "does not use trajectory drift and does not by itself establish persistence."
        ),
    }
    payload["result_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
