#!/usr/bin/env python3
"""Repeated RMS/support-matched random null for the Phi lm-head repair."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
from torch._inductor.codecache import PyCodeCache

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "archive/round1_code/src"), str(ROOT / "src")]

from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_frozen_candidate_fp32_screen import wrapper_modules  # noqa: E402
from scripts.run_generated_fp32_screen import load_model  # noqa: E402
from scripts.run_phi64_lmhead_dx_repair import MMRepair  # noqa: E402
from scripts.run_targeted_full_coordinate import validate_release  # noqa: E402


def summarize(vectors: list[torch.Tensor]) -> dict[str, float]:
    resultant = torch.stack(vectors).double().sum(0)
    energy = sum(float(torch.dot(row.double(), row.double())) for row in vectors)
    scale = math.sqrt(max(energy, 0.0))
    distance = float(torch.linalg.vector_norm(resultant))
    return {
        "final_distance_l2": distance,
        "diffusive_step_scale": scale,
        "coherence_amplification": distance / max(scale, 1e-30),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, choices=(2, 8, 16, 32), default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 211, 307, 401, 503])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bank = json.loads((ROOT / "results/coverage/phi4_seq64_input_bank.json").read_text())
    states = list(bank.get("states", bank.get("records")))[:args.steps]
    if len(states) != args.steps:
        raise RuntimeError("input bank is incomplete")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    configure_candidate_runtime(24000)
    model = load_model("phi", Path("/data1/tzh/models/microsoft/Phi-4-mini-instruct"), device)
    model.eval()
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=False, dynamic=False)
    warm = torch.tensor([states[0]["token_ids"]], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True); candidate(warm).backward(); torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    capture = json.loads((ROOT / "results/coverage/runtime_releases/phi4_seq64_r1/capture.json").read_text())
    validate_release(wrapper_modules(modules), capture)
    parameter = model.model.norm.weight
    initial = parameter.detach().float().clone()
    reference = initial.clone()
    random_masters = [initial.clone() for _ in args.seeds]
    random_generators = [torch.Generator(device=device).manual_seed(seed) for seed in args.seeds]
    natural_vectors: list[torch.Tensor] = []
    random_vectors: list[list[torch.Tensor]] = [[] for _ in args.seeds]
    previous_random_drifts = [torch.zeros_like(initial) for _ in args.seeds]
    records = []

    def gradient(master: torch.Tensor, state: dict[str, Any], repair: bool) -> torch.Tensor:
        with torch.no_grad():
            parameter.copy_(master.to(parameter.dtype))
        model.zero_grad(set_to_none=True)
        values = torch.tensor([state["token_ids"]], dtype=torch.long, device=device)
        observer = MMRepair(modules, "REPAIR_FP32_CAST_BF16") if repair else None
        if observer is None:
            candidate(values).backward()
        else:
            with observer:
                candidate(values).backward()
        torch.cuda.synchronize(device)
        return parameter.grad.detach().float().clone()

    for index, state in enumerate(states):
        gc = gradient(reference, state, False)
        gr = gradient(reference, state, True)
        natural = -args.learning_rate * (gc - gr)
        natural_vectors.append(natural.detach().cpu())
        next_reference = reference - args.learning_rate * gr
        row = {"step": index + 1, "state_id": str(state.get("state_id", index)),
               "natural_local_l2": float(torch.linalg.vector_norm(natural)), "random": []}
        for arm_index, generator in enumerate(random_generators):
            master = random_masters[arm_index]
            base_gradient = gradient(master, state, True)
            signs = torch.randint(
                0, 2, natural.shape, generator=generator, device=device, dtype=torch.int8
            ).mul_(2).sub_(1).to(natural.dtype)
            injection = natural * signs
            next_master = master - args.learning_rate * base_gradient + injection
            drift = next_master - next_reference
            increment = drift - previous_random_drifts[arm_index]
            random_vectors[arm_index].append(increment.detach().cpu())
            random_masters[arm_index] = next_master
            previous_random_drifts[arm_index] = drift
            row["random"].append({
                "seed": args.seeds[arm_index],
                "requested_injection_l2": float(torch.linalg.vector_norm(natural)),
                "realized_injection_l2": float(torch.linalg.vector_norm(injection)),
                "drift_l2": float(torch.linalg.vector_norm(drift)),
            })
        reference = next_reference
        records.append(row)
        print(json.dumps({"event": "PHI_REPEATED_RANDOM_STEP", "step": index + 1}), flush=True)

    natural_summary = summarize(natural_vectors)
    nulls = [{"seed": seed, **summarize(rows)} for seed, rows in zip(args.seeds, random_vectors)]
    amplitudes = [row["coherence_amplification"] for row in nulls]
    payload = {
        "schema": "kernel-analyzer-phi-repeated-random-null-v1",
        "status": "COMPLETE" if args.steps == 32 else "ENGINEERING_DRY_RUN",
        "case": "phi4_seq64_lmhead_dx",
        "carrier": "model.norm.weight",
        "steps": args.steps,
        "natural_common_state_local_effect": natural_summary,
        "repeated_random_nulls": nulls,
        "null_amplification": {
            "minimum": min(amplitudes), "maximum": max(amplitudes),
            "mean": sum(amplitudes) / len(amplitudes),
        },
        "invariants": {
            "injected_every_step": True,
            "rms_and_support_matched_by_coordinatewise_rademacher_signs": True,
            "same_repair_background_and_data_stream": True,
            "random_seed_count": len(args.seeds),
        },
        "records": records,
        "claim_boundary": (
            "The random arms preserve each natural common-state local effective-update norm "
            "and support while destroying its coordinate signs. Closed-loop feedback is retained."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "natural": natural_summary,
                      "null_amplification": payload["null_amplification"]}, sort_keys=True))


if __name__ == "__main__":
    main()
