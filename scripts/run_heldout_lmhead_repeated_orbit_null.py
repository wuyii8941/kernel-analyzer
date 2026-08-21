#!/usr/bin/env python3
"""Repeated real-kernel reduction-orbit null for revealed lm-head cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch._inductor.codecache import PyCodeCache

from kernel_analyzer.reduction_orbit import frozen_crossfit_permutations
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime
from scripts.run_generated_fp32_screen import load_model
from scripts.run_heldout_lmhead_consequence import adam_delta, resolve_parameter
from scripts.run_qwen256_lmhead_property_confirmation import ShapeObserver


NULL_SEEDS = (20260821, 20260822, 20260823, 20260824, 20260825)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("generic", "mistral3"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--steps", type=int, choices=(2, 32), default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    bank = json.loads(args.input_bank.read_text())
    states = [row for row in bank["states"] if row["role"] == "TRAJECTORY"][:args.steps]
    if len(states) != args.steps:
        raise RuntimeError("trajectory population is incomplete")
    device = torch.device(args.device)
    configure_candidate_runtime(33_000)
    model = load_model(args.architecture, args.model, device)
    model.eval()
    resolved_carrier, carrier = resolve_parameter(model, args.carrier)
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=False, dynamic=False)
    warm = torch.tensor([states[0]["token_ids"]], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True); candidate(warm).backward(); torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    orbit = frozen_crossfit_permutations(args.vocab_size, 20260820)
    permutations = orbit["permutations"]
    left = (bank["sequence_length"], args.vocab_size)
    right = (args.vocab_size, args.hidden_size)

    choices: dict[int, list[int]] = {}
    for seed in NULL_SEEDS:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        choices[seed] = torch.randint(1, 9, (args.steps,), generator=generator).tolist()

    def gradient(master: torch.Tensor, state: dict, mode: str, permutation=None) -> torch.Tensor:
        with torch.no_grad(): carrier.copy_(master.to(carrier.dtype))
        values = torch.tensor([state["token_ids"]], dtype=torch.long, device=device)
        model.zero_grad(set_to_none=True)
        if mode == "default":
            candidate(values).backward()
        else:
            observer = ShapeObserver(
                modules, mode, permutations, left_shape=left, right_shape=right,
                selected_permutation=permutation,
            )
            with observer: candidate(values).backward()
        torch.cuda.synchronize(device)
        if carrier.grad is None: raise RuntimeError("carrier gradient is absent")
        return carrier.grad.detach().float().clone()

    initial = carrier.detach().float().clone()
    candidate_master = initial.clone(); repair_master = initial.clone()
    candidate_m = torch.zeros_like(initial); candidate_v = torch.zeros_like(initial)
    repair_m = torch.zeros_like(initial); repair_v = torch.zeros_like(initial)
    nulls = {
        seed: {
            "master": initial.clone(), "m": torch.zeros_like(initial),
            "v": torch.zeros_like(initial), "local_norms": [],
        } for seed in NULL_SEEDS
    }
    natural_local_norms = []
    records = []
    for index, state in enumerate(states):
        step = index + 1
        gc = gradient(candidate_master, state, "default")
        gr_at_c = gradient(candidate_master, state, "fp32")
        gr = gradient(repair_master, state, "fp32")
        raw_uc, next_cm, next_cv = adam_delta(
            gc, candidate_m, candidate_v, step, learning_rate=args.learning_rate
        )
        raw_ur_at_c, _, _ = adam_delta(
            gr_at_c, candidate_m, candidate_v, step, learning_rate=args.learning_rate
        )
        raw_ur, next_rm, next_rv = adam_delta(
            gr, repair_m, repair_v, step, learning_rate=args.learning_rate
        )
        next_candidate = candidate_master + raw_uc
        next_repair = repair_master + raw_ur
        natural_local = (next_candidate - candidate_master) - (
            (candidate_master + raw_ur_at_c) - candidate_master
        )
        natural_local_norm = float(torch.linalg.vector_norm(natural_local))
        natural_local_norms.append(natural_local_norm)
        candidate_master, repair_master = next_candidate, next_repair
        candidate_m, candidate_v = next_cm, next_cv
        repair_m, repair_v = next_rm, next_rv

        null_step = {}
        for seed in NULL_SEEDS:
            branch = nulls[seed]
            permutation_index = choices[seed][index]
            gp = gradient(branch["master"], state, "permuted", permutations[permutation_index])
            grn = gradient(branch["master"], state, "fp32")
            raw_up, next_nm, next_nv = adam_delta(
                gp, branch["m"], branch["v"], step, learning_rate=args.learning_rate
            )
            raw_un, _, _ = adam_delta(
                grn, branch["m"], branch["v"], step, learning_rate=args.learning_rate
            )
            next_null = branch["master"] + raw_up
            local_null = (next_null - branch["master"]) - (
                (branch["master"] + raw_un) - branch["master"]
            )
            null_norm = float(torch.linalg.vector_norm(local_null))
            branch["local_norms"].append(null_norm)
            branch["master"], branch["m"], branch["v"] = next_null, next_nm, next_nv
            null_step[str(seed)] = {
                "orbit_variant": orbit["variant_ids"][permutation_index],
                "local_update_error_l2": null_norm,
            }
            del gp, grn, raw_up, raw_un, local_null
        records.append({
            "step": step, "state_id": state["state_id"],
            "natural_local_update_error_l2": natural_local_norm,
            "nulls": null_step,
        })
        print(json.dumps({"event": "REPEATED_ORBIT_NULL_STEP", "step": step}), flush=True)
        del gc, gr_at_c, gr, raw_uc, raw_ur_at_c, raw_ur, natural_local
        torch.cuda.empty_cache()

    natural_drift = candidate_master - repair_master
    drift_vectors = [natural_drift]
    null_summaries = []
    for seed in NULL_SEEDS:
        null_drift = nulls[seed]["master"] - repair_master
        drift_vectors.append(null_drift)
        natural_norm = float(torch.linalg.vector_norm(natural_drift))
        null_norm = float(torch.linalg.vector_norm(null_drift))
        cosine = float(torch.sum(natural_drift * null_drift) / max(natural_norm * null_norm, 1e-30))
        ratios = [
            value / max(reference, 1e-30)
            for value, reference in zip(nulls[seed]["local_norms"], natural_local_norms)
        ]
        null_summaries.append({
            "seed": seed, "final_drift_l2": null_norm,
            "final_norm_over_natural": null_norm / max(natural_norm, 1e-30),
            "final_cosine_with_natural": cosine,
            "local_error_norm_ratio_mean": sum(ratios) / len(ratios),
            "local_error_norm_ratio_min": min(ratios), "local_error_norm_ratio_max": max(ratios),
        })
    drift_matrix = torch.stack(drift_vectors).double()
    gram = drift_matrix @ drift_matrix.T
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0)
    effective_rank = float(eigenvalues.sum().square() / max(float(eigenvalues.square().sum()), 1e-30))
    payload = {
        "schema": "kernel-analyzer-heldout-repeated-real-orbit-null-v1",
        "status": "RETROSPECTIVE_MECHANISM_DIAGNOSTIC" if args.steps == 32 else "ENGINEERING_DRY_RUN",
        "model": str(args.model.resolve()), "architecture": args.architecture,
        "carrier": {"declared": args.carrier, "resolved_runtime_name": resolved_carrier},
        "carrier_coordinates": carrier.numel(), "steps": args.steps,
        "null_seeds": list(NULL_SEEDS), "orbit_variant_choices": choices,
        "natural_final_drift_l2": float(torch.linalg.vector_norm(natural_drift)),
        "nulls": null_summaries, "drift_gram": gram.tolist(),
        "drift_effective_rank_participation_ratio": effective_rank,
        "records": records,
        "claim_boundary": (
            "Every null uses a real semantics-preserving captured-operand reduction orbit at "
            "every step. This is retrospective on revealed models and is not a confirmatory null."
        ),
    }
    payload["result_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
