#!/usr/bin/env python3
"""Four-arm paired consequence trajectory for a held-out lm-head dX repair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch._inductor.codecache import PyCodeCache

from kernel_analyzer.persistence_property import aligned_level_statistics_from_gram
from kernel_analyzer.reduction_orbit import frozen_permutations
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime
from scripts.run_generated_fp32_screen import load_model
from scripts.run_qwen256_lmhead_property_confirmation import ShapeObserver
from kernel_analyzer.short_persistence import SharedShortPersistenceScreen


def adam_delta(
    gradient: torch.Tensor, first: torch.Tensor, second: torch.Tensor, step: int,
    *, learning_rate: float, beta1: float = 0.9, beta2: float = 0.95,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    next_first = first.mul(beta1).add(gradient, alpha=1.0 - beta1)
    next_second = second.mul(beta2).addcmul(gradient, gradient, value=1.0 - beta2)
    delta = -learning_rate * (next_first / (1.0 - beta1**step)) / (
        (next_second / (1.0 - beta2**step)).sqrt() + epsilon
    )
    return delta, next_first, next_second


def resolve_parameter(model: torch.nn.Module, declared: str) -> tuple[str, torch.nn.Parameter]:
    parameters = dict(model.named_parameters())
    if declared in parameters:
        return declared, parameters[declared]
    matches = [name for name in parameters if name.endswith(f".{declared}")]
    if len(matches) != 1:
        raise RuntimeError(f"carrier absent or ambiguous: {declared}; matches={matches}")
    return matches[0], parameters[matches[0]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("generic", "mistral3"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument(
        "--state-role", choices=("TRAJECTORY", "CONFIRMATION", "SCREENING"),
        default="TRAJECTORY",
        help="Frozen input-bank role used for the consequence/screen run.",
    )
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--steps", type=int, choices=(2, 8, 16, 32), default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--matched-random-null", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--short-screen-output", type=Path)
    parser.add_argument("--short-screen-steps", type=int, default=8)
    parser.add_argument("--short-screen-projection-dim", type=int, default=256)
    parser.add_argument("--short-screen-null-draws", type=int, default=2000)
    args = parser.parse_args()

    prediction = json.loads(args.prediction.read_text())
    if prediction["status"] != "PREDICTION_FROZEN":
        raise RuntimeError("held-out predictor was not frozen before consequence")
    bank = json.loads(args.input_bank.read_text())
    states = [row for row in bank["states"] if row["role"] == args.state_role][:args.steps]
    if len(states) != args.steps:
        raise RuntimeError(f"{args.state_role} population is incomplete")
    if args.short_screen_output is not None and not 4 <= args.short_screen_steps <= min(16, args.steps):
        raise ValueError("short screen steps must be in [4, min(16, trajectory steps)]")
    state_ids = [str(row["state_id"]) for row in states]
    left = (bank["sequence_length"], args.vocab_size)
    right = (args.vocab_size, args.hidden_size)
    device = torch.device(args.device)
    configure_candidate_runtime(32_000)
    model = load_model(args.architecture, args.model, device)
    model.eval()
    resolved_carrier, carrier = resolve_parameter(model, args.carrier)
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=False, dynamic=False)
    warm = torch.tensor([states[0]["token_ids"]], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True); candidate(warm).backward(); torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    dummy_permutations = frozen_permutations(args.vocab_size, 2, 20260820)

    initial = carrier.detach().float().clone()
    candidate_master = initial.clone(); repair_master = initial.clone()
    candidate_m = torch.zeros_like(initial); candidate_v = torch.zeros_like(initial)
    repair_m = torch.zeros_like(initial); repair_v = torch.zeros_like(initial)
    null_master = initial.clone()
    null_m = torch.zeros_like(initial); null_v = torch.zeros_like(initial)
    null_generator = torch.Generator(device=device).manual_seed(20260820)
    short_screen = (
        SharedShortPersistenceScreen(
            projection_dim=args.short_screen_projection_dim,
            projection_seed=20260822,
            expected_steps=args.short_screen_steps,
            null_draws=args.short_screen_null_draws,
            prefix_growth_mode="after_warmup",
        ) if args.short_screen_output is not None else None
    )

    def gradient(master: torch.Tensor, state: dict, repair: bool) -> torch.Tensor:
        with torch.no_grad():
            carrier.copy_(master.to(carrier.dtype))
        values = torch.tensor([state["token_ids"]], dtype=torch.long, device=device)
        model.zero_grad(set_to_none=True)
        if repair:
            observer = ShapeObserver(
                modules, "fp32", dummy_permutations,
                left_shape=left, right_shape=right,
            )
            with observer:
                candidate(values).backward()
        else:
            candidate(values).backward()
        torch.cuda.synchronize(device)
        if carrier.grad is None:
            raise RuntimeError("declared carrier gradient is absent")
        return carrier.grad.detach().float().clone()

    vectors: list[torch.Tensor] = []
    actual_sum = torch.zeros_like(initial)
    records = []
    initial_drift = candidate_master - repair_master
    for index, state in enumerate(states):
        step = index + 1
        gc_c = gradient(candidate_master, state, False)
        gr_c = gradient(candidate_master, state, True)
        gc_r = gradient(repair_master, state, False)
        gr_r = gradient(repair_master, state, True)
        raw_uc_c, next_cm, next_cv = adam_delta(
            gc_c, candidate_m, candidate_v, step, learning_rate=args.learning_rate
        )
        raw_ur_c, _, _ = adam_delta(
            gr_c, candidate_m, candidate_v, step, learning_rate=args.learning_rate
        )
        raw_uc_r, _, _ = adam_delta(
            gc_r, repair_m, repair_v, step, learning_rate=args.learning_rate
        )
        raw_ur_r, next_rm, next_rv = adam_delta(
            gr_r, repair_m, repair_v, step, learning_rate=args.learning_rate
        )
        # Step means the update actually written to the FP32 master, including
        # master-write rounding.  Pre-write Adam deltas do not telescope to the
        # observed parameter drift and therefore are not valid consequence arms.
        next_candidate_master = candidate_master + raw_uc_c
        next_repair_master = repair_master + raw_ur_r
        uc_c = next_candidate_master - candidate_master
        ur_c = (candidate_master + raw_ur_c) - candidate_master
        uc_r = (repair_master + raw_uc_r) - repair_master
        ur_r = next_repair_master - repair_master
        local = 0.5 * ((uc_c - ur_c) + (uc_r - ur_r))
        feedback = 0.5 * ((uc_c - uc_r) + (ur_c - ur_r))
        drift_before = candidate_master - repair_master
        drift_after = next_candidate_master - next_repair_master
        actual = drift_after - drift_before
        residual = actual - local - feedback
        if float(torch.linalg.vector_norm(residual)) > 2e-6 * max(
            float(torch.linalg.vector_norm(actual)), 1e-30
        ):
            raise RuntimeError("symmetric consequence recurrence failed to close")
        null_record = {}
        if args.matched_random_null:
            gr_n = gradient(null_master, state, True)
            raw_ur_n, next_nm, next_nv = adam_delta(
                gr_n, null_m, null_v, step, learning_rate=args.learning_rate
            )
            null_base = null_master + raw_ur_n
            if step == 1:
                signs = torch.randint(
                    0, 2, local.shape, generator=null_generator, device=device,
                    dtype=torch.int8,
                ).mul_(2).sub_(1).to(local.dtype)
                selected_request = local * signs
                next_null_master = null_base + selected_request
                realized_injection = next_null_master - null_base
            else:
                selected_request = torch.zeros_like(local)
                next_null_master = null_base
                realized_injection = torch.zeros_like(local)
            requested_norm = float(torch.linalg.vector_norm(selected_request))
            realized_norm = float(torch.linalg.vector_norm(realized_injection))
            null_norm_relative_error = (
                abs(realized_norm - requested_norm) / max(requested_norm, 1e-30)
                if step == 1 else 0.0
            )
            null_master = next_null_master
            null_m, null_v = next_nm, next_nv
            null_drift = null_master - next_repair_master
            null_record = {
                "matched_random_requested_l2": float(torch.linalg.vector_norm(selected_request)),
                "matched_random_realized_l2": realized_norm,
                "matched_random_norm_relative_error": null_norm_relative_error,
                "matched_random_master_drift_l2": float(torch.linalg.vector_norm(null_drift)),
            }
            del gr_n, raw_ur_n, selected_request, realized_injection
        if short_screen is not None and step <= args.short_screen_steps:
            short_screen.add(
                f"{resolved_carrier}::local",
                local.detach().float().cpu().numpy().reshape(-1),
            )
            short_screen.add(
                f"{resolved_carrier}::feedback",
                feedback.detach().float().cpu().numpy().reshape(-1),
            )
            short_screen.add(
                f"{resolved_carrier}::actual",
                actual.detach().float().cpu().numpy().reshape(-1),
            )
        if short_screen is None:
            vectors.extend((local.cpu(), feedback.cpu(), actual.cpu()))
        candidate_master = next_candidate_master
        repair_master = next_repair_master
        candidate_m, candidate_v = next_cm, next_cv
        repair_m, repair_v = next_rm, next_rv
        drift = drift_after
        actual_sum.add_(actual)
        records.append({
            "step": step, "state_id": state_ids[index],
            "local_l2": float(torch.linalg.vector_norm(local)),
            "feedback_l2": float(torch.linalg.vector_norm(feedback)),
            "actual_increment_l2": float(torch.linalg.vector_norm(actual)),
            "master_drift_l2": float(torch.linalg.vector_norm(drift)),
            "recurrence_residual_l2": float(torch.linalg.vector_norm(residual)),
            **null_record,
        })
        print(json.dumps({"event": "HELDOUT_CONSEQUENCE_STEP", **records[-1]}), flush=True)
        del gc_c, gr_c, gc_r, gr_r, uc_c, ur_c, uc_r, ur_r, local, feedback, actual
        torch.cuda.empty_cache()

    statistics = None
    if vectors:
        matrix = torch.stack(vectors).double()
        statistics = aligned_level_statistics_from_gram(
            (matrix @ matrix.T).numpy(), state_ids=state_ids,
            level_ids=("local", "feedback", "actual"), sign_flip_draws=4000,
            seed=20260820,
        )
    final_drift = candidate_master - repair_master
    summed_actual = (
        matrix.reshape(args.steps, 3, -1)[:, 2].sum(0)
        if vectors else actual_sum.cpu().double()
    )
    telescoping_residual = summed_actual - (final_drift - initial_drift).cpu().double()
    null_summary = None
    if args.matched_random_null:
        final_null_drift = null_master - repair_master
        null_summary = {
            "kind": "ONE_SHOT_RADEMACHER_SIGN_SCRAMBLE_OF_STEP1_LOCAL_EFFECTIVE_UPDATE",
            "same_step1_support_and_requested_l2": True,
            "max_realized_norm_relative_error": max(
                row["matched_random_norm_relative_error"] for row in records
            ),
            "realized_norm_match_within_10_percent_every_step": all(
                row["matched_random_norm_relative_error"] <= 0.10 for row in records
            ),
            "final_null_drift_l2": float(torch.linalg.vector_norm(final_null_drift)),
            "natural_null_final_drift_cosine": float(
                torch.sum(final_drift * final_null_drift)
                / max(
                    float(torch.linalg.vector_norm(final_drift)
                          * torch.linalg.vector_norm(final_null_drift)),
                    1e-30,
                )
            ),
        }
    payload = {
        "schema": "kernel-analyzer-heldout-lmhead-consequence-v1",
        "status": "COMPLETE" if args.steps == 32 else "ENGINEERING_DRY_RUN",
        "model": str(args.model.resolve()), "architecture": args.architecture,
        "prediction_file": str(args.prediction.resolve()),
        "prediction_revealed_before_trajectory": prediction["prediction"],
        "input_bank": str(args.input_bank.resolve()),
        "state_role": args.state_role,
        "carrier": {"declared": args.carrier, "resolved_runtime_name": resolved_carrier},
        "carrier_coordinates": carrier.numel(), "steps": args.steps,
        "optimizer": {"name": "AdamW", "learning_rate": args.learning_rate,
                      "betas": [0.9, 0.95], "epsilon": 1e-8, "weight_decay": 0.0},
        "records": records, "statistics": statistics,
        "final_master_drift_l2": float(torch.linalg.vector_norm(final_drift)),
        "telescoping_residual_l2": float(torch.linalg.vector_norm(telescoping_residual)),
        "matched_random_feedback_null": null_summary,
        "only_declared_parameter_updated": True,
        "claim_boundary": (
            "Four-arm symmetric one-parameter consequence trajectory.  COMPLETE measures "
            "32-step drift but does not by itself establish an empirical-random-null threshold."
        ),
    }
    payload["result_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if short_screen is not None:
        short_payload = short_screen.finalize()
        short_payload["input"] = {
            "kind": "LIVE_PAIRED_TRAJECTORY_EFFECTIVE_UPDATE",
            "case_id": f"{args.architecture}_lmhead_dx",
            "carrier_parameters": [resolved_carrier],
            "evaluation_steps_used": args.short_screen_steps,
            "raw_vectors_retained": False,
            "source_result_sha256": payload["result_sha256"],
        }
        args.short_screen_output.parent.mkdir(parents=True, exist_ok=True)
        args.short_screen_output.write_text(
            json.dumps(short_payload, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
