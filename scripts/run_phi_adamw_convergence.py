#!/usr/bin/env python3
"""Train paired Phi candidate/repair carrier trajectories to a frozen loss plateau."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
os.environ.setdefault("XDG_CACHE_HOME", "/data1/tzh/cache/xdg")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data1/tzh/cache/torchinductor")

import numpy as np
import torch
from torch._inductor.codecache import PyCodeCache

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "archive/round1_code/src"):
    sys.path.insert(0, str(path))

from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_generated_fp32_screen import load_model, tensor_digest  # noqa: E402
from scripts.run_heldout_lmhead_consequence import adam_delta  # noqa: E402
from scripts.run_phi64_lmhead_dx_repair import MMRepair  # noqa: E402


def relative_span(values: list[float]) -> float:
    scale = max(abs(float(np.mean(values))), 1e-12)
    return (max(values) - min(values)) / scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--train-states", type=int, default=32)
    parser.add_argument("--validation-states", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=4096)
    parser.add_argument("--min-steps", type=int, default=1024)
    parser.add_argument("--eval-every", type=int, default=128)
    parser.add_argument("--convergence-points", type=int, default=4)
    parser.add_argument("--loss-relative-span", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.train_states < 2 or args.validation_states < 2:
        raise ValueError("training and validation banks need at least two states")
    if args.min_steps < args.eval_every * args.convergence_points:
        raise ValueError("minimum steps do not contain a full convergence window")
    if args.max_steps < args.min_steps or args.eval_every <= 0:
        raise ValueError("invalid convergence horizon")

    bank = json.loads(args.input_bank.read_text())
    states = bank.get("states", bank.get("records"))
    needed = args.train_states + args.validation_states
    if len(states) < needed:
        raise RuntimeError(f"input bank needs at least {needed} unique states")
    train = states[:args.train_states]
    validation = states[args.train_states:needed]
    train_ids = [str(row.get("state_id", index)) for index, row in enumerate(train)]
    validation_ids = [
        str(row.get("state_id", args.train_states + index))
        for index, row in enumerate(validation)
    ]
    if set(train_ids) & set(validation_ids):
        raise RuntimeError("training and validation state IDs overlap")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    configure_candidate_runtime(24_000)
    model = load_model(
        "phi", Path("/data1/tzh/models/microsoft/Phi-4-mini-instruct"), device
    )
    model.eval()
    carrier = model.model.norm.weight
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=False, dynamic=False)
    warm = torch.tensor([train[0]["token_ids"]], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    candidate(warm).backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])

    initial = carrier.detach().float().clone()
    candidate_master = initial.clone()
    repair_master = initial.clone()
    candidate_m = torch.zeros_like(initial)
    candidate_v = torch.zeros_like(initial)
    repair_m = torch.zeros_like(initial)
    repair_v = torch.zeros_like(initial)

    def run_gradient(
        master: torch.Tensor,
        state: dict[str, Any],
        *,
        repair: bool,
        seed: int,
    ) -> tuple[str, float, torch.Tensor, Any]:
        with torch.no_grad():
            carrier.copy_(master.to(carrier.dtype))
        values = torch.tensor([state["token_ids"]], dtype=torch.long, device=device)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True)
        observer = MMRepair(modules, "REPAIR_FP32_CAST_BF16") if repair else None
        if observer is None:
            loss = candidate(values)
            loss.backward()
        else:
            with observer:
                loss = candidate(values)
                loss.backward()
        torch.cuda.synchronize(device)
        if carrier.grad is None or not torch.isfinite(carrier.grad).all():
            raise RuntimeError("carrier gradient is absent or nonfinite")
        return (
            tensor_digest(loss),
            float(loss.detach().float().item()),
            carrier.grad.detach().float().clone(),
            observer,
        )

    def validation_checkpoint(step: int) -> dict[str, Any]:
        candidate_losses: list[float] = []
        repair_losses: list[float] = []
        candidate_grads: list[torch.Tensor] = []
        repair_grads: list[torch.Tensor] = []
        for offset, state in enumerate(validation):
            seed = 90_000 + step * args.validation_states + offset
            _, loss_c, grad_c, _ = run_gradient(
                candidate_master, state, repair=False, seed=seed
            )
            _, loss_r, grad_r, observer = run_gradient(
                repair_master, state, repair=True, seed=seed
            )
            if observer is None or observer.calls != 1:
                raise RuntimeError("validation repair missed the exact endpoint")
            candidate_losses.append(loss_c)
            repair_losses.append(loss_r)
            candidate_grads.append(grad_c)
            repair_grads.append(grad_r)
        mean_candidate_grad = torch.stack(candidate_grads).mean(0)
        mean_repair_grad = torch.stack(repair_grads).mean(0)
        grad_delta = mean_candidate_grad - mean_repair_grad
        parameter_delta = candidate_master - repair_master
        return {
            "step": step,
            "candidate_loss": float(np.mean(candidate_losses)),
            "repair_loss": float(np.mean(repair_losses)),
            "loss_gap_candidate_minus_repair": float(
                np.mean(candidate_losses) - np.mean(repair_losses)
            ),
            "candidate_mean_gradient_l2": float(torch.linalg.vector_norm(mean_candidate_grad)),
            "repair_mean_gradient_l2": float(torch.linalg.vector_norm(mean_repair_grad)),
            "mean_gradient_difference_l2": float(torch.linalg.vector_norm(grad_delta)),
            "parameter_distance_l2": float(torch.linalg.vector_norm(parameter_delta)),
            "candidate_from_initial_l2": float(torch.linalg.vector_norm(candidate_master - initial)),
            "repair_from_initial_l2": float(torch.linalg.vector_norm(repair_master - initial)),
        }

    checkpoints = [validation_checkpoint(0)]
    train_rows: list[dict[str, Any]] = []
    converged = False
    stop_step = args.max_steps
    for index in range(args.max_steps):
        step = index + 1
        learning_rate = args.learning_rate
        if args.lr_schedule == "cosine":
            progress = (step - 1) / max(args.max_steps - 1, 1)
            learning_rate *= 0.5 * (1.0 + math.cos(math.pi * progress))
        state_index = index % args.train_states
        state = train[state_index]
        seed = 60_000 + step
        loss_c_digest, loss_c, grad_c, _ = run_gradient(
            candidate_master, state, repair=False, seed=seed
        )
        loss_r_digest, loss_r, grad_r, observer = run_gradient(
            repair_master, state, repair=True, seed=seed
        )
        if observer is None or observer.calls != 1 or observer.local is None:
            raise RuntimeError("training repair missed the exact endpoint")
        # Candidate and repair have different live weights after step one, so
        # their forward losses may differ.  At step one they must still match.
        if step == 1 and loss_c_digest != loss_r_digest:
            raise RuntimeError("backward-only repair changed the initial forward loss")
        update_c, next_cm, next_cv = adam_delta(
            grad_c,
            candidate_m,
            candidate_v,
            step,
            learning_rate=learning_rate,
            beta1=0.9,
            beta2=0.95,
        )
        update_r, next_rm, next_rv = adam_delta(
            grad_r,
            repair_m,
            repair_v,
            step,
            learning_rate=learning_rate,
            beta1=0.9,
            beta2=0.95,
        )
        candidate_master.add_(update_c)
        repair_master.add_(update_r)
        candidate_m, candidate_v = next_cm, next_cv
        repair_m, repair_v = next_rm, next_rv
        train_rows.append({
            "step": step,
            "state_id": train_ids[state_index],
            "candidate_loss": loss_c,
            "repair_loss": loss_r,
            "loss_gap_candidate_minus_repair": loss_c - loss_r,
            "parameter_distance_l2": float(torch.linalg.vector_norm(candidate_master - repair_master)),
            "candidate_update_l2": float(torch.linalg.vector_norm(update_c)),
            "repair_update_l2": float(torch.linalg.vector_norm(update_r)),
            "learning_rate": learning_rate,
            "endpoint_changed_coordinates": int(observer.local["changed_coordinates"]),
        })
        if step % args.eval_every == 0:
            checkpoint = validation_checkpoint(step)
            checkpoints.append(checkpoint)
            recent = checkpoints[-args.convergence_points:]
            if step >= args.min_steps and len(recent) == args.convergence_points:
                candidate_span = relative_span([row["candidate_loss"] for row in recent])
                repair_span = relative_span([row["repair_loss"] for row in recent])
                converged = (
                    candidate_span <= args.loss_relative_span
                    and repair_span <= args.loss_relative_span
                )
                checkpoint["candidate_recent_relative_loss_span"] = candidate_span
                checkpoint["repair_recent_relative_loss_span"] = repair_span
                checkpoint["loss_plateau_gate"] = converged
            if not args.quiet:
                print(json.dumps({"event": "PHI_CONVERGENCE_CHECKPOINT", **checkpoint}), flush=True)
            if converged:
                stop_step = step
                break
        del grad_c, grad_r, update_c, update_r

    final = checkpoints[-1]
    payload = {
        "schema": "kernel-analyzer-phi-adamw-carrier-convergence-v1",
        "status": "LOSS_PLATEAU_REACHED" if converged else "MAX_STEPS_WITHOUT_FROZEN_LOSS_PLATEAU",
        "case_id": "phi4_seq64_lmhead_dx",
        "protocol": {
            "input_bank": str(args.input_bank.resolve()),
            "training_state_ids": train_ids,
            "validation_state_ids": validation_ids,
            "training_schedule": "repeat the frozen training-state order",
            "optimizer": {
                "name": "AdamW",
                "learning_rate": args.learning_rate,
                "schedule": args.lr_schedule,
                "betas": [0.9, 0.95],
                "epsilon": 1e-8,
                "initial_moments": "ZERO_THEN_EVOLVED_SEPARATELY_PER_ARM",
            },
            "candidate": "deterministic compiled BF16 lm_head dX",
            "repair": "FP32 MM followed by the required BF16 ABI cast",
            "carrier": "model.norm.weight",
            "other_parameters_frozen": True,
            "max_steps": args.max_steps,
            "min_steps": args.min_steps,
            "validation_every": args.eval_every,
            "convergence_points": args.convergence_points,
            "loss_relative_span_threshold": args.loss_relative_span,
        },
        "steps_completed": stop_step,
        "loss_plateau_reached": converged,
        "checkpoints": checkpoints,
        "final": final,
        "train_rows": train_rows,
        "claim_boundary": (
            "This is convergence of a frozen finite-data, one-parameter-carrier training "
            "problem. It can show whether candidate and repair reach different loss, gradient, "
            "or parameter values under that controlled objective. It is not full-parameter "
            "pretraining convergence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "event": "PHI_CONVERGENCE_COMPLETE",
        "status": payload["status"],
        "steps": stop_step,
        "candidate_loss": final["candidate_loss"],
        "repair_loss": final["repair_loss"],
        "loss_gap": final["loss_gap_candidate_minus_repair"],
        "gradient_difference_l2": final["mean_gradient_difference_l2"],
        "parameter_distance_l2": final["parameter_distance_l2"],
        "output": str(args.output),
    }), flush=True)


if __name__ == "__main__":
    main()
