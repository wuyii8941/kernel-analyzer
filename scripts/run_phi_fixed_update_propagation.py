#!/usr/bin/env python3
"""Inject a fixed real Phi update-error sequence into an alternate checkpoint.

Phase 1 extracts local update errors from the real candidate/repair boundary.
Phase 2 evolves two repair arms from a slightly perturbed checkpoint while
injecting the frozen errors only into one arm.  This is a propagation probe,
not a new formation label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch._inductor.codecache import PyCodeCache

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "archive/round1_code/src")]

from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_frozen_candidate_fp32_screen import wrapper_modules  # noqa: E402
from scripts.run_generated_fp32_screen import load_model, tensor_digest  # noqa: E402
from scripts.run_phi64_lmhead_dx_repair import MMRepair  # noqa: E402
from scripts.run_phi64_lmhead_dx_trajectory import evaluate  # noqa: E402


MODEL = Path("/data1/tzh/models/microsoft/Phi-4-mini-instruct")
BANK = ROOT / "results/coverage/phi4_seq64_input_bank.json"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--checkpoint-perturbation", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use host GPU")
    if not 2 <= args.steps <= 16:
        raise ValueError("steps must be in [2,16]")

    bank = json.loads(BANK.read_text())
    states = bank.get("states", bank.get("records"))[: args.steps]
    device = torch.device(args.device)
    configure_candidate_runtime(24000)
    model = load_model("phi", MODEL, device)
    model.eval()
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=False, dynamic=False)
    warm = torch.tensor([states[0]["token_ids"]], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    candidate(warm).backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    parameter = model.model.norm.weight
    initial = parameter.detach().float().clone()
    baseline_master = initial.clone()
    fixed_errors: list[torch.Tensor] = []
    extraction_rows: list[dict[str, Any]] = []

    # Extract the real candidate-minus-repair update-error sequence.
    for step, state in enumerate(states):
        values = torch.tensor([state["token_ids"]], dtype=torch.long, device=device)
        with torch.no_grad():
            parameter.copy_(baseline_master.to(parameter.dtype))
        seed = 24000 + step
        loss_c, grad_c, _ = evaluate(model, candidate, values, modules, seed, False)
        loss_r, grad_r, local = evaluate(model, candidate, values, modules, seed, True)
        if loss_c != loss_r or local is None:
            raise RuntimeError("baseline repair boundary changed or was not captured")
        fixed = (grad_c - grad_r).mul(-args.learning_rate).detach().cpu()
        fixed_errors.append(fixed)
        baseline_master.add_(grad_c, alpha=-args.learning_rate)
        extraction_rows.append({
            "step": step + 1,
            "state_id": str(state.get("state_id", state.get("sequence_id", step))),
            "fixed_error_l2": float(torch.linalg.vector_norm(fixed).item()),
            "repair_changed_coordinates": local["changed_coordinates"],
        })
        del values, grad_c, grad_r, fixed
        torch.cuda.empty_cache()

    # Alternate checkpoint: same model, but a declared small perturbation.
    generator = torch.Generator(device="cpu").manual_seed(20260822)
    direction = torch.randn(initial.shape, generator=generator, dtype=torch.float32)
    direction.mul_(args.checkpoint_perturbation / max(float(torch.linalg.vector_norm(direction).item()), 1e-30))
    direction = direction.to(initial.device)
    alternate = initial + direction
    candidate_master = alternate.clone()
    repair_master = alternate.clone()
    direct_sum = torch.zeros_like(initial, device="cpu")
    rows: list[dict[str, Any]] = []
    for step, state in enumerate(states):
        values = torch.tensor([state["token_ids"]], dtype=torch.long, device=device)
        fixed = fixed_errors[step].to(device)
        with torch.no_grad():
            parameter.copy_(candidate_master.to(parameter.dtype))
        _, grad_candidate_repair, _ = evaluate(model, candidate, values, modules, 44000 + step, True)
        with torch.no_grad():
            parameter.copy_(repair_master.to(parameter.dtype))
        _, grad_repair, _ = evaluate(model, candidate, values, modules, 44000 + step, True)
        before = candidate_master - repair_master
        candidate_master.add_(grad_candidate_repair, alpha=-args.learning_rate)
        candidate_master.add_(fixed)
        repair_master.add_(grad_repair, alpha=-args.learning_rate)
        direct_sum.add_(fixed_errors[step])
        drift = candidate_master - repair_master
        rows.append({
            "step": step + 1,
            "state_id": str(state.get("state_id", state.get("sequence_id", step))),
            "fixed_error_l2": float(torch.linalg.vector_norm(fixed).item()),
            "drift_before_l2": float(torch.linalg.vector_norm(before).item()),
            "drift_after_l2": float(torch.linalg.vector_norm(drift).item()),
            "direct_fixed_sum_l2": float(torch.linalg.vector_norm(direct_sum).item()),
        })
        print(json.dumps({"event": "PHI_FIXED_UPDATE_STEP", **rows[-1]}, sort_keys=True), flush=True)
        del values, fixed, grad_candidate_repair, grad_repair, before, drift
        torch.cuda.empty_cache()

    payload = {
        "schema": "kernel-analyzer-phi-fixed-update-propagation-v1",
        "status": "COMPLETE",
        "case_id": "phi4_lm_head_dx_seq64",
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "checkpoint_perturbation_l2": float(torch.linalg.vector_norm(direction).item()),
        "extraction": extraction_rows,
        "propagation": rows,
        "final": {
            "fixed_sum_l2": rows[-1]["direct_fixed_sum_l2"],
            "alternate_feedback_drift_l2": rows[-1]["drift_after_l2"],
            "feedback_over_direct_ratio": rows[-1]["drift_after_l2"] / max(rows[-1]["direct_fixed_sum_l2"], 1e-30),
        },
        "claim_boundary": "fixed real candidate-minus-repair update-error sequence injected into an alternate checkpoint through repair dynamics; not a universal predictor or a new formation label",
    }
    payload["result_sha256"] = digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "status": payload["status"], **payload["final"]}, sort_keys=True))


if __name__ == "__main__":
    main()
