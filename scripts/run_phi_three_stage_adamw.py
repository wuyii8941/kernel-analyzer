#!/usr/bin/env python3
"""Measure the three-stage Phi contrast with a common-state AdamW map.

The endpoint output and gradient stages are identical to the frozen ordered
reference run.  Only the effective-update stage changes: candidate and repair
gradients are passed through the same pre-step AdamW moments at each common
state, and the repair arm advances the reference state and moments.  This is
an optimizer-response comparison, not a second trajectory label.
"""

from __future__ import annotations

import argparse
import hashlib
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
from scripts.run_generated_fp32_screen import load_model, tensor_digest  # noqa: E402
from scripts.run_phi64_lmhead_dx_repair import MMRepair  # noqa: E402
from scripts.run_targeted_full_coordinate import validate_release  # noqa: E402


def metric_curve(vectors: list[torch.Tensor]) -> list[dict[str, float | int]]:
    total = torch.zeros_like(vectors[0], dtype=torch.float64)
    energy = 0.0
    rows = []
    for index, vector in enumerate(vectors, start=1):
        value = vector.double()
        total.add_(value)
        energy += float(torch.dot(value.reshape(-1), value.reshape(-1)))
        if index in (2, 4, 8, 16, 32):
            scale = math.sqrt(max(energy, 0.0))
            rows.append({
                "horizon": index,
                "resultant_l2": float(torch.linalg.vector_norm(total)),
                "path_rms_l2": scale,
                "coherence_amplification": float(torch.linalg.vector_norm(total)) / max(scale, 1e-30),
            })
    return rows


def adamw_update(parameter: torch.Tensor, gradient: torch.Tensor,
                 first: torch.Tensor, second: torch.Tensor, step: int,
                 learning_rate: float, beta1: float, beta2: float,
                 epsilon: float, weight_decay: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    first_next = beta1 * first + (1.0 - beta1) * gradient
    second_next = beta2 * second + (1.0 - beta2) * gradient.square()
    bias1 = 1.0 - beta1 ** step
    bias2 = 1.0 - beta2 ** step
    direction = (first_next / bias1) / ((second_next / bias2).sqrt() + epsilon)
    delta = -learning_rate * (direction + weight_decay * parameter)
    return parameter + delta, first_next, second_next, delta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("host GPU required")
    bank = json.loads((ROOT / "results/coverage/phi4_seq64_input_bank.json").read_text())
    states = list(bank.get("states", bank.get("records")))
    if len(states) != 32:
        raise RuntimeError("the frozen Phi reference trajectory must contain 32 states")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    configure_candidate_runtime(24000)
    model = load_model("phi", Path("/data1/tzh/models/microsoft/Phi-4-mini-instruct"), device)
    model.eval()
    start = len(PyCodeCache.modules)
    step_fn = torch.compile(LossStep(model), backend="inductor", fullgraph=False, dynamic=False)
    warm = torch.tensor([states[0]["token_ids"]], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    step_fn(warm).backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    capture = json.loads((args.release_dir / "capture.json").read_text())
    validate_release(wrapper_modules(modules), capture)
    parameter = model.model.norm.weight
    reference = parameter.detach().float().clone()
    first = torch.zeros_like(reference)
    second = torch.zeros_like(reference)
    local_vectors, gradient_vectors, update_vectors, records = [], [], [], []

    def gradient(state: dict[str, Any], repair: bool) -> tuple[torch.Tensor, str]:
        with torch.no_grad():
            parameter.copy_(reference.to(parameter.dtype))
        model.zero_grad(set_to_none=True)
        values = torch.tensor([state["token_ids"]], dtype=torch.long, device=device)
        observer = MMRepair(modules, "REPAIR_FP32_CAST_BF16") if repair else None
        if observer is None:
            loss = step_fn(values)
            loss.backward()
        else:
            with observer:
                loss = step_fn(values)
                loss.backward()
        torch.cuda.synchronize(device)
        if observer is not None and tensor_digest(loss) == "":
            raise RuntimeError("unreachable digest guard")
        return parameter.grad.detach().float().clone(), tensor_digest(loss)

    for index, state in enumerate(states):
        gc, candidate_loss_digest = gradient(state, False)
        gr, repair_loss_digest = gradient(state, True)
        if candidate_loss_digest != repair_loss_digest:
            raise RuntimeError(f"repair changed forward loss at state {index}")
        # Source and gradient stages remain the same common-state quantities.
        local_vectors.append(torch.zeros(1, dtype=torch.float64))
        gradient_vectors.append((gc - gr).cpu())
        candidate_next, candidate_first, candidate_second, candidate_delta = adamw_update(
            reference, gc, first, second, index + 1,
            args.learning_rate, args.beta1, args.beta2, args.epsilon, args.weight_decay,
        )
        repair_next, repair_first, repair_second, repair_delta = adamw_update(
            reference, gr, first, second, index + 1,
            args.learning_rate, args.beta1, args.beta2, args.epsilon, args.weight_decay,
        )
        update_vectors.append((candidate_delta - repair_delta).cpu())
        reference, first, second = repair_next, repair_first, repair_second
        records.append({
            "step": index + 1,
            "state_id": str(state.get("state_id", index)),
            "gradient_l2": float(torch.linalg.vector_norm((gc - gr).double())),
            "effective_update_l2": float(torch.linalg.vector_norm((candidate_delta - repair_delta).double())),
        })
        print(json.dumps({"event": "PHI_THREE_STAGE_ADAMW_STEP", **records[-1]}), flush=True)

    payload = {
        "schema": "kernel-analyzer-phi-three-stage-adamw-v1",
        "status": "COMPLETE_ORDERED_32_STATE_COMMON_STATE_ADAMW",
        "case_id": "phi4_seq64_lmhead_dx",
        "state_count": 32,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "beta1": args.beta1, "beta2": args.beta2,
            "epsilon": args.epsilon, "weight_decay": args.weight_decay,
            "moment_state": "candidate and repair use identical pre-step moments; repair advances reference moments",
        },
        "stages": {
            "operator_output_error": {"status": "NOT_RECAPTURED_IN_ADAMW_RUN", "source": "phi_three_stage_reference.json"},
            "parameter_gradient_error": {"coherence_curve": metric_curve(gradient_vectors), "source": "this_run"},
            "effective_update_error": {"coherence_curve": metric_curve(update_vectors), "source": "this_run"},
        },
        "rows": records,
        "claim_boundary": "AdamW response mapping on the same declared Phi carrier and ordered states; it is not a new formation label or a full-model training result.",
    }
    payload["result_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
