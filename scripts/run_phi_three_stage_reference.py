#!/usr/bin/env python3
"""Measure the same Phi operator contrast at three training stages.

For one ordered 32-state reference trajectory, the candidate and matched repair
are evaluated at the same carrier state.  We accumulate the local endpoint
output error, the parameter-gradient difference, and the resulting SGD update
difference separately.  No stage is inferred from the other two.
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


def sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def metric_curve(vectors: list[torch.Tensor]) -> list[dict[str, float | int]]:
    total = torch.zeros_like(vectors[0], dtype=torch.float64)
    energy = 0.0
    rows = []
    for index, vector in enumerate(vectors, start=1):
        value = vector.double()
        total.add_(value)
        energy += float(torch.dot(value.reshape(-1), value.reshape(-1)))
        if index in (2, 4, 8, 16, 32):
            rows.append({
                "horizon": index,
                "resultant_l2": float(torch.linalg.vector_norm(total)),
                "path_rms_l2": math.sqrt(max(energy, 0.0)),
                "coherence_amplification": float(torch.linalg.vector_norm(total)) / max(math.sqrt(max(energy, 0.0)), 1e-30),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("the three-stage reference run requires the host GPU")
    bank = json.loads((ROOT / "results/coverage/phi4_seq64_input_bank.json").read_text())
    states = list(bank.get("states", bank.get("records")))
    if len(states) != 32:
        raise RuntimeError("the frozen Phi reference trajectory must contain exactly 32 states")
    device = torch.device(args.device)
    configure_candidate_runtime(24000)
    model = load_model("phi", Path("/data1/tzh/models/microsoft/Phi-4-mini-instruct"), device)
    model.eval()
    start = len(PyCodeCache.modules)
    step = torch.compile(LossStep(model), backend="inductor", fullgraph=False, dynamic=False)
    warm = torch.tensor([states[0]["token_ids"]], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    step(warm).backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    capture = json.loads((args.release_dir / "capture.json").read_text())
    validate_release(wrapper_modules(modules), capture)

    parameter = model.model.norm.weight
    reference_master = parameter.detach().float().clone()
    local_vectors: list[torch.Tensor] = []
    gradient_vectors: list[torch.Tensor] = []
    update_vectors: list[torch.Tensor] = []
    rows = []
    for index, state in enumerate(states):
        with torch.no_grad():
            parameter.copy_(reference_master.to(parameter.dtype))
        values = torch.tensor([state["token_ids"]], dtype=torch.long, device=device)
        seed = 24000 + index
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True)
        loss_candidate = step(values)
        loss_candidate.backward()
        torch.cuda.synchronize(device)
        grad_candidate = parameter.grad.detach().float().cpu().clone()
        candidate_loss_digest = tensor_digest(loss_candidate)

        with torch.no_grad():
            parameter.copy_(reference_master.to(parameter.dtype))
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model.zero_grad(set_to_none=True)
        observer = MMRepair(modules, "REPAIR_FP32_CAST_BF16")
        with observer:
            loss_repair = step(values)
            loss_repair.backward()
        torch.cuda.synchronize(device)
        if tensor_digest(loss_repair) != candidate_loss_digest:
            raise RuntimeError(f"repair changed forward loss at state {index}")
        if observer.local_vector is None:
            raise RuntimeError(f"missing local endpoint vector at state {index}")
        local = torch.from_numpy(observer.local_vector).float()
        grad_repair = parameter.grad.detach().float().cpu().clone()
        gradient = grad_candidate - grad_repair
        update = -args.learning_rate * gradient
        local_vectors.append(local)
        gradient_vectors.append(gradient)
        update_vectors.append(update)
        with torch.no_grad():
            reference_master.add_((-args.learning_rate * grad_repair).to(reference_master.device))
        rows.append({
            "step": index + 1,
            "state_id": str(state.get("state_id", index)),
            "local_l2": float(torch.linalg.vector_norm(local.double())),
            "gradient_l2": float(torch.linalg.vector_norm(gradient.double())),
            "effective_update_l2": float(torch.linalg.vector_norm(update.double())),
            "endpoint_changed_coordinates": observer.local["changed_coordinates"],
        })
        print(json.dumps({"event": "PHI_THREE_STAGE_STEP", **rows[-1]}), flush=True)
        del values, loss_candidate, loss_repair, grad_candidate, grad_repair, gradient, update
        torch.cuda.empty_cache()

    payload = {
        "schema": "kernel-analyzer-phi-three-stage-reference-v1",
        "status": "COMPLETE_ORDERED_32_STATE_REFERENCE",
        "case_id": "phi4_seq64_lmhead_dx",
        "release_dir": str(args.release_dir),
        "state_count": len(states),
        "state_order": [str(row.get("state_id", i)) for i, row in enumerate(states)],
        "reference_trajectory": "repair-gradient SGD master; candidate and repair evaluated at identical pre-step carrier state",
        "learning_rate": args.learning_rate,
        "stages": {
            "operator_output_error": {"coherence_curve": metric_curve(local_vectors)},
            "parameter_gradient_error": {"coherence_curve": metric_curve(gradient_vectors)},
            "effective_update_error": {"coherence_curve": metric_curve(update_vectors)},
        },
        "rows": rows,
        "claim_boundary": "The three stages are measured separately on one ordered reference trajectory and are not a universal full-model claim.",
    }
    payload["result_sha256"] = sha(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
