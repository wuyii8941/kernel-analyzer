#!/usr/bin/env python3
"""Paired live-weight trajectory for one frozen external-MM repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data1/tzh/cache/huggingface/transformers")
os.environ.setdefault("XDG_CACHE_HOME", "/data1/tzh/cache/xdg")

import torch
from torch._inductor.codecache import PyCodeCache

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "archive/round1_code/src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_frozen_candidate_fp32_screen import wrapper_modules  # noqa: E402
from scripts.run_generated_fp32_screen import file_digest, load_model  # noqa: E402
from scripts.run_qwen128_vproj_repair import VProjRepair  # noqa: E402
from scripts.run_targeted_full_coordinate import validate_release  # noqa: E402


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def tensor_digest(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def adam_step(
    master: torch.Tensor, grad: torch.Tensor,
    first: torch.Tensor, second: torch.Tensor, step: int,
    *, lr: float, beta1: float, beta2: float, epsilon: float,
) -> None:
    first.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    second.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    master.addcdiv_(
        first / (1.0 - beta1**step),
        (second / (1.0 - beta2**step)).sqrt().add_(epsilon),
        value=-lr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("qwen", "mamba", "phi", "deepseek8"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--target-sha", required=True)
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--repair-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    args = parser.parse_args()
    if not 1 <= args.steps <= 32:
        raise ValueError("steps must be in [1, 32]")

    queue = json.loads((ROOT / "results/coverage/bias_candidate_queue.json").read_text())
    bound = next(row for row in queue["candidates"]
                 if row["candidate_id"] == args.candidate_id)
    if bound["exact_generated_call"]["source_line_sha256"] != args.target_sha:
        raise RuntimeError("candidate source identity differs")
    repair_evidence = json.loads(args.repair_evidence.read_text())
    if repair_evidence["candidate_id"] != args.candidate_id:
        raise RuntimeError("repair evidence belongs to another candidate")
    required_repair_gates = (
        "restoration_sham_exact",
        "accumulation_intervention_nonnull_every_state",
        "accumulation_intervention_reduces_fp32_sse_every_state",
        "direct_weight_carrier_nonnull_every_state",
    )
    if not all(repair_evidence["gates"][key] for key in required_repair_gates):
        raise RuntimeError("local repair evidence failed a required causal gate")

    bank = json.loads(args.input_bank.read_text())
    states = bank.get("states", bank.get("records"))
    if len(states) < args.steps:
        raise RuntimeError("input bank is shorter than trajectory")
    capture = json.loads((args.release_dir / "capture.json").read_text())
    if file_digest(args.input_bank) != capture["input"]["input_bank_sha256"]:
        raise RuntimeError("input bank differs from frozen release")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    configure_candidate_runtime(24000)
    model = load_model(args.architecture, args.model, device)
    parameters = dict(model.named_parameters())
    if args.carrier not in parameters:
        raise RuntimeError("carrier parameter is absent")
    target = parameters[args.carrier]
    start = len(PyCodeCache.modules)
    candidate = torch.compile(
        LossStep(model), backend="inductor",
        fullgraph=not bool(capture.get("allow_graph_breaks", False)), dynamic=False,
    )
    warm_tokens = states[0].get("token_ids", states[0].get("input_ids"))
    warm = torch.tensor([warm_tokens], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    candidate(warm).backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    validate_release(wrapper_modules(modules), capture)

    initial = target.detach().float().clone()
    candidate_master = initial.clone()
    repair_master = initial.clone()
    beta1, beta2, epsilon = 0.9, 0.95, 1e-8
    candidate_m = torch.zeros_like(initial); candidate_v = torch.zeros_like(initial)
    repair_m = torch.zeros_like(initial); repair_v = torch.zeros_like(initial)

    def gradient(master: torch.Tensor, tokens: list[int], mode: str | None) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            target.copy_(master.to(target.dtype))
        values = torch.tensor([tokens], dtype=torch.long, device=device)
        model.zero_grad(set_to_none=True)
        observer = VProjRepair(modules, mode, args.target_sha) if mode else None
        if observer is None:
            loss = candidate(values); loss.backward()
        else:
            with observer:
                loss = candidate(values); loss.backward()
        torch.cuda.synchronize(device)
        if target.grad is None:
            raise RuntimeError("carrier gradient is absent")
        result = target.grad.detach().float().clone()
        target.grad = None
        return loss.detach(), result

    first_tokens = states[0].get("token_ids", states[0].get("input_ids"))
    initial_loss, initial_grad = gradient(initial, first_tokens, None)
    sham_loss, sham_grad = gradient(initial, first_tokens, "SHAM")
    repair_loss, initial_repair_grad = gradient(
        initial, first_tokens, "REPAIR_FP32_CAST_BF16"
    )
    controls = {
        "matched_sham_exact": (
            torch.equal(initial_loss, sham_loss)
            and tensor_digest(initial_grad) == tensor_digest(sham_grad)
        ),
        "repair_forward_effect_observed": (
            not torch.equal(initial_loss, repair_loss)
            or not torch.equal(initial_grad, initial_repair_grad)
        ),
        "repair_gradient_nonzero": not torch.equal(initial_grad, initial_repair_grad),
    }
    del initial_grad, sham_grad, initial_repair_grad

    records = []
    frozen_direction: torch.Tensor | None = None
    for index in range(args.steps):
        state = states[index]
        tokens = state.get("token_ids", state.get("input_ids"))
        state_id = str(state.get("sequence_id", state.get("state_id", index)))
        cand_loss_c, cand_grad_c = gradient(candidate_master, tokens, None)
        repair_loss_c, repair_grad_c = gradient(
            candidate_master, tokens, "REPAIR_FP32_CAST_BF16"
        )
        cand_loss_r, cand_grad_r = gradient(repair_master, tokens, None)
        repair_loss_r, repair_grad_r = gradient(
            repair_master, tokens, "REPAIR_FP32_CAST_BF16"
        )
        removal_c = cand_grad_c - repair_grad_c
        removal_r = cand_grad_r - repair_grad_r
        adam_step(
            candidate_master, cand_grad_c, candidate_m, candidate_v, index + 1,
            lr=args.learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon,
        )
        adam_step(
            repair_master, repair_grad_r, repair_m, repair_v, index + 1,
            lr=args.learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon,
        )
        delta = candidate_master - repair_master
        if frozen_direction is None:
            norm = torch.linalg.vector_norm(delta)
            if not bool(norm > 0):
                raise RuntimeError("step-1 master divergence is zero")
            frozen_direction = delta / norm
        records.append({
            "step": index + 1,
            "state_id": state_id,
            "candidate_current_loss": float(cand_loss_c.cpu()),
            "repair_current_loss": float(repair_loss_r.cpu()),
            "candidate_current_removal_l2": float(torch.linalg.vector_norm(removal_c).cpu()),
            "repair_current_removal_l2": float(torch.linalg.vector_norm(removal_r).cpu()),
            "candidate_current_removal_nonzero": not torch.equal(cand_grad_c, repair_grad_c),
            "repair_current_removal_nonzero": not torch.equal(cand_grad_r, repair_grad_r),
            "fp32_master_l2": float(torch.linalg.vector_norm(delta).cpu()),
            "fp32_master_projection": float(torch.sum(delta * frozen_direction).cpu()),
            "bf16_materialized_nonzero": int(torch.count_nonzero(
                candidate_master.to(torch.bfloat16) != repair_master.to(torch.bfloat16)
            ).cpu()),
        })
        print(json.dumps({"event": "STEP_COMPLETE", **records[-1]}), flush=True)
        del cand_grad_c, repair_grad_c, cand_grad_r, repair_grad_r

    checkpoints = [step for step in (1, 8, 16, 32) if step <= args.steps]
    projections = [records[step - 1]["fp32_master_projection"] for step in checkpoints]
    grows = args.steps == 32 and all(
        right > left for left, right in zip(projections, projections[1:])
    )
    gates = {
        **controls,
        "only_declared_parameter_updated": True,
        "paired_same_weight_measurement": True,
        "all_steps_repair_nonzero": all(
            row["candidate_current_removal_nonzero"]
            and row["repair_current_removal_nonzero"] for row in records
        ),
        "directional_live_weight_accumulation": grows,
    }
    payload = {
        "schema": "kernel-analyzer-external-mm-repair-trajectory-v1",
        "status": (
            "PASS_STRICT_FLASH_STYLE_CASE" if all(gates.values()) else
            "COMPLETE_PILOT" if args.steps < 32 else
            "FAIL_DIRECTIONAL_ACCUMULATION"
        ),
        "candidate_id": args.candidate_id,
        "architecture": args.architecture,
        "carrier_parameter": args.carrier,
        "repair": "same-input FP32 MM accumulation followed by original BF16 output ABI",
        "steps": args.steps,
        "optimizer": {
            "name": "AdamW", "learning_rate": args.learning_rate,
            "betas": [beta1, beta2], "epsilon": epsilon, "weight_decay": 0.0,
        },
        "initial_controls": controls,
        "records": records,
        "directional_projection_checkpoints": checkpoints,
        "directional_projections": projections,
        "gates": gates,
        "bindings": {
            "source_line_sha256": args.target_sha,
            "release_capture_sha256": capture["result_sha256"],
            "input_bank_sha256": file_digest(args.input_bank),
            "repair_evidence_sha256": repair_evidence["result_sha256"],
        },
        "claim_boundary": (
            "The repair isolates local MM accumulation at fixed operands and restores the "
            "original BF16 output ABI. Output-rounding and inherited-operand mechanisms are "
            "not repaired by this trajectory. Cross-state coherence remains a separate verdict."
        ),
    }
    payload["result_sha256"] = canonical(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "TRAJECTORY_COMPLETE", "status": payload["status"],
                      "projections": projections}, sort_keys=True))


if __name__ == "__main__":
    main()
