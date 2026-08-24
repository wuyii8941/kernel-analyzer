#!/usr/bin/env python3
"""Classify Qwen3-VL SiLU separation by four-arm optimizer recurrence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data1/tzh/cache/huggingface/transformers")
os.environ.setdefault("XDG_CACHE_HOME", "/data1/tzh/cache/xdg")

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from kernel_analyzer.trajectory_persistence import OrderedVectorPath, cosine  # noqa: E402
from scripts.round2_vl_bias import STATES  # noqa: E402
from scripts.round2_vl_silu_cause import DecomposedSiluModule  # noqa: E402
from scripts.round2_vl_smoke import prepare_step  # noqa: E402
from scripts.round2_vl_static import specialize_fixed_grid  # noqa: E402
from scripts.run_vl_silu_invocation_trajectory import LossStep, PROOF_UNIT, TARGET  # noqa: E402


def norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value).item())


def adam_arm(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    step: int,
    *,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    next_first = first * beta1 + gradient * (1.0 - beta1)
    next_second = second * beta2 + gradient.square() * (1.0 - beta2)
    update = -learning_rate * (next_first / (1.0 - beta1**step)) / (
        (next_second / (1.0 - beta2**step)).sqrt() + epsilon
    )
    return update, next_first, next_second


def materialized_update(master: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    """Return the increment actually representable in the FP32 master."""
    return (master + update) - master


def write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path,
        default=Path("/data1/tzh/models/Qwen/Qwen3-VL-Reranker-2B"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--calibration-steps", type=int, default=8)
    parser.add_argument("--pad-length", type=int, default=160)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    args = parser.parse_args()
    if args.steps < 2:
        raise ValueError("steps must be at least 2")
    if not 1 <= args.calibration_steps < args.steps:
        raise ValueError("calibration steps must precede evaluation")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    prepared = []
    for state in STATES:
        inputs, labels, metadata = prepare_step(
            processor, Path(state["image"]), width=args.width, height=args.height,
            question=state["question"], answer=state["answer"],
            pad_length=args.pad_length,
        )
        prepared.append((state, inputs, labels, metadata))
    if not all(row[1]["input_ids"].shape == prepared[0][1]["input_ids"].shape
               for row in prepared):
        raise RuntimeError("trajectory states do not share one specialized shape")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager",
        trust_remote_code=True,
    ).to(device)
    model.config.use_cache = False
    model.eval()
    target = dict(model.named_parameters())[TARGET]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    target.requires_grad_(True)

    text_mlps: dict[int, torch.nn.Module] = {}
    originals: dict[int, object] = {}
    for name, module in model.named_modules():
        if module.__class__.__name__ == "Qwen3VLTextMLP":
            layer = int(name.split(".layers.", 1)[1].split(".", 1)[0])
            text_mlps[layer] = module
            originals[layer] = module.act_fn
    if sorted(text_mlps) != list(range(28)):
        raise RuntimeError("expected all 28 text MLPs")
    for module in text_mlps.values():
        module.act_fn = DecomposedSiluModule()

    first_values = {key: value.to(device) for key, value in prepared[0][1].items()}
    specialization = specialize_fixed_grid(
        model, first_values["image_grid_thw"], first_values["input_ids"]
    )
    step_module = LossStep(model)
    device_states = []
    for state, inputs, labels, metadata in prepared:
        values = {key: value.to(device) for key, value in inputs.items()}
        positions, _ = model.model.get_rope_index(
            values["input_ids"], values["image_grid_thw"],
            attention_mask=values["attention_mask"],
        )
        device_states.append((state, (
            values["input_ids"], values["attention_mask"], values["pixel_values"],
            values["image_grid_thw"], positions, labels.to(device),
        ), metadata))

    initial = target.detach().float().clone()
    candidate_master = initial.clone()
    repair_master = initial.clone()
    candidate_first = torch.zeros_like(initial)
    candidate_second = torch.zeros_like(initial)
    repair_first = torch.zeros_like(initial)
    repair_second = torch.zeros_like(initial)
    beta1, beta2, epsilon = 0.9, 0.95, 1e-8

    def gradient(
        master: torch.Tensor,
        arguments: tuple[torch.Tensor, ...],
        *,
        decomposed: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            target.copy_(master.to(torch.bfloat16))
        text_mlps[0].act_fn = DecomposedSiluModule() if decomposed else originals[0]
        model.zero_grad(set_to_none=True)
        loss = step_module(*arguments)
        loss.backward()
        if target.grad is None:
            raise RuntimeError("declared carrier gradient is absent")
        result = target.grad.detach().float().clone()
        target.grad = None
        return loss.detach(), result

    local_path = OrderedVectorPath(
        total_steps=args.steps, calibration_steps=args.calibration_steps
    )
    feedback_path = OrderedVectorPath(
        total_steps=args.steps, calibration_steps=args.calibration_steps
    )
    actual_path = OrderedVectorPath(
        total_steps=args.steps, calibration_steps=args.calibration_steps
    )
    max_recurrence_relative = 0.0
    records = []

    for offset in range(args.steps):
        state, arguments, metadata = device_states[offset % len(device_states)]
        drift_before = candidate_master - repair_master
        candidate_loss_c, candidate_grad_c = gradient(
            candidate_master, arguments, decomposed=True
        )
        repair_loss_c, repair_grad_c = gradient(
            candidate_master, arguments, decomposed=False
        )
        candidate_loss_r, candidate_grad_r = gradient(
            repair_master, arguments, decomposed=True
        )
        repair_loss_r, repair_grad_r = gradient(
            repair_master, arguments, decomposed=False
        )
        if not torch.equal(candidate_loss_c, repair_loss_c):
            raise RuntimeError("SiLU backward repair changed candidate-state forward loss")
        if not torch.equal(candidate_loss_r, repair_loss_r):
            raise RuntimeError("SiLU backward repair changed repair-state forward loss")

        step = offset + 1
        ucc_raw, next_candidate_first, next_candidate_second = adam_arm(
            candidate_master, candidate_grad_c, candidate_first, candidate_second,
            step, learning_rate=args.learning_rate, beta1=beta1, beta2=beta2,
            epsilon=epsilon,
        )
        urc_raw, _, _ = adam_arm(
            candidate_master, repair_grad_c, candidate_first, candidate_second,
            step, learning_rate=args.learning_rate, beta1=beta1, beta2=beta2,
            epsilon=epsilon,
        )
        ucr_raw, _, _ = adam_arm(
            repair_master, candidate_grad_r, repair_first, repair_second,
            step, learning_rate=args.learning_rate, beta1=beta1, beta2=beta2,
            epsilon=epsilon,
        )
        urr_raw, next_repair_first, next_repair_second = adam_arm(
            repair_master, repair_grad_r, repair_first, repair_second,
            step, learning_rate=args.learning_rate, beta1=beta1, beta2=beta2,
            epsilon=epsilon,
        )
        ucc = materialized_update(candidate_master, ucc_raw)
        urc = materialized_update(candidate_master, urc_raw)
        ucr = materialized_update(repair_master, ucr_raw)
        urr = materialized_update(repair_master, urr_raw)
        local = ((ucc - urc) + (ucr - urr)) * 0.5
        feedback = ((ucc - ucr) + (urc - urr)) * 0.5

        candidate_master.add_(ucc_raw)
        repair_master.add_(urr_raw)
        candidate_first, candidate_second = next_candidate_first, next_candidate_second
        repair_first, repair_second = next_repair_first, next_repair_second
        drift_after = candidate_master - repair_master
        actual_increment = ucc - urr
        master_difference_residual = (drift_after - drift_before) - actual_increment
        recurrence = actual_increment - local - feedback
        recurrence_relative = norm(recurrence) / max(
            norm(actual_increment), norm(local), norm(feedback), 1e-30
        )
        max_recurrence_relative = max(max_recurrence_relative, recurrence_relative)

        local_row = local_path.add(local)
        feedback_row = feedback_path.add(feedback)
        actual_row = actual_path.add(actual_increment)
        row = {
            "step": step,
            "state_id": state["id"],
            "input_sequence_length": metadata["sequence_length"],
            "candidate_loss": float(candidate_loss_c.detach().float().item()),
            "repair_loss": float(repair_loss_r.detach().float().item()),
            "paired_loss_gap_candidate_minus_repair": float(
                candidate_loss_c.detach().float().item()
                - repair_loss_r.detach().float().item()
            ),
            "forward_loss_equal_at_candidate_state": bool(torch.equal(candidate_loss_c, repair_loss_c)),
            "forward_loss_equal_at_repair_state": bool(torch.equal(candidate_loss_r, repair_loss_r)),
            "candidate_state_gradient_effect_l2": norm(candidate_grad_c - repair_grad_c),
            "repair_state_gradient_effect_l2": norm(candidate_grad_r - repair_grad_r),
            "local_effect_l2": local_row["l2"],
            "feedback_effect_l2": feedback_row["l2"],
            "actual_drift_increment_l2": actual_row["l2"],
            "local_frozen_carrier_projection": local_row["frozen_carrier_projection"],
            "feedback_frozen_carrier_projection": feedback_row["frozen_carrier_projection"],
            "recurrence_residual_l2": norm(recurrence),
            "recurrence_relative": recurrence_relative,
            "master_difference_residual_l2": norm(master_difference_residual),
            "drift_l2": norm(drift_after),
        }
        records.append(row)
        write(args.output, {
            "schema": "kernel-analyzer-vl-silu-persistence-recurrence-v1",
            "status": "RUNNING",
            "steps_complete": step,
            "records": records,
        })
        print(json.dumps({"event": "SILU_RECURRENCE_STEP", **row}), flush=True)
        del candidate_grad_c, repair_grad_c, candidate_grad_r, repair_grad_r
        del ucc, urc, ucr, urr, ucc_raw, urc_raw, ucr_raw, urr_raw
        del local, feedback, recurrence, actual_increment, master_difference_residual
        torch.cuda.empty_cache()

    local_summary = local_path.finalize()
    feedback_summary = feedback_path.finalize()
    actual_summary = actual_path.finalize()
    final_drift = candidate_master - repair_master
    loss_gaps = [float(row["paired_loss_gap_candidate_minus_repair"]) for row in records]
    assert local_path.total is not None and feedback_path.total is not None
    local_final_cosine = cosine(local_path.total, final_drift)
    feedback_final_cosine = cosine(feedback_path.total, final_drift)
    recurrence_closed = max_recurrence_relative <= 1e-5
    persistent_local = bool(
        recurrence_closed
        and local_summary["coherence_amplification"] >= 2.0
        and actual_summary["coherence_amplification"] >= 2.0
        and local_final_cosine is not None and local_final_cosine >= 0.5
    )
    feedback_sustained = bool(
        not persistent_local and recurrence_closed
        and feedback_summary["coherence_amplification"] >= 2.0
        and actual_summary["coherence_amplification"] >= 2.0
        and feedback_final_cosine is not None and feedback_final_cosine >= 0.5
    )
    verdict = (
        "PERSISTENT_LOCAL_BIAS" if persistent_local else
        "FEEDBACK_SUSTAINED_SEPARATION" if feedback_sustained else
        "DIFFUSIVE_OR_CANCELING_SEPARATION"
    )
    payload = {
        "schema": "kernel-analyzer-vl-silu-persistence-recurrence-v1",
        "status": "COMPLETE",
        "verdict": verdict,
        "case": {
            "model": str(args.model),
            "proof_unit": PROOF_UNIT,
            "target_parameter": TARGET,
            "candidate": "decomposed eight-node SiLU backward",
            "repair": "native aten.silu_backward for layer 0",
        },
        "protocol": {
            "steps": args.steps,
            "calibration_steps": args.calibration_steps,
            "state_order": [records[i]["state_id"] for i in range(args.steps)],
            "optimizer": {
                "name": "AdamW", "learning_rate": args.learning_rate,
                "betas": [beta1, beta2], "epsilon": epsilon, "weight_decay": 0.0,
            },
            "symmetric_four_counterfactual_recurrence": True,
            "fixed_grid_specialization": specialization,
            "predeclared_classification": {
                "coherence_amplification_min": 2.0,
                "final_alignment_cosine_min": 0.5,
                "recurrence_relative_max": 1e-5,
            },
        },
        "summaries": {
            "local": local_summary,
            "feedback": feedback_summary,
            "actual_drift_increment": actual_summary,
            "final_drift_l2": norm(final_drift),
            "local_final_drift_cosine": local_final_cosine,
            "feedback_final_drift_cosine": feedback_final_cosine,
            "max_recurrence_relative": max_recurrence_relative,
            "global_master_difference_closure_l2": norm(
                actual_path.total - final_drift.reshape(-1)
            ),
            "paired_loss_gap_final": loss_gaps[-1] if loss_gaps else None,
            "paired_loss_gap_mean_last_512": (
                float(np.mean(loss_gaps[-512:])) if loss_gaps else None
            ),
            "paired_loss_gap_std_last_512": (
                float(np.std(loss_gaps[-512:])) if loss_gaps else None
            ),
        },
        "records": records,
        "claim_boundary": (
            "This classifies the existing SiLU candidate-repair separation as an "
            "ordered effective-update mechanism. It does not use mere norm growth "
            "or a post-hoc final-drift direction as the persistence gate."
        ),
    }
    write(args.output, payload)
    print(json.dumps({
        "event": "SILU_RECURRENCE_COMPLETE", "verdict": verdict,
        "local_amplification": local_summary["coherence_amplification"],
        "feedback_amplification": feedback_summary["coherence_amplification"],
        "actual_amplification": actual_summary["coherence_amplification"],
    }), flush=True)


if __name__ == "__main__":
    main()
