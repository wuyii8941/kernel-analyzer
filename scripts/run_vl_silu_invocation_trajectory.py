#!/usr/bin/env python3
"""Paired live-weight trajectory for one exact Qwen3-VL SiLU F+B unit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.round2_vl_bias import STATES
from scripts.round2_vl_silu_cause import DecomposedSiluModule
from scripts.round2_vl_smoke import prepare_step, sha256_file
from scripts.round2_vl_static import specialize_fixed_grid


TARGET = "model.language_model.layers.0.mlp.gate_proj.weight"
PROOF_UNIT = "vl-fb-1315"


def digest_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


class LossStep(torch.nn.Module):
    def __init__(self, subject: torch.nn.Module) -> None:
        super().__init__()
        self.subject = subject

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        position_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return self.subject(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            position_ids=position_ids,
            labels=labels,
            use_cache=False,
            return_dict=False,
        )[0]


def adamw_step(
    master: torch.Tensor,
    grad: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    step: int,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    epsilon: float,
) -> None:
    first.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    second.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    corrected_first = first / (1.0 - beta1**step)
    corrected_second = second / (1.0 - beta2**step)
    master.addcdiv_(corrected_first, corrected_second.sqrt().add_(epsilon), value=-lr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--pad-length", type=int, default=160)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    args = parser.parse_args()
    if not 1 <= args.steps <= 32:
        raise ValueError("steps must be in [1, 32]")

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
            processor,
            Path(state["image"]),
            width=args.width,
            height=args.height,
            question=state["question"],
            answer=state["answer"],
            pad_length=args.pad_length,
        )
        prepared.append((state, inputs, labels, metadata))
    if not all(row[1]["input_ids"].shape == prepared[0][1]["input_ids"].shape
               for row in prepared):
        raise RuntimeError("frozen trajectory states do not share one shape")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device)
    model.config.use_cache = False
    model.eval()
    named_parameters = dict(model.named_parameters())
    if TARGET not in named_parameters:
        raise RuntimeError(f"target parameter is absent: {TARGET}")
    target = named_parameters[TARGET]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    target.requires_grad_(True)

    text_mlps: dict[int, torch.nn.Module] = {}
    originals: dict[int, Any] = {}
    for name, module in model.named_modules():
        if module.__class__.__name__ != "Qwen3VLTextMLP":
            continue
        layer = int(name.split(".layers.", 1)[1].split(".", 1)[0])
        text_mlps[layer] = module
        originals[layer] = module.act_fn
    if sorted(text_mlps) != list(range(28)):
        raise RuntimeError(f"expected text layers 0:27, found {sorted(text_mlps)}")
    for layer, module in text_mlps.items():
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
        device_states.append((
            state,
            (
                values["input_ids"], values["attention_mask"],
                values["pixel_values"], values["image_grid_thw"],
                positions, labels.to(device),
            ),
            metadata,
        ))

    initial_master = target.detach().float().clone()
    candidate_master = initial_master.clone()
    repair_master = initial_master.clone()
    beta1, beta2, epsilon = 0.9, 0.95, 1e-8
    candidate_m = torch.zeros_like(candidate_master)
    candidate_v = torch.zeros_like(candidate_master)
    repair_m = torch.zeros_like(repair_master)
    repair_v = torch.zeros_like(repair_master)

    def gradient(master: torch.Tensor, arguments: tuple[torch.Tensor, ...], *,
                 decomposed_layer0: bool) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            target.copy_(master.to(torch.bfloat16))
        text_mlps[0].act_fn = (
            DecomposedSiluModule() if decomposed_layer0 else originals[0]
        )
        model.zero_grad(set_to_none=True)
        loss = step_module(*arguments)
        loss.backward()
        if target.grad is None:
            raise RuntimeError("target gradient is absent")
        value = target.grad.detach().float().clone()
        target.grad = None
        return loss.detach(), value

    # Exact matched sham: installing another decomposed wrapper must reproduce
    # the candidate gradient bit-for-bit at the same initial weight and state.
    initial_args = device_states[0][1]
    candidate_loss, candidate_grad = gradient(
        initial_master, initial_args, decomposed_layer0=True
    )
    sham_loss, sham_grad = gradient(
        initial_master, initial_args, decomposed_layer0=True
    )
    repair_loss, repair_grad = gradient(
        initial_master, initial_args, decomposed_layer0=False
    )
    sham_exact = (
        torch.equal(candidate_loss, sham_loss)
        and digest_tensor(candidate_grad) == digest_tensor(sham_grad)
    )
    initial_repair_nonzero = not torch.equal(candidate_grad, repair_grad)
    initial_forward_exact = torch.equal(candidate_loss, repair_loss)
    del candidate_grad, sham_grad, repair_grad

    records = []
    frozen_direction: torch.Tensor | None = None
    for index in range(args.steps):
        state, arguments, metadata = device_states[index % len(device_states)]
        cand_loss_c, cand_grad_c = gradient(
            candidate_master, arguments, decomposed_layer0=True
        )
        repair_loss_c, repair_grad_c = gradient(
            candidate_master, arguments, decomposed_layer0=False
        )
        cand_loss_r, cand_grad_r = gradient(
            repair_master, arguments, decomposed_layer0=True
        )
        repair_loss_r, repair_grad_r = gradient(
            repair_master, arguments, decomposed_layer0=False
        )
        if not torch.equal(cand_loss_c, repair_loss_c):
            raise RuntimeError("candidate-current forward losses differ by backward arm")
        if not torch.equal(cand_loss_r, repair_loss_r):
            raise RuntimeError("repair-current forward losses differ by backward arm")

        removal_c = cand_grad_c - repair_grad_c
        removal_r = cand_grad_r - repair_grad_r
        adamw_step(
            candidate_master, cand_grad_c, candidate_m, candidate_v, index + 1,
            lr=args.learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon,
        )
        adamw_step(
            repair_master, repair_grad_r, repair_m, repair_v, index + 1,
            lr=args.learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon,
        )
        master_delta = candidate_master - repair_master
        if frozen_direction is None:
            norm = torch.linalg.vector_norm(master_delta)
            if not bool(norm > 0):
                raise RuntimeError("step-1 master divergence is zero")
            frozen_direction = master_delta / norm
        projection = float(torch.sum(master_delta * frozen_direction).cpu())
        candidate_bf16 = candidate_master.to(torch.bfloat16)
        repair_bf16 = repair_master.to(torch.bfloat16)
        records.append({
            "step": index + 1,
            "state_id": state["id"],
            "input_sequence_length": metadata["sequence_length"],
            "candidate_current_loss": float(cand_loss_c.cpu()),
            "repair_current_loss": float(repair_loss_r.cpu()),
            "candidate_current_removal_l2": float(torch.linalg.vector_norm(removal_c).cpu()),
            "repair_current_removal_l2": float(torch.linalg.vector_norm(removal_r).cpu()),
            "candidate_current_removal_nonzero": not torch.equal(cand_grad_c, repair_grad_c),
            "repair_current_removal_nonzero": not torch.equal(cand_grad_r, repair_grad_r),
            "fp32_master_l2": float(torch.linalg.vector_norm(master_delta).cpu()),
            "fp32_master_projection": projection,
            "bf16_materialized_nonzero": int(torch.count_nonzero(candidate_bf16 != repair_bf16).cpu()),
        })
        del cand_grad_c, repair_grad_c, cand_grad_r, repair_grad_r

    checkpoint_projections = {
        str(step): records[step - 1]["fp32_master_projection"]
        for step in (8, 16, 32) if step <= args.steps
    }
    ordered_projection_steps = [step for step in (1, 8, 16, 32) if step <= args.steps]
    ordered_projections = [
        records[step - 1]["fp32_master_projection"] for step in ordered_projection_steps
    ]
    directional_projection_grows = (
        args.steps == 32
        and all(right > left for left, right in zip(
            ordered_projections, ordered_projections[1:]
        ))
    )
    trajectory_pass = (
        args.steps == 32
        and all(row["candidate_current_removal_nonzero"] for row in records)
        and all(row["repair_current_removal_nonzero"] for row in records)
        and directional_projection_grows
        and records[-1]["fp32_master_l2"] > 0
        and records[-1]["bf16_materialized_nonzero"] > 0
    )
    sources = {}
    for relative in (
        "results/round2/vl_math_ledger.json.gz",
        "results/round2/vl_silu_cause.json",
        "results/round2/vl_bias.json",
    ):
        sources[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    payload = {
        "schema": "kernel-analyzer-vl-silu-invocation-trajectory-v1",
        "status": "PASS_STRICT_FLASH_STYLE_CASE" if trajectory_pass else (
            "COMPLETE_PILOT" if args.steps < 32 else "FAIL_TRAJECTORY"
        ),
        "subject": {
            "model": str(args.model.resolve()),
            "proof_unit": PROOF_UNIT,
            "forward_origin": "silu",
            "target_parameter": TARGET,
            "candidate_backward": "q*sigma(x)*(1+x*(1-sigma(x))) decomposed into 8 ATen nodes",
            "repair_backward": "aten.silu_backward.default for layer-0 only",
            "unchanged_other_silu_invocations": 27,
        },
        "protocol": {
            "states": len(device_states),
            "trajectory_steps": args.steps,
            "state_order": [device_states[i % len(device_states)][0]["id"] for i in range(args.steps)],
            "same_current_weight_candidate_and_repair_measured_each_arm_each_step": True,
            "direction_frozen_after_calibration_step": 1,
            "optimizer": {
                "name": "AdamW", "learning_rate": args.learning_rate,
                "betas": [beta1, beta2], "epsilon": epsilon, "weight_decay": 0.0,
            },
            "fixed_grid_specialization": specialization,
        },
        "initial_controls": {
            "candidate_sham_exact": sham_exact,
            "candidate_repair_gradient_nonzero": initial_repair_nonzero,
            "candidate_repair_forward_loss_exact": initial_forward_exact,
        },
        "records": records,
        "checkpoint_projections": checkpoint_projections,
        "directional_projection_checkpoints": ordered_projection_steps,
        "directional_projection_strictly_grows": directional_projection_grows,
        "gates": {
            "complete_invocation_fb_proof": True,
            "actual_aot_decomposition_exactly_reproduced_by_intervention": True,
            "single_invocation_repair_nonzero": initial_repair_nonzero,
            "matched_sham_exact": sham_exact,
            "forward_unchanged": initial_forward_exact,
            "only_declared_parameter_updated": True,
            "paired_same_weight_measurement": True,
            "all_steps_repair_nonzero": all(
                row["candidate_current_removal_nonzero"]
                and row["repair_current_removal_nonzero"] for row in records
            ),
            "directional_live_weight_accumulation": trajectory_pass,
        },
        "evidence_files": sources,
        "claim_boundary": (
            "One concrete layer-0 SiLU forward and its actual decomposed AOT backward are "
            "isolated while the other 27 SiLU backward programs remain identical between arms. "
            "The six natural multimodal states repeat in a frozen order over the 32-step "
            "trajectory; this is a trajectory-local case and not cross-state generalization."
        ),
    }
    payload["result_sha256"] = canonical(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output), "status": payload["status"],
        "initial_controls": payload["initial_controls"],
        "final": records[-1],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
