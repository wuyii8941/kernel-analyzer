#!/usr/bin/env python3
"""Measure AdamW rectification of the exact Qwen3-VL SiLU gradient residual."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

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

from scripts.round2_vl_bias import STATES  # noqa: E402
from scripts.round2_vl_silu_cause import DecomposedSiluModule  # noqa: E402
from scripts.round2_vl_smoke import prepare_step  # noqa: E402
from scripts.round2_vl_static import specialize_fixed_grid  # noqa: E402
from scripts.run_vl_silu_invocation_trajectory import (  # noqa: E402
    LossStep,
    TARGET,
    adamw_step,
)


def adam_update(
    grad: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    step: int,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    epsilon: float,
) -> torch.Tensor:
    m = first * beta1 + grad * (1.0 - beta1)
    v = second * beta2 + grad.square() * (1.0 - beta2)
    return -(lr * (m / (1.0 - beta1**step)) /
             ((v / (1.0 - beta2**step)).sqrt() + epsilon))


def norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value).item())


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = norm(left) * norm(right)
    return float(torch.sum(left * right).item() / denominator) if denominator else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path,
        default=Path("/data1/tzh/models/Qwen/Qwen3-VL-Reranker-2B"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--pad-length", type=int, default=160)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/property/bias_property_search/vl_silu_optimizer_oddness.json",
    )
    args = parser.parse_args()
    if not 2 <= args.steps <= 32:
        raise ValueError("steps must be in [2, 32]")

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
    for module in text_mlps.values():
        module.act_fn = DecomposedSiluModule()

    first_values = {key: value.to(device) for key, value in prepared[0][1].items()}
    specialize_fixed_grid(model, first_values["image_grid_thw"], first_values["input_ids"])
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

    master = target.detach().float().clone()
    first = torch.zeros_like(master)
    second = torch.zeros_like(master)
    beta1, beta2, epsilon = 0.9, 0.95, 1e-8

    def gradient(arguments: tuple[torch.Tensor, ...], decomposed: bool):
        with torch.no_grad():
            target.copy_(master.to(torch.bfloat16))
        text_mlps[0].act_fn = DecomposedSiluModule() if decomposed else originals[0]
        model.zero_grad(set_to_none=True)
        loss = step_module(*arguments)
        loss.backward()
        if target.grad is None:
            raise RuntimeError("target gradient is absent")
        result = target.grad.detach().float().clone()
        target.grad = None
        return loss.detach(), result

    gradient_sum = torch.zeros_like(master)
    plus_sum = torch.zeros_like(master)
    minus_sum = torch.zeros_like(master)
    oddness_sum = torch.zeros_like(master)
    gradient_path = plus_path = minus_path = 0.0
    rows = []
    for offset in range(args.steps):
        state, arguments, metadata = device_states[offset % len(device_states)]
        candidate_loss, candidate_grad = gradient(arguments, True)
        repair_loss, repair_grad = gradient(arguments, False)
        delta = candidate_grad - repair_grad
        anti_grad = repair_grad - delta
        repair_update = adam_update(
            repair_grad, first, second, offset + 1, lr=args.learning_rate,
            beta1=beta1, beta2=beta2, epsilon=epsilon,
        )
        plus = adam_update(
            candidate_grad, first, second, offset + 1, lr=args.learning_rate,
            beta1=beta1, beta2=beta2, epsilon=epsilon,
        ) - repair_update
        minus = adam_update(
            anti_grad, first, second, offset + 1, lr=args.learning_rate,
            beta1=beta1, beta2=beta2, epsilon=epsilon,
        ) - repair_update
        oddness = plus + minus
        response_even = oddness * 0.5
        response_odd = (plus - minus) * 0.5
        active_mask = delta != 0
        crossing_mask = active_mask & (candidate_grad * anti_grad <= 0)
        active_coordinates = int(active_mask.sum().item())
        crossing_coordinates = int(crossing_mask.sum().item())
        delta_energy = float(delta.square().sum().item())
        even_energy = float(response_even.square().sum().item())
        odd_energy = float(response_odd.square().sum().item())
        dn, pn, mn, on = norm(delta), norm(plus), norm(minus), norm(oddness)
        gradient_sum.add_(delta)
        plus_sum.add_(plus)
        minus_sum.add_(minus)
        oddness_sum.add_(oddness)
        gradient_path += dn
        plus_path += pn
        minus_path += mn
        rows.append({
            "step": offset + 1,
            "state_id": state["id"],
            "input_sequence_length": metadata["sequence_length"],
            "forward_loss_equal": bool(torch.equal(candidate_loss, repair_loss)),
            "gradient_residual_l2": dn,
            "antithetic_gradient_residual_l2": norm(anti_grad - repair_grad),
            "natural_update_residual_l2": pn,
            "antithetic_update_residual_l2": mn,
            "optimizer_oddness_l2": on,
            "optimizer_oddness_ratio": on / max(pn + mn, 1e-30),
            "active_gradient_residual_coordinates": active_coordinates,
            "antithetic_gradient_sign_crossing_coordinates": crossing_coordinates,
            "sign_crossing_fraction": crossing_coordinates / max(active_coordinates, 1),
            "delta_energy_on_sign_crossings": (
                float(delta[crossing_mask].square().sum().item()) / max(delta_energy, 1e-30)
            ),
            "response_even_l2": math.sqrt(max(0.0, even_energy)),
            "response_odd_l2": math.sqrt(max(0.0, odd_energy)),
            "response_even_energy_fraction": even_energy / max(even_energy + odd_energy, 1e-30),
            "response_even_energy_on_sign_crossings": (
                float(response_even[crossing_mask].square().sum().item()) / max(even_energy, 1e-30)
            ),
            "response_even_alignment_with_natural": (
                float((response_even * plus).sum().item())
                / max(norm(response_even) * pn, 1e-30)
            ),
            "natural_antithetic_update_cosine": cosine(plus, minus),
        })
        print(json.dumps({"event": "SILU_ODDNESS_STEP", **rows[-1]}), flush=True)
        adamw_step(
            master, repair_grad, first, second, offset + 1,
            lr=args.learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon,
        )
        del candidate_grad, repair_grad, delta, anti_grad, repair_update
        del plus, minus, oddness, response_even, response_odd
        torch.cuda.empty_cache()

    gradient_resultant = norm(gradient_sum)
    plus_resultant = norm(plus_sum)
    minus_resultant = norm(minus_sum)
    oddness_resultant = norm(oddness_sum)
    aggregate = {
        "gradient_resultant_l2": gradient_resultant,
        "gradient_persistence": gradient_resultant / max(gradient_path, 1e-30),
        "natural_update_resultant_l2": plus_resultant,
        "natural_update_persistence": plus_resultant / max(plus_path, 1e-30),
        "antithetic_update_resultant_l2": minus_resultant,
        "antithetic_update_persistence": minus_resultant / max(minus_path, 1e-30),
        "optimizer_oddness_resultant_l2": oddness_resultant,
        "optimizer_oddness_resultant_ratio": oddness_resultant / max(
            plus_resultant + minus_resultant, 1e-30
        ),
        "optimizer_nonoddness_resultant_l2": oddness_resultant,
        "optimizer_nonoddness_resultant_ratio": oddness_resultant / max(
            plus_resultant + minus_resultant, 1e-30
        ),
        "mean_step_optimizer_oddness_ratio": sum(
            row["optimizer_oddness_ratio"] for row in rows
        ) / len(rows),
        "mean_step_sign_crossing_fraction": sum(
            row["sign_crossing_fraction"] for row in rows
        ) / len(rows),
        "mean_step_delta_energy_on_sign_crossings": sum(
            row["delta_energy_on_sign_crossings"] for row in rows
        ) / len(rows),
        "mean_step_response_even_energy_fraction": sum(
            row["response_even_energy_fraction"] for row in rows
        ) / len(rows),
        "mean_step_response_even_energy_on_sign_crossings": sum(
            row["response_even_energy_on_sign_crossings"] for row in rows
        ) / len(rows),
        "energy_weighted_response_even_on_sign_crossings": sum(
            row["response_even_l2"] ** 2
            * row["response_even_energy_on_sign_crossings"]
            for row in rows
        ) / max(sum(row["response_even_l2"] ** 2 for row in rows), 1e-30),
        "response_even_energy_in_first_two_steps": sum(
            row["response_even_l2"] ** 2 for row in rows[:2]
        ) / max(sum(row["response_even_l2"] ** 2 for row in rows), 1e-30),
        "step_integrated_response_even_energy_fraction": sum(
            row["response_even_l2"] ** 2 for row in rows
        ) / max(sum(
            row["response_even_l2"] ** 2 + row["response_odd_l2"] ** 2
            for row in rows
        ), 1e-30),
        "natural_antithetic_update_resultant_cosine": cosine(plus_sum, minus_sum),
        "stateless_sgd_resultant_l2": args.learning_rate * gradient_resultant,
        "adam_over_stateless_sgd_resultant": plus_resultant / max(
            args.learning_rate * gradient_resultant, 1e-30
        ),
        "all_forward_losses_equal": all(row["forward_loss_equal"] for row in rows),
    }
    payload = {
        "schema": "kernel-analyzer-vl-silu-optimizer-oddness-v2",
        "case": "qwen3vl_layer0_silu_backward",
        "conditioning": "repair-driven six-state cyclic training trajectory",
        "closed_boundary": (
            "same aten.silu forward; candidate is the actual decomposed AOT VJP; "
            "repair is native aten.silu_backward for layer 0 only"
        ),
        "intervention": (
            "replace natural delta_g by exact -delta_g at the same repair gradient, "
            "weights, Adam moments, input, and RNG"
        ),
        "optimizer": {
            "name": "AdamW", "learning_rate": args.learning_rate,
            "betas": [beta1, beta2], "epsilon": epsilon, "weight_decay": 0.0,
        },
        "steps": args.steps,
        "records": rows,
        "aggregate": aggregate,
        "interpretation": (
            "Nonzero oddness is optimizer rectification of an exactly sign-symmetric "
            "gradient residual pair; stateless SGD is the exact linear null."
        ),
        "claim_boundary": (
            "This tests optimizer-map curvature for one already closed F+B case; it "
            "does not use trajectory drift as a formation label."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "SILU_ODDNESS_COMPLETE", **aggregate}), flush=True)


if __name__ == "__main__":
    main()
