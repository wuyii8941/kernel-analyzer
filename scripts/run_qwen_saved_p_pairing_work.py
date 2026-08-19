#!/usr/bin/env python3
"""Test whether saved-P error needs its real head/transport pairing to accumulate.

The experiment follows one repair-driven reference trajectory.  At every
pre-step state it evaluates three backward programs from identical weights and
optimizer moments:

* natural candidate dS;
* exact saved-P repair;
* repair plus the natural dS residual rolled across attention heads.

The roll preserves the complete local residual multiset, support and L2 norm,
but breaks its pairing with head-specific Q/K transport.  The scientific
outcome is the accumulated candidate-minus-repair effective optimizer update,
not a cross-state raw residual mean.
"""

from __future__ import annotations

import argparse
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
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data1/tzh/cache/torchinductor")

import torch
from torch._inductor.codecache import PyCodeCache

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "archive/round1_code/src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from kernel_analyzer.seup import adamw_effective_update_delta  # noqa: E402
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_generated_fp32_screen import file_digest, load_model  # noqa: E402
from scripts.run_qwen128_softmax_saved_p_trajectory import (  # noqa: E402
    CANDIDATE_ID,
    CARRIERS,
    SavedProbabilityRepair,
    adam_step,
    joint_dot,
    joint_norm,
    state_tokens,
)


def clone_tree(value: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: tensor.detach().float().clone() for key, tensor in value.items()}


def zero_tree(value: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: torch.zeros_like(tensor, dtype=torch.float32) for key, tensor in value.items()}


def add_tree_(target: dict[str, torch.Tensor], source: dict[str, torch.Tensor]) -> None:
    for key in target:
        target[key].add_(source[key])


def cosine(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    ln = float(joint_norm(left).item())
    rn = float(joint_norm(right).item())
    if ln == 0.0 or rn == 0.0:
        return float("nan")
    return float((joint_dot(left, right) / (ln * rn)).item())


def optimizer_rectification_geometry(
    repair_grad: dict[str, torch.Tensor],
    delta: dict[str, torch.Tensor],
    plus: dict[str, torch.Tensor],
    minus: dict[str, torch.Tensor],
) -> dict[str, float | int]:
    """Locate Adam's response-even energy relative to gradient sign crossings."""

    active = crossing = 0
    delta_energy = crossing_delta_energy = 0.0
    even_energy = crossing_even_energy = odd_energy = 0.0
    plus_even_inner = 0.0
    for name in delta:
        residual = delta[name]
        reference = repair_grad[name]
        natural = reference + residual
        antithetic = reference - residual
        active_mask = residual != 0
        crossing_mask = active_mask & (natural * antithetic <= 0)
        even = (plus[name] + minus[name]) * 0.5
        odd = (plus[name] - minus[name]) * 0.5
        active += int(active_mask.sum().item())
        crossing += int(crossing_mask.sum().item())
        delta_energy += float(residual.square().sum().item())
        crossing_delta_energy += float(residual[crossing_mask].square().sum().item())
        even_energy += float(even.square().sum().item())
        crossing_even_energy += float(even[crossing_mask].square().sum().item())
        odd_energy += float(odd.square().sum().item())
        plus_even_inner += float((plus[name] * even).sum().item())
    plus_norm = float(joint_norm(plus).item())
    even_norm = math.sqrt(max(0.0, even_energy))
    return {
        "active_gradient_residual_coordinates": active,
        "antithetic_gradient_sign_crossing_coordinates": crossing,
        "sign_crossing_fraction": crossing / max(active, 1),
        "delta_energy_on_sign_crossings": crossing_delta_energy / max(delta_energy, 1e-30),
        "response_even_l2": even_norm,
        "response_odd_l2": math.sqrt(max(0.0, odd_energy)),
        "response_even_energy_fraction": even_energy / max(even_energy + odd_energy, 1e-30),
        "response_even_energy_on_sign_crossings": crossing_even_energy / max(even_energy, 1e-30),
        "response_even_alignment_with_natural": plus_even_inner / max(plus_norm * even_norm, 1e-30),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/property/bias_property_search/saved_p_pairing_work.json",
    )
    args = parser.parse_args()
    if not 2 <= args.steps <= 32:
        raise ValueError("steps must be in [2, 32]")

    proof_path = ROOT / "results/coverage/cases/qwen128_softmax_fb.json"
    proof = json.loads(proof_path.read_text())
    if proof["candidate_id"] != CANDIDATE_ID or not all(
        proof["concrete_program_proof"].values()
    ):
        raise RuntimeError("saved-P F+B semantic boundary is incomplete")
    bank_path = ROOT / "results/coverage/qwen_seq128_input_bank.json"
    states = json.loads(bank_path.read_text())["states"]
    if len(states) < args.steps:
        raise RuntimeError("input bank is shorter than requested trajectory")
    release = ROOT / "results/coverage/runtime_releases/qwen_seq128_r1"
    capture = json.loads((release / "capture.json").read_text())
    if file_digest(bank_path) != capture["input"]["input_bank_sha256"]:
        raise RuntimeError("input bank differs from the frozen compiled release")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    configure_candidate_runtime(24000)
    model = load_model("qwen", Path("/data1/tzh/models/Qwen/Qwen3-1.7B"), device)
    parameters = dict(model.named_parameters())
    targets = {name: parameters[name] for name in CARRIERS}
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=True, dynamic=False)
    warm = torch.tensor([state_tokens(states[0])], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    candidate(warm).backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    # The observer binds the exact generated source-line identities itself.
    # Requiring byte identity for the entire recompiled wrapper would reject
    # harmless cache/codegen drift unrelated to this semantic boundary.

    master = {name: value.detach().float().clone() for name, value in targets.items()}
    first = zero_tree(master)
    second = zero_tree(master)

    def gradient(tokens: list[int], mode: str | None):
        with torch.no_grad():
            for name in CARRIERS:
                targets[name].copy_(master[name].to(targets[name].dtype))
        values = torch.tensor([tokens], dtype=torch.long, device=device)
        model.zero_grad(set_to_none=True)
        observer = SavedProbabilityRepair(modules, mode) if mode else None
        if observer is None:
            loss = candidate(values)
            loss.backward()
        else:
            with observer:
                loss = candidate(values)
                loss.backward()
        torch.cuda.synchronize(device)
        gradients = {}
        for name in CARRIERS:
            if targets[name].grad is None:
                raise RuntimeError(f"missing carrier gradient: {name}")
            gradients[name] = targets[name].grad.detach().float().clone()
            targets[name].grad = None
        return loss.detach(), gradients, observer

    natural_update_sum = zero_tree(master)
    shuffled_update_sum = zero_tree(master)
    negative_update_sum = zero_tree(master)
    optimizer_oddness_sum = zero_tree(master)
    natural_gradient_sum = zero_tree(master)
    shuffled_gradient_sum = zero_tree(master)
    natural_update_path = 0.0
    shuffled_update_path = 0.0
    negative_update_path = 0.0
    natural_gradient_path = 0.0
    shuffled_gradient_path = 0.0
    rows: list[dict[str, Any]] = []

    for offset, state in enumerate(states[:args.steps]):
        step = offset + 1
        tokens = state_tokens(state)
        loss_n, grad_n, _ = gradient(tokens, None)
        loss_r, grad_r, repair = gradient(tokens, "REPAIR_SAVED_P")
        loss_s, grad_s, shuffled = gradient(tokens, "PERMUTE_SAVED_P_RESIDUAL")
        if repair is None or shuffled is None:
            raise RuntimeError("missing saved-P observer")

        grad_delta_n = {name: grad_n[name] - grad_r[name] for name in CARRIERS}
        grad_delta_s = {name: grad_s[name] - grad_r[name] for name in CARRIERS}
        update_delta_n = adamw_effective_update_delta(
            grad_n, grad_r, first, second, master,
            step=step, learning_rate=args.learning_rate,
            betas=(0.9, 0.95), epsilon=1e-8, weight_decay=0.0,
        )
        update_delta_s = adamw_effective_update_delta(
            grad_s, grad_r, first, second, master,
            step=step, learning_rate=args.learning_rate,
            betas=(0.9, 0.95), epsilon=1e-8, weight_decay=0.0,
        )
        # Exact antithetic gradient residual around the same repair gradient.
        # No extra model execution is needed.  A locally odd optimizer map
        # would produce update_delta_minus == -update_delta_n.
        grad_anti = {
            name: grad_r[name].mul(2.0).sub(grad_n[name]) for name in CARRIERS
        }
        update_delta_minus = adamw_effective_update_delta(
            grad_anti, grad_r, first, second, master,
            step=step, learning_rate=args.learning_rate,
            betas=(0.9, 0.95), epsilon=1e-8, weight_decay=0.0,
        )
        oddness = {
            name: update_delta_n[name] + update_delta_minus[name]
            for name in CARRIERS
        }

        gn = float(joint_norm(grad_delta_n).item())
        gs = float(joint_norm(grad_delta_s).item())
        un = float(joint_norm(update_delta_n).item())
        us = float(joint_norm(update_delta_s).item())
        um = float(joint_norm(update_delta_minus).item())
        odd = float(joint_norm(oddness).item())
        rectification = optimizer_rectification_geometry(
            grad_r, grad_delta_n, update_delta_n, update_delta_minus
        )
        natural_gradient_path += gn
        shuffled_gradient_path += gs
        natural_update_path += un
        shuffled_update_path += us
        negative_update_path += um
        add_tree_(natural_gradient_sum, grad_delta_n)
        add_tree_(shuffled_gradient_sum, grad_delta_s)
        add_tree_(natural_update_sum, update_delta_n)
        add_tree_(shuffled_update_sum, update_delta_s)
        add_tree_(negative_update_sum, update_delta_minus)
        add_tree_(optimizer_oddness_sum, oddness)

        pre_norm_rel = abs(
            shuffled.permuted_residual_l2 - shuffled.natural_residual_l2
        ) / max(shuffled.natural_residual_l2, 1e-30)
        post_norm_rel = abs(
            shuffled.delivered_residual_l2 - shuffled.natural_residual_l2
        ) / max(shuffled.natural_residual_l2, 1e-30)
        rows.append({
            "step": step,
            "state_id": str(state.get("sequence_id", offset)),
            "forward_loss_equal": bool(torch.equal(loss_n, loss_r) and torch.equal(loss_n, loss_s)),
            "natural_local_residual_l2": shuffled.natural_residual_l2,
            "permuted_local_residual_l2": shuffled.permuted_residual_l2,
            "delivered_local_residual_l2": shuffled.delivered_residual_l2,
            "precast_norm_relative_error": pre_norm_rel,
            "postcast_norm_relative_error": post_norm_rel,
            "natural_local_residual_sum": shuffled.residual_sum,
            "permuted_local_residual_sum": shuffled.permuted_residual_sum,
            "natural_gradient_effect_l2": gn,
            "shuffled_gradient_effect_l2": gs,
            "natural_update_effect_l2": un,
            "shuffled_update_effect_l2": us,
            "antithetic_gradient_update_effect_l2": um,
            "optimizer_oddness_l2": odd,
            "optimizer_oddness_ratio": odd / max(un + um, 1e-30),
            **rectification,
            "natural_shuffled_gradient_cosine": cosine(grad_delta_n, grad_delta_s),
            "natural_shuffled_update_cosine": cosine(update_delta_n, update_delta_s),
            "repair_changed_coordinates": repair.changed_coordinates,
            "shuffle_changed_coordinates": shuffled.changed_coordinates,
        })
        print(json.dumps({"event": "PAIRING_STEP", **rows[-1]}), flush=True)

        # The conditioning trajectory is defined independently by the repair
        # program; neither natural nor shuffled measurements alter its states.
        for name in CARRIERS:
            adam_step(master[name], grad_r[name], first[name], second[name], step, args.learning_rate)
        del grad_n, grad_r, grad_s, grad_anti, grad_delta_n, grad_delta_s
        del update_delta_n, update_delta_s, update_delta_minus, oddness
        torch.cuda.empty_cache()

    natural_update_resultant = float(joint_norm(natural_update_sum).item())
    shuffled_update_resultant = float(joint_norm(shuffled_update_sum).item())
    natural_gradient_resultant = float(joint_norm(natural_gradient_sum).item())
    shuffled_gradient_resultant = float(joint_norm(shuffled_gradient_sum).item())
    negative_update_resultant = float(joint_norm(negative_update_sum).item())
    optimizer_oddness_resultant = float(joint_norm(optimizer_oddness_sum).item())
    update_suppression = 1.0 - shuffled_update_resultant / max(natural_update_resultant, 1e-30)
    gradient_suppression = 1.0 - shuffled_gradient_resultant / max(natural_gradient_resultant, 1e-30)
    payload = {
        "schema": "kernel-analyzer-trajectory-conditioned-pairing-work-v2",
        "case_id": CANDIDATE_ID,
        "scientific_question": (
            "Does the saved-P local dS residual require its real head-specific backward "
            "transport pairing to form accumulated effective-update work?"
        ),
        "conditioning": (
            "repair-driven training trajectory; natural, repair, and intervention are "
            "evaluated from the identical pre-step weights and AdamW moments"
        ),
        "intervention": {
            "operation": "fixed cyclic roll of the natural dS residual across 16 attention heads",
            "head_shift": 1,
            "preserves": ["complete residual multiset", "causal support", "precast L2 norm"],
            "breaks": "residual pairing with head-specific downstream Q/K transport",
            "is_candidate_implementation": False,
        },
        "optimizer": {
            "name": "AdamW", "learning_rate": args.learning_rate,
            "betas": [0.9, 0.95], "epsilon": 1e-8, "weight_decay": 0.0,
        },
        "steps": args.steps,
        "records": rows,
        "aggregate": {
            "natural_gradient_resultant_l2": natural_gradient_resultant,
            "shuffled_gradient_resultant_l2": shuffled_gradient_resultant,
            "gradient_pairing_suppression": gradient_suppression,
            "natural_gradient_persistence": natural_gradient_resultant / max(natural_gradient_path, 1e-30),
            "shuffled_gradient_persistence": shuffled_gradient_resultant / max(shuffled_gradient_path, 1e-30),
            "natural_update_resultant_l2": natural_update_resultant,
            "shuffled_update_resultant_l2": shuffled_update_resultant,
            "update_pairing_suppression": update_suppression,
            "natural_update_persistence": natural_update_resultant / max(natural_update_path, 1e-30),
            "shuffled_update_persistence": shuffled_update_resultant / max(shuffled_update_path, 1e-30),
            "antithetic_gradient_update_resultant_l2": negative_update_resultant,
            "antithetic_gradient_update_persistence": negative_update_resultant / max(negative_update_path, 1e-30),
            "optimizer_oddness_resultant_l2": optimizer_oddness_resultant,
            "optimizer_oddness_resultant_ratio": optimizer_oddness_resultant / max(
                natural_update_resultant + negative_update_resultant, 1e-30
            ),
            "optimizer_nonoddness_resultant_l2": optimizer_oddness_resultant,
            "optimizer_nonoddness_resultant_ratio": optimizer_oddness_resultant / max(
                natural_update_resultant + negative_update_resultant, 1e-30
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
            "natural_antithetic_update_resultant_cosine": cosine(
                natural_update_sum, negative_update_sum
            ),
            "stateless_sgd_natural_resultant_l2": (
                args.learning_rate * natural_gradient_resultant
            ),
            "adam_over_stateless_sgd_resultant": natural_update_resultant / max(
                args.learning_rate * natural_gradient_resultant, 1e-30
            ),
            "natural_shuffled_update_resultant_cosine": cosine(
                natural_update_sum, shuffled_update_sum
            ),
            "all_forward_losses_equal": all(row["forward_loss_equal"] for row in rows),
            "max_precast_norm_relative_error": max(row["precast_norm_relative_error"] for row in rows),
            "max_postcast_norm_relative_error": max(row["postcast_norm_relative_error"] for row in rows),
        },
        "interpretation_rule": {
            "supports_pairing_mechanism": (
                "The shuffled arm materially reduces accumulated gradient and effective-update "
                "resultants while the local residual multiset and precast norm remain fixed."
            ),
            "rejects_pairing_mechanism": (
                "The shuffled arm leaves accumulated effective-update work unchanged or larger."
            ),
            "no_universal_threshold_claimed": True,
        },
        "claim_boundary": (
            "A positive result supports residual/transport coupling in this closed saved-P "
            "semantic region. It does not by itself prove a universal operator property."
        ),
    }
    if not all(math.isfinite(value) for value in (
        natural_update_resultant, shuffled_update_resultant,
        natural_gradient_resultant, shuffled_gradient_resultant,
        negative_update_resultant, optimizer_oddness_resultant,
    )):
        raise RuntimeError("nonfinite pairing-work aggregate")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "PAIRING_WORK_COMPLETE", **payload["aggregate"]}), flush=True)


if __name__ == "__main__":
    main()
