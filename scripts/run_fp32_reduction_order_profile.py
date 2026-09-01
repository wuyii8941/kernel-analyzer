#!/usr/bin/env python3
"""Run the frozen FP32-only RMSNorm weight-gradient order experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from analyze_three_mechanism_profiles import _profile  # noqa: E402
from kernel_analyzer.fp32_reduction_order import ordered_rms_norm  # noqa: E402
from scripts.run_heldout_lmhead_consequence import adam_delta  # noqa: E402


PROTOCOL = ROOT / "results/property/fp32_reduction_order_v1/protocol.json"
STATE_PROTOCOL = ROOT / "results/trajectory/liger_protocol.json"
STATE_DESIGN = ROOT / "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/supplementary_state_design_v1.json"
MODEL = Path("/data1/tzh/models/Qwen/Qwen3-1.7B")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_digest(value: torch.Tensor) -> str:
    array = value.detach().contiguous().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _zero_stage(repair_vectors: list[np.ndarray]) -> dict[str, Any]:
    repair = np.stack(repair_vectors).astype(np.float64)
    return {
        "status": "EXACT_ZERO_EFFECT",
        "state_count": len(repair_vectors),
        "coordinate_count": int(repair_vectors[0].size),
        "total_effect_rms": 0.0,
        "repair_rms": float(np.sqrt(np.mean(np.sum(repair * repair, axis=1)))),
        "decision": "NO_FORWARD_DIFFERENCE_BY_CONSTRUCTION_AND_BITWISE_CHECK",
    }


def _profile_or_zero(
    effects: list[np.ndarray], repairs: list[np.ndarray], *, seed: int
) -> dict[str, Any]:
    if all(np.count_nonzero(value) == 0 for value in effects):
        return _zero_stage(repairs)
    return _profile(effects, repairs, seed=seed)


def _holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[name] = running
    return adjusted


def _branch(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    upstream: torch.Tensor,
    *,
    epsilon: float,
    reduction: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    branch_hidden = hidden.detach().clone().requires_grad_(True)
    branch_weight = weight.detach().clone().requires_grad_(True)
    output = ordered_rms_norm(
        branch_hidden,
        branch_weight,
        epsilon=epsilon,
        reduction=reduction,
    )
    output.backward(upstream)
    if branch_hidden.grad is None or branch_weight.grad is None:
        raise RuntimeError("ordered RMSNorm backward did not produce both gradients")
    return output.detach(), branch_hidden.grad.detach(), branch_weight.grad.detach()


def _confidence_excludes_zero(profile: dict[str, Any], effect: str) -> bool:
    interval = profile[effect]["bootstrap_95"]
    return bool(interval[0] > 0.0 or interval[-1] < 0.0)


def run(device: torch.device) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM

    protocol = json.loads(PROTOCOL.read_text())
    if protocol["status"] != "FROZEN_BEFORE_EMPIRICAL_RESULTS":
        raise RuntimeError("FP32 reduction-order protocol is not frozen")
    trajectory = json.loads(STATE_PROTOCOL.read_text())
    design = json.loads(STATE_DESIGN.read_text())
    records = {row["sequence_id"]: row for row in design["records"]}
    state_ids = list(trajectory["trajectory"]["state_order"])
    states = [records[state_id] for state_id in state_ids]
    if len(states) != 32:
        raise RuntimeError("the frozen experiment requires exactly 32 states")

    torch.manual_seed(20260901)
    torch.cuda.manual_seed_all(20260901)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
    ).to(device).eval()
    model.config.use_cache = False
    norm = model.model.norm
    parameter = norm.weight
    epsilon = float(norm.variance_epsilon)
    master = parameter.detach().float().clone()
    first = torch.zeros_like(master)
    second = torch.zeros_like(master)

    stages: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {
        name: ([], []) for name in ("LOCAL", "PARAMETER_GRADIENT", "ADAMW_UPDATE")
    }
    diagnostics: list[dict[str, Any]] = []

    for index, (state_id, state) in enumerate(zip(state_ids, states, strict=True)):
        with torch.no_grad():
            parameter.copy_(master)
        captured: list[torch.Tensor] = []

        def capture_input(_module, arguments):
            captured.append(arguments[0].detach())

        handle = norm.register_forward_pre_hook(capture_input)
        input_ids = torch.tensor([state["input_ids"]], dtype=torch.long, device=device)
        with torch.no_grad():
            model.model(input_ids=input_ids, use_cache=False, return_dict=True)
        handle.remove()
        if len(captured) != 1:
            raise RuntimeError(f"final RMSNorm input was captured {len(captured)} times")
        hidden = captured[0].float()

        with torch.no_grad():
            standard_output = norm(hidden).float()
        loss_input = standard_output.detach().clone().requires_grad_(True)
        logits = model.lm_head(loss_input)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            input_ids[:, 1:].reshape(-1),
        )
        (upstream,) = torch.autograd.grad(loss, loss_input)
        upstream = upstream.detach().float()

        candidate_output, candidate_hidden_grad, candidate_gradient = _branch(
            hidden, master, upstream, epsilon=epsilon, reduction="sequential"
        )
        repair_output, repair_hidden_grad, repair_gradient = _branch(
            hidden, master, upstream, epsilon=epsilon, reduction="balanced"
        )
        sham_output, sham_hidden_grad, sham_gradient = _branch(
            hidden, master, upstream, epsilon=epsilon, reduction="sequential"
        )
        if not torch.equal(candidate_output, repair_output):
            raise RuntimeError("the two FP32 reduction orders changed the forward output")
        if not torch.equal(candidate_output, standard_output):
            max_abs = float((candidate_output - standard_output).abs().max().item())
            raise RuntimeError(f"custom RMSNorm forward differs from model RMSNorm: {max_abs}")
        if not torch.equal(candidate_hidden_grad, repair_hidden_grad):
            raise RuntimeError("the two reduction orders changed the hidden-state gradient")
        if not torch.equal(candidate_output, sham_output):
            raise RuntimeError("matched sham changed the forward output")
        if not torch.equal(candidate_hidden_grad, sham_hidden_grad):
            raise RuntimeError("matched sham changed the hidden-state gradient")
        if not torch.equal(candidate_gradient, sham_gradient):
            raise RuntimeError("matched sham did not reproduce the candidate gradient")

        normalized = hidden * torch.rsqrt(hidden.square().mean(dim=-1, keepdim=True) + epsilon)
        contributions = (upstream * normalized).reshape(-1, master.numel())
        high_precision = contributions.double().sum(dim=0)
        candidate_update, _, _ = adam_delta(
            candidate_gradient, first, second, index + 1,
            learning_rate=1e-4, beta1=0.9, beta2=0.95,
        )
        repair_update, next_first, next_second = adam_delta(
            repair_gradient, first, second, index + 1,
            learning_rate=1e-4, beta1=0.9, beta2=0.95,
        )

        local_effect = (candidate_output - repair_output).reshape(-1).cpu().numpy()
        local_repair = repair_output.reshape(-1).cpu().numpy()
        gradient_effect = (candidate_gradient - repair_gradient).cpu().numpy()
        gradient_repair = repair_gradient.cpu().numpy()
        update_effect = (candidate_update - repair_update).cpu().numpy()
        update_repair = repair_update.cpu().numpy()
        stages["LOCAL"][0].append(local_effect)
        stages["LOCAL"][1].append(local_repair)
        stages["PARAMETER_GRADIENT"][0].append(gradient_effect)
        stages["PARAMETER_GRADIENT"][1].append(gradient_repair)
        stages["ADAMW_UPDATE"][0].append(update_effect)
        stages["ADAMW_UPDATE"][1].append(update_repair)
        diagnostics.append({
            "state_id": state_id,
            "loss": float(loss.detach().item()),
            "forward_digest": _tensor_digest(candidate_output),
            "upstream_digest": _tensor_digest(upstream),
            "candidate_gradient_digest": _tensor_digest(candidate_gradient),
            "repair_gradient_digest": _tensor_digest(repair_gradient),
            "candidate_minus_repair_l2": float(torch.linalg.vector_norm(candidate_gradient - repair_gradient).item()),
            "candidate_error_vs_fp64_l2": float(torch.linalg.vector_norm(candidate_gradient.double() - high_precision).item()),
            "repair_error_vs_fp64_l2": float(torch.linalg.vector_norm(repair_gradient.double() - high_precision).item()),
            "candidate_and_sham_bitwise_equal": True,
            "forward_outputs_bitwise_equal": True,
            "hidden_gradients_bitwise_equal": True,
            "all_declared_operands_fp32": bool(
                hidden.dtype == upstream.dtype == candidate_gradient.dtype == repair_gradient.dtype == torch.float32
            ),
        })
        master.add_(repair_update)
        first, second = next_first, next_second
        print(json.dumps({
            "event": "FP32_REDUCTION_ORDER_STATE_COMPLETE",
            "step": index + 1,
            "state_id": state_id,
            "gradient_difference_l2": diagnostics[-1]["candidate_minus_repair_l2"],
        }), flush=True)
        del logits, loss, loss_input, hidden, upstream
        torch.cuda.empty_cache()

    profiles = {
        stage: _profile_or_zero(effects, repairs, seed=20260920 + stage_index)
        for stage_index, (stage, (effects, repairs)) in enumerate(stages.items())
    }
    update = profiles["ADAMW_UPDATE"]
    if update.get("status") == "EXACT_ZERO_EFFECT":
        primary = {
            "status": "SOURCE_PREDICTOR_NOT_CONFIRMED",
            "reason": "the two declared FP32 reduction orders produced exact-zero AdamW update difference",
        }
    else:
        effect_keys = {
            "fixed_additive_direction": "additive_heldout_effect",
            "repair_aligned_scaling": "aligned_effect",
            "remaining_direction": "orthogonal_heldout_effect",
        }
        raw_p = {name: float(update[key]["signflip_p"]) for name, key in effect_keys.items()}
        adjusted = _holm(raw_p)
        decisions = {}
        for name, key in effect_keys.items():
            interval_excludes_zero = _confidence_excludes_zero(update, key)
            direction_ok = True
            if name in {"fixed_additive_direction", "remaining_direction"}:
                direction_ok = bool(update[key]["bootstrap_95"][0] > 0.0)
            decisions[name] = {
                "estimate": float(update[key]["estimate"]),
                "confidence_interval_95": [float(update[key]["bootstrap_95"][0]), float(update[key]["bootstrap_95"][-1])],
                "raw_p": raw_p[name],
                "holm_adjusted_p": adjusted[name],
                "confirmed": bool(interval_excludes_zero and adjusted[name] <= 0.05 and direction_ok),
            }
        confirmed = [name for name, row in decisions.items() if row["confirmed"]]
        primary = {
            "status": "SOURCE_PREDICTOR_CONFIRMED_FOR_DECLARED_ORDER_PAIR" if confirmed else "SOURCE_PREDICTOR_NOT_CONFIRMED",
            "confirmed_update_effects": confirmed,
            "effects": decisions,
            "multiplicity_family_size": 3,
        }

    return {
        "schema": "kernel-analyzer-fp32-reduction-order-profile-v1",
        "status": "COMPLETE",
        "case_id": protocol["case"]["case_id"],
        "protocol_sha256": _sha256(PROTOCOL),
        "state_protocol_sha256": _sha256(STATE_PROTOCOL),
        "state_design_sha256": _sha256(STATE_DESIGN),
        "model_config_sha256": _sha256(MODEL / "config.json"),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "tf32_allowed": False,
        "dtype": "FP32",
        "state_ids": state_ids,
        "split": {"calibration": state_ids[:16], "confirmation": state_ids[16:]},
        "implementations": protocol["implementations"],
        "invariants": {
            "all_states_forward_outputs_bitwise_equal": all(row["forward_outputs_bitwise_equal"] for row in diagnostics),
            "all_states_hidden_gradients_bitwise_equal": all(row["hidden_gradients_bitwise_equal"] for row in diagnostics),
            "all_states_candidate_and_sham_bitwise_equal": all(row["candidate_and_sham_bitwise_equal"] for row in diagnostics),
            "all_declared_operands_fp32": all(row["all_declared_operands_fp32"] for row in diagnostics),
        },
        "profiles": profiles,
        "primary_update_decision": primary,
        "diagnostics": diagnostics,
        "claim_boundary": "One Qwen checkpoint and 32 frozen sequence inputs. This tests a same-FP32-format implementation-order difference; it is not a long-run or cross-model conclusion.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/property/fp32_reduction_order_v1/profile.json",
    )
    args = parser.parse_args()
    result = run(torch.device(args.device))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "event": "FP32_REDUCTION_ORDER_PROFILE_WRITTEN",
        "output": str(args.output),
        "decision": result["primary_update_decision"]["status"],
    }), flush=True)


if __name__ == "__main__":
    main()
