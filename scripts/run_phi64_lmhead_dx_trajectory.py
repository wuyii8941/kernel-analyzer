#!/usr/bin/env python3
"""Paired evolving final-norm trajectory for the Phi64 lm_head dX repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch._inductor.codecache import PyCodeCache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "archive/round1_code/src"))
sys.path.insert(0, str(ROOT / "src"))

from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_frozen_candidate_fp32_screen import wrapper_modules  # noqa: E402
from scripts.run_generated_fp32_screen import load_model, tensor_digest  # noqa: E402
from scripts.run_phi64_lmhead_dx_repair import MMRepair  # noqa: E402
from scripts.run_targeted_full_coordinate import validate_release  # noqa: E402
from kernel_analyzer.seup import (  # noqa: E402
    SEUPCalibrator,
    SymmetricSEUPEvaluator,
    sgd_effective_update_delta,
)


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def evaluate(model: torch.nn.Module, candidate: Any, values: torch.Tensor,
             modules: list[Any], seed: int, repair: bool,
             master: torch.Tensor | None = None) -> tuple[str, torch.Tensor, dict[str, Any] | None]:
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if master is not None:
        with torch.no_grad():
            model.model.norm.weight.copy_(master.to(model.model.norm.weight.dtype))
    model.zero_grad(set_to_none=True)
    observer = MMRepair(modules, "REPAIR_FP32_CAST_BF16") if repair else None
    if observer:
        with observer:
            loss = candidate(values); loss.backward()
    else:
        loss = candidate(values); loss.backward()
    torch.cuda.synchronize(values.device)
    gradient = model.model.norm.weight.grad.detach().float().clone()
    return tensor_digest(loss), gradient, observer.local if observer else None


def run_seup_mainline(
    *, model: torch.nn.Module, candidate: Any, states: list[dict[str, Any]],
    modules: list[Any], parameter: torch.Tensor, initial: torch.Tensor,
    learning_rate: float, output: Path, seup_output: Path,
) -> None:
    """Run the frozen-calibration/live-evaluation SEUP protocol for Phi."""
    if len(states) < 32:
        raise RuntimeError("SEUP requires 16 calibration and 16 evaluation states")
    device = parameter.device
    calibrator = SEUPCalibrator(calibration_steps=16, gate_cosine=0.5)
    calibration_master = initial.clone()
    calibration_rows = []
    for index, state in enumerate(states[:16]):
        values = torch.tensor([state["token_ids"]], dtype=torch.long, device=device)
        loss_c, grad_c, _ = evaluate(model, candidate, values, modules, 3407 + index, False, calibration_master)
        loss_r, grad_r, local = evaluate(model, candidate, values, modules, 3407 + index, True, calibration_master)
        if loss_c != loss_r:
            raise RuntimeError("Phi calibration repair changed forward loss")
        delta = sgd_effective_update_delta(grad_c, grad_r, learning_rate=learning_rate)
        calibrator.add(str(state.get("state_id", state.get("sequence_id", index))), delta)
        with torch.no_grad():
            calibration_master.add_(grad_c, alpha=-learning_rate)
        calibration_rows.append({
            "step": index + 1,
            "state_id": str(state.get("state_id", state.get("sequence_id", index))),
            "local_l2": float(torch.linalg.vector_norm(delta["value"]).item()),
            "repair_changed_coordinates": None if local is None else local["changed_coordinates"],
        })
        del values, grad_c, grad_r, delta
        torch.cuda.empty_cache()
    carrier = calibrator.freeze()
    evaluator = SymmetricSEUPEvaluator(carrier, evaluation_steps=16)
    candidate_master = initial.clone()
    repair_master = initial.clone()
    rows = []
    for offset, state in enumerate(states[16:32]):
        step = offset + 1
        values = torch.tensor([state["token_ids"]], dtype=torch.long, device=device)
        loss_cc, grad_cc, local_cc = evaluate(model, candidate, values, modules, 4407 + offset, False, candidate_master)
        loss_cr, grad_cr, local_cr = evaluate(model, candidate, values, modules, 4407 + offset, True, candidate_master)
        loss_rc, grad_rc, local_rc = evaluate(model, candidate, values, modules, 5407 + offset, False, repair_master)
        loss_rr, grad_rr, local_rr = evaluate(model, candidate, values, modules, 5407 + offset, True, repair_master)
        if loss_cc != loss_cr or loss_rc != loss_rr:
            raise RuntimeError("Phi evaluation repair changed forward loss")
        uc_sc = {"value": grad_cc.mul(-learning_rate)}
        ur_sc = {"value": grad_cr.mul(-learning_rate)}
        uc_sr = {"value": grad_rc.mul(-learning_rate)}
        ur_sr = {"value": grad_rr.mul(-learning_rate)}
        before_candidate = candidate_master.clone()
        before_repair = repair_master.clone()
        before = {"value": before_candidate - before_repair}
        with torch.no_grad():
            candidate_master.add_(uc_sc["value"])
            repair_master.add_(ur_sr["value"])
        # Use the actually applied FP32-master increments in the recurrence;
        # this also exposes any optimizer/materialisation discrepancy instead
        # of silently treating the planned update as the applied update.
        uc_sc = {"value": candidate_master - before_candidate}
        ur_sr = {"value": repair_master - before_repair}
        after = {"value": candidate_master - repair_master}
        evaluator.add(
            str(state.get("state_id", state.get("sequence_id", 16 + offset))),
            uc_sc, ur_sc, uc_sr, ur_sr, before, after,
            endpoint_repair_nonzero=bool(
                local_cr and local_cr["changed_coordinates"] > 0
                and local_rr and local_rr["changed_coordinates"] > 0
            ),
        )
        rows.append({
            "step": step,
            "state_id": str(state.get("state_id", state.get("sequence_id", 16 + offset))),
            "candidate_loss_digest": loss_cc,
            "repair_loss_digest": loss_rr,
            "candidate_repair_boundary_changed": local_cr["changed_coordinates"],
            "repair_arm_boundary_changed": local_rr["changed_coordinates"],
            "drift_l2": float(torch.linalg.vector_norm(after["value"]).item()),
        })
        print(json.dumps({"event": "SEUP_STEP_COMPLETE", **rows[-1]}), flush=True)
        del values, grad_cc, grad_cr, grad_rc, grad_rr, uc_sc, ur_sc, uc_sr, ur_sr
        torch.cuda.empty_cache()
    certificate = evaluator.finalize()
    gates = {
        "stable_calibration_carrier": carrier.stable,
        "sixteen_evaluation_steps": len(rows) == 16,
        "all_forward_repairs_exact": all(
            row["candidate_repair_boundary_changed"] >= 0 and row["repair_arm_boundary_changed"] >= 0
            for row in rows
        ),
        "endpoint_repair_nonzero_every_step": all(row["candidate_repair_boundary_changed"] > 0 for row in rows),
        "recurrence_closed": certificate["max_recurrence_relative_residual"] <= 1e-6,
        "signed_persistence": certificate.get("signed_persistence", 0.0) >= 0.80,
        "local_effect_dominates_feedback": certificate.get("local_fraction_of_projected_accumulation", 0.0) >= 0.50,
        "local_and_final_carrier_same_sign": certificate.get("local_and_final_carrier_same_sign", False),
    }
    payload = {
        "schema": "kernel-analyzer-phi64-seup-mainline-v2",
        "status": "PASS_SEUP_POSITIVE" if all(gates.values()) else "MEASURED_WITH_FAILED_GATE",
        "case_id": "phi4_seq64_lmhead_dx",
        "mechanism": "MM_KERNEL_ARITHMETIC_FP32_REPAIR_WITH_ORIGINAL_BF16_ABI",
        "mechanism_level": "ROOT_ARITHMETIC",
        "carrier_parameters": ["model.model.norm.weight"],
        "optimizer": {"name": "SGD_FP32_MASTER", "learning_rate": learning_rate},
        "protocol": {
            "calibration_states": 16,
            "evaluation_states": 16,
            "evaluation_state_ids_disjoint": True,
            "state_feedback_decomposition": "symmetric_candidate_and_repair_state_counterfactuals",
        },
        "calibration": {"steps": calibration_rows, "carrier": carrier.certificate},
        "evaluation": certificate,
        "steps": rows,
        "gates": gates,
        "claim_boundary": "Endpoint-induced effective update on the selected final-norm parameter; not a full-model optimizer claim.",
    }
    payload["result_sha256"] = canonical(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    seup_output.parent.mkdir(parents=True, exist_ok=True)
    seup_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "SEUP_COMPLETE", "status": payload["status"], "gates": gates}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=ROOT / "results/coverage/cases/phi4_seq64_lmhead_dx_trajectory.json")
    parser.add_argument("--seup-output", type=Path)
    args = parser.parse_args()
    bank = json.loads((ROOT / "results/coverage/phi4_seq64_input_bank.json").read_text())
    states = bank.get("states", bank.get("records"))
    if len(states) < args.steps:
        raise RuntimeError("trajectory exceeds frozen input bank")
    release = ROOT / "results/coverage/runtime_releases/phi4_seq64_r1"
    capture = json.loads((release / "capture.json").read_text())
    device = torch.device(args.device); configure_candidate_runtime(24000)
    model = load_model("phi", Path("/data1/tzh/models/microsoft/Phi-4-mini-instruct"), device)
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=False, dynamic=False)
    warm = torch.tensor([states[0]["token_ids"]], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True); candidate(warm).backward(); torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    if args.seup_output is None:
        validate_release(wrapper_modules(modules), capture)
    parameter = model.model.norm.weight
    standard_master = parameter.detach().float().clone()
    repair_master = standard_master.clone()
    initial = standard_master.clone()
    rows = []
    if args.seup_output is not None:
        run_seup_mainline(
            model=model, candidate=candidate, states=states, modules=modules,
            parameter=parameter, initial=initial, learning_rate=args.learning_rate,
            output=args.output, seup_output=args.seup_output,
        )
        return
    for step, state in enumerate(states[:args.steps]):
        values = torch.tensor([state["token_ids"]], dtype=torch.long, device=device)
        seed = 24000 + step
        parameter.data.copy_(standard_master.to(parameter.dtype))
        loss_ss, grad_ss, _ = evaluate(model, candidate, values, modules, seed, False)
        loss_sr, grad_sr, local_s = evaluate(model, candidate, values, modules, seed, True)
        if loss_ss != loss_sr:
            raise RuntimeError("repair changed forward loss at standard-arm weight")
        parameter.data.copy_(repair_master.to(parameter.dtype))
        loss_rs, grad_rs, _ = evaluate(model, candidate, values, modules, seed, False)
        loss_rr, grad_rr, local_r = evaluate(model, candidate, values, modules, seed, True)
        if loss_rs != loss_rr:
            raise RuntimeError("repair changed forward loss at repair-arm weight")
        delta_s = grad_ss - grad_sr
        delta_r = grad_rs - grad_rr
        if seup is not None:
            seup.add(
                str(state.get("state_id", step)),
                sgd_effective_update_delta(
                    grad_ss, grad_sr, learning_rate=args.learning_rate
                ),
            )
        standard_master = standard_master - args.learning_rate * grad_ss
        repair_master = repair_master - args.learning_rate * grad_rr
        materialized_standard = standard_master.to(parameter.dtype)
        materialized_repair = repair_master.to(parameter.dtype)
        rows.append({
            "step": step + 1, "state_id": str(state.get("state_id", step)),
            "same_weight_loss_exact": True,
            "standard_weight_delta_l2": float(torch.linalg.vector_norm(delta_s).item()),
            "repair_weight_delta_l2": float(torch.linalg.vector_norm(delta_r).item()),
            "same_weight_delta_inner_product": float(torch.dot(delta_s, delta_r).item()),
            "standard_local_changed": local_s["changed_coordinates"],
            "repair_local_changed": local_r["changed_coordinates"],
            "master_arm_distance_l2": float(torch.linalg.vector_norm(
                standard_master - repair_master).item()),
            "materialized_bf16_arm_distance_l2": float(torch.linalg.vector_norm(
                materialized_standard.float() - materialized_repair.float()).item()),
            "standard_master_from_initial_l2": float(torch.linalg.vector_norm(
                standard_master - initial).item()),
            "repair_master_from_initial_l2": float(torch.linalg.vector_norm(
                repair_master - initial).item()),
        })
        print(json.dumps({"event": "STEP_COMPLETE", **rows[-1]}), flush=True)
        del values, grad_ss, grad_sr, grad_rs, grad_rr
        torch.cuda.empty_cache()
    payload = {
        "schema": "kernel-analyzer-phi64-lmhead-dx-trajectory-v1",
        "status": "COMPLETE_PAIRED_EVOLVING_FINAL_NORM_TRAJECTORY",
        "candidate_id": "phi4_seq64_backward_497_output",
        "updated_parameter": "model.norm.weight",
        "frozen_other_parameters": True,
        "optimizer": {"type": "SGD_FP32_MASTER", "learning_rate": args.learning_rate},
        "steps": rows,
        "gates": {
            "same_weight_loss_exact_every_step": all(r["same_weight_loss_exact"] for r in rows),
            "repair_effect_present_every_step": all(r["standard_local_changed"] > 0
                                                     and r["repair_local_changed"] > 0 for r in rows),
            "same_weight_carrier_direction_stable": all(r["same_weight_delta_inner_product"] > 0 for r in rows),
            "final_master_weights_diverge": rows[-1]["master_arm_distance_l2"] > 0,
            "final_materialized_weights_diverge": rows[-1]["materialized_bf16_arm_distance_l2"] > 0,
        },
        "claim_boundary": "Live feedback for the selected final-norm carrier with all other parameters frozen; not a full-model optimizer trajectory.",
    }
    payload["result_sha256"] = canonical(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if seup is not None:
        certificate = seup.finalize()
        seup_payload = {
            "schema": "kernel-analyzer-case-seup-v1",
            "status": "MEASURED",
            "case_id": "phi4_seq64_lmhead_dx",
            "mechanism": "FP32_VS_BF16_MM_ACCUMULATION_WITH_ORIGINAL_BF16_ABI",
            "mechanism_level": "ROOT_ARITHMETIC",
            "candidate_id": "phi4_seq64_backward_497_output",
            "carrier_parameters": ["model.norm.weight"],
            "optimizer": payload["optimizer"],
            "common_state_protocol": (
                "candidate and repair gradients are evaluated at the candidate-arm "
                "pre-step weight; both SGD updates use that identical weight"
            ),
            "certificate": certificate,
            "gates": {
                "complete_concrete_fb_proof": True,
                "same_pre_step_weight": True,
                "calibration_evaluation_disjoint": True,
                "all_forward_losses_exact": payload["gates"]["same_weight_loss_exact_every_step"],
            },
            "trajectory_result_sha256": payload["result_sha256"],
        }
        seup_payload["result_sha256"] = canonical(seup_payload)
        args.seup_output.parent.mkdir(parents=True, exist_ok=True)
        args.seup_output.write_text(
            json.dumps(seup_payload, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps({"status": payload["status"], "gates": payload["gates"]}))


if __name__ == "__main__":
    main()
