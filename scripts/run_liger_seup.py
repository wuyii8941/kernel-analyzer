#!/usr/bin/env python3
"""Minimal mainline SEUP campaign for the causally closed Liger anchor.

The carrier is the tied embedding parameter, the only parameter whose
same-weight gradient changes under the established Liger accumulator repair.
Calibration and evaluation use disjoint states.  The natural evaluation uses
the symmetric four-counterfactual decomposition; the intervention replays a
third live arm with an alternating carrier sign while preserving the
endpoint-induced update-error norm exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data1/tzh/cache/huggingface/transformers")
os.environ.setdefault("XDG_CACHE_HOME", "/data1/tzh/cache/xdg")

import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/data1/tzh").resolve()
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from kernel_analyzer.seup import (  # noqa: E402
    SEUPCalibrator,
    SymmetricSEUPEvaluator,
    alternating_sign_schedule,
    force_carrier_sign,
)
from scripts.liger_trajectory import full_step  # noqa: E402


TIED = "model.embed_tokens.weight"
NORM_MATCH_TOLERANCE = 1e-3  # FP32 carrier arithmetic; raw relative errors remain recorded.


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def checked(path: Path) -> Path:
    result = path.resolve()
    if DATA_ROOT not in (result, *result.parents):
        raise RuntimeError(f"path must stay under /data1/tzh: {result}")
    return result


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def set_masters(model: torch.nn.Module, masters: dict[str, torch.Tensor], device: torch.device) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(masters[name].to(device=device, dtype=parameter.dtype))


def cpu_gradients(value: dict[str, Any]) -> dict[str, torch.Tensor | None]:
    return {
        name: None if gradient is None else gradient.detach().float().cpu()
        for name, gradient in value["gradients"].items()
    }


def update_masters(masters: dict[str, torch.Tensor], gradients: dict[str, torch.Tensor | None], learning_rate: float) -> None:
    with torch.no_grad():
        for name, gradient in gradients.items():
            if gradient is not None:
                masters[name].add_(gradient, alpha=-learning_rate)


def local_tied_delta(default: dict[str, torch.Tensor | None], repair: dict[str, torch.Tensor | None], learning_rate: float) -> dict[str, torch.Tensor]:
    if default[TIED] is None or repair[TIED] is None:
        raise RuntimeError("Liger tied gradient is absent")
    # The tied embedding has 311M entries.  Avoid repeating a full finite-value
    # census in the generic tree helper; autograd gradients were already
    # materialised and the carrier gate performs the finite check once per
    # calibration update.
    return {TIED: (default[TIED].detach().float() - repair[TIED].detach().float()).mul(-learning_rate)}


def single_tied_update(gradients: dict[str, torch.Tensor | None], learning_rate: float) -> dict[str, torch.Tensor]:
    if gradients[TIED] is None:
        raise RuntimeError("Liger tied gradient is absent")
    return {TIED: gradients[TIED].detach().float().mul(-learning_rate)}


def run(args: argparse.Namespace) -> None:
    if args.steps != 32:
        raise ValueError("SEUP mainline requires exactly 32 states")
    design = load_json(checked(args.design))
    records = design["records"]
    if len(records) < 32:
        raise RuntimeError("Liger state design has fewer than 32 states")
    states = records[:32]
    state_ids = [str(row["sequence_id"]) for row in states]
    if len(set(state_ids)) != 32:
        raise RuntimeError("Liger state IDs are not unique")

    import transformers
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    from transformers import AutoModelForCausalLM

    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        checked(args.model), dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True
    ).to(device)
    model.config.use_cache = False
    model.eval()
    named = dict(model.named_parameters())
    if TIED not in named:
        raise RuntimeError("tied embedding parameter is absent")
    if model.lm_head.weight.untyped_storage().data_ptr() != model.model.embed_tokens.weight.untyped_storage().data_ptr():
        raise RuntimeError("Liger model weights are not tied")
    initial = {name: parameter.detach().cpu().float().clone() for name, parameter in named.items()}
    default_module = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction="mean", accum_dtype=None).to(device)
    repair_module = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction="mean", accum_dtype=torch.float32).to(device)

    def pair(masters: dict[str, torch.Tensor], state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, torch.Tensor | None], dict[str, torch.Tensor | None]]:
        set_masters(model, masters, device)
        input_ids = torch.tensor([state["input_ids"]], device=device, dtype=torch.long)
        default = full_step(model, default_module, input_ids)
        repair = full_step(model, repair_module, input_ids)
        common = {key: default[key] == repair[key] for key in ("hidden_digest", "labels_digest", "dH_digest")}
        common["loss"] = bool(torch.equal(default["loss"], repair["loss"]))
        if not all(common.values()):
            raise RuntimeError(f"Liger same-weight controls failed at {state['sequence_id']}: {common}")
        gradients_default = cpu_gradients(default)
        gradients_repair = cpu_gradients(repair)
        del default["gradients"], repair["gradients"], default, repair, input_ids
        torch.cuda.empty_cache()
        return {"controls": common}, {"controls": common}, gradients_default, gradients_repair

    calibrator = SEUPCalibrator(16, 0.5)
    calibration_masters = {name: value.clone() for name, value in initial.items()}
    calibration_rows = []
    for index, state in enumerate(states[:16]):
        _, _, gradients_default, gradients_repair = pair(calibration_masters, state)
        delta = local_tied_delta(gradients_default, gradients_repair, args.learning_rate)
        calibrator.add(state_ids[index], delta)
        update_masters(calibration_masters, gradients_default, args.learning_rate)
        calibration_rows.append({"step": index + 1, "state_id": state_ids[index], "delta_l2": float(torch.linalg.vector_norm(delta[TIED]).item())})
        del gradients_default, gradients_repair, delta
    carrier = calibrator.freeze()
    if carrier.basis is None:
        raise RuntimeError("Liger anchor failed calibration carrier stability gate")

    evaluator = SymmetricSEUPEvaluator(carrier, 16)
    candidate_masters = {name: value.clone() for name, value in initial.items()}
    repair_masters = {name: value.clone() for name, value in initial.items()}
    natural_repair_history: list[torch.Tensor] = []
    evaluation_rows = []
    for offset, state in enumerate(states[16:32]):
        _, _, g_cc, g_cr = pair(candidate_masters, state)
        _, _, g_rc, g_rr = pair(repair_masters, state)
        uc_sc_planned = single_tied_update(g_cc, args.learning_rate)
        ur_sc = single_tied_update(g_cr, args.learning_rate)
        uc_sr = single_tied_update(g_rc, args.learning_rate)
        ur_sr_planned = single_tied_update(g_rr, args.learning_rate)
        candidate_before = candidate_masters[TIED].clone()
        repair_before = repair_masters[TIED].clone()
        update_masters(candidate_masters, g_cc, args.learning_rate)
        update_masters(repair_masters, g_rr, args.learning_rate)
        uc_sc = {TIED: candidate_masters[TIED] - candidate_before}
        ur_sr = {TIED: repair_masters[TIED] - repair_before}
        before = {TIED: candidate_before - repair_before}
        after = {TIED: candidate_masters[TIED] - repair_masters[TIED]}
        evaluator.add(state_ids[16 + offset], uc_sc, ur_sc, uc_sr, ur_sr, before, after,
                       endpoint_repair_nonzero=bool(torch.count_nonzero(g_cc[TIED] - g_cr[TIED]).item() > 0
                                                    and torch.count_nonzero(g_rc[TIED] - g_rr[TIED]).item() > 0))
        natural_repair_history.append(repair_masters[TIED].clone())
        evaluation_rows.append({"step": offset + 1, "state_id": state_ids[16 + offset], "drift_l2": float(torch.linalg.vector_norm(after[TIED]).item())})
        print(json.dumps({"event": "SEUP_STEP_COMPLETE", **evaluation_rows[-1]}), flush=True)
        del g_cc, g_cr, g_rc, g_rr, uc_sc_planned, ur_sc, uc_sr, ur_sr_planned, uc_sc, ur_sr, before, after

    natural = evaluator.finalize()
    intervention_masters = {name: value.clone() for name, value in initial.items()}
    intervention_rows = []
    signs = alternating_sign_schedule(16, first=1)
    for offset, state in enumerate(states[16:32]):
        _, _, g_i, g_ir = pair(intervention_masters, state)
        natural_delta = local_tied_delta(g_i, g_ir, args.learning_rate)
        forced = force_carrier_sign(natural_delta, carrier.basis, signs[offset])
        norm_error = abs(float(torch.linalg.vector_norm(forced[TIED]).item()) - float(torch.linalg.vector_norm(natural_delta[TIED]).item())) / max(float(torch.linalg.vector_norm(natural_delta[TIED]).item()), 1e-30)
        update_masters(intervention_masters, g_ir, args.learning_rate)
        intervention_masters[TIED].add_(forced[TIED])
        drift = {TIED: intervention_masters[TIED] - natural_repair_history[offset]}
        projection = float(torch.sum(drift[TIED] * carrier.basis[TIED]).item())
        intervention_rows.append({"step": offset + 1, "state_id": state_ids[16 + offset], "forced_sign": signs[offset],
                                  "natural_delta_l2": float(torch.linalg.vector_norm(natural_delta[TIED]).item()),
                                  "norm_relative_error": norm_error, "repair_relative_drift_projection": projection})
        print(json.dumps({"event": "INTERVENTION_STEP_COMPLETE", **intervention_rows[-1]}), flush=True)
        del g_i, g_ir, natural_delta, forced, drift

    natural_projection = abs(float(natural["final_carrier_projection"]))
    intervention_projection = abs(float(intervention_rows[-1]["repair_relative_drift_projection"]))
    reduction = 1.0 - intervention_projection / max(natural_projection, 1e-30)
    gates = {
        "stable_calibration_carrier": carrier.stable,
        "sixteen_evaluation_steps": len(evaluation_rows) == 16,
        "recurrence_closed": natural["max_recurrence_relative_residual"] <= 1e-6,
        "natural_signed_persistence_ge_0_80": natural.get("signed_persistence", 0.0) >= 0.80,
        "natural_local_fraction_ge_0_50": natural.get("local_fraction_of_projected_accumulation", 0.0) >= 0.50,
        "norm_matched_intervention_every_step": all(row["norm_relative_error"] <= NORM_MATCH_TOLERANCE for row in intervention_rows),
        "intervention_projection_reduction_ge_0_70": reduction >= 0.70,
        "intervention_has_alternating_signs": all(row["forced_sign"] == signs[row["step"] - 1] for row in intervention_rows),
    }
    payload = {
        "schema": "kernel-analyzer-liger-seup-mainline-v2",
        "status": "PASS_SEUP_ANCHOR" if all(gates.values()) else "MEASURED_WITH_FAILED_GATE",
        "case_id": "qwen3_liger_fused_linear_ce",
        "mechanism": "CHUNK_GEOMETRY_BF16_ACCUMULATION",
        "mechanism_level": "ROOT_ARITHMETIC",
        "carrier_parameters": [TIED],
        "optimizer": {"name": "STATELESS_SGD_FP32_MASTER", "learning_rate": args.learning_rate},
        "protocol": {"calibration_states": 16, "evaluation_states": 16, "evaluation_state_ids_disjoint": True,
                     "state_feedback_decomposition": "symmetric_candidate_and_repair_state_counterfactuals",
                     "intervention": "alternating_force_carrier_sign_preserving_endpoint_delta_norm",
                     "norm_match_relative_tolerance": NORM_MATCH_TOLERANCE},
        "calibration": {"steps": calibration_rows, "carrier": carrier.certificate},
        "evaluation": natural,
        "natural_steps": evaluation_rows,
        "intervention": {"steps": intervention_rows, "final_projection_reduction": reduction},
        "gates": gates,
        "environment": {"torch": torch.__version__, "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(device), "dtype": "torch.bfloat16"},
        "claim_boundary": "Full Liger training-step endpoint is closed; SEUP and intervention are measured on the tied embedding parameter, the only same-weight local-gradient carrier. No AdamW or cross-model claim.",
    }
    payload["result_sha256"] = canonical(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.seup_output.parent.mkdir(parents=True, exist_ok=True)
    args.seup_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "SEUP_COMPLETE", "status": payload["status"], "gates": gates}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DATA_ROOT / "models/Qwen/Qwen3-1.7B")
    parser.add_argument("--design", type=Path, default=ROOT / "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/supplementary_state_design_v1.json")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=ROOT / "results/property/seup_mainline/liger.json")
    parser.add_argument("--seup-output", type=Path, default=ROOT / "results/property/seup_mainline/liger_seup.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
