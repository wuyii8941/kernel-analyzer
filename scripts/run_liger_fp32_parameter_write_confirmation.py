#!/usr/bin/env python3
"""Measure actual FP32 parameter writes for the Liger reduction-order case.

This is deliberately a one-step, state-bank confirmation.  The hidden states
are held fixed, the fused loss is evaluated with identical FP32 operands, and
only the dW chunk-addition order is changed.  Unlike the historical Liger
confirmation, this script computes parameter_after - parameter_before with a
real torch AdamW step in original coordinates; sketches are retained only for
the held-out direction diagnostic.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kernel_analyzer.training_bias_profile import matched_training_bias_profile  # noqa: E402
from scripts.run_liger_fp32_chunk_order_length import (  # noqa: E402
    DESIGN,
    MODEL,
    SKETCH_SIZE,
    _install_order,
    _sketch,
)


LEARNING_RATE = 1e-4
BETAS = (0.9, 0.95)
EPSILON = 1e-8
SEED = 20260917


def _one_step_write(
    module: Any,
    hidden: torch.Tensor,
    weight0: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return loss, hidden gradient, weight gradient, and actual AdamW write."""
    weight = torch.nn.Parameter(weight0.detach().clone())
    h = hidden.detach().clone().reshape(-1, hidden.shape[-1]).requires_grad_(True)
    loss = module(weight, h, labels)
    grad_hidden, grad_weight = torch.autograd.grad(loss, (h, weight), retain_graph=False)
    before = weight.detach().clone()
    optimizer = torch.optim.AdamW(
        [weight], lr=LEARNING_RATE, betas=BETAS, eps=EPSILON,
        weight_decay=0.0, foreach=False, fused=False,
    )
    weight.grad = grad_weight.detach()
    optimizer.step()
    write = weight.detach() - before
    result = (loss.detach(), grad_hidden.detach(), grad_weight.detach(), write.detach())
    del optimizer, weight, h, grad_weight
    return result


def _exact_stats(effect: torch.Tensor, repair: torch.Tensor) -> dict[str, Any]:
    if effect.shape != repair.shape:
        raise ValueError("effect and reference must refer to the same parameter coordinates")
    effect64 = effect.detach().double()
    repair64 = repair.detach().double()
    return {
        "accumulation_dtype": "float64",
        "effect_energy": float(torch.sum(effect64 * effect64).item()),
        "repair_energy": float(torch.sum(repair64 * repair64).item()),
        "effect_repair_inner_product": float(torch.sum(effect64 * repair64).item()),
        "nonzero_effect_coordinates": int(torch.count_nonzero(effect).item()),
        "effect_l2": float(torch.linalg.vector_norm(effect64).item()),
        "repair_l2": float(torch.linalg.vector_norm(repair64).item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, choices=(64, 256), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"preserve the existing measurement: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("host GPU required")

    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    import liger_kernel.ops.fused_linear_cross_entropy as fused
    from transformers import AutoModelForCausalLM

    design = json.loads(DESIGN.read_text())
    records = [row for row in design["records"] if int(row["length"]) == args.length]
    if len(records) != 32:
        raise RuntimeError(f"expected 32 length-{args.length} records, got {len(records)}")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager", local_files_only=True,
    ).to(device).eval()
    model.config.use_cache = False
    weight0 = model.lm_head.weight.detach().clone()
    original_source = __import__("inspect").getsource(fused.fused_linear_cross_entropy_forward)
    kinds = ("original", "reverse", "even_then_odd", "frozen_permutation")

    exact_rows: list[dict[str, Any]] = []
    write_effects: list[np.ndarray] = []
    write_repairs: list[np.ndarray] = []
    gradient_effects: list[np.ndarray] = []
    gradient_repairs: list[np.ndarray] = []
    for index, row in enumerate(records):
        ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
        with torch.no_grad():
            hidden = model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state.detach()
        labels = torch.nn.functional.pad(ids, (0, 1), value=-100)[..., 1:].contiguous().reshape(-1)
        outputs: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for kind in kinds:
            _install_order(kind, original_source, fused)
            module = LigerFusedLinearCrossEntropyLoss(
                ignore_index=-100, reduction="mean", accum_dtype=torch.float32,
            ).to(device)
            outputs[kind] = _one_step_write(module, hidden, weight0, labels, device)
            del module

        candidate_loss, candidate_hidden, candidate_gradient, candidate_write = outputs["original"]
        reference_loss, reference_hidden, reference_gradient, reference_write = outputs["reverse"]
        if not torch.equal(candidate_loss, reference_loss):
            raise RuntimeError("chunk order changed forward loss")
        if not torch.equal(candidate_hidden, reference_hidden):
            raise RuntimeError("chunk order changed hidden-state gradient")

        gradient_delta = candidate_gradient - reference_gradient
        write_delta = candidate_write - reference_write
        row_stats = _exact_stats(write_delta, reference_write)
        row_stats.update({
            "state_id": str(row["sequence_id"]),
            "forward_and_hidden_gradient_bitwise_equal": True,
            "actual_optimizer": "torch.optim.AdamW",
            "write_measurement": "parameter_after_step_minus_parameter_before_step",
            "gradient_effect_l2": float(torch.linalg.vector_norm(gradient_delta.double()).item()),
        })
        exact_rows.append(row_stats)
        write_effects.append(_sketch(write_delta))
        write_repairs.append(_sketch(reference_write))
        gradient_effects.append(_sketch(gradient_delta))
        gradient_repairs.append(_sketch(reference_gradient))
        print(json.dumps({"event": "LIGER_FP32_PARAMETER_WRITE_STATE", "index": index,
                          "state_id": row_stats["state_id"],
                          "write_effect_l2": row_stats["effect_l2"]}), flush=True)
        del ids, hidden, labels, outputs, gradient_delta, write_delta
        gc.collect()
        torch.cuda.empty_cache()

    calibration = list(range(16))
    confirmation = list(range(16, 32))
    write_profile = matched_training_bias_profile(
        np.stack(write_effects), np.stack(write_repairs),
        calibration_indices=calibration, confirmation_indices=confirmation,
        inference_unit_ids=None, include_joint_gram=True, seed=SEED,
    )
    gradient_profile = matched_training_bias_profile(
        np.stack(gradient_effects), np.stack(gradient_repairs),
        calibration_indices=calibration, confirmation_indices=confirmation,
        inference_unit_ids=None, include_joint_gram=True, seed=SEED + 1,
    )
    write_mean = np.mean(np.stack(write_effects)[calibration], axis=0)
    direction_norm = float(np.linalg.norm(write_mean))
    if direction_norm > 0:
        write_mean = write_mean / direction_norm
        confirmation_projection = np.stack(write_effects)[confirmation] @ write_mean
    else:
        confirmation_projection = np.zeros(len(confirmation), dtype=np.float64)

    effect_energy = sum(row["effect_energy"] for row in exact_rows)
    repair_energy = sum(row["repair_energy"] for row in exact_rows)
    payload = {
        "schema": "kernel-analyzer-liger-fp32-parameter-write-confirmation-v1",
        "status": "COMPLETE",
        "case_id": "liger_fused_ce_fp32_chunk_order",
        "sequence_length": args.length,
        "state_ids": [str(row["sequence_id"]) for row in records],
        "dtype": "FP32_FOR_BOTH_PRIMARY_IMPLEMENTATIONS",
        "measurement_geometry": "ORIGINAL_COORDINATE_PARAMETER_WRITE_PLUS_SKETCH_DIRECTION",
        "comparison": {
            "candidate": "original sequential FP32 chunk order",
            "reference": "reverse FP32 chunk order",
            "only_declared_difference": "dW chunk-addition order",
            "optimizer": "torch.optim.AdamW, zero initial moments, weight_decay=0",
        },
        "profiles": {"PARAMETER_WRITE": write_profile, "PARAMETER_GRADIENT": gradient_profile},
        "fixed_suite_total_write_rms": float(np.sqrt(effect_energy / repair_energy)) if repair_energy else 0.0,
        "fixed_suite_total_write_effect_energy": effect_energy,
        "fixed_suite_total_reference_write_energy": repair_energy,
        "direction_diagnostic": {
            "geometry": "COUNT_SKETCH_8192",
            "calibration_count": 16,
            "confirmation_count": 16,
            "confirmation_projection_mean": float(confirmation_projection.mean()),
            "confirmation_positive_count": int(np.count_nonzero(confirmation_projection > 0)),
            "confirmation_mean_projection_positive": bool(direction_norm > 0 and confirmation_projection.mean() > 0),
            "population_mean_bias_decision": "NOT_ASSESSED_FIXED_SUITE",
            "role": "DESCRIPTIVE_ONLY_NOT_USED_FOR_FIXED_SUITE_TOTAL_WRITE_DECISION",
        },
        "rows": exact_rows,
        "claim_boundary": (
            "This is a 32-state one-step confirmation on a fixed hidden-state bank. "
            "The reported write effect is measured in original parameter coordinates after "
            "an actual torch AdamW step; it is not a long-run quality result or a population guarantee."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "output": str(args.output),
                      "fixed_suite_total_write_rms": payload["fixed_suite_total_write_rms"],
                      "confirmation_positive_count": payload["direction_diagnostic"]["confirmation_positive_count"]}))


if __name__ == "__main__":
    main()
