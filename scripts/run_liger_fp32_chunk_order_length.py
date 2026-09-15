#!/usr/bin/env python3
"""Confirm the FP32 chunk-order effect on a disjoint sequence-length bank.

This is a continuation of the frozen same-dtype Liger reduction experiment.
It deliberately uses a sequence length not used by the original length-128
run, while keeping the candidate/reference distinction and source prediction
unchanged.  The output contains no source identity metadata; the protocol and
input bank are the declared provenance for this measurement.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kernel_analyzer.training_bias_profile import matched_training_bias_profile  # noqa: E402


MODEL = Path("/data1/tzh/models/Qwen/Qwen3-1.7B")
DESIGN = ROOT / "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/supplementary_state_design_v1.json"
SKETCH_SIZE = 8192
ORDER_SEED = 20260903


def _order_values(kind: str, count: int) -> list[int]:
    if kind == "original":
        return list(range(count))
    if kind == "reverse":
        return list(reversed(range(count)))
    if kind == "even_then_odd":
        return list(range(0, count, 2)) + list(range(1, count, 2))
    if kind == "frozen_permutation":
        return torch.randperm(count, generator=torch.Generator().manual_seed(ORDER_SEED)).tolist()
    raise ValueError(kind)


def _install_order(kind: str, original_source: str, fused: Any) -> None:
    marker = "    for chunk_id in range(num_chunks):"
    replacement = "    for chunk_id in _declared_chunk_order(num_chunks):"
    if original_source.count(marker) != 1:
        raise RuntimeError("Liger chunk loop was not uniquely identified")
    namespace = dict(fused.__dict__)
    namespace["_declared_chunk_order"] = lambda count: _order_values(kind, count)
    transformed = original_source.replace(marker, replacement, 1)
    exec(compile(transformed, f"<liger-fp32-order-{kind}>", "exec"), namespace)
    fused.fused_linear_cross_entropy_forward = namespace["fused_linear_cross_entropy_forward"]


def _run_region(module: Any, hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor):
    h = hidden.detach().clone().reshape(-1, hidden.shape[-1]).requires_grad_(True)
    loss = module(weight, h, labels)
    grad_hidden, grad_weight = torch.autograd.grad(loss, (h, weight), retain_graph=False)
    return loss.detach(), grad_hidden.detach(), grad_weight.detach()


def _adam_first_step(gradient: torch.Tensor, learning_rate: float = 1e-4) -> torch.Tensor:
    # This is the same zero-moment, no-decay AdamW proposed update used by the
    # original source-order experiment; it is only a common response probe.
    return -learning_rate * gradient / (gradient.abs() + 1e-8)


def _sketch(value: torch.Tensor) -> np.ndarray:
    flat = value.detach().reshape(-1)
    positions = torch.arange(SKETCH_SIZE, device=flat.device, dtype=torch.int64)
    signs = torch.remainder(positions * 1_103_515_245 + ORDER_SEED, 2).float().mul_(2).sub_(1)
    buckets = torch.remainder(positions * 2_654_435_761 + ORDER_SEED, SKETCH_SIZE)
    columns = torch.zeros(SKETCH_SIZE, device=flat.device, dtype=torch.float64)
    blocks = math.ceil(flat.numel() / SKETCH_SIZE)
    for block_start in range(0, blocks, 512):
        start = block_start * SKETCH_SIZE
        stop = min(flat.numel(), (block_start + 512) * SKETCH_SIZE)
        chunk = flat[start:stop].float()
        if chunk.numel() % SKETCH_SIZE:
            chunk = torch.nn.functional.pad(chunk, (0, SKETCH_SIZE - chunk.numel() % SKETCH_SIZE))
        columns += (chunk.reshape(-1, SKETCH_SIZE) * signs).sum(dim=0, dtype=torch.float64)
    output = torch.zeros(SKETCH_SIZE, device=flat.device, dtype=torch.float64)
    output[buckets] += columns
    return output.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, choices=(64, 256), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("host GPU required")

    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    import liger_kernel.ops.fused_linear_cross_entropy as fused
    from transformers import AutoModelForCausalLM

    design = json.loads(DESIGN.read_text())
    records = [row for row in design["records"] if int(row["length"]) == args.length]
    if len(records) != 32:
        raise RuntimeError(f"expected 32 length-{args.length} records, got {len(records)}")
    state_ids = [str(row["sequence_id"]) for row in records]

    device = torch.device(args.device)
    torch.manual_seed(20260903)
    torch.cuda.manual_seed_all(20260903)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager", local_files_only=True,
    ).to(device).eval()
    model.config.use_cache = False
    weight = model.lm_head.weight
    original_source = inspect.getsource(fused.fused_linear_cross_entropy_forward)
    kinds = ("original", "reverse", "even_then_odd", "frozen_permutation")

    gradient_effects: list[np.ndarray] = []
    gradient_repairs: list[np.ndarray] = []
    update_effects: list[np.ndarray] = []
    update_repairs: list[np.ndarray] = []
    orbit_effects: list[np.ndarray] = []
    orbit_repair_effects: list[np.ndarray] = []
    orbit_repair_references: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(records):
        ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
        with torch.no_grad():
            hidden = model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state.detach()
        labels = torch.nn.functional.pad(ids, (0, 1), value=-100)[..., 1:].contiguous().reshape(-1)
        outputs: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for kind in kinds:
            _install_order(kind, original_source, fused)
            module = LigerFusedLinearCrossEntropyLoss(
                ignore_index=-100, reduction="mean", accum_dtype=torch.float32,
            ).to(device)
            outputs[kind] = _run_region(module, hidden, weight, labels)
            del module

        original_loss, original_hidden_grad, original_grad = outputs["original"]
        reverse_loss, reverse_hidden_grad, reverse_grad = outputs["reverse"]
        if not torch.equal(original_loss, reverse_loss):
            raise RuntimeError("chunk order changed forward loss")
        if not torch.equal(original_hidden_grad, reverse_hidden_grad):
            raise RuntimeError("chunk order changed hidden-state gradient")

        _install_order("original", original_source, fused)
        sham_module = LigerFusedLinearCrossEntropyLoss(
            ignore_index=-100, reduction="mean", accum_dtype=torch.float32,
        ).to(device)
        sham_loss, sham_hidden_grad, sham_grad = _run_region(sham_module, hidden, weight, labels)
        if not (torch.equal(sham_loss, original_loss)
                and torch.equal(sham_hidden_grad, original_hidden_grad)
                and torch.equal(sham_grad, original_grad)):
            raise RuntimeError("matched sham did not reproduce candidate")

        gradient_delta = original_grad - reverse_grad
        update_delta = _adam_first_step(original_grad) - _adam_first_step(reverse_grad)
        schedule_gradients = torch.stack([outputs[kind][2] for kind in kinds])
        orbit_delta = schedule_gradients.double().mean(dim=0).float() - reverse_grad
        orbit_repair_delta = original_grad - schedule_gradients.double().mean(dim=0).float()
        orbit_repair_update_delta = _adam_first_step(original_grad) - _adam_first_step(
            schedule_gradients.double().mean(dim=0).float()
        )
        gradient_effects.append(_sketch(gradient_delta))
        gradient_repairs.append(_sketch(reverse_grad))
        update_effects.append(_sketch(update_delta))
        update_repairs.append(_sketch(_adam_first_step(reverse_grad)))
        orbit_effects.append(_sketch(orbit_delta))
        orbit_repair_effects.append(_sketch(orbit_repair_update_delta))
        orbit_repair_references.append(_sketch(_adam_first_step(
            schedule_gradients.double().mean(dim=0).float()
        )))
        rows.append({
            "state_id": state_ids[index],
            "forward_and_hidden_gradient_bitwise_equal": True,
            "matched_sham_exact": True,
            "candidate_minus_repair_gradient_l2": float(torch.linalg.vector_norm(gradient_delta.double()).item()),
            "candidate_minus_repair_update_l2": float(torch.linalg.vector_norm(update_delta.double()).item()),
            "candidate_minus_schedule_mean_update_l2": float(
                torch.linalg.vector_norm(orbit_repair_update_delta.double()).item()
            ),
        })
        print(json.dumps({"event": "LIGER_FP32_ORDER_HELDOUT_STATE", "index": index,
                          "state_id": state_ids[index]}), flush=True)
        del ids, hidden, labels, outputs, original_grad, reverse_grad, sham_grad
        del gradient_delta, update_delta, schedule_gradients, orbit_delta
        del orbit_repair_delta, orbit_repair_update_delta, sham_module
        gc.collect()
        torch.cuda.empty_cache()

    calibration = list(range(16))
    confirmation = list(range(16, 32))
    unit_ids = [f"length-{args.length}-state-{i:02d}" for i in range(32)]
    gradient_profile = matched_training_bias_profile(
        np.stack(gradient_effects), np.stack(gradient_repairs),
        calibration_indices=calibration, confirmation_indices=confirmation,
        inference_unit_ids=unit_ids, include_joint_gram=True, seed=20260903,
    )
    update_profile = matched_training_bias_profile(
        np.stack(update_effects), np.stack(update_repairs),
        calibration_indices=calibration, confirmation_indices=confirmation,
        inference_unit_ids=unit_ids, include_joint_gram=True, seed=20260904,
    )
    orbit_repair_profile = matched_training_bias_profile(
        np.stack(orbit_repair_effects), np.stack(orbit_repair_references),
        calibration_indices=calibration, confirmation_indices=confirmation,
        inference_unit_ids=unit_ids, include_joint_gram=True, seed=20260905,
    )
    orbit = np.stack(orbit_effects)
    predictor = orbit[calibration].mean(axis=0)
    predictor /= max(float(np.linalg.norm(predictor)), 1e-30)
    confirmation_projection = np.stack(gradient_effects)[confirmation] @ predictor
    payload = {
        "schema": "kernel-analyzer-liger-fp32-chunk-order-length-confirmation-v1",
        "status": "COMPLETE",
        "case_id": "liger_fused_ce_fp32_chunk_order",
        "sequence_length": args.length,
        "dtype": "FP32_FOR_BOTH_PRIMARY_IMPLEMENTATIONS",
        "state_ids": state_ids,
        "measurement_geometry": "COUNT_SKETCH_8192",
        "profiles": {
            "PARAMETER_GRADIENT": gradient_profile,
            "ADAMW_UPDATE": update_profile,
            "ADAMW_UPDATE_ORBIT_AVERAGED_REPAIR": orbit_repair_profile,
        },
        "source_prediction": {
            "confirmation_projection_mean": float(confirmation_projection.mean()),
            "confirmation_positive_count": int(np.count_nonzero(confirmation_projection > 0)),
            "confirmation_count": 16,
            "direction_repeated": bool(float(confirmation_projection.mean()) > 0),
        },
        "rows": rows,
        "source_intervention": {
            "repair": "mean of original, reverse, even-then-odd, and frozen-permutation FP32 dW schedules",
            "prediction": "averaging equivalent schedules reduces the order-dependent update effect relative to original-versus-reverse",
            "comparison": "candidate original order versus schedule-averaged gradient and zero-moment AdamW response",
        },
        "claim_boundary": (
            "Both implementations use FP32 and differ only in the declared Liger dW chunk-addition order. "
            "This is a disjoint sequence-length confirmation of the same source prediction; it is not a long training run."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
