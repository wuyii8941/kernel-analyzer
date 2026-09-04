#!/usr/bin/env python3
"""Test a Liger dW reduction-order change with FP32 on both sides."""

from __future__ import annotations

import argparse
import gc
import hashlib
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
from scripts.run_heldout_lmhead_consequence import adam_delta  # noqa: E402


MODEL = Path("/data1/tzh/models/Qwen/Qwen3-1.7B")
DESIGN = ROOT / "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/supplementary_state_design_v1.json"
TRAJECTORY = ROOT / "results/trajectory/liger_protocol.json"
SKETCH_SIZE = 8192
SKETCH_SEED = 20260903


def order_values(kind: str, count: int) -> list[int]:
    if kind == "original":
        return list(range(count))
    if kind == "reverse":
        return list(reversed(range(count)))
    if kind == "even_then_odd":
        return list(range(0, count, 2)) + list(range(1, count, 2))
    if kind == "frozen_permutation":
        return torch.randperm(count, generator=torch.Generator().manual_seed(20260903)).tolist()
    raise ValueError(kind)


def install_order(kind: str, original_source: str, fused: Any) -> dict[str, str]:
    old = "    for chunk_id in range(num_chunks):"
    new = "    for chunk_id in _declared_chunk_order(num_chunks):"
    if original_source.count(old) != 1:
        raise RuntimeError("Liger chunk loop was not uniquely identified")
    namespace = dict(fused.__dict__)
    namespace["_declared_chunk_order"] = lambda count: order_values(kind, count)
    transformed = original_source.replace(old, new, 1)
    exec(compile(transformed, f"<liger-fp32-order-{kind}>", "exec"), namespace)
    fused.fused_linear_cross_entropy_forward = namespace["fused_linear_cross_entropy_forward"]
    return {
        "kind": kind,
        "original_source_sha256": hashlib.sha256(original_source.encode()).hexdigest(),
        "transformed_source_sha256": hashlib.sha256(transformed.encode()).hexdigest(),
    }


def run_region(module: Any, hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h = hidden.detach().clone().reshape(-1, hidden.shape[-1]).requires_grad_(True)
    loss = module(weight, h, labels)
    grad_hidden, grad_weight = torch.autograd.grad(loss, (h, weight), retain_graph=False)
    return loss.detach(), grad_hidden.detach(), grad_weight.detach()


def sketch(value: torch.Tensor) -> np.ndarray:
    flat = value.detach().reshape(-1)
    position = torch.arange(SKETCH_SIZE, device=flat.device, dtype=torch.int64)
    signs = torch.remainder(position * 1_103_515_245 + SKETCH_SEED, 2).float().mul_(2).sub_(1)
    buckets = torch.remainder(position * 2_654_435_761 + SKETCH_SEED, SKETCH_SIZE)
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--states", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.states != 32:
        raise ValueError("the frozen experiment requires 32 states")
    if not torch.cuda.is_available():
        raise RuntimeError("host GPU required")

    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    import liger_kernel.ops.fused_linear_cross_entropy as fused
    from transformers import AutoModelForCausalLM

    design = json.loads(DESIGN.read_text())
    trajectory = json.loads(TRAJECTORY.read_text())
    records = {str(row["sequence_id"]): row for row in design["records"]}
    state_ids = list(trajectory["trajectory"]["state_order"])
    if len(state_ids) != 32:
        raise RuntimeError("the frozen state order changed")

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
    provenance = {kind: install_order(kind, original_source, fused) for kind in kinds}

    gradient_effects: list[np.ndarray] = []
    gradient_repairs: list[np.ndarray] = []
    update_effects: list[np.ndarray] = []
    update_repairs: list[np.ndarray] = []
    orbit_means: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for index, state_id in enumerate(state_ids):
        ids = torch.tensor([records[state_id]["input_ids"]], dtype=torch.long, device=device)
        with torch.no_grad():
            hidden = model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state.detach()
        labels = torch.nn.functional.pad(ids, (0, 1), value=-100)[..., 1:].contiguous().reshape(-1)
        outputs: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for kind in kinds:
            install_order(kind, original_source, fused)
            module = LigerFusedLinearCrossEntropyLoss(
                ignore_index=-100, reduction="mean", accum_dtype=torch.float32,
            ).to(device)
            outputs[kind] = run_region(module, hidden, weight, labels)
            del module
        original_loss, original_hidden_grad, original_grad = outputs["original"]
        reverse_loss, reverse_hidden_grad, reverse_grad = outputs["reverse"]
        if not torch.equal(original_loss, reverse_loss):
            raise RuntimeError("chunk order changed the forward loss")
        if not torch.equal(original_hidden_grad, reverse_hidden_grad):
            raise RuntimeError("chunk order changed the hidden-state gradient")
        install_order("original", original_source, fused)
        sham_module = LigerFusedLinearCrossEntropyLoss(
            ignore_index=-100, reduction="mean", accum_dtype=torch.float32,
        ).to(device)
        sham_loss, sham_hidden_grad, sham_grad = run_region(sham_module, hidden, weight, labels)
        if not (torch.equal(sham_loss, original_loss) and torch.equal(sham_hidden_grad, original_hidden_grad)
                and torch.equal(sham_grad, original_grad)):
            raise RuntimeError("matched sham did not exactly reproduce the candidate")

        gradient_delta = original_grad - reverse_grad
        zeros = torch.zeros_like(original_grad)
        candidate_update, _, _ = adam_delta(original_grad, zeros, zeros, 1, learning_rate=1e-4)
        repair_update, _, _ = adam_delta(reverse_grad, zeros, zeros, 1, learning_rate=1e-4)
        update_delta = candidate_update - repair_update
        schedule_gradients = torch.stack([outputs[kind][2] for kind in kinds])
        schedule_mean = schedule_gradients.double().mean(dim=0).float()
        orbit_delta = schedule_mean - reverse_grad

        gradient_effects.append(sketch(gradient_delta))
        gradient_repairs.append(sketch(reverse_grad))
        update_effects.append(sketch(update_delta))
        update_repairs.append(sketch(repair_update))
        orbit_means.append(sketch(orbit_delta))
        rows.append({
            "state_id": state_id,
            "candidate_minus_repair_gradient_l2": float(torch.linalg.vector_norm(gradient_delta.double()).item()),
            "candidate_minus_repair_update_l2": float(torch.linalg.vector_norm(update_delta.double()).item()),
            "orbit_mean_minus_repair_l2": float(torch.linalg.vector_norm(orbit_delta.double()).item()),
            "forward_and_hidden_gradient_bitwise_equal": True,
            "matched_sham_exact": True,
        })
        print(json.dumps({"event": "LIGER_FP32_ORDER_STATE", "index": index, "state_id": state_id}), flush=True)
        del ids, hidden, labels, outputs, original_grad, reverse_grad, sham_grad
        del gradient_delta, zeros, candidate_update, repair_update, update_delta
        del schedule_gradients, schedule_mean, orbit_delta, sham_module
        gc.collect(); torch.cuda.empty_cache()

    indices_a = list(range(16)); indices_b = list(range(16, 32))
    units = [f"state-{index:02d}" for index in range(32)]
    gradient_profile = matched_training_bias_profile(
        np.stack(gradient_effects), np.stack(gradient_repairs),
        calibration_indices=indices_a, confirmation_indices=indices_b,
        inference_unit_ids=units, include_joint_gram=True, seed=20260903,
    )
    update_profile = matched_training_bias_profile(
        np.stack(update_effects), np.stack(update_repairs),
        calibration_indices=indices_a, confirmation_indices=indices_b,
        inference_unit_ids=units, include_joint_gram=True, seed=20260904,
    )
    orbit = np.stack(orbit_means)
    primary = np.stack(gradient_effects)
    predictor = orbit[indices_a].mean(axis=0)
    predictor /= max(float(np.linalg.norm(predictor)), 1e-30)
    confirmation_projection = primary[indices_b] @ predictor
    payload = {
        "schema": "kernel-analyzer-liger-fp32-chunk-order-v1",
        "status": "COMPLETE",
        "case_id": "liger_fused_ce_fp32_chunk_order",
        "dtype": "FP32_FOR_BOTH_PRIMARY_IMPLEMENTATIONS",
        "state_ids": state_ids,
        "measurement_geometry": "COUNT_SKETCH_8192",
        "implementation_provenance": provenance,
        "profiles": {"PARAMETER_GRADIENT": gradient_profile, "ADAMW_UPDATE": update_profile},
        "source_prediction": {
            "confirmation_projection_mean": float(confirmation_projection.mean()),
            "confirmation_positive_count": int(np.count_nonzero(confirmation_projection > 0)),
            "confirmation_count": 16,
            "direction_repeated": bool(float(confirmation_projection.mean()) > 0),
        },
        "rows": rows,
        "claim_boundary": (
            "Both implementations use FP32 and differ only in Liger dW chunk-addition order. "
            "The 8192-coordinate sketch is fixed before results and this is not a long training run."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
