#!/usr/bin/env python3
"""Test a same-dtype compensated chunk accumulation on the real Liger path.

The candidate is the original FP32 chunk accumulation.  The intervention keeps
the same FP32 contribution tensors and chunk order, but uses an FP32 Kahan
compensation tensor for the outer chunk sum.  The run is intentionally a
mechanism intervention, not a claim that compensated summation is a universal
repair.
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
        return torch.randperm(
            count, generator=torch.Generator().manual_seed(ORDER_SEED)
        ).tolist()
    raise ValueError(kind)


def _install_source(kind: str, original_source: str, fused: Any) -> None:
    marker = "    for chunk_id in range(num_chunks):"
    replacement = "    for chunk_id in _declared_chunk_order(num_chunks):"
    if original_source.count(marker) != 1:
        raise RuntimeError("Liger chunk loop was not uniquely identified")
    transformed = original_source.replace(marker, replacement, 1)
    if kind == "kahan":
        init_marker = "    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)"
        init_replacement = (
            "    grad_weight_compensation = (torch.zeros_like(grad_weight) "
            "if grad_weight is not None else None)\n"
            + init_marker
        )
        if transformed.count(init_marker) != 1:
            raise RuntimeError("Liger loss accumulator was not uniquely identified")
        transformed = transformed.replace(init_marker, init_replacement, 1)
        sum_marker = "            grad_weight += torch.mm(grad_logits_chunk.t(), _input_chunk).float()"
        sum_replacement = (
            "            contribution = torch.mm(grad_logits_chunk.t(), _input_chunk).float()\n"
            "            corrected = contribution - grad_weight_compensation\n"
            "            updated = grad_weight + corrected\n"
            "            grad_weight_compensation = (updated - grad_weight) - corrected\n"
            "            grad_weight = updated"
        )
        if transformed.count(sum_marker) != 1:
            raise RuntimeError("Liger gradient accumulation was not uniquely identified")
        transformed = transformed.replace(sum_marker, sum_replacement, 1)
    namespace = dict(fused.__dict__)
    namespace["_declared_chunk_order"] = lambda count: _order_values(
        "original" if kind == "kahan" else kind, count
    )
    exec(compile(transformed, f"<liger-fp32-{kind}>", "exec"), namespace)
    fused.fused_linear_cross_entropy_forward = namespace["fused_linear_cross_entropy_forward"]


def _run_region(module: Any, hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor):
    h = hidden.detach().clone().reshape(-1, hidden.shape[-1]).requires_grad_(True)
    loss = module(weight, h, labels)
    grad_hidden, grad_weight = torch.autograd.grad(loss, (h, weight), retain_graph=False)
    return loss.detach(), grad_hidden.detach(), grad_weight.detach()


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

    variant_kinds = ("reverse", "even_then_odd", "frozen_permutation", "kahan")
    gradient_effects: dict[str, list[np.ndarray]] = {kind: [] for kind in variant_kinds}
    update_effects: dict[str, list[np.ndarray]] = {kind: [] for kind in variant_kinds}
    gradient_repairs: dict[str, list[np.ndarray]] = {kind: [] for kind in variant_kinds}
    update_repairs: dict[str, list[np.ndarray]] = {kind: [] for kind in variant_kinds}
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(records):
        ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
        with torch.no_grad():
            hidden = model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state.detach()
        labels = torch.nn.functional.pad(ids, (0, 1), value=-100)[..., 1:].contiguous().reshape(-1)
        outputs: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for kind in ("original", "reverse", "even_then_odd", "frozen_permutation", "kahan"):
            _install_source(kind, original_source, fused)
            module = LigerFusedLinearCrossEntropyLoss(
                ignore_index=-100, reduction="mean", accum_dtype=torch.float32,
            ).to(device)
            outputs[kind] = _run_region(module, hidden, weight, labels)
            del module

        original_loss, original_hidden_grad, original_grad = outputs["original"]
        for kind in variant_kinds:
            variant_loss, variant_hidden_grad, variant_grad = outputs[kind]
            if not torch.equal(original_loss, variant_loss):
                raise RuntimeError(f"{kind} intervention changed forward loss")
            if not torch.equal(original_hidden_grad, variant_hidden_grad):
                raise RuntimeError(f"{kind} intervention changed hidden-state gradient")
            delta = original_grad - variant_grad
            candidate_update = -1e-4 * original_grad / (original_grad.abs() + 1e-8)
            variant_update = -1e-4 * variant_grad / (variant_grad.abs() + 1e-8)
            gradient_effects[kind].append(_sketch(delta))
            update_effects[kind].append(_sketch(candidate_update - variant_update))
            gradient_repairs[kind].append(_sketch(variant_grad))
            update_repairs[kind].append(_sketch(variant_update))
        reverse_grad = outputs["reverse"][2]
        kahan_grad = outputs["kahan"][2]
        order_delta = original_grad - reverse_grad
        kahan_delta = original_grad - kahan_grad
        rows.append({
            "state_index": index,
            "forward_and_hidden_gradient_bitwise_equal": True,
            "gradient_l2": {
                kind: float(torch.linalg.vector_norm((original_grad - outputs[kind][2]).double()).item())
                for kind in variant_kinds
            },
            "kahan_to_reverse_gradient_l2_ratio": float(
                torch.linalg.vector_norm(kahan_delta.double()).item()
                / max(torch.linalg.vector_norm(order_delta.double()).item(), 1e-30)
            ),
        })
        print(json.dumps({"event": "LIGER_KAHAN_STATE", "index": index}), flush=True)
        del ids, hidden, labels, outputs, original_grad, reverse_grad, kahan_grad
        del order_delta, kahan_delta
        gc.collect()
        torch.cuda.empty_cache()

    calibration = list(range(16))
    confirmation = list(range(16, 32))
    units = [f"length-{args.length}-state-{i:02d}" for i in range(32)]

    def profile(effect, repair, seed):
        return matched_training_bias_profile(
            np.stack(effect), np.stack(repair), calibration_indices=calibration,
            confirmation_indices=confirmation, inference_unit_ids=units,
            include_joint_gram=True, seed=seed,
        )

    payload = {
        "schema": "kernel-analyzer-liger-fp32-kahan-intervention-v1",
        "status": "COMPLETE",
        "case_id": "liger_fused_ce_fp32_chunk_accumulation",
        "sequence_length": args.length,
        "dtype": "FP32_FOR_ALL_VARIANTS",
        "comparison": {
            "candidate": "original sequential FP32 chunk sum",
            "source_reference": "reverse sequential FP32 chunk sum",
            "intervention": "original chunk order with FP32 Kahan compensation for the outer chunk sum",
            "prediction": "Kahan compensation should reduce the original-versus-reverse gradient and update effect",
        },
        "profiles": {
            f"ORIGINAL_MINUS_{kind.upper()}": {
                "PARAMETER_GRADIENT": profile(gradient_effects[kind], gradient_repairs[kind], 20260911 + index),
                "ADAMW_UPDATE": profile(update_effects[kind], update_repairs[kind], 20260921 + index),
            }
            for index, kind in enumerate(variant_kinds)
        },
        "rows": rows,
        "claim_boundary": (
            "All variants use FP32 inputs, FP32 contribution tensors, and FP32 outer accumulation. "
            "The Kahan run is a targeted source intervention on a real Liger implementation; "
            "it does not establish a training-loss consequence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
