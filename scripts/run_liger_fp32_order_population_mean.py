#!/usr/bin/env python3
"""Test a signed mean for the FP32 Liger chunk-order effect.

The empirical input bank is sampled with replacement.  A calibration mean
direction is accumulated in original lm_head coordinates, then confirmation
units are scored on that frozen direction.  This is deliberately a scoped
projected-mean experiment; it does not call a fixed suite a natural
population and it does not use a sketch for the primary endpoint.
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

from scripts.run_liger_fp32_chunk_order_length import DESIGN, MODEL, _install_order  # noqa: E402


LEARNING_RATE = 1e-4
BETAS = (0.9, 0.95)
EPSILON = 1e-8
SEED = 20260919
CALIBRATION_COUNT = 32
CONFIRMATION_COUNT = 64


def one_step(module: Any, hidden: torch.Tensor, weight0: torch.Tensor,
             labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    out = (loss.detach(), grad_hidden.detach(), write.detach())
    del optimizer, weight, h, grad_weight
    return out


def exact_stats(effect: torch.Tensor, repair: torch.Tensor) -> dict[str, Any]:
    e = effect.detach().double()
    r = repair.detach().double()
    return {
        "effect_energy": float(torch.sum(e * e).item()),
        "repair_energy": float(torch.sum(r * r).item()),
        "effect_l2": float(torch.linalg.vector_norm(e).item()),
        "repair_l2": float(torch.linalg.vector_norm(r).item()),
        "nonzero_effect_coordinates": int(torch.count_nonzero(effect).item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, choices=(64, 256), default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"preserve existing output: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("host GPU required")

    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    import liger_kernel.ops.fused_linear_cross_entropy as fused
    from transformers import AutoModelForCausalLM
    from scipy.stats import t

    records = [
        row for row in json.loads(DESIGN.read_text())["records"]
        if int(row["length"]) == args.length
    ]
    if len(records) != 32:
        raise RuntimeError(f"expected 32 bank records, got {len(records)}")
    rng = np.random.default_rng(SEED)
    draw_indices = rng.integers(0, len(records), size=CALIBRATION_COUNT + CONFIRMATION_COUNT)
    draws = [records[int(index)] for index in draw_indices]

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
    source = inspect.getsource(fused.fused_linear_cross_entropy_forward)

    direction_sum = torch.zeros_like(weight0)
    calibration_rows: list[dict[str, Any]] = []
    confirmation_rows: list[dict[str, Any]] = []
    confirmation_projection: list[float] = []
    direction: torch.Tensor | None = None
    total_effect_energy = 0.0
    total_repair_energy = 0.0

    for index, record in enumerate(draws):
        ids = torch.tensor([record["input_ids"]], dtype=torch.long, device=device)
        with torch.no_grad():
            hidden = model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state.detach()
        labels = torch.nn.functional.pad(ids, (0, 1), value=-100)[..., 1:].contiguous().reshape(-1)
        outputs: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for kind in ("original", "reverse"):
            _install_order(kind, source, fused)
            module = LigerFusedLinearCrossEntropyLoss(
                ignore_index=-100, reduction="mean", accum_dtype=torch.float32,
            ).to(device)
            outputs[kind] = one_step(module, hidden, weight0, labels)
            del module
        candidate_loss, candidate_hidden, candidate_write = outputs["original"]
        reference_loss, reference_hidden, reference_write = outputs["reverse"]
        if not torch.equal(candidate_loss, reference_loss):
            raise RuntimeError("chunk order changed forward loss")
        if not torch.equal(candidate_hidden, reference_hidden):
            raise RuntimeError("chunk order changed hidden-state gradient")
        effect = candidate_write - reference_write
        stats = exact_stats(effect, reference_write)
        stats.update({
            "unit": index,
            "bank_index": int(draw_indices[index]),
            "state_id": str(record["sequence_id"]),
            "forward_and_hidden_gradient_bitwise_equal": True,
        })
        total_effect_energy += stats["effect_energy"]
        total_repair_energy += stats["repair_energy"]
        if index < CALIBRATION_COUNT:
            direction_sum.add_(effect)
            calibration_rows.append(stats)
        else:
            if direction is None:
                direction = direction_sum / max(float(CALIBRATION_COUNT), 1.0)
                norm = torch.linalg.vector_norm(direction)
                if not bool(norm > 0):
                    raise RuntimeError("calibration direction is zero")
                direction = direction / norm
            projection = torch.sum(effect.double() * direction.double()).item()
            stats["frozen_calibration_direction_projection"] = float(projection)
            confirmation_projection.append(float(projection))
            confirmation_rows.append(stats)
        print(json.dumps({
            "event": "LIGER_FP32_ORDER_POPULATION_UNIT",
            "unit": index,
            "bank_index": int(draw_indices[index]),
            "phase": "calibration" if index < CALIBRATION_COUNT else "confirmation",
            "effect_l2": stats["effect_l2"],
        }), flush=True)
        del ids, hidden, labels, outputs, effect, candidate_loss, reference_loss
        del candidate_hidden, reference_hidden, candidate_write, reference_write
        gc.collect()
        torch.cuda.empty_cache()

    if direction is None:
        raise RuntimeError("no confirmation direction")
    values = np.asarray(confirmation_projection, dtype=np.float64)
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    two_sided_half = float(t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    one_sided_half = float(t.ppf(0.95, n - 1)) * sd / math.sqrt(n)
    payload = {
        "schema": "kernel-analyzer-liger-fp32-order-population-mean-v1",
        "status": "COMPLETE",
        "case_id": "liger_fused_ce_fp32_chunk_order",
        "sequence_length": args.length,
        "candidate": "original sequential FP32 chunk order",
        "reference": "reverse FP32 chunk order",
        "dtype": "FP32_FOR_BOTH_IMPLEMENTATIONS",
        "population": "with-replacement draws from the declared 32-state empirical Liger bank",
        "draw_seed": SEED,
        "draw_indices": [int(x) for x in draw_indices],
        "calibration_count": CALIBRATION_COUNT,
        "confirmation_count": CONFIRMATION_COUNT,
        "target": "lm_head.weight actual torch AdamW parameter write",
        "calibration_direction": {
            "geometry": "ORIGINAL_COORDINATE_EXACT",
            "norm_before_normalization": float(torch.linalg.vector_norm(direction_sum / CALIBRATION_COUNT).item()),
            "coordinate_count": int(weight0.numel()),
        },
        "confirmation": {
            "mean_projection": mean,
            "sample_sd": sd,
            "two_sided_95_interval": [mean - two_sided_half, mean + two_sided_half],
            "one_sided_95_lower_bound": mean - one_sided_half,
            "positive_count": int(np.count_nonzero(values > 0)),
            "negative_count": int(np.count_nonzero(values < 0)),
            "zero_count": int(np.count_nonzero(values == 0)),
            "scope": "DECLARED_IID_WITH_REPLACEMENT_FROM_EMPIRICAL_LIGER_BANK",
            "implies_vector_mean_nonzero_if_one_sided_bound_positive": bool(mean - one_sided_half > 0),
        },
        "fixed_draw_energy_ratio": float(math.sqrt(total_effect_energy / total_repair_energy))
        if total_repair_energy else 0.0,
        "calibration_rows": calibration_rows,
        "confirmation_rows": confirmation_rows,
        "claim_boundary": (
            "The signed projection is measured in original parameter coordinates after "
            "a real zero-moment torch AdamW step.  The inference population is the "
            "declared empirical bank sampled with replacement; this is not a claim "
            "about arbitrary natural LLM data or long-run quality."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "output": str(args.output),
        "confirmation": payload["confirmation"],
    }))


if __name__ == "__main__":
    main()
