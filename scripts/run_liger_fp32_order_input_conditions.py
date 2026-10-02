#!/usr/bin/env python3
"""Probe when same-FP32 chunk-order effects cancel or survive.

The condition set is fixed before measurement:
  natural labels, reversed labels, and one constant valid label.
Hidden states and all model parameters are held fixed.  Only the label
contribution pattern and the declared dW chunk-addition order vary.  This is a
mechanism probe, not a natural-data population claim.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.run_liger_fp32_chunk_order_length import DESIGN, MODEL, _install_order, _sketch  # noqa: E402
from scripts.run_liger_fp32_parameter_write_confirmation import _one_step_write, _exact_stats  # noqa: E402


SEED = 20260918
CONDITIONS = ("natural", "reversed_labels", "constant_label")


def _labels_for(condition: str, ids: torch.Tensor) -> torch.Tensor:
    labels = torch.nn.functional.pad(ids, (0, 1), value=-100)[..., 1:].contiguous().reshape(-1)
    if condition == "natural":
        return labels
    if condition == "reversed_labels":
        return labels.flip(0).contiguous()
    if condition == "constant_label":
        return torch.full_like(labels, int(labels[0].item()))
    raise ValueError(condition)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, choices=(64, 256), default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("host GPU required")

    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    import liger_kernel.ops.fused_linear_cross_entropy as fused
    from transformers import AutoModelForCausalLM

    records = [row for row in json.loads(DESIGN.read_text())["records"] if int(row["length"]) == args.length]
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
    source = inspect.getsource(fused.fused_linear_cross_entropy_forward)

    rows: list[dict[str, Any]] = []
    sketches: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {
        condition: ([], []) for condition in CONDITIONS
    }
    for index, record in enumerate(records):
        ids = torch.tensor([record["input_ids"]], dtype=torch.long, device=device)
        with torch.no_grad():
            hidden = model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state.detach()
        for condition in CONDITIONS:
            labels = _labels_for(condition, ids)
            outputs: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
            for kind in ("original", "reverse"):
                _install_order(kind, source, fused)
                module = LigerFusedLinearCrossEntropyLoss(
                    ignore_index=-100, reduction="mean", accum_dtype=torch.float32,
                ).to(device)
                outputs[kind] = _one_step_write(module, hidden, weight0, labels, device)
                del module
            candidate_loss, candidate_hidden, _, candidate_write = outputs["original"]
            reference_loss, reference_hidden, _, reference_write = outputs["reverse"]
            if not torch.equal(candidate_loss, reference_loss):
                raise RuntimeError(f"forward loss changed for {condition}")
            if not torch.equal(candidate_hidden, reference_hidden):
                raise RuntimeError(f"hidden gradient changed for {condition}")
            effect = candidate_write - reference_write
            stats = _exact_stats(effect, reference_write)
            stats.update({
                "state_id": str(record["sequence_id"]),
                "condition": condition,
                "forward_and_hidden_gradient_bitwise_equal": True,
            })
            rows.append(stats)
            sketches[condition][0].append(_sketch(effect))
            sketches[condition][1].append(_sketch(reference_write))
            del labels, outputs, effect
        print(json.dumps({"event": "LIGER_ORDER_INPUT_CONDITION_STATE", "index": index,
                          "state_id": str(record["sequence_id"])}), flush=True)
        del ids, hidden
        gc.collect()
        torch.cuda.empty_cache()

    summaries: dict[str, Any] = {}
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        effect_energy = sum(row["effect_energy"] for row in selected)
        repair_energy = sum(row["repair_energy"] for row in selected)
        effect_sketch = np.stack(sketches[condition][0])
        repair_sketch = np.stack(sketches[condition][1])
        calibration_mean = effect_sketch[:16].mean(axis=0)
        norm = float(np.linalg.norm(calibration_mean))
        if norm > 0:
            projection = effect_sketch[16:] @ (calibration_mean / norm)
        else:
            projection = np.zeros(16, dtype=np.float64)
        summaries[condition] = {
            "state_count": len(selected),
            "fixed_suite_total_write_rms": float(np.sqrt(effect_energy / repair_energy)) if repair_energy else 0.0,
            "mean_effect_l2": float(np.mean([row["effect_l2"] for row in selected])),
            "nonzero_effect_state_count": int(sum(row["nonzero_effect_coordinates"] > 0 for row in selected)),
            "calibration_count": 16,
            "confirmation_count": 16,
            "confirmation_projection_mean": float(projection.mean()),
            "confirmation_positive_count": int(np.count_nonzero(projection > 0)),
            "direction_diagnostic": "COUNT_SKETCH_8192_DESCRIPTIVE_ONLY",
        }
    payload = {
        "schema": "kernel-analyzer-liger-fp32-order-input-conditions-v1",
        "status": "COMPLETE",
        "case_id": "liger_fused_ce_fp32_chunk_order",
        "sequence_length": args.length,
        "condition_policy": {
            "conditions": list(CONDITIONS),
            "natural": "original next-token labels",
            "reversed_labels": "reverse the same label vector; hidden states unchanged",
            "constant_label": "replace every target with the first valid natural label; hidden states unchanged",
            "reason": "predeclared label-contribution patterns to test cancellation without changing the operator or model operands",
        },
        "comparison": {
            "candidate": "original sequential FP32 chunk order",
            "reference": "reverse FP32 chunk order",
            "optimizer": "real torch.optim.AdamW, zero initial moments, weight_decay=0",
        },
        "measurement_geometry": "ORIGINAL_COORDINATE_PARAMETER_WRITE_PLUS_SKETCH_DIRECTION",
        "summaries": summaries,
        "rows": rows,
        "claim_boundary": (
            "This is a controlled input-condition probe on one fixed hidden-state bank. "
            "It tests when order effects cancel or survive; it does not estimate a natural "
            "training-population probability and does not establish a loss consequence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "output": str(args.output), "summaries": summaries}))


if __name__ == "__main__":
    main()
