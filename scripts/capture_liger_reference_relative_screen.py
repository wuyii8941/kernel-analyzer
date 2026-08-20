#!/usr/bin/env python3
"""Test Liger accumulation error in the same-state reference-update frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "archive/round1_code/src"),
    str(ROOT / "archive/nonprecision_v1/code"),
]

from kernel_analyzer.reference_relative_oracle import (  # noqa: E402
    ReferenceRelativeObservation,
    certify_reference_relative,
)
from run_liger_fused_ce import capture_lmhead, hidden_from_boundary  # noqa: E402


def fused_gradient(module: Any, hidden: Any, weight: Any, labels: Any) -> tuple[Any, Any, Any]:
    import torch

    h = hidden.detach().clone().reshape(-1, hidden.shape[-1]).requires_grad_(True)
    loss = module(weight, h, labels)
    d_h, d_w = torch.autograd.grad(loss, (h, weight))
    return loss.detach(), d_h.detach(), d_w.detach()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--coordinates", type=int, default=65_536)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/property/bias_oracle_recovery/liger_reference_relative.json",
    )
    args = parser.parse_args()
    if args.states < 4:
        raise ValueError("at least four states are required")

    import torch
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    device = torch.device(args.device)
    protocol = json.loads((
        ROOT / "archive/nonprecision_v1/runs/liger.fused_ce.protocol.json"
    ).read_text(encoding="utf-8"))
    design = json.loads(Path(
        protocol["bindings"]["state_design"]["path"]
    ).read_text(encoding="utf-8"))
    allocations = protocol["state_allocations"]["confirmation"][:args.states]
    records = {row["sequence_id"]: row for row in design["records"]}

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        "/data1/tzh/models/Qwen/Qwen3-1.7B",
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    ).to(device)
    model.config.use_cache = False
    model.eval()
    weight = model.lm_head.weight
    total_coordinates = int(weight.numel())
    rng = np.random.default_rng(20260820)
    sampled = np.sort(rng.choice(
        total_coordinates, size=args.coordinates, replace=False
    )).astype(np.int64)
    indices = torch.from_numpy(sampled).to(device)
    default_module = LigerFusedLinearCrossEntropyLoss(
        ignore_index=-100, reduction="mean", accum_dtype=None
    ).to(device)
    repair_module = LigerFusedLinearCrossEntropyLoss(
        ignore_index=-100, reduction="mean", accum_dtype=torch.float32
    ).to(device)
    observations = []
    rows = []
    for index, allocation in enumerate(allocations):
        record = records[allocation["state_id"]]
        values = torch.tensor([record["input_ids"]], dtype=torch.long, device=device)
        boundary, _, observation_gates = capture_lmhead(model, values)
        hidden = hidden_from_boundary(boundary, device)
        labels = torch.nn.functional.pad(
            values, (0, 1), value=-100
        )[..., 1:].reshape(-1)
        default_loss, default_dh, default_dw = fused_gradient(
            default_module, hidden, weight, labels
        )
        repair_loss, repair_dh, repair_dw = fused_gradient(
            repair_module, hidden, weight, labels
        )
        error = (
            default_dw.reshape(-1).index_select(0, indices).float()
            - repair_dw.reshape(-1).index_select(0, indices).float()
        )
        reference = repair_dw.reshape(-1).index_select(0, indices).float()
        dot = float(torch.dot(error, reference).item())
        error_energy = float(torch.dot(error, error).item())
        reference_energy = float(torch.dot(reference, reference).item())
        observation = ReferenceRelativeObservation(
            condition_id=allocation["state_id"],
            error_reference_dot=dot,
            error_energy=error_energy,
            reference_energy=reference_energy,
        )
        gates = {
            **observation_gates,
            "loss_equal": bool(torch.equal(default_loss, repair_loss)),
            "dH_equal": bool(torch.equal(default_dh, repair_dh)),
            "error_nonzero": error_energy > 0.0,
            "reference_nonzero": reference_energy > 0.0,
        }
        observations.append(observation)
        rows.append({
            **observation.as_dict(),
            "gates": gates,
        })
        if not all(gates.values()):
            raise RuntimeError(f"Liger reference-relative boundary failed: {gates}")
        del values, boundary, hidden, labels, default_dw, repair_dw, error, reference
        torch.cuda.empty_cache()
        print(json.dumps({
            "event": "STATE_COMPLETE", "index": index,
            "coefficient": observation.coefficient,
            "cosine": observation.cosine,
        }, sort_keys=True), flush=True)
    certificate = certify_reference_relative(observations)
    payload = {
        "schema": "kernel-analyzer-liger-reference-relative-screen-v1",
        "case_id": "liger_fused_ce",
        "coordinate_sampling": {
            "method": "fixed_seed_uniform_without_replacement",
            "seed": 20260820,
            "sampled": args.coordinates,
            "population": total_coordinates,
        },
        "rows": rows,
        "certificate": certificate.as_dict(),
        "status": certificate.status,
        "claim_boundary": (
            "development recovery measurement on frozen natural states; the "
            "reference direction is the same-state FP32-accumulator dW, not a "
            "candidate-error or trajectory-fitted carrier"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output), "status": payload["status"],
        "certificate": certificate.as_dict(),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
