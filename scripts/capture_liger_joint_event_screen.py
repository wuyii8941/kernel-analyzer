#!/usr/bin/env python3
"""Capture per-chunk Liger dW accumulation-error atoms for one natural state.

Only a deterministic coordinate sample is retained.  The sampled event atoms
sum exactly to the sampled default-BF16 minus FP32-accumulator dW contrast; the
complete event Gram then supplies a direction-free joint-coherence screen.
"""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
import json
import math
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

from kernel_analyzer.joint_event_oracle import (  # noqa: E402
    certify_joint_event_gram,
    gram_from_event_vectors,
)
from run_liger_fused_ce import capture_lmhead, hidden_from_boundary  # noqa: E402


class ChunkEventObserver(AbstractContextManager["ChunkEventObserver"]):
    def __init__(self, torch: Any, indices: Any) -> None:
        self.torch = torch
        self.indices = indices
        self.original_mm = torch.mm
        self.bf16_accumulator = torch.zeros(
            len(indices), dtype=torch.bfloat16, device=indices.device
        )
        self.fp32_accumulator = torch.zeros(
            len(indices), dtype=torch.float32, device=indices.device
        )
        self.atoms: list[np.ndarray] = []
        self.contribution_energy = 0.0
        self.rounding_atom_energy = 0.0
        self.rounding_atom_contribution_dot = 0.0
        self.calls = 0

    def __enter__(self) -> "ChunkEventObserver":
        def wrapped(left: Any, right: Any, *args: Any, **kwargs: Any) -> Any:
            output = self.original_mm(left, right, *args, **kwargs)
            # Liger's only explicit torch.mm in this region is the chunk dW.
            if output.ndim == 2 and output.numel() > int(self.indices.max().item()):
                contribution = output.detach().reshape(-1).index_select(
                    0, self.indices
                ).float()
                previous = self.bf16_accumulator.float()
                updated = (previous + contribution).to(self.torch.bfloat16)
                rounding_atom = updated.float() - previous - contribution
                self.atoms.append(rounding_atom.cpu().numpy().copy())
                self.contribution_energy += float(
                    self.torch.dot(contribution, contribution).item()
                )
                self.rounding_atom_energy += float(
                    self.torch.dot(rounding_atom, rounding_atom).item()
                )
                self.rounding_atom_contribution_dot += float(
                    self.torch.dot(rounding_atom, contribution).item()
                )
                self.bf16_accumulator = updated
                self.fp32_accumulator.add_(contribution)
                self.calls += 1
            return output

        self.torch.mm = wrapped
        return self

    def __exit__(self, *unused: Any) -> None:
        del unused
        self.torch.mm = self.original_mm

    def finalize(self) -> list[np.ndarray]:
        # The repair arm casts its FP32 accumulator back to BF16 once.  Its cast
        # error must be subtracted from the sequential-BF16 error atoms so that
        # their sum equals default_dW - repaired_dW exactly.
        repair_cast = self.fp32_accumulator.to(self.torch.bfloat16).float()
        final_cast_correction = self.fp32_accumulator - repair_cast
        return [*self.atoms, final_cast_correction.cpu().numpy().copy()]


def fused_gradient(module: Any, hidden: Any, weight: Any, labels: Any) -> tuple[Any, Any, Any]:
    import torch

    h = hidden.detach().clone().reshape(-1, hidden.shape[-1]).requires_grad_(True)
    loss = module(weight, h, labels)
    d_h, d_w = torch.autograd.grad(loss, (h, weight))
    return loss.detach(), d_h.detach(), d_w.detach()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--coordinates", type=int, default=65_536)
    parser.add_argument("--state-offset", type=int, default=0)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/property/bias_oracle_recovery/liger_joint_event.json",
    )
    args = parser.parse_args()

    import torch
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    device = torch.device(args.device)
    protocol = json.loads((
        ROOT / "archive/nonprecision_v1/runs/liger.fused_ce.protocol.json"
    ).read_text(encoding="utf-8"))
    design_path = Path(protocol["bindings"]["state_design"]["path"])
    design = json.loads(design_path.read_text(encoding="utf-8"))
    allocation = protocol["state_allocations"]["confirmation"][args.state_offset]
    records = {row["sequence_id"]: row for row in design["records"]}
    record = records[allocation["state_id"]]

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
    values = torch.tensor([record["input_ids"]], dtype=torch.long, device=device)
    boundary, _, observation_gates = capture_lmhead(model, values)
    hidden = hidden_from_boundary(boundary, device)
    labels = torch.nn.functional.pad(values, (0, 1), value=-100)[..., 1:].reshape(-1)
    weight = model.lm_head.weight
    total_coordinates = int(weight.numel())
    if args.coordinates < 1024 or args.coordinates > total_coordinates:
        raise ValueError("invalid coordinate sample size")
    rng = np.random.default_rng(20260820)
    sampled = np.sort(rng.choice(
        total_coordinates, size=args.coordinates, replace=False
    )).astype(np.int64)
    indices = torch.from_numpy(sampled).to(device)
    default_module = LigerFusedLinearCrossEntropyLoss(
        ignore_index=-100, reduction="mean", accum_dtype=None
    ).to(device)
    fp32_module = LigerFusedLinearCrossEntropyLoss(
        ignore_index=-100, reduction="mean", accum_dtype=torch.float32
    ).to(device)

    observer = ChunkEventObserver(torch, indices)
    with observer:
        default_loss, default_dh, default_dw = fused_gradient(
            default_module, hidden, weight, labels
        )
    fp32_loss, fp32_dh, fp32_dw = fused_gradient(fp32_module, hidden, weight, labels)
    torch.cuda.synchronize(device)
    atoms = observer.finalize()
    default_sample = default_dw.reshape(-1).index_select(0, indices).float()
    fp32_sample = fp32_dw.reshape(-1).index_select(0, indices).float()
    atom_sum = np.sum(np.stack(atoms, axis=0), axis=0, dtype=np.float64)
    actual = (default_sample - fp32_sample).cpu().numpy().astype(np.float64)
    reference = fp32_sample.cpu().numpy().astype(np.float64)
    closure_max_abs = float(np.max(np.abs(atom_sum - actual)))
    closure_relative_l2 = float(
        np.linalg.norm(atom_sum - actual) / max(np.linalg.norm(actual), 1e-30)
    )
    gram = gram_from_event_vectors(atoms)
    certificate = certify_joint_event_gram(
        gram, coordinate_count=args.coordinates, random_sign_draws=4000
    )
    actual_norm = float(np.linalg.norm(actual))
    reference_norm = float(np.linalg.norm(reference))
    actual_reference_dot = float(np.dot(actual, reference))
    reference_relative = {
        "error_l2": actual_norm,
        "reference_update_l2": reference_norm,
        "cosine": actual_reference_dot / max(actual_norm * reference_norm, 1e-30),
        "multiplicative_coefficient": actual_reference_dot / max(reference_norm**2, 1e-30),
        "error_energy_in_reference_direction": (
            actual_reference_dot**2
            / max(actual_norm**2 * reference_norm**2, 1e-30)
        ),
    }
    event_absorption = {
        "rounding_atom_contribution_dot": observer.rounding_atom_contribution_dot,
        "rounding_atom_energy": observer.rounding_atom_energy,
        "contribution_energy": observer.contribution_energy,
        "cosine": observer.rounding_atom_contribution_dot / max(
            math.sqrt(observer.rounding_atom_energy * observer.contribution_energy),
            1e-30,
        ),
    }
    gates = {
        **observation_gates,
        "chunk_count_64": observer.calls == 64,
        "loss_equal_between_accumulator_arms": bool(torch.equal(default_loss, fp32_loss)),
        "dH_equal_between_accumulator_arms": bool(torch.equal(default_dh, fp32_dh)),
        "sampled_default_accumulator_exact": bool(torch.equal(
            default_sample, observer.bf16_accumulator.float()
        )),
        "sampled_fp32_accumulator_exact": bool(torch.equal(
            fp32_sample, observer.fp32_accumulator.to(torch.bfloat16).float()
        )),
        "event_sum_closes_actual_contrast": closure_relative_l2 <= 5e-5,
    }
    payload = {
        "schema": "kernel-analyzer-liger-joint-event-screen-v1",
        "case_id": "liger_fused_ce",
        "state_id": allocation["state_id"],
        "event_definition": (
            "64 sequential BF16 dW-add rounding increments plus the negated "
            "single final BF16 cast error of the FP32-accumulator repair"
        ),
        "coordinate_sampling": {
            "method": "fixed_seed_uniform_without_replacement",
            "seed": 20260820,
            "sampled": args.coordinates,
            "population": total_coordinates,
        },
        "closure": {
            "max_abs": closure_max_abs,
            "relative_l2": closure_relative_l2,
        },
        "gates": gates,
        "joint_event_certificate": certificate.as_dict(),
        "reference_relative_error": reference_relative,
        "event_rounding_vs_chunk_contribution": event_absorption,
        "complete_event_gram": gram,
        "status": (
            certificate.status if all(gates.values()) else "INVALID_BOUNDARY"
        ),
        "claim_boundary": (
            "single natural state and a fixed coordinate sample; this is an "
            "oracle-input feasibility measurement, not held-out validation"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output), "status": payload["status"],
        "coherence": certificate.normalized_cross_event_coherence,
        "pvalue": certificate.random_sign_pvalue,
        "reference_relative": reference_relative,
        "event_absorption": event_absorption,
        "gates": gates,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
