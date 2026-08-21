#!/usr/bin/env python3
"""Open-loop transported reduction-orbit predictor for held-out lm-head dX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch._inductor.codecache import PyCodeCache

from kernel_analyzer.persistence_property import transported_orbit_certificate_from_gram
from kernel_analyzer.reduction_orbit import frozen_permutations
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime
from scripts.run_generated_fp32_screen import load_model, tensor_digest
from scripts.run_qwen256_lmhead_property_confirmation import ShapeObserver


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("generic", "mistral3"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--steps", type=int, choices=(2, 16), default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    bank = json.loads(args.input_bank.read_text())
    states = [row for row in bank["states"] if row["role"] == "CONFIRMATION"][:args.steps]
    if len(states) != args.steps:
        raise RuntimeError("frozen confirmation population is incomplete")
    ids = [str(row["state_id"]) for row in states]
    left = (bank["sequence_length"], args.vocab_size)
    right = (args.vocab_size, args.hidden_size)
    device = torch.device(args.device)
    configure_candidate_runtime(29000)
    model = load_model(args.architecture, args.model, device)
    model.eval()
    parameters = dict(model.named_parameters())
    resolved_carrier = args.carrier
    if resolved_carrier not in parameters:
        suffix_matches = [name for name in parameters if name.endswith(f".{args.carrier}")]
        if len(suffix_matches) != 1:
            raise RuntimeError(
                f"declared carrier absent or ambiguous: {args.carrier}; "
                f"suffix_matches={suffix_matches}"
            )
        resolved_carrier = suffix_matches[0]
    carrier = parameters[resolved_carrier]
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=False, dynamic=False)
    warm = torch.tensor([states[0]["token_ids"]], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True); candidate(warm).backward(); torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    permutations = frozen_permutations(args.vocab_size, 8, 20260820)
    vectors: list[torch.Tensor] = []
    rows = []

    def evaluate(state: dict, index: int, mode: str, permutation=None):
        values = torch.tensor([state["token_ids"]], dtype=torch.long, device=device)
        torch.manual_seed(29_000 + index); torch.cuda.manual_seed_all(29_000 + index)
        model.zero_grad(set_to_none=True)
        observer = ShapeObserver(
            modules, mode, permutations, left_shape=left, right_shape=right,
            selected_permutation=permutation,
        )
        with observer:
            loss = candidate(values); loss.backward()
        torch.cuda.synchronize(device)
        return tensor_digest(loss), carrier.grad.detach().float().clone(), observer

    for index, state in enumerate(states):
        loss_reference, gradient_reference, repaired = evaluate(state, index, "fp32")
        state_vectors = []
        for permutation in permutations:
            loss_variant, gradient_variant, variant = evaluate(
                state, index, "permuted", permutation
            )
            if loss_variant != loss_reference:
                raise RuntimeError("backward-only orbit changed forward loss")
            state_vectors.append(
                (gradient_variant - gradient_reference).mul(-args.learning_rate).cpu()
            )
            del gradient_variant, variant
        vectors.extend(state_vectors)
        rows.append({
            "state_id": ids[index], "step": index + 1,
            "fp32_endpoint_delta_l2": repaired.changed_l2,
            "orbit_mean_update_l2": float(
                torch.linalg.vector_norm(torch.stack(state_vectors).mean(0)).item()
            ),
        })
        print(json.dumps({"event": "HELDOUT_ORBIT_STATE", **rows[-1]}), flush=True)
        del gradient_reference, repaired, state_vectors
        torch.cuda.empty_cache()

    matrix = torch.stack(vectors).double()
    certificate = transported_orbit_certificate_from_gram(
        (matrix @ matrix.T).numpy(), state_ids=ids,
        variant_ids=["identity"] + [f"perm_{index:02d}" for index in range(1, 8)],
        reference_variant="identity", sign_flip_draws=4000, seed=20260820,
    )
    payload = {
        "schema": "kernel-analyzer-heldout-lmhead-transported-orbit-v1",
        "status": "PREDICTION_FROZEN" if args.steps == 16 else "ENGINEERING_DRY_RUN",
        "model": str(args.model.resolve()), "architecture": args.architecture,
        "input_bank": str(args.input_bank.resolve()),
        "endpoint_binding": {"left_shape": list(left), "right_shape": list(right)},
        "carrier": {
            "declared": args.carrier,
            "resolved_runtime_name": resolved_carrier,
        },
        "carrier_coordinates": carrier.numel(),
        "common_weight_open_loop": True, "weight_advancement_used": False,
        "states": rows, "certificate": certificate,
        "prediction": (
            "SOURCE_PERSISTENCE_RISK" if certificate["status"] == "PERSISTENT_TRANSPORTED_CONDITIONAL_MEAN"
            else "NO_DETECTABLE_SOURCE_PERSISTENCE_UNDER_PROTOCOL"
        ),
        "claim_boundary": (
            "One held-out exact lm-head dX implementation; this is an open-loop predictor, "
            "not a revealed trajectory consequence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
