#!/usr/bin/env python3
"""Check Gemma-4 per-layer input gating on real text states.

This is a small natural model-component probe.  It keeps the Gemma-4
checkpoint, token bank, embeddings, PLE inputs, and parameter carrier fixed,
and compares the native BF16 gate-times-input product with the same product
evaluated in float32 and written back to BF16.  An exact result is retained as
negative evidence; it is not silently promoted to a new problem group.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import Gemma4ForConditionalGeneration


def run(args: argparse.Namespace) -> dict[str, Any]:
    bank = json.loads(args.input_bank.read_text(encoding="utf-8"))["states"][: args.states]
    if len(bank) != args.states or args.states < 2:
        raise ValueError("requested states are not available")
    model = Gemma4ForConditionalGeneration.from_pretrained(
        str(args.model), local_files_only=True, dtype=torch.bfloat16
    ).to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layer = model.model.language_model.layers[0]
    layer.per_layer_input_gate.weight.requires_grad_(True)
    layer.per_layer_projection.weight.requires_grad_(True)
    rows = []
    for state in bank:
        input_ids = torch.tensor([state["token_ids"][: args.sequence_length]], dtype=torch.long, device=args.device)
        with torch.no_grad():
            inputs = model.model.language_model.embed_tokens(input_ids)
            per_layer = model.model.language_model.get_per_layer_inputs(input_ids, inputs)
            per_layer = model.model.language_model.project_per_layer_inputs(inputs, per_layer)
            per_layer = per_layer[:, :, 0, :]
        residual = inputs.detach()
        carrier = inputs.detach()
        outputs = []
        for reference in (False, True):
            model.zero_grad(set_to_none=True)
            gate = layer.act_fn(layer.per_layer_input_gate(carrier))
            if reference:
                product = (gate.float() * per_layer.float()).to(gate.dtype)
            else:
                product = gate * per_layer
            transformed = layer.post_per_layer_input_norm(layer.per_layer_projection(product))
            hidden = residual + transformed
            logits = model.lm_head(hidden).float()
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), input_ids.reshape(-1)
            )
            loss.backward()
            outputs.append(
                (
                    loss.detach().cpu(),
                    layer.per_layer_input_gate.weight.grad.detach().clone(),
                    layer.per_layer_projection.weight.grad.detach().clone(),
                )
            )
        candidate, reference = outputs
        gate_update = -args.learning_rate * (candidate[1] - reference[1])
        projection_update = -args.learning_rate * (candidate[2] - reference[2])
        rows.append(
            {
                "state_id": state.get("state_id"),
                "loss_difference_native_minus_fp32_product": float(candidate[0] - reference[0]),
                "gate_gradient_max_abs": float((candidate[1] - reference[1]).float().abs().max()),
                "projection_gradient_max_abs": float((candidate[2] - reference[2]).float().abs().max()),
                "gate_write_rms_over_reference": float(
                    gate_update.float().norm() / (args.learning_rate * reference[1].float().norm() + 1e-30)
                ),
                "projection_write_rms_over_reference": float(
                    projection_update.float().norm()
                    / (args.learning_rate * reference[2].float().norm() + 1e-30)
                ),
            }
        )
    return {
        "schema": "kernel-analyzer-gemma4-per-layer-input-probe-v1",
        "status": "SCREENED_EXACT_IDENTITY_BF16_PATH",
        "model": str(args.model),
        "input_source": "declared real Gemma-4 text input bank",
        "operator": "Gemma-4 per-layer input gate-times-input path",
        "candidate": "native BF16 gate output multiplied by BF16 per-layer input",
        "reference": "same gate and input multiplied in float32, then written to BF16",
        "comparison_scope": {
            "same_model_weights": True,
            "same_input_ids": True,
            "same_parameter_carriers": True,
            "layer": 0,
            "sequence_length": args.sequence_length,
        },
        "rows": rows,
        "summary": {
            "state_count": len(rows),
            "nonzero_loss_difference": sum(r["loss_difference_native_minus_fp32_product"] != 0.0 for r in rows),
            "nonzero_gate_gradient_difference": sum(r["gate_gradient_max_abs"] != 0.0 for r in rows),
            "nonzero_projection_gradient_difference": sum(
                r["projection_gradient_max_abs"] != 0.0 for r in rows
            ),
            "max_gate_write_rms": max(r["gate_write_rms_over_reference"] for r in rows),
            "max_projection_write_rms": max(r["projection_write_rms_over_reference"] for r in rows),
        },
        "claim_boundary": (
            "A component-level negative screen on one Gemma-4 checkpoint; exact identity under this "
            "declared path is not a universal claim and does not add a problem group."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/google/gemma-4-E2B"))
    parser.add_argument(
        "--input-bank",
        type=Path,
        default=Path("results/property/tcmp_allop_v1/input_banks/gemma4_e2b_text128.json"),
    )
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
