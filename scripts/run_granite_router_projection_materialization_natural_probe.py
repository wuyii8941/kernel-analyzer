#!/usr/bin/env python3
"""Probe Granite MoE router-score precision on real text states.

The native router computes its linear scores in model dtype and converts them
to float before top-k.  The reference computes the same linear scores in
float32.  The probe keeps the rest of the MoE block unchanged and records
whether the score change alters routing, router gradients, or one-step writes.
It is promoted only if the declared real-text bank gives a stable parameter
direction; a changed top-k set is reported rather than hidden.
"""

from __future__ import annotations

import argparse
import json
import types
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _windows(tokenizer: Any, source: Path, seq_len: int, count: int) -> list[torch.Tensor]:
    ids = tokenizer(source.read_text(encoding="utf-8"), add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    stride = 2 * seq_len
    required = count * stride + seq_len
    if ids.numel() < required:
        raise RuntimeError(f"text source has {ids.numel()} tokens, need {required}")
    return [ids[i * stride : i * stride + seq_len].clone() for i in range(count)]


def _reference_router(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    frozen_index: torch.Tensor | None = None,
):
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    scores = torch.nn.functional.linear(flat.float(), self.weight.float()).float()
    if frozen_index is None:
        top_scores, top_index = scores.topk(self.top_k, dim=-1)
    else:
        top_index = frozen_index.to(scores.device)
        top_scores = scores.gather(1, top_index)
    weights = torch.softmax(top_scores, dim=-1).type_as(hidden_states)
    return top_index, weights, scores


def _relative(delta: torch.Tensor, reference: torch.Tensor) -> float:
    return float(delta.norm()) / max(float(reference.norm()), 1e-30)


def run(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    bank = _windows(tokenizer, args.text_source, args.sequence_length, args.states)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16
    ).to(args.device).eval()
    model.config.use_cache = False
    router = model.model.layers[args.layer].block_sparse_moe.router
    target = router.weight
    native_forward = router.forward

    def run_once(
        ids: torch.Tensor,
        reference: bool,
        frozen_index: torch.Tensor | None = None,
    ) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
        if reference and frozen_index is None:
            router.forward = types.MethodType(_reference_router, router)
        elif reference:
            router.forward = types.MethodType(
                lambda self, hidden: _reference_router(self, hidden, frozen_index=frozen_index), router
            )
        else:
            router.forward = native_forward
        model.zero_grad(set_to_none=True)
        output = model(input_ids=ids, labels=ids, use_cache=False, return_dict=True)
        output.loss.backward()
        if target.grad is None:
            raise RuntimeError("router gradient is missing")
        with torch.no_grad():
            hidden = model.model.embed_tokens(ids).reshape(-1, model.config.hidden_size)
            if reference:
                top_index, top_weights, scores = _reference_router(
                    router, hidden, frozen_index=frozen_index
                )
            else:
                top_index, top_weights, scores = native_forward(hidden)
        return (
            float(output.loss.detach().cpu()),
            target.grad.detach().float().cpu().clone(),
            top_index.detach().cpu().clone(),
            top_weights.detach().float().cpu().clone(),
        )

    rows: list[dict[str, Any]] = []
    effects: list[torch.Tensor] = []
    writes: list[torch.Tensor] = []
    references: list[torch.Tensor] = []
    for state_id, tokens in enumerate(bank):
        ids = tokens.unsqueeze(0).to(args.device)
        native_loss, native_grad, native_index, native_weights = run_once(ids, False)
        reference_loss, reference_grad, reference_index, reference_weights = run_once(
            ids, True, native_index if args.freeze_selection else None
        )
        router.forward = native_forward
        gradient_effect = reference_grad - native_grad
        native_write = -args.learning_rate * native_grad
        reference_write = -args.learning_rate * reference_grad
        write_effect = reference_write - native_write
        selected_changed = not torch.equal(native_index, reference_index)
        effects.append(gradient_effect.reshape(-1).double())
        writes.append(write_effect.reshape(-1).double())
        references.append(native_write.reshape(-1).double())
        rows.append(
            {
                "state_id": state_id,
                "native_loss": native_loss,
                "reference_loss": reference_loss,
                "loss_difference_reference_minus_native": reference_loss - native_loss,
                "selected_expert_set_changed": selected_changed,
                "selected_token_count_changed": int((native_index != reference_index).sum()),
                "routing_weight_rms_relative": _relative(reference_weights - native_weights, native_weights),
                "router_gradient_rms_relative": _relative(gradient_effect, native_grad),
                "router_write_rms_relative": _relative(write_effect, native_write),
            }
        )

    split = args.states // 2
    direction = torch.stack(effects[:split]).mean(dim=0)
    direction_norm = float(direction.norm())
    projections: list[float] = []
    if direction_norm > 0:
        direction = direction / direction_norm
        projections = [float(torch.dot(effect, direction)) for effect in effects[split:]]
    aligned = [
        float(torch.dot(effect, reference)) / max(float(torch.dot(reference, reference)), 1e-30)
        for effect, reference in zip(writes[split:], references[split:])
    ]
    return {
        "schema": "kernel-analyzer-granite-router-projection-materialization-natural-probe-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "layer": args.layer,
        "parameter": f"model.layers.{args.layer}.block_sparse_moe.router.weight",
        "operator": "Granite MoE router score projection",
        "candidate": "native model-dtype router linear followed by float conversion",
        "reference": "same router linear evaluated in float32 before top-k",
        "input_source": str(args.text_source),
        "comparison_scope": {
            "same_model_weights": True,
            "same_input_ids": True,
            "single_changed_boundary": "router score linear projection",
            "expert_body_unchanged": True,
            "freeze_native_top_k_selection": args.freeze_selection,
        },
        "rows": rows,
        "summary": {
            "state_count": len(rows),
            "calibration_count": split,
            "confirmation_count": len(rows) - split,
            "selected_set_changed_states": sum(row["selected_expert_set_changed"] for row in rows),
            "mean_router_write_rms_relative": sum(row["router_write_rms_relative"] for row in rows) / len(rows),
            "confirmation_projection_positive": sum(value > 0 for value in projections),
            "confirmation_projection_negative": sum(value < 0 for value in projections),
            "confirmation_projection_values": projections,
            "confirmation_aligned_write_values": aligned,
            "direction_norm": direction_norm,
        },
        "claim_boundary": (
            "One Granite MoE checkpoint and the declared real text bank. A changed routing set is "
            "part of the observed router effect, not evidence of a pure arithmetic-only bias. "
            "No population or long-run loss claim is made."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/ibm-granite/granite-3.1-1b-a400m-base"))
    parser.add_argument("--text-source", type=Path, default=Path("docs/root_cause_closure_current.md"))
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--freeze-selection", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
