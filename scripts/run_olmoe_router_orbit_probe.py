#!/usr/bin/env python3
"""Strict eager F+B orbit probe for the OLMoE expert accumulation order.

The orbit changes only the order in which expert contributions are added to
the same destination rows.  It preserves the real-valued mathematical sum,
routing indices, routing weights, model parameters, and loss.  This is an
engineering/formation probe: it does not advance weights or issue a
trajectory verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM

from run_olmoe_router_accum_probe import (
    capture_gradients,
    patch_target,
)


def digest_tensor(t: torch.Tensor) -> str:
    x = t.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(str(x.dtype).encode())
    h.update(repr(tuple(x.shape)).encode())
    h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def flatten_difference(
    baseline: dict[str, torch.Tensor],
    current: dict[str, torch.Tensor],
) -> torch.Tensor:
    if set(baseline) != set(current):
        raise RuntimeError("orbit gradient reach differs")
    return torch.cat([(current[name] - baseline[name]).reshape(-1) for name in baseline])


def run_variant(
    model: torch.nn.Module,
    ids: torch.Tensor,
    layer: int,
    prefixes: tuple[str, ...],
    expert_order: list[int],
) -> tuple[torch.Tensor, dict[str, Any], dict[str, torch.Tensor]]:
    trace: dict[str, Any] = {"expert_order": expert_order}
    patch_target(model, layer, repair=False, trace=trace, expert_order=expert_order)
    model.zero_grad(set_to_none=True)
    out = model(input_ids=ids, labels=ids, use_cache=False, return_dict=False)
    loss = out[0]
    loss.backward()
    return loss.detach(), trace, capture_gradients(model, prefixes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--input-bank", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--states", type=int, default=4)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--variants", type=int, default=4)
    ap.add_argument(
        "--scope",
        choices=("router_gate", "layer_attn", "layer_norm", "final_norm"),
        default="router_gate",
    )
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--sign-flip-draws", type=int, default=4000)
    ap.add_argument("--save-state-means", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.states < 2 or args.variants < 2:
        raise ValueError("orbit probe needs at least two states and variants")
    bank = json.loads(args.input_bank.read_text())
    states = bank.get("states", bank.get("records"))
    if args.start < 0 or args.start + args.states > len(states):
        raise ValueError("input bank shorter than requested range")
    torch.manual_seed(24000)
    torch.cuda.manual_seed_all(24000)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True
    ).to(device).train()
    model.config.use_cache = False
    if not 0 <= args.layer < len(model.model.layers):
        raise ValueError("layer outside model")
    if args.scope == "router_gate":
        prefixes = (f"model.layers.{args.layer}.mlp.gate.",)
    elif args.scope == "layer_attn":
        prefixes = (f"model.layers.{args.layer}.self_attn.",)
    elif args.scope == "layer_norm":
        prefixes = (f"model.layers.{args.layer}.input_layernorm.",)
    else:
        prefixes = ("model.norm.",)
    num_experts = len(model.model.layers[args.layer].mlp.experts)
    permutations: list[list[int]] = []
    for variant in range(args.variants):
        order = list(range(num_experts))
        if variant:
            random.Random(args.seed + variant).shuffle(order)
        permutations.append(order)

    rows: list[dict[str, Any]] = []
    state_vectors: list[torch.Tensor] = []
    for state in states[args.start : args.start + args.states]:
        ids = torch.tensor(
            [state.get("token_ids", state.get("input_ids"))],
            dtype=torch.long,
            device=device,
        )
        losses: list[float] = []
        traces: list[dict[str, Any]] = []
        grads: list[dict[str, torch.Tensor]] = []
        for order in permutations:
            loss, trace, gradient = run_variant(
                model, ids, args.layer, prefixes, order
            )
            losses.append(float(loss.cpu()))
            traces.append(trace)
            grads.append(gradient)
        baseline = grads[0]
        deltas = torch.stack([flatten_difference(baseline, gradient) for gradient in grads[1:]])
        state_vectors.append(deltas)
        routing_same = all(
            trace["selected_experts"] == traces[0]["selected_experts"]
            for trace in traces
        )
        weight_same = all(
            trace["routing_weights_digest"] == traces[0]["routing_weights_digest"]
            for trace in traces
        )
        rows.append({
            "state_id": state.get("state_id", state.get("sequence_id")),
            "losses": losses,
            "loss_range": max(losses) - min(losses),
            "routing_same": routing_same,
            "routing_weights_same": weight_same,
            "variant_delta_norms": [float(x.norm()) for x in deltas],
            "variant_count": args.variants,
        })
        del ids, grads, deltas
        torch.cuda.empty_cache()

    # [states, non-default variants, coordinates].  The non-default orbit
    # mean is computed from disjoint halves to avoid plug-in self-correlation.
    cube = torch.stack(state_vectors).double()
    half = cube.shape[1] // 2
    mean_a = cube[:, :half].mean(dim=1)
    mean_b = cube[:, half:].mean(dim=1)
    all_mean = cube.mean(dim=1)
    total = all_mean.sum(dim=0)
    energy = float((all_mean * all_mean).sum())
    observed_norm = float(total.norm())
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    signs = torch.where(
        torch.randint(0, 2, (args.sign_flip_draws, all_mean.shape[0]), generator=rng) == 0,
        -torch.ones((), dtype=torch.float64),
        torch.ones((), dtype=torch.float64),
    )
    null_norms = torch.linalg.vector_norm(signs @ all_mean.cpu(), dim=1)
    sign_flip_p = float((1 + int((null_norms >= observed_norm).sum())) / (args.sign_flip_draws + 1))
    cross_numerator = float((mean_a.sum(dim=0) * mean_b.sum(dim=0)).sum())
    cross_denominator = float((mean_a * mean_b).sum())
    geometry = {
        "scope": args.scope,
        "parameter_prefixes": list(prefixes),
        "state_count": int(cube.shape[0]),
        "nondefault_orbit_count": int(cube.shape[1]),
        "coordinate_count": int(cube.shape[2]),
        "coherence_amplification": float(total.norm() / (energy ** 0.5 + 1e-30)),
        "crossfit_mean_numerator": cross_numerator,
        "crossfit_mean_denominator": cross_denominator,
        "crossfit_mean_amplification": cross_numerator / (cross_denominator + 1e-30),
        "mean_pair_dot_across_states": float(
            ((all_mean @ all_mean.T).sum() - (all_mean * all_mean).sum())
            / max(cube.shape[0] * (cube.shape[0] - 1), 1)
        ),
        "variant_order_seed": args.seed,
        "default_order_excluded_from_mean": True,
        "mathematical_semantics_preserved": True,
        "sign_flip_null": {
            "draws": args.sign_flip_draws,
            "seed": args.seed,
            "observed_sum_norm": observed_norm,
            "null_mean_norm": float(null_norms.mean()),
            "null_95_percentile": float(torch.quantile(null_norms, 0.95)),
            "one_sided_p": sign_flip_p,
            "above_95": bool(observed_norm > float(torch.quantile(null_norms, 0.95))),
        },
    }
    if args.save_state_means:
        geometry["state_mean_vectors"] = all_mean.tolist()
    payload = {
        "schema": "kernel-analyzer-olmoe-router-orbit-probe-v1",
        "status": "ENGINEERING_ONLY",
        "claim_boundary": "same-real-semantics expert-order orbit; formation screen only; no trajectory verdict",
        "model": str(args.model.resolve()),
        "input_bank": str(args.input_bank.resolve()),
        "layer": args.layer,
        "state_start": args.start,
        "state_count": args.states,
        "orbit_variants": args.variants,
        "candidate": "native BF16 expert index_add_ with expert-loop order variant",
        "repair": "none; each non-default orbit variant is compared to default BF16",
        "rows": rows,
        "all_routing_same": all(row["routing_same"] for row in rows),
        "all_routing_weights_same": all(row["routing_weights_same"] for row in rows),
        "formation_geometry": geometry,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
