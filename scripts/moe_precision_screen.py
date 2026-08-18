#!/usr/bin/env python3
"""Screen Granite MoE routing and router-gradient bias in full LM F+B."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[1]


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/ibm-granite/granite-3.1-1b-a400m-base"))
    parser.add_argument("--input-bank", type=Path, default=ROOT / "results/moe/input_bank.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/moe/precision_screen.json")
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--calibration-states", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def router_modules(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    found = {
        name: module
        for name, module in model.named_modules()
        if type(module).__name__ == "GraniteMoeTopKGating"
    }
    if not found:
        raise RuntimeError("no GraniteMoeTopKGating modules")
    return found


def load_model(model_path: Path, dtype: torch.dtype, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        local_files_only=True,
        attn_implementation="eager",
    ).to(device).train()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    routers = router_modules(model)
    for module in routers.values():
        module.layer.weight.requires_grad_(True)
    return model, routers


def run_arm(model, routers, input_ids, rng_seed: int):
    torch.manual_seed(rng_seed)
    torch.cuda.manual_seed_all(rng_seed)
    captured: dict[str, dict[str, torch.Tensor]] = {}
    handles = []
    for name, module in routers.items():
        def hook(_module, _inputs, output, *, key=name):
            logits = output[-1].detach().float()
            top = torch.topk(logits, _module.top_k + 1, dim=-1)
            captured[key] = {
                "selected": top.indices[:, : _module.top_k].cpu(),
                "boundary_margin": (top.values[:, _module.top_k - 1] - top.values[:, _module.top_k]).cpu(),
            }
        handles.append(module.register_forward_hook(hook))
    model.zero_grad(set_to_none=True)
    loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
    loss.backward()
    for handle in handles:
        handle.remove()
    gradients = {
        name: module.layer.weight.grad.detach().float().cpu().clone()
        for name, module in routers.items()
    }
    if set(captured) != set(routers):
        raise RuntimeError("router capture incomplete")
    return float(loss.detach().cpu()), gradients, captured


def route_difference(left: dict[str, dict[str, torch.Tensor]], right: dict[str, dict[str, torch.Tensor]]):
    flipped = 0
    denominator = 0
    for name in left:
        left_ids = torch.sort(left[name]["selected"], dim=-1).values
        right_ids = torch.sort(right[name]["selected"], dim=-1).values
        token_flip = torch.any(left_ids != right_ids, dim=-1)
        flipped += int(token_flip.sum())
        denominator += token_flip.numel()
    return flipped, denominator


def main() -> None:
    args = parse_args()
    bank = json.loads(args.input_bank.read_text())
    if args.states > len(bank["states"]):
        raise RuntimeError("input bank too small")
    state_ids = list(range(args.states))

    reference_gradients = {}
    reference_routes = {}
    reference_losses = {}
    reference_repeat_gate = None
    model, routers = load_model(args.model, torch.float32, args.device)
    router_names = sorted(routers)
    for state_id in state_ids:
        row = bank["states"][state_id]
        ids_cpu = torch.tensor(row["token_ids"], dtype=torch.long)
        if hashlib.sha256(ids_cpu.numpy().tobytes()).hexdigest() != row["token_sha256"]:
            raise RuntimeError(f"token digest mismatch: {state_id}")
        loss, gradients, routes = run_arm(
            model, routers, ids_cpu.unsqueeze(0).to(args.device), 17000 + state_id
        )
        reference_losses[state_id] = loss
        reference_gradients[state_id] = gradients
        reference_routes[state_id] = routes
        if state_id == 0:
            repeat_loss, repeat_gradients, repeat_routes = run_arm(
                model, routers, ids_cpu.unsqueeze(0).to(args.device), 17000 + state_id
            )
            repeat_flips, repeat_denominator = route_difference(routes, repeat_routes)
            reference_repeat_gate = {
                "loss_exact": repeat_loss == loss,
                "loss_delta": repeat_loss - loss,
                "router_gradients_bitwise_exact": all(
                    torch.equal(gradients[name], repeat_gradients[name]) for name in router_names
                ),
                "router_selections_exact": repeat_flips == 0,
                "routing_repeat_flipped_tokens": repeat_flips,
                "routing_repeat_denominator": repeat_denominator,
            }
        torch.cuda.empty_cache()
    del model, routers
    torch.cuda.empty_cache()

    candidate_gradients = {}
    candidate_routes = {}
    candidate_losses = {}
    candidate_repeat_gradients = {}
    candidate_repeat_routes = {}
    candidate_repeat_losses = {}
    model, routers = load_model(args.model, torch.bfloat16, args.device)
    for state_id in state_ids:
        row = bank["states"][state_id]
        ids_cpu = torch.tensor(row["token_ids"], dtype=torch.long)
        loss, gradients, routes = run_arm(
            model, routers, ids_cpu.unsqueeze(0).to(args.device), 17000 + state_id
        )
        candidate_losses[state_id] = loss
        candidate_gradients[state_id] = gradients
        candidate_routes[state_id] = routes
        repeat_loss, repeat_gradients, repeat_routes = run_arm(
            model, routers, ids_cpu.unsqueeze(0).to(args.device), 17000 + state_id
        )
        candidate_repeat_losses[state_id] = repeat_loss
        candidate_repeat_gradients[state_id] = repeat_gradients
        candidate_repeat_routes[state_id] = repeat_routes
        torch.cuda.empty_cache()

    deltas = {
        state_id: {
            name: (
                candidate_gradients[state_id][name].double()
                + candidate_repeat_gradients[state_id][name].double()
            ) / 2 - reference_gradients[state_id][name].double()
            for name in router_names
        }
        for state_id in state_ids
    }
    runtime_deltas = {
        state_id: {
            name: candidate_repeat_gradients[state_id][name].double()
            - candidate_gradients[state_id][name].double()
            for name in router_names
        }
        for state_id in state_ids
    }
    state_rows = []
    for state_id in state_ids:
        total_tokens = 0
        flipped_tokens = 0
        flipped_assignments = 0
        margins = []
        flipped_margins = []
        layer_flips = []
        runtime_flipped_tokens, runtime_token_denominator = route_difference(
            candidate_routes[state_id], candidate_repeat_routes[state_id]
        )
        for name in router_names:
            ref = torch.sort(reference_routes[state_id][name]["selected"], dim=-1).values
            cand = torch.sort(candidate_routes[state_id][name]["selected"], dim=-1).values
            token_flip = torch.any(ref != cand, dim=-1)
            assignment_difference = torch.sum(ref != cand).item()
            margin = reference_routes[state_id][name]["boundary_margin"]
            total_tokens += token_flip.numel()
            flipped_tokens += int(token_flip.sum())
            flipped_assignments += int(assignment_difference)
            margins.append(margin)
            if token_flip.any():
                flipped_margins.append(margin[token_flip])
            layer_flips.append({"router": name, "flipped_tokens": int(token_flip.sum()), "tokens": token_flip.numel()})
        all_margins = torch.cat(margins)
        flip_margin = torch.cat(flipped_margins) if flipped_margins else torch.empty(0)
        signal_l2 = torch.sqrt(sum(torch.sum(value ** 2) for value in deltas[state_id].values()))
        runtime_l2 = torch.sqrt(sum(torch.sum(value ** 2) for value in runtime_deltas[state_id].values()))
        state_rows.append({
            "state_id": state_id,
            "split": "CALIBRATION" if state_id < args.calibration_states else "HELDOUT",
            "token_sha256": bank["states"][state_id]["token_sha256"],
            "fp32_loss": reference_losses[state_id],
            "bf16_loss_repeats": [candidate_losses[state_id], candidate_repeat_losses[state_id]],
            "bf16_mean_loss_delta": (
                candidate_losses[state_id] + candidate_repeat_losses[state_id]
            ) / 2 - reference_losses[state_id],
            "bf16_runtime_loss_delta": candidate_repeat_losses[state_id] - candidate_losses[state_id],
            "router_gradient_delta_l2": float(signal_l2),
            "router_gradient_runtime_delta_l2": float(runtime_l2),
            "runtime_over_precision_signal_l2": float(runtime_l2 / signal_l2) if signal_l2 else None,
            "routing_token_denominator": total_tokens,
            "routing_flipped_tokens": flipped_tokens,
            "routing_flip_rate": flipped_tokens / total_tokens,
            "bf16_repeat_routing_flipped_tokens": runtime_flipped_tokens,
            "bf16_repeat_routing_denominator": runtime_token_denominator,
            "bf16_repeat_routing_flip_rate": runtime_flipped_tokens / runtime_token_denominator,
            "sorted_assignment_difference_count": flipped_assignments,
            "fp32_boundary_margin_median": float(torch.quantile(all_margins, 0.5)),
            "fp32_boundary_margin_p01": float(torch.quantile(all_margins, 0.01)),
            "flipped_token_margin_median": float(torch.quantile(flip_margin, 0.5)) if len(flip_margin) else None,
            "layer_flips": layer_flips,
        })

    parameter_rows = []
    for name in router_names:
        raw = sum(deltas[i][name] for i in range(args.calibration_states))
        norm = torch.linalg.vector_norm(raw)
        if norm == 0:
            continue
        direction = raw / norm
        projections = [float(torch.sum(deltas[i][name] * direction)) for i in state_ids]
        heldout = projections[args.calibration_states :]
        runtime_projections = [
            float(torch.sum(runtime_deltas[i][name] * direction)) for i in state_ids
        ]
        heldout_mean = sum(heldout) / len(heldout)
        max_runtime = max(abs(value) for value in runtime_projections)
        parameter_rows.append({
            "parameter": f"{name}.layer.weight",
            "numel": direction.numel(),
            "calibration_projections": projections[: args.calibration_states],
            "heldout_projections": heldout,
            "heldout_positive": sum(value > 0 for value in heldout),
            "heldout_negative": sum(value < 0 for value in heldout),
            "heldout_mean": heldout_mean,
            "max_abs_runtime_projection": max_runtime,
            "runtime_projection_over_abs_heldout_mean": (
                max_runtime / abs(heldout_mean) if heldout_mean else None
            ),
            "persistent_positive": all(value > 0 for value in heldout),
            "persistent_negative": all(value < 0 for value in heldout),
        })
    parameter_rows.sort(key=lambda row: abs(row["heldout_mean"]), reverse=True)
    output = {
        "schema": "kernel-analyzer-granite-moe-precision-routing-screen-v1",
        "status": "COMPLETE",
        "model": str(args.model),
        "reference": "FP32 eager full causal-LM F+B",
        "candidate": "BF16 eager full causal-LM F+B",
        "precision_is_primary_intervention": True,
        "dropout_rng_common_random_numbers": True,
        "seq_len": bank["seq_len"],
        "states": state_ids,
        "calibration_states": list(range(args.calibration_states)),
        "heldout_states": list(range(args.calibration_states, args.states)),
        "router_parameter_denominator": len(router_names),
        "reference_repeat_gate": reference_repeat_gate,
        "state_rows": state_rows,
        "parameter_rows": parameter_rows,
        "persistent_positive_count": sum(row["persistent_positive"] for row in parameter_rows),
        "persistent_negative_count": sum(row["persistent_negative"] for row in parameter_rows),
        "tensor_values_saved": False,
        "claim_boundary": "SCREEN_ONLY; ROUTING_FLIPS_AND_DIRECTIONS_REQUIRE_INDEPENDENT_CONFIRMATION_AND_SAME_INPUT_ROUTER_INTERVENTION",
    }
    output["result_sha256"] = canonical_hash(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "persistent_positive": output["persistent_positive_count"],
        "persistent_negative": output["persistent_negative_count"],
        "flip_rates": [row["routing_flip_rate"] for row in state_rows],
        "top": parameter_rows[:5],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
