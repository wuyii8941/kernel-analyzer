#!/usr/bin/env python3
"""Two-pass exact full-parameter T3 confirmation for the seq128 lm_head dX VJP."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "archive/round1_code/src"))

from forkcert.analytic_local_vjp_reference import analytic_local_vjp
from forkcert.autograd_local_vjp_intervention import (
    run_analytic_local_vjp_edge_intervention,
    validate_analytic_local_vjp_edge_intervention,
)


CHUNK = 50_000_000


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def square(value: Any) -> float:
    import torch
    flat = value.reshape(-1)
    total = 0.0
    for start in range(0, flat.numel(), CHUNK):
        part = flat[start:start + CHUNK].to(torch.float64)
        total += float((part * part).sum())
    return total


def dot(left: Any, right: Any) -> float:
    import torch
    a, b = left.reshape(-1), right.reshape(-1)
    total = 0.0
    for start in range(0, a.numel(), CHUNK):
        av = a[start:start + CHUNK].to(torch.float64)
        bv = b[start:start + CHUNK].to(torch.float64)
        total += float((av * bv).sum())
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=ROOT / "results/coverage/lmhead_t3_confirmation_design.json")
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=ROOT / "results/coverage/lmhead_t3_confirmation.json")
    args = parser.parse_args()
    design = json.loads(args.design.read_text())
    hypothesis = design["hypothesis"]
    if design["status"] != "FROZEN_BEFORE_CANDIDATE_EXECUTION" or len(design["records"]) != 32:
        raise RuntimeError("lm_head confirmation design is not frozen and complete")

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True
    ).to(device).eval()
    model.config.use_cache = False
    parameter_names = [name for name, _ in model.named_parameters()]
    lm_head_outputs: list[Any] = []

    def capture_lm_head(_module: Any, _inputs: Any, output: Any) -> None:
        lm_head_outputs.append(output)

    origin_handle = model.lm_head.register_forward_hook(capture_lm_head)
    sums: dict[str, Any] = {}
    state_squares: list[float] = []
    first_pass_rows = []
    started = time.monotonic()

    def measure(state: dict[str, Any], accumulate: bool) -> tuple[float, float, dict[str, float]]:
        values = torch.tensor([state["input_ids"]], dtype=torch.long, device=device)

        def loss_closure():
            return model(input_ids=values, labels=values, use_cache=False, return_dict=True).loss

        model.zero_grad(set_to_none=True)
        baseline_loss = loss_closure()
        baseline_value = baseline_loss.detach().clone()
        baseline_loss.backward()
        baseline = {
            name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
            for name, parameter in model.named_parameters()
        }
        model.zero_grad(set_to_none=True)
        def resolve_lm_head_mm(_loss: Any) -> Any:
            if not lm_head_outputs:
                raise RuntimeError("lm_head forward origin was not captured")
            root = lm_head_outputs[-1].grad_fn
            level = [root]
            seen = set()
            while level:
                matches = []
                next_level = []
                for node in level:
                    if node is None or id(node) in seen:
                        continue
                    seen.add(id(node))
                    name = str(node.name()) if hasattr(node, "name") else type(node).__name__
                    if name == hypothesis["runtime_autograd_node"]:
                        matches.append(node)
                    next_level.extend(
                        value for value, _index in getattr(node, "next_functions", ())
                        if value is not None
                    )
                if matches:
                    if len(matches) != 1:
                        raise RuntimeError(
                            f"lm_head forward origin has {len(matches)} nearest MmBackward nodes"
                        )
                    return matches[0]
                level = next_level
            raise RuntimeError("lm_head forward origin has no MmBackward node")

        lm_head_outputs.clear()
        repaired_loss, intervention = run_analytic_local_vjp_edge_intervention(
            loss_closure=loss_closure,
            target_sequence_nr=None,
            target_node_resolver=resolve_lm_head_mm,
            target_tuple_index=int(hypothesis["tuple_index"]),
            allowed_runtime_autograd_nodes=(hypothesis["runtime_autograd_node"],),
            analytic_vjp_evaluator=analytic_local_vjp,
        )
        validate_analytic_local_vjp_edge_intervention(intervention)
        if not torch.equal(baseline_value, repaired_loss.detach()):
            raise RuntimeError("backward-only repair changed loss")
        total_square = 0.0
        total_dot = 0.0
        changed_parameters = 0
        for name, parameter in model.named_parameters():
            before = baseline[name]
            after = None if parameter.grad is None else parameter.grad.detach().cpu()
            if before is None or after is None:
                if before is not None or after is not None:
                    raise RuntimeError("gradient None pattern changed")
                continue
            delta = after.float() - before.float()
            value_square = square(delta)
            total_square += value_square
            changed_parameters += value_square > 0
            if accumulate:
                if name not in sums:
                    sums[name] = delta.clone()
                else:
                    sums[name].add_(delta)
            else:
                total_dot += dot(delta, sums[name])
            del delta
        local = intervention["local_edge"]["replacement_minus_actual"]
        if local["max_abs"] <= 0 or total_square <= 0:
            raise RuntimeError("zero local repair or carrier")
        model.zero_grad(set_to_none=True)
        del baseline, baseline_loss, repaired_loss, values
        gc.collect()
        torch.cuda.empty_cache()
        return total_square, total_dot, {
            "local_max_abs": float(local["max_abs"]),
            "local_rms": float(local["rms"]),
            "changed_parameters": changed_parameters,
        }

    for index, state in enumerate(design["records"]):
        state_square, _, local = measure(state, True)
        state_squares.append(state_square)
        first_pass_rows.append({
            "state_id": state["sequence_id"], "record_sha256": state["record_sha256"],
            "carrier_l2": math.sqrt(state_square), **local,
        })
        print(json.dumps({"pass": 1, "state": index + 1, "states": 32}), flush=True)

    sum_square = sum(square(value) for value in sums.values())
    square_sum = sum(state_squares)
    count = len(state_squares)
    u_value = (sum_square - square_sum) / (count * (count - 1))
    dots = []
    second_pass_rows = []
    for index, state in enumerate(design["records"]):
        repeated_square, state_dot_sum, local = measure(state, False)
        if not math.isclose(repeated_square, state_squares[index], rel_tol=0.0, abs_tol=0.0):
            raise RuntimeError(f"carrier repeat changed for {state['sequence_id']}")
        if local != {key: first_pass_rows[index][key] for key in local}:
            raise RuntimeError(f"local repeat changed for {state['sequence_id']}")
        dots.append(state_dot_sum)
        second_pass_rows.append({"state_id": state["sequence_id"], "dot_with_global_sum": state_dot_sum})
        print(json.dumps({"pass": 2, "state": index + 1, "states": 32}), flush=True)

    leave_one = []
    for index in range(count):
        reduced_sum_square = sum_square - 2.0 * dots[index] + state_squares[index]
        reduced_square_sum = square_sum - state_squares[index]
        leave_one.append(
            (reduced_sum_square - reduced_square_sum) / ((count - 1) * (count - 2))
        )
    pseudovalues = np.asarray([
        count * u_value - (count - 1) * value for value in leave_one
    ], dtype=np.float64)
    rng = np.random.default_rng(3454)
    bootstrap = np.asarray([
        rng.choice(pseudovalues, size=count, replace=True).mean() for _ in range(10_000)
    ])
    lower, median, upper = np.quantile(bootstrap, [0.025, 0.5, 0.975])
    passed = float(lower) > 0
    payload = {
        "schema": "kernel-analyzer-lmhead-t3-full-carrier-confirmation-v1",
        "status": "COMPLETE_TWO_PASS_FULL_CARRIER_CONFIRMATION",
        "design_sha256": design["design_sha256"],
        "states": count,
        "passes": 2,
        "reachable_parameters": len(parameter_names),
        "all_parameter_coordinates_used": True,
        "per_state_tensors_saved": False,
        "repeat_exact": True,
        "cross_state_full_vector_u": u_value,
        "cluster_pseudovalue_bootstrap_95": {
            "method": "LEAVE_ONE_STATE_U_PSEUDOVALUE_CLUSTER_BOOTSTRAP",
            "draws": 10_000, "seed": 3454,
            "lower_95": float(lower), "median": float(median), "upper_95": float(upper),
        },
        "multiplicity_family": hypothesis["multiplicity_family"],
        "frozen_success_gate_passed": passed,
        "verdict": "T3_COHERENT_PASS" if passed else "CAUSAL_NONCOHERENT",
        "state_rows": first_pass_rows,
        "second_pass_dot_rows": second_pass_rows,
        "resource_usage": {"wall_time_seconds": time.monotonic() - started},
        "claim_boundary": "Two exact executions per frozen state; complete 310-parameter carrier, no coordinate selection, no per-state tensors retained. This is a T3 certificate only.",
    }
    payload["result_sha256"] = digest(payload)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    origin_handle.remove()
    print(json.dumps({"output": str(args.output), "verdict": payload["verdict"], "lower_95": float(lower)}))


if __name__ == "__main__":
    main()
