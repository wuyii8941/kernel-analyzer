#!/usr/bin/env python
"""Independent-reference numerical Oracle for a complete linear SGD step."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from dataclasses import dataclass, field
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = "forkcert.analytic-linear-step-oracle.v0.1"
DIMENSION = 257
LEARNING_RATE = Fraction(1, 32)
U = Decimal(2) ** Decimal(-24)


@dataclass
class CompileAudit:
    backend_compiles: int = 0
    runtime_invocations: int = 0
    graph_hashes: list[str] = field(default_factory=list)
    graph_nodes: list[int] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mode", choices=["correct", "reverse", "drop_last"], required=True)
    parser.add_argument("--states", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    return parser.parse_args()


def fraction_decimal(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def decimal_float(value: float) -> Decimal:
    return Decimal.from_float(float(value))


def gamma(count: int) -> Decimal:
    amount = Decimal(count) * U
    if amount >= 1:
        raise ValueError("rounding-error model requires count*u < 1")
    return amount / (Decimal(1) - amount)


def state_values(state: int) -> tuple[list[float], list[float], float]:
    weights = [(((53 * j + 97) % 997) - 498) / 2048.0 for j in range(DIMENSION)]
    features = [
        (((37 * j + 17 * state + 11) % 1009) - 504) / 4096.0
        for j in range(DIMENSION)
    ]
    target = (((71 * state + 29) % 509) - 254) / 1024.0
    return weights, features, target


def exact_fraction(value: float) -> Fraction:
    numerator, denominator = float(value).as_integer_ratio()
    return Fraction(numerator, denominator)


def exact_transition(weights: list[float], features: list[float], target: float) -> dict[str, Any]:
    w = [exact_fraction(value) for value in weights]
    x = [exact_fraction(value) for value in features]
    y = exact_fraction(target)
    products = [left * right for left, right in zip(w, x, strict=True)]
    prediction = sum(products, Fraction(0))
    residual = prediction - y
    loss = Fraction(1, 2) * residual * residual
    gradient = [residual * value for value in x]
    next_weight = [
        value - LEARNING_RATE * grad
        for value, grad in zip(w, gradient, strict=True)
    ]
    return {
        "prediction": prediction,
        "residual": residual,
        "target": y,
        "loss": loss,
        "gradient": gradient,
        "next_weight": next_weight,
        "sum_abs_products": sum((abs(value) for value in products), Fraction(0)),
    }


def analytic_bounds(exact: dict[str, Any], features: list[float], weights: list[float]) -> dict[str, Any]:
    prediction = abs(fraction_decimal(exact["prediction"]))
    residual = abs(fraction_decimal(exact["residual"]))
    target = abs(fraction_decimal(exact["target"]))
    sum_abs = fraction_decimal(exact["sum_abs_products"])
    prediction_error = gamma(2 * DIMENSION) * sum_abs
    residual_error = prediction_error + U * (prediction + prediction_error + target)
    residual_upper = residual + residual_error
    loss_error = (
        Decimal("0.5") * (Decimal(2) * residual * residual_error + residual_error * residual_error)
        + gamma(4) * Decimal("0.5") * residual_upper * residual_upper
    )
    gradient_error = []
    update_error = []
    lr = fraction_decimal(LEARNING_RATE)
    for feature, weight, exact_gradient in zip(features, weights, exact["gradient"], strict=True):
        x = abs(fraction_decimal(exact_fraction(feature)))
        w = abs(fraction_decimal(exact_fraction(weight)))
        gradient = abs(fraction_decimal(exact_gradient))
        g_error = x * residual_error + gamma(8) * x * residual_upper
        gradient_error.append(g_error)
        update_error.append(
            lr * g_error + gamma(6) * (w + lr * (gradient + g_error))
        )
    return {
        "prediction": prediction_error,
        "loss": loss_error,
        "gradient": gradient_error,
        "next_weight": update_error,
    }


def tracking_backend(torch: Any, audit: CompileAudit) -> Callable[..., Any]:
    from torch._dynamo.backends.registry import lookup_backend

    inductor = lookup_backend("inductor")

    def backend(graph_module: Any, example_inputs: list[Any]) -> Callable[..., Any]:
        audit.backend_compiles += 1
        audit.graph_hashes.append(hashlib.sha256(graph_module.code.encode()).hexdigest())
        audit.graph_nodes.append(sum(1 for _ in graph_module.graph.nodes))
        compiled = inductor(graph_module, example_inputs)

        def counted(*args: Any) -> Any:
            audit.runtime_invocations += 1
            return compiled(*args)

        return counted

    return backend


def optimizer_options(optimizer: Any) -> list[dict[str, Any]]:
    return [
        {
            key: value if isinstance(value, (bool, int, float, str, type(None))) else repr(value)
            for key, value in group.items()
            if key != "params"
        }
        for group in optimizer.param_groups
    ]


def error_record(observed: float, truth: Fraction, bound: Decimal) -> dict[str, Any]:
    error = abs(decimal_float(observed) - fraction_decimal(truth))
    return {
        "absolute_error": float(error),
        "bound": float(bound),
        "ratio": float(error / bound) if bound > 0 else (0.0 if error == 0 else math.inf),
        "within_bound": bool(error <= bound),
    }


def vector_error_records(observed: list[float], truth: list[Fraction], bounds: list[Decimal]) -> dict[str, Any]:
    records = [
        error_record(value, exact, bound)
        for value, exact, bound in zip(observed, truth, bounds, strict=True)
    ]
    return {
        "all_within_bound": all(item["within_bound"] for item in records),
        "max_absolute_error": max(item["absolute_error"] for item in records),
        "max_bound": max(item["bound"] for item in records),
        "max_ratio": max(item["ratio"] for item in records),
        "violating_indices_head": [
            index for index, item in enumerate(records) if not item["within_bound"]
        ][:20],
    }


def main() -> None:
    args = parse_args()
    if args.states < 1 or args.repeats < 2:
        raise ValueError("states must be positive and repeats must be at least two")
    getcontext().prec = 100
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU fallback")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch._dynamo.reset()
    torch._dynamo.config.suppress_errors = False

    initial_weights, _, _ = state_values(0)
    parameter = torch.nn.Parameter(torch.tensor(initial_weights, dtype=torch.float32, device="cuda"))

    def eager_core(features: Any, target: Any) -> tuple[Any, Any]:
        prediction = torch.sum(parameter * features)
        residual = prediction - target
        return residual * residual * 0.5, prediction

    def candidate_core(features: Any, target: Any) -> tuple[Any, Any]:
        terms = parameter * features
        if args.mode == "reverse":
            terms = torch.flip(terms, dims=(0,))
        elif args.mode == "drop_last":
            terms = terms[:-1]
        prediction = torch.sum(terms)
        residual = prediction - target
        return residual * residual * 0.5, prediction

    audit = CompileAudit()
    compiled_core = torch.compile(
        candidate_core, backend=tracking_backend(torch, audit), fullgraph=True, dynamic=False
    )
    baseline = parameter.detach().clone()

    def run_arm(function: Callable[..., Any], features: Any, target: Any, compiled: bool) -> dict[str, Any]:
        with torch.no_grad():
            parameter.copy_(baseline)
        parameter.grad = None
        optimizer = torch.optim.SGD(
            [parameter], lr=float(LEARNING_RATE), momentum=0.0, dampening=0.0,
            weight_decay=0.0, nesterov=False, maximize=False, foreach=False,
            differentiable=False, fused=False,
        )
        before = audit.runtime_invocations
        loss, prediction = function(features, target)
        loss.backward()
        gradient = parameter.grad.detach().cpu().tolist()
        options = optimizer_options(optimizer)
        optimizer.step()
        torch.cuda.synchronize()
        return {
            "prediction": float(prediction.detach().item()),
            "loss": float(loss.detach().item()),
            "gradient": gradient,
            "next_weight": parameter.detach().cpu().tolist(),
            "gradient_shape": list(parameter.grad.shape),
            "gradient_dtype": str(parameter.grad.dtype),
            "optimizer_options": options,
            "optimizer_state_empty": not bool(optimizer.state),
            "compiled_invocations": audit.runtime_invocations - before if compiled else 0,
        }

    warm_weights, warm_features, warm_target = state_values(0)
    if warm_weights != initial_weights:
        raise RuntimeError("weight generator unexpectedly depends on state")
    warm_x = torch.tensor(warm_features, dtype=torch.float32, device="cuda")
    warm_y = torch.tensor(warm_target, dtype=torch.float32, device="cuda")
    warm = run_arm(compiled_core, warm_x, warm_y, True)
    del warm

    rows = []
    with (out_dir / "states.jsonl").open("w", encoding="utf-8") as handle:
        for state in range(args.states):
            weights, features, target = state_values(state)
            exact = exact_transition(weights, features, target)
            bounds = analytic_bounds(exact, features, weights)
            x = torch.tensor(features, dtype=torch.float32, device="cuda")
            y = torch.tensor(target, dtype=torch.float32, device="cuda")
            for repeat in range(args.repeats):
                eager = run_arm(eager_core, x, y, False)
                candidate = run_arm(compiled_core, x, y, True)
                arm_records = {}
                for name, arm in (("eager", eager), ("candidate", candidate)):
                    prediction = error_record(arm["prediction"], exact["prediction"], bounds["prediction"])
                    loss = error_record(arm["loss"], exact["loss"], bounds["loss"])
                    gradient = vector_error_records(arm["gradient"], exact["gradient"], bounds["gradient"])
                    next_weight = vector_error_records(
                        arm["next_weight"], exact["next_weight"], bounds["next_weight"]
                    )
                    numerical_accept = (
                        prediction["within_bound"] and loss["within_bound"]
                        and gradient["all_within_bound"] and next_weight["all_within_bound"]
                    )
                    arm_records[name] = {
                        "prediction": prediction,
                        "loss": loss,
                        "gradient": gradient,
                        "next_weight": next_weight,
                        "numerical_verdict": "ACCEPT" if numerical_accept else "REJECT",
                    }
                candidate_identity = candidate["compiled_invocations"] == 1
                exact_structure = (
                    eager["gradient_shape"] == candidate["gradient_shape"] == [DIMENSION]
                    and eager["gradient_dtype"] == candidate["gradient_dtype"] == "torch.float32"
                    and eager["optimizer_options"] == candidate["optimizer_options"]
                    and eager["optimizer_state_empty"] and candidate["optimizer_state_empty"]
                )
                row = {
                    "state": state,
                    "repeat": repeat,
                    "mode": args.mode,
                    "candidate_identity_valid": candidate_identity,
                    "exact_structure_verdict": (
                        "INVALID" if not candidate_identity else "ACCEPT" if exact_structure else "REJECT"
                    ),
                    "arms": arm_records,
                }
                rows.append(row)
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["state"], []).append(row)
    repeat_stable = 0
    for state_rows in grouped.values():
        signatures = {
            json.dumps(row["arms"], sort_keys=True, allow_nan=False) for row in state_rows
        }
        repeat_stable += int(len(signatures) == 1)
    primary = [row for row in rows if row["repeat"] == 0]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "contract": str((Path(__file__).parent / "ANALYTIC_LINEAR_STEP_CONTRACT_V0_1_2026-07-17.md").resolve()),
        "mode": args.mode,
        "dimension": DIMENSION,
        "states": args.states,
        "repeats": args.repeats,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "compile_audit": {
            "backend_compiles": audit.backend_compiles,
            "runtime_invocations": audit.runtime_invocations,
            "graph_hashes": audit.graph_hashes,
            "graph_nodes": audit.graph_nodes,
        },
        "candidate_identity_all_valid": all(row["candidate_identity_valid"] for row in rows),
        "exact_structure_all_accept": all(row["exact_structure_verdict"] == "ACCEPT" for row in rows),
        "eager_numerical_accept_states": sum(
            row["arms"]["eager"]["numerical_verdict"] == "ACCEPT" for row in primary
        ),
        "candidate_numerical_accept_states": sum(
            row["arms"]["candidate"]["numerical_verdict"] == "ACCEPT" for row in primary
        ),
        "candidate_numerical_reject_states": sum(
            row["arms"]["candidate"]["numerical_verdict"] == "REJECT" for row in primary
        ),
        "repeat_stable_states": repeat_stable,
        "max_candidate_error_bound_ratio": max(
            max(
                row["arms"]["candidate"][field]["ratio" if field in ("prediction", "loss") else "max_ratio"]
                for field in ("prediction", "loss", "gradient", "next_weight")
            )
            for row in primary
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
