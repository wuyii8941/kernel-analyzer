#!/usr/bin/env python3
"""Analyze same-state optimizer ablations from an explicit raw replay capture.

The input must contain the candidate and repair gradients, updates, and
pre-step moments from the same state.  Missing vectors are an abstention, not
an invitation to reconstruct them from norms or digests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def subtract(left: list[float], right: list[float]) -> list[float]:
    return [float(a) - float(b) for a, b in zip(left, right)]


def scale(values: list[float], factor: float) -> list[float]:
    return [factor * float(value) for value in values]


def add(left: list[float], right: list[float]) -> list[float]:
    return [float(a) + float(b) for a, b in zip(left, right)]


def adam_step(
    gradient: list[float],
    first: list[float],
    second: list[float],
    step: int,
    *,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
) -> tuple[list[float], list[float], list[float]]:
    next_first = [beta1 * m + (1.0 - beta1) * g for m, g in zip(first, gradient)]
    next_second = [beta2 * v + (1.0 - beta2) * g * g for v, g in zip(second, gradient)]
    first_bias = 1.0 - beta1**step
    second_bias = 1.0 - beta2**step
    update = [
        -learning_rate * (m / first_bias) /
        (math.sqrt(v / second_bias) + epsilon)
        for m, v in zip(next_first, next_second)
    ]
    return update, next_first, next_second


def coherence(vectors: list[list[float]]) -> float:
    if not vectors or not vectors[0]:
        raise ValueError("empty vector sequence")
    total = [0.0] * len(vectors[0])
    energy = 0.0
    for vector in vectors:
        if len(vector) != len(total):
            raise ValueError("coordinate count changed within a sequence")
        total = add(total, vector)
        energy += sum(value * value for value in vector)
    denominator = math.sqrt(energy)
    return math.sqrt(sum(value * value for value in total)) / max(denominator, 1e-30)


def validate_capture(payload: dict[str, Any]) -> tuple[list[str], dict[str, list[list[float]]]]:
    errors: list[str] = []
    if payload.get("schema") != "kernel-analyzer-bound-endpoint-raw-stage-v1":
        errors.append("wrong raw-stage schema")
    if payload.get("status") != "COMPLETE":
        errors.append("raw capture is not a complete 32-step capture")
    state_ids = payload.get("state_ids")
    if not isinstance(state_ids, list) or len(state_ids) != 32 or len(set(state_ids)) != 32:
        errors.append("state_ids must contain 32 unique states")
    vectors = payload.get("vectors")
    required = {
        "candidate_gradient",
        "repair_gradient",
        "candidate_update",
        "repair_update",
        "candidate_first_moment_before_step",
        "candidate_second_moment_before_step",
        "repair_first_moment_before_step",
        "repair_second_moment_before_step",
    }
    if not isinstance(vectors, dict):
        errors.append("vectors object is missing")
        return errors, {}
    missing = sorted(required - set(vectors))
    if missing:
        errors.append("missing raw vectors: " + ",".join(missing))
    normalized: dict[str, list[list[float]]] = {}
    for name in required:
        value = vectors.get(name)
        if not isinstance(value, list) or len(value) != 32:
            errors.append(f"{name} must contain 32 vectors")
            continue
        if any(not isinstance(row, list) or not row for row in value):
            errors.append(f"{name} contains a malformed vector")
            continue
        normalized[name] = [[float(item) for item in row] for row in value]
    if normalized:
        coordinate_counts = {len(row) for rows in normalized.values() for row in rows}
        if len(coordinate_counts) != 1:
            errors.append("raw vector coordinate counts are inconsistent")
    return errors, normalized


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    errors, vectors = validate_capture(payload)
    if errors:
        return {
            "schema": "kernel-analyzer-direct-persistence-v4-optimizer-state-result-v1",
            "status": "ABSTAIN_MISSING_OR_MALFORMED_CAPTURE",
            "case_id": payload.get("case_id"),
            "errors": errors,
            "claim_boundary": "No optimizer conclusion is emitted from incomplete raw replay data.",
        }
    optimizer = payload.get("optimizer", {})
    learning_rate = float(optimizer.get("learning_rate", 0.0))
    beta1, beta2 = [float(value) for value in optimizer.get("betas", [0.9, 0.95])]
    epsilon = float(optimizer.get("epsilon", 1e-8))
    gradient_diff = [
        subtract(candidate, repair)
        for candidate, repair in zip(vectors["candidate_gradient"], vectors["repair_gradient"])
    ]
    captured_adamw = [
        subtract(candidate, repair)
        for candidate, repair in zip(vectors["candidate_update"], vectors["repair_update"])
    ]
    stateless_sgd = [scale(vector, -learning_rate) for vector in gradient_diff]
    moment_reset = []
    for step, (candidate, repair) in enumerate(
        zip(vectors["candidate_gradient"], vectors["repair_gradient"]), 1
    ):
        zeros = [0.0] * len(candidate)
        candidate_update, _, _ = adam_step(
            candidate, zeros, zeros, 1, learning_rate=learning_rate,
            beta1=beta1, beta2=beta2, epsilon=epsilon,
        )
        repair_update, _, _ = adam_step(
            repair, zeros, zeros, 1, learning_rate=learning_rate,
            beta1=beta1, beta2=beta2, epsilon=epsilon,
        )
        moment_reset.append(subtract(candidate_update, repair_update))
    return {
        "schema": "kernel-analyzer-direct-persistence-v4-optimizer-state-result-v1",
        "status": "COMPLETE_SAME_STATE_OPTIMIZER_ABLATION",
        "case_id": payload["case_id"],
        "state_ids": payload["state_ids"],
        "optimizer": optimizer,
        "arms": {
            "gradient_difference": {"A32": coherence(gradient_diff)},
            "stateless_sgd": {"A32": coherence(stateless_sgd)},
            "captured_adamw_moments": {"A32": coherence(captured_adamw)},
            "moment_reset_each_step": {"A32": coherence(moment_reset)},
        },
        "claim_boundary": (
            "This is a same-state optimizer response comparison. It does not "
            "represent a natural early/middle/late training phase unless the "
            "capture itself came from that phase."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(load(args.input))
    result["input"] = {
        "path": str(args.input),
        "sha256": sha256(args.input),
        "raw_vectors_are_external_to_repository": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "case_id": result.get("case_id")}, sort_keys=True))


if __name__ == "__main__":
    main()
