#!/usr/bin/env python
"""Conformance controls for production/mediation evidence.

These controls are deliberately independent of TorchInductor.  They calibrate
the evidence logic before it is used on a compiler artifact: a seeded local
producer, a benign numerical change, a propagation-only change, and a no-op.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import torch

from forkcert.operator_evidence import production_mediation_interpretation


def endpoint(value: torch.Tensor) -> torch.Tensor:
    return value.sum(dim=-1)


def event(value: torch.Tensor) -> torch.Tensor:
    return endpoint(value) > 0


def contrast(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    difference = (right - left).abs()
    left_event, right_event = event(left), event(right)
    upward = (~left_event) & right_event
    downward = left_event & (~right_event)
    return {
        "max_abs": float(difference.max().item()),
        "nonzero": int((difference != 0).sum().item()),
        "off_to_on": int(upward.sum().item()),
        "on_to_off": int(downward.sum().item()),
        "disagreement": int((upward | downward).sum().item()),
    }


def run_case(
    name: str,
    reference_region: Callable[[torch.Tensor], torch.Tensor],
    candidate_region: Callable[[torch.Tensor], torch.Tensor],
    reference_boundary: torch.Tensor,
    candidate_boundary: torch.Tensor,
) -> dict[str, Any]:
    # Production uses the same input to both regions.
    ref_local = reference_region(reference_boundary)
    cand_local = candidate_region(reference_boundary)
    production = bool(not torch.equal(ref_local, cand_local))
    # Mediation uses a fixed suffix and only changes the boundary value.
    ref_endpoint = endpoint(reference_boundary)
    candidate_endpoint = endpoint(candidate_boundary)
    mediation = bool(not torch.equal(event(ref_endpoint), event(candidate_endpoint)))
    row = {
        "name": name,
        "production_observed": production,
        "mediation_observed": mediation,
        "local_continuous": contrast(ref_local, cand_local),
        "boundary_endpoint": contrast(ref_endpoint, candidate_endpoint),
        "interpretation": production_mediation_interpretation(production, mediation),
    }
    return row


def calibration_rows() -> list[dict[str, Any]]:
    base = torch.tensor([[0.10, -0.05], [-0.20, 0.10]], dtype=torch.float64)

    def identity(value: torch.Tensor) -> torch.Tensor:
        return value

    def producer_fault(value: torch.Tensor) -> torch.Tensor:
        output = value.clone()
        output[1, 0] += 0.20
        return output

    def benign_mutation(value: torch.Tensor) -> torch.Tensor:
        return value + 1e-8

    def same_region(value: torch.Tensor) -> torch.Tensor:
        return value

    propagated_boundary = base.clone()
    propagated_boundary[1, 0] += 0.20
    return [
        run_case("seeded_local_producer", identity, producer_fault, base, producer_fault(base)),
        run_case("benign_continuous_mutation", identity, benign_mutation, base, benign_mutation(base)),
        run_case("propagation_only", same_region, same_region, base, propagated_boundary),
        run_case("no_op", identity, identity, base, base),
    ]


EXPECTED = {
    "seeded_local_producer": (True, True),
    "benign_continuous_mutation": (True, False),
    "propagation_only": (False, True),
    "no_op": (False, False),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = calibration_rows()
    errors = []
    for row in rows:
        expected = EXPECTED[row["name"]]
        observed = (row["production_observed"], row["mediation_observed"])
        if observed != expected:
            errors.append(f"{row['name']}: observed={observed} expected={expected}")
    report = {
        "schema_version": "forkcert.seeded_fault_conformance.v0.1",
        "purpose": "pipeline conformance, not compiler-bug evidence",
        "rows": rows,
        "valid": not errors,
        "errors": errors,
        "claim": "the evidence logic distinguishes producer, benign mutation, propagation-only, and no-op controls",
    }
    path = Path(args.out).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
