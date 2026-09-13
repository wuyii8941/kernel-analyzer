#!/usr/bin/env python3
"""Check that the production bounded-population path can pass, reject, and abstain."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from kernel_analyzer.training_numerical_analysis import analyze_bounded_population_artifact


ROOT = Path(__file__).resolve().parents[1]


def artifact(effect: np.ndarray, repair: np.ndarray) -> dict:
    import torch

    written = []
    for proposed in (repair + effect, repair):
        parameter = torch.nn.Parameter(torch.zeros(proposed.shape, dtype=torch.float64))
        before = parameter.detach().clone()
        parameter.grad = -torch.from_numpy(proposed.copy())
        torch.optim.SGD([parameter], lr=1.0, foreach=False).step()
        written.append((parameter.detach() - before).numpy())
    effect, repair = written[0] - written[1], written[1]
    return {
        "case_id": "bounded-population-capability-v2",
        "contrast_id": "CONTROLLED_SYNTHETIC_UPDATE",
        "status": "COMPLETE",
        "state_ids": list(range(effect.shape[0])),
        "inference_unit_ids": [f"unit-{index}" for index in range(effect.shape[0])],
        "original_coordinate_statistics": {
            "PARAMETER_WRITE": [
                {
                    "effect_energy": float(u @ u),
                    "repair_energy": float(r @ r),
                    "effect_repair_inner_product": float(u @ r),
                    "nonzero_effect_coordinates": int(np.count_nonzero(u)),
                }
                for u, r in zip(effect, repair)
            ],
        },
        "parameter_write_protocol": {
            "version": "optimizer-implementation-readback-v1",
            "measurement": "parameter_after_step_minus_parameter_before_step",
            "implementation": "torch.optim.SGD",
            "dtype": "float64",
            "scope": "CONTROLLED_VECTOR_REALIZATION_NOT_NATURAL_LLM_STATES",
        },
        "runtime_boundary": {"kind": "SYNTHETIC_NOT_A_NATURAL_TRAINING_CASE"},
    }


def protocol(x_max: float, b_max: float, *, margin: float = 0.01) -> dict:
    return {
        "schema": "bounded-population-capability-v2",
        "claim_scope": "DECLARED_STATE_POPULATION_UPDATE",
        "primary_stage": "PARAMETER_WRITE",
        "data_use": "METHOD_VALIDATION",
        "mandatory_population_endpoints": ["TOTAL_RMS"],
        "population_margins": {"full_update_rms": margin},
        "family_alpha": 0.05,
        "population_energy_bounds": {
            "effect_energy_upper_bound": x_max,
            "repair_energy_upper_bound": b_max,
            "provenance": "support fixed by this controlled synthetic generator",
            "fixed_before_observation": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT):
        parser.error("choose a new output inside kernel-analyzer")
    n = 4096
    repair = np.ones((n, 1), dtype=np.float64)
    cases = {
        "NONZERO_WITHIN_MARGIN": (np.full((n, 1), 0.005), 0.005 ** 2, "EQUIVALENT"),
        "CLEARLY_OUTSIDE_MARGIN": (np.full((n, 1), 0.02), 0.02 ** 2, "NON_EQUIVALENT"),
    }
    results = {}
    for name, (effect, x_max, expected) in cases.items():
        result = analyze_bounded_population_artifact(
            artifact(effect, repair), protocol(x_max, 1.0),
        )
        results[name] = {
            "expected": expected,
            "measurement_status": result["measurement_status"],
            "decision": result["equivalence_decision"],
            "analysis": result,
        }
    # An all-zero observation cannot pass when the declared population support
    # still permits rare boundary events with a large effect.
    zero = np.zeros((64, 1), dtype=np.float64)
    rare_probability = 1.0 / (20.0 * len(zero))
    wide_support = analyze_bounded_population_artifact(
        artifact(zero, np.ones_like(zero)),
        protocol((0.01 / rare_probability) ** 2, 1.0),
    )
    results["UNOBSERVED_WIDE_SUPPORT"] = {
        "expected": "INCONCLUSIVE",
        "measurement_status": wide_support["measurement_status"],
        "decision": wide_support["equivalence_decision"],
        "analysis": wide_support,
    }
    passed = all(row["measurement_status"] == "VALID" and
                 row["decision"] == row["expected"] for row in results.values())
    payload = {
        "schema": "population-decision-capability-v2",
        "status": "PASS" if passed else "FAIL",
        "valid_judgment_count": sum(row["decision"] != "NOT_ASSESSED" for row in results.values()),
        "case_count": len(results),
        "production_path": "analyze_bounded_population_artifact",
        "unobserved_boundary_rare_event_probability": rare_probability,
        "cases": results,
        "scope": "CONTROLLED_BOUNDED_SYNTHETIC_CAPABILITY_NOT_REAL_TRAINING_POWER",
        "source_sha256": {
            str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            str((ROOT / "src/kernel_analyzer/training_numerical_analysis.py").resolve()):
                hashlib.sha256((ROOT / "src/kernel_analyzer/training_numerical_analysis.py").read_bytes()).hexdigest(),
            str((ROOT / "src/kernel_analyzer/training_equivalence.py").resolve()):
                hashlib.sha256((ROOT / "src/kernel_analyzer/training_equivalence.py").read_bytes()).hexdigest(),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": payload["status"],
                      "decisions": {name: row["decision"] for name, row in results.items()}}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
