"""Run three small reference-free diagnostics.

The examples mirror mechanisms already present in the repository:

* a controlled low-precision arithmetic variant,
* a saved-probability normalization defect, and
* a declared specification mismatch.

They intentionally never turn these observations into a bias verdict.  This
script is a calibration of the evidence mode, not a new natural training
case.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kernel_analyzer import (
    check_reduction_order,
    check_softmax_saved_state,
    diagnose_kernel,
)
from kernel_analyzer.fp32_reduction_order import balanced_fp32_sum, sequential_fp32_sum
from kernel_analyzer.softmax_saved_state_diagnostic import evaluate as evaluate_saved_state


def _rounding_case():
    return check_reduction_order(
        sequential_fp32_sum,
        balanced_fp32_sum,
        lambda index: (torch.tensor([1.0e8, 1.0, -1.0e8, 1.0], dtype=torch.float32),),
        samples=8,
    )


def _saved_state_case():
    def candidate(scores, maximum, denominator):
        return evaluate_saved_state(
            scores.to(torch.bfloat16), maximum, denominator, scale=1.0
        )["reconstructed_probability"]

    def make_inputs(index):
        del index
        return (
            torch.zeros((1, 4), dtype=torch.float32),
            torch.zeros((1,), dtype=torch.float32),
            torch.full((1,), 8.0, dtype=torch.float32),
        )

    return check_softmax_saved_state(
        candidate,
        make_inputs,
        samples=8,
    )


def _spec_case():
    def candidate(x):
        return x * torch.sigmoid(x)

    def spec(index, args, kwargs, output):
        del index, kwargs
        error = (output - args[0]).abs().max()
        return {"passed": bool(error <= 1e-6), "max_abs_spec_error": float(error.item())}

    return diagnose_kernel(
        candidate,
        lambda index: (torch.tensor([1.0 + index]),),
        specification_check=spec,
        samples=8,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/property/reference_free_diagnostics_v1/examples.json"),
    )
    args = parser.parse_args()
    payload = {
        "schema": "kernel-analyzer-reference-free-examples-v1",
        "purpose": "evidence_mode_calibration_not_natural_training_confirmation",
        "cases": {
            "controlled_rounding": _rounding_case(),
            "saved_state_consistency": _saved_state_case(),
            "declared_specification": _spec_case(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        name: {
            "diagnostic_status": value["diagnostic_status"],
            "bias_decision": value["bias_decision"],
            "evidence_classes": value["evidence_classes"],
        }
        for name, value in payload["cases"].items()
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
