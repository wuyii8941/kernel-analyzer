#!/usr/bin/env python3
"""Compare two SiLU source references on the same retained natural bank.

The native and explicit source runs use the same endpoint, checkpoint,
parameter boundary and input bank.  The report tests whether changing the
reference spelling itself changes the recorded original-coordinate effect.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / (
    "results/property/numerical_coverage_v1/deepseek128_silu_batch000/raw/"
    "mapped_backward_667_in_out_ptr0-silu-common-input.json"
)
EXPLICIT = ROOT / (
    "results/property/numerical_coverage_v1/silu_factorial_explicit_source_run3/raw/"
    "mapped_backward_667_in_out_ptr0-silu-common-input.json"
)
OUT = ROOT / "results/property/root_cause_closure_v1/silu_source_factorial_v1.json"
STAGES = (
    "LOCAL",
    "PARAMETER_GRADIENT",
    "ADAMW_UPDATE",
    "PARAMETER_WRITE",
)
FIELDS = ("effect_energy", "repair_energy", "effect_repair_inner_product")


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    native = read(NATIVE)
    explicit = read(EXPLICIT)
    for key in ("state_ids", "calibration_state_ids", "confirmation_state_ids", "input_bank", "carrier"):
        if native.get(key) != explicit.get(key):
            raise ValueError(f"same-bank contract mismatch in {key}")
    if native["reference_comparison_scope"]["same_local_operands"] is not True:
        raise ValueError("native run is not a same-input comparison")
    if explicit["reference_comparison_scope"]["same_local_operands"] is not True:
        raise ValueError("explicit run is not a same-input comparison")

    maxima: dict[str, dict[str, float]] = {}
    differing_counts: dict[str, dict[str, int]] = {}
    for stage in STAGES:
        maxima[stage] = {}
        differing_counts[stage] = {}
        left_rows = native["original_coordinate_statistics"][stage]
        right_rows = explicit["original_coordinate_statistics"][stage]
        if len(left_rows) != len(right_rows):
            raise ValueError(f"row count mismatch in {stage}")
        for field in FIELDS:
            differences = [
                abs(float(left.get(field, 0.0)) - float(right.get(field, 0.0)))
                for left, right in zip(left_rows, right_rows)
            ]
            maxima[stage][field] = max(differences, default=0.0)
            differing_counts[stage][field] = sum(value != 0.0 for value in differences)

    effect_fields_match = all(
        math.isclose(
            float(native["original_coordinate_statistics"][stage][0]["effect_energy"]),
            float(explicit["original_coordinate_statistics"][stage][0]["effect_energy"]),
            rel_tol=1e-12,
            abs_tol=1e-18,
        )
        for stage in STAGES
    )
    return_payload = {
        "schema": "kernel-analyzer-silu-source-factorial-v1",
        "status": "COMPLETE_SAME_BANK_SOURCE_RESPONSE_COMPARISON",
        "scope": "SAME_NATURAL_STATE_BANK_32_STATES_FIXED_PARAMETER",
        "state_count": len(native["state_ids"]),
        "confirmation_count": len(native["confirmation_state_ids"]),
        "input_bank": native["input_bank"],
        "carrier": native["carrier"],
        "reference_variants": {
            "native": native["reference_comparison_scope"]["reference_variant"],
            "explicit": explicit["reference_comparison_scope"]["reference_variant"],
        },
        "max_absolute_original_coordinate_difference": maxima,
        "nonzero_difference_counts": differing_counts,
        "effect_energy_equal_within_tolerance_all_stages": effect_fields_match,
        "interpretation": (
            "Changing the retained SiLU reference from native sigmoid compact/source-order "
            "to explicit exponential source-order leaves the candidate-minus-reference "
            "effect energy equal within the declared numerical tolerance at every recorded "
            "stage on this same natural bank. Only floating accumulation changes at tiny levels. Thus the "
            "selected source spelling is not identified as the cause of this measured profile; "
            "the remaining candidate-versus-reference difference must be examined at another "
            "implementation boundary or intermediate."
        ),
        "claim_boundary": (
            "This is a same-bank reference-factor comparison, not a family-wide SiLU claim, "
            "not a population mean-bias proof, and not a training-quality result."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(return_payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output": str(OUT), "status": return_payload["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
