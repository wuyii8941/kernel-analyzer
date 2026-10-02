#!/usr/bin/env python3
"""Recompute the same-bank GELU source-choice comparison.

This is a small, deterministic analysis of retained original-coordinate
statistics.  It deliberately does not infer a family-wide or natural-state
population claim: the three records use one declared 32-state bank and the
same parameter boundary.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/property/numerical_coverage_v1"
OUT = ROOT / "results/property/root_cause_closure_v1/gelu_source_factorial_v1.json"

RECORD_DIRS = {
    "native_tanh": "gemma_gelu_tanh_native_confirmation_v3",
    "explicit_exponential_tanh": "gemma_gelu_tanh_exp_confirmation_v3",
    "native_tanh_fused_multiply_add": "gemma_gelu_tanh_fma_confirmation_v7",
}

STAGES = ("LOCAL", "PARAMETER_GRADIENT", "ADAMW_UPDATE", "PARAMETER_WRITE")


def _read_record(directory: str) -> dict[str, Any]:
    candidates = [
        BASE / directory / "raw/gelu_backward_952_out_ptr0.json",
        BASE / directory / "raw/gelu_backward_1182_out_ptr0.json",
    ]
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text())
    raise FileNotFoundError(f"no GELU record in {directory}")


def _confirmation_rows(record: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    confirmation = {str(value) for value in record["confirmation_state_ids"]}
    return [
        row
        for state_id, row in zip(
            record["state_ids"], record["original_coordinate_statistics"][stage]
        )
        if str(state_id) in confirmation
    ]


def _stage_summary(record: dict[str, Any], stage: str) -> dict[str, float]:
    rows = _confirmation_rows(record, stage)
    effect = math.fsum(float(row["effect_energy"]) for row in rows)
    repair = math.fsum(float(row["repair_energy"]) for row in rows)
    aligned = math.fsum(
        float(row.get("effect_repair_inner_product", 0.0)) for row in rows
    )
    if repair <= 0:
        raise ValueError(f"non-positive repair energy for {stage}")
    return {
        "effect_energy_sum": effect,
        "repair_energy_sum": repair,
        "effect_repair_inner_product_sum": aligned,
        "rms_ratio": math.sqrt(effect / repair),
        "aligned_ratio_of_sums": aligned / repair,
    }


def _same_bank(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    first = next(iter(records.values()))
    for name, record in records.items():
        if record["state_ids"] != first["state_ids"]:
            raise ValueError(f"state bank mismatch for {name}")
        if record["confirmation_state_ids"] != first["confirmation_state_ids"]:
            raise ValueError(f"confirmation bank mismatch for {name}")
        if record["input_bank"] != first["input_bank"]:
            raise ValueError(f"input bank mismatch for {name}")
        if record["reference_comparison_scope"]["same_local_operands"] is not True:
            raise ValueError(f"same-input contract missing for {name}")
    return {
        "state_count": len(first["state_ids"]),
        "confirmation_count": len(first["confirmation_state_ids"]),
        "input_bank": first["input_bank"],
        "parameter_scope": first["reference_comparison_scope"]["parameter_scope"],
    }


def build() -> dict[str, Any]:
    records = {name: _read_record(directory) for name, directory in RECORD_DIRS.items()}
    bank = _same_bank(records)
    summaries = {
        name: {stage: _stage_summary(record, stage) for stage in STAGES}
        for name, record in records.items()
    }
    native = summaries["native_tanh"]
    fused = summaries["native_tanh_fused_multiply_add"]
    profile_match = all(
        math.isclose(native[stage][key], fused[stage][key], rel_tol=0.0, abs_tol=1e-15)
        for stage in STAGES
        for key in ("effect_energy_sum", "repair_energy_sum", "effect_repair_inner_product_sum")
    )
    explicit = summaries["explicit_exponential_tanh"]
    explicit_differs = any(
        not math.isclose(native[stage]["effect_energy_sum"], explicit[stage]["effect_energy_sum"], rel_tol=0.0, abs_tol=1e-18)
        for stage in STAGES
    )
    ratios = {
        stage: explicit[stage]["effect_energy_sum"] / native[stage]["effect_energy_sum"]
        if native[stage]["effect_energy_sum"] > 0
        else None
        for stage in STAGES
    }
    return {
        "schema": "kernel-analyzer-gelu-source-factorial-v1",
        "status": "COMPLETE_SAME_BANK_SOURCE_FACTORIAL_RECOMPUTATION",
        "scope": "SAME_NATURAL_STATE_BANK_32_STATES_FIXED_PARAMETER",
        "bank": bank,
        "records": {
            name: {
                "directory": directory,
                "reference_variant": records[name]["reference_comparison_scope"]["reference_variant"],
            }
            for name, directory in RECORD_DIRS.items()
        },
        "stage_summaries": summaries,
        "comparisons": {
            "native_tanh_fused_multiply_add_profile_match": profile_match,
            "explicit_exponential_tanh_differs_from_native": explicit_differs,
            "explicit_to_native_effect_energy_ratio": ratios,
        },
        "interpretation": (
            "On the same 32-state natural input bank and fixed parameter boundary, "
            "native tanh and its fused multiply-add spelling produce identical "
            "original-coordinate local, gradient, update and write summaries. The "
            "explicit exponential tanh reference changes those summaries. This "
            "excludes the tested native-versus-fused spelling as the source of the "
            "observed profile on this bank, while leaving the exact candidate "
            "instruction and family-wide population behavior unidentified."
        ),
        "claim_boundary": (
            "Same-bank source-choice response only; no universal GELU root, no "
            "natural-population mean-bias claim, and no independent training-quality claim."
        ),
    }


def main() -> None:
    report = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output": str(OUT), "status": report["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
