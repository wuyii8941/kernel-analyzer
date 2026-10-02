#!/usr/bin/env python3
"""Join the additional mainstream scan with its source probes.

This report is intentionally conservative.  It records every nonzero stage
from the generated-input scan, but only calls a source "closed" when the
dedicated behavioural probe matches the declared arithmetic alternatives.
Neither a nonzero stage nor a directional checker result is promoted to a
model-wide or training-level claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _nonzero_stages(operator: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for stage, data in operator.get("stages", {}).items():
        rms = data.get("total_rms")
        if rms is not None and float(rms) > 0.0:
            result.append({
                "stage": stage,
                "total_rms": float(rms),
                "decision": data.get("decision"),
                "aligned_ratio_of_sums": data.get("aligned_ratio_of_sums"),
                "aligned_statewise_gain_interval": data.get("aligned_statewise_gain_interval"),
                "aligned_projection_interval": data.get("aligned_projection_interval"),
                "aligned_estimand": data.get("aligned_estimand", "legacy_mean_of_statewise_ratios"),
                "systematic_bias_reasons": data.get("systematic_bias_reasons", []),
            })
    return result


def _source_status(scan_row: dict[str, Any], operator: str, stage: str,
                   sources: dict[str, dict[str, Any]]) -> tuple[str, str]:
    """Return (status, explanation) without upgrading ambiguous probes."""
    model = scan_row["model"]
    if operator == "rotary_position_embedding":
        row = sources.get("rotary", {}).get("models", {}).get(model)
        if not row:
            return "SOURCE_NOT_ASSESSED", "No model-specific rotary source probe was recorded."
        if row.get("all_reference_matches_bf16_formula") and row.get("all_candidate_matches_fp32_formula"):
            return (
                "INTERMEDIATE_ARITHMETIC_MATERIALIZATION_BOUNDARY",
                "Eager matches BF16 stepwise rotation; compiled candidate matches the explicit "
                "FP32-intermediate rotation followed by BF16 output storage.",
            )
        return "SOURCE_NOT_UNIQUELY_IDENTIFIED", "The tested rotary formula alternatives did not match both routes."

    if operator == "activation":
        row = sources.get("activation", {}).get("models", {}).get(model)
        if not row:
            return "SOURCE_NOT_ASSESSED", "No activation source probe was recorded."
        if stage == "OUTPUT":
            if not row.get("all_candidate_output_differs", False):
                return "NO_DISCREPANCY_IN_SOURCE_PROBE", "The source probe found no output difference."
            # GPT-2's declared gelu_new spelling is matched by eager but not
            # by the compiled route, which is the strongest available source
            # evidence.  Other GELU spellings do not uniquely match the
            # explicit Python expression on this runtime.
            if row.get("all_reference_output_matches_explicit"):
                return (
                    "COMPILED_ACTIVATION_EVALUATION_ORDER",
                    "The eager registered activation matches its declared explicit formula, while "
                    "the compiled route differs; this supports a compiler/native evaluation-order "
                    "difference, not an instruction-level attribution.",
                )
            return (
                "ACTIVATION_SOURCE_NOT_UNIQUELY_IDENTIFIED",
                "A tiny compiled/eager output difference is reproducible, but the native GELU "
                "route and tested explicit spelling do not uniquely identify its arithmetic source.",
            )
        if stage == "BACKWARD" and row.get("all_candidate_backward_differs", False):
            return (
                "COMPILED_ACTIVATION_BACKWARD_EVALUATION_VARIANT",
                "The compiled and eager VJPs differ on the same BF16 inputs; the analytic FP32 "
                "comparison is reported as evidence about evaluation order, not a unique kernel proof.",
            )
        return "ACTIVATION_SOURCE_NOT_UNIQUELY_IDENTIFIED", "No matching backward source alternative was established."

    if operator == "layernorm":
        row = sources.get("layernorm", {}).get("models", {}).get(model)
        if not row:
            return "SOURCE_NOT_ASSESSED", "No LayerNorm source probe was recorded."
        if stage == "OUTPUT":
            if row.get("all_candidate_matches_fp32_formula"):
                return (
                    "LAYERNORM_FP32_REDUCTION_MATERIALIZATION_BEHAVIOUR",
                    "The compiled candidate matches the explicit FP32 reduction/write-back formula; "
                    "the eager route does not match the naïve BF16 reduction formula. This is a "
                    "behavioural arithmetic attribution, not an instruction trace.",
                )
            return "LAYERNORM_OUTPUT_SOURCE_NOT_UNIQUELY_IDENTIFIED", "Candidate/reference differ, but no tested formula uniquely matches the candidate."
        if stage == "BACKWARD":
            if row.get("all_candidate_backward_matches_analytic"):
                return (
                    "LAYERNORM_BACKWARD_ANALYTIC_FP32_CONTRACT",
                    "The compiled backward matches the closed-form FP32 LayerNorm derivative for every probe row.",
                )
            cast_rows = [
                item for probe in row.get("rows", []) for item in probe.get("backward", [])
                if "candidate_vs_analytic_fp32_cast" in item
                and "reference_vs_analytic_fp32_cast" in item
            ]
            if cast_rows:
                candidate_exact = sum(item["candidate_vs_analytic_fp32_cast"].get("exact", False) for item in cast_rows)
                reference_exact = sum(item["reference_vs_analytic_fp32_cast"].get("exact", False) for item in cast_rows)
                candidate_max = max(item["candidate_vs_analytic_fp32_cast"].get("relative_l2", 0.0) for item in cast_rows)
                reference_max = max(item["reference_vs_analytic_fp32_cast"].get("relative_l2", 0.0) for item in cast_rows)
                if candidate_exact >= reference_exact and candidate_max <= reference_max:
                    return (
                        "LAYERNORM_BACKWARD_FP32_DERIVATIVE_CAST_WITH_SPARSE_ROUNDING",
                        "After accounting for BF16 gradient write-back, the compiled route is at least "
                        "as close to the closed-form FP32 derivative as eager (more exact rows and no "
                        "larger maximum residual); the remaining sparse reduction/cast rounding is "
                        "not uniquely assigned to one generated operation.",
                    )
            return (
                "LAYERNORM_BACKWARD_REDUCTION_SOURCE_NOT_UNIQUELY_IDENTIFIED",
                "The backward discrepancy is sparse and reproducible, but neither route exactly "
                "matches the closed-form derivative on all cotangents; a lower-level reduction/cast "
                "probe would be needed for unique attribution.",
            )

    if operator == "softmax":
        row = sources.get("softmax", {}).get("analysis", {})
        if row.get("all_reference_matches_fp32_formula"):
            return (
                "COMPILED_BF16_SOFTMAX_REDUCTION_MATERIALIZATION",
                "Eager softmax matches the explicit FP32-intermediate formula; sparse compiled/eager "
                "differences occur in the BF16 route and are reproduced with and without autograd. "
                "This does not establish a directional bias.",
            )
        return "SOFTMAX_SOURCE_NOT_UNIQUELY_IDENTIFIED", "A nonzero softmax stage was observed, but the explicit formula check did not close the source."

    return "SOURCE_NOT_ASSESSED", "No dedicated source probe is registered for this operator."


def build(scan: dict[str, Any], *, activation: dict[str, Any], layernorm: dict[str, Any],
          rotary: dict[str, Any], softmax: dict[str, Any]) -> dict[str, Any]:
    sources = {
        "activation": activation,
        "layernorm": layernorm,
        "rotary": rotary,
        "softmax": softmax,
    }
    rows: list[dict[str, Any]] = []
    counts: dict[str, Any] = {
        "models": len(scan.get("cases", {})),
        "valid_model_cases": 0,
        "operators": {},
        "nonzero_stage_rows": 0,
        "direction_confirmed_rows": 0,
        "source_closed_rows": 0,
        "source_partial_or_unresolved_rows": 0,
        "measurement_unresolved_model_cases": 0,
    }
    closed_prefixes = (
        "INTERMEDIATE_", "COMPILED_", "LAYERNORM_FP32_", "NO_DISCREPANCY_IN_SOURCE_PROBE",
    )
    for model, case in scan.get("cases", {}).items():
        if case.get("status") == "UNRESOLVED_MEASUREMENT":
            counts["measurement_unresolved_model_cases"] += 1
            continue
        counts["valid_model_cases"] += 1
        for operator, value in case.get("operators", {}).items():
            counts["operators"][operator] = counts["operators"].get(operator, 0) + 1
            for stage in _nonzero_stages(value):
                counts["nonzero_stage_rows"] += 1
                if stage.get("decision") == "SYSTEMATIC_BIAS_CONFIRMED":
                    counts["direction_confirmed_rows"] += 1
                status, explanation = _source_status(
                    {"model": model}, operator, stage["stage"], sources,
                )
                if status.startswith(closed_prefixes):
                    counts["source_closed_rows"] += 1
                else:
                    counts["source_partial_or_unresolved_rows"] += 1
                rows.append({
                    "model": model,
                    "operator": operator,
                    "stage": stage["stage"],
                    "total_rms": stage["total_rms"],
                    "decision": stage.get("decision"),
                    "aligned_ratio_of_sums": stage.get("aligned_ratio_of_sums"),
                    "aligned_statewise_gain_interval": stage.get("aligned_statewise_gain_interval"),
                    "aligned_projection_interval": stage.get("aligned_projection_interval"),
                    "aligned_estimand": stage.get("aligned_estimand"),
                    "systematic_bias_reasons": stage.get("systematic_bias_reasons", []),
                    "root_cause_status": status,
                    "root_cause_explanation": explanation,
                    "scope": "generated_model_config_inputs_and_declared_compiled_route",
                    "training_claim": False,
                })
    return {
        "schema": "additional-mainstream-model-root-cause-summary-v1",
        "status": "COMPLETE",
        "scan": {
            "schema": scan.get("schema"),
            "selected_models": scan.get("selected_models", []),
            "samples": scan.get("samples"),
            "dtype": scan.get("dtype"),
            "scope": scan.get("scope"),
        },
        "source_probe_schemas": {
            "activation": activation.get("schema"),
            "layernorm": layernorm.get("schema"),
            "rotary": rotary.get("schema"),
            "softmax": softmax.get("schema"),
        },
        "counts": counts,
        "rows": rows,
        "limitations": [
            "All measurements use declared generated inputs and a compiled Inductor route; no full model weights are constructed.",
            "A confirmed directional endpoint is conditional on this probe distribution and does not prove a model-wide or training-level bias.",
            "Formula matches are behavioural source evidence, not instruction-level traces.",
            "A nonzero stage whose source status is partial or unresolved remains explicitly unresolved.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--layernorm", type=Path, required=True)
    parser.add_argument("--rotary", type=Path, required=True)
    parser.add_argument("--softmax", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not output.is_relative_to(root):
        raise ValueError("output must be inside the repository")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = build(
        _load(args.scan), activation=_load(args.activation), layernorm=_load(args.layernorm),
        rotary=_load(args.rotary), softmax=_load(args.softmax),
    )
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
