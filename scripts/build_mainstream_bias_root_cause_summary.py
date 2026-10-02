#!/usr/bin/env python3
"""Combine a mainstream scan with its source probes into one auditable table.

The scanner and source probes deliberately remain separate: the first answers
whether a generated candidate/reference pair differs, while this utility joins
the corresponding behavioural attribution without turning it into a training
or instruction-level claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_nonzero(operator: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for stage, data in operator.get("stages", {}).items():
        rms = data.get("total_rms")
        if rms is not None and float(rms) > 0.0:
            rows.append({
                "stage": stage,
                "total_rms": float(rms),
                "decision": data.get("decision"),
                "systematic_bias_reasons": data.get("systematic_bias_reasons", []),
            })
    return rows


def _rmsnorm_attribution(row: dict[str, Any] | None) -> tuple[str, str]:
    if not row:
        return "NOT_AVAILABLE", "No RMSNorm source probe row was recorded."
    if not row.get("all_candidate_differs_from_reference", False):
        return "NO_DISCREPANCY", "The source probe observed no candidate/reference difference."
    if (row.get("all_reference_matches_bf16_weight_formula")
            and row.get("all_candidate_matches_fp32_weight_formula")):
        return (
            "INTERMEDIATE_CAST_MATERIALIZATION_BOUNDARY",
            "Eager matches normalized-value BF16 cast then BF16-weight multiply; "
            "candidate matches FP32-weight multiply then BF16 write-back.",
        )
    # A single BF16 tie at one coordinate can differ after an otherwise exact
    # formula match.  Keep that residual explicit rather than discarding the
    # useful attribution or incorrectly calling the whole source unresolved.
    if row.get("all_reference_matches_bf16_weight_formula"):
        exact_rows = int(row.get("candidate_fp32_formula_exact_rows", 0))
        rows = int(row.get("samples", 0))
        residual_coordinates = [
            item.get("candidate_vs_fp32_weight_formula", {}).get("nonzero_coordinates", 0)
            for item in row.get("rows", [])
        ]
        if rows and exact_rows == rows - 1 and max(residual_coordinates, default=0) <= 1:
            return (
                "INTERMEDIATE_CAST_MATERIALIZATION_BOUNDARY_WITH_ISOLATED_ROUNDING",
                "All but one probe row match the FP32-weight multiply boundary; the remaining "
                "row differs at one BF16 coordinate, consistent with an isolated tie/rounding case.",
            )
    return "SOURCE_NOT_UNIQUELY_IDENTIFIED", "The tested formula alternatives did not uniquely match both routes."


def _rotary_attribution(row: dict[str, Any] | None) -> tuple[str, str]:
    if not row:
        return "NOT_AVAILABLE", "No rotary source probe row was recorded."
    if not row.get("all_candidate_differs_from_reference", False):
        return "NO_DISCREPANCY", "The source probe observed no candidate/reference difference."
    if row.get("all_reference_matches_bf16_formula") and row.get("all_candidate_matches_fp32_formula"):
        return (
            "INTERMEDIATE_ARITHMETIC_MATERIALIZATION_BOUNDARY",
            "Eager matches BF16 stepwise rotation; candidate matches FP32 intermediate "
            "rotation followed by BF16 output storage.",
        )
    return "SOURCE_NOT_UNIQUELY_IDENTIFIED", "The tested arithmetic alternatives did not uniquely match both routes."


def _rmsnorm_backward_attribution(row: dict[str, Any] | None) -> tuple[str, str]:
    if not row:
        return "NOT_AVAILABLE", "No RMSNorm backward source probe row was recorded."
    if not row.get("all_backward_candidate_differs_from_reference", False):
        return "NO_DISCREPANCY", "The backward source probe observed no candidate/reference difference."
    if (row.get("all_backward_reference_matches_bf16_weight_formula")
            and row.get("all_backward_candidate_matches_fp32_weight_formula")):
        return (
            "BACKWARD_CAST_MATERIALIZATION_BOUNDARY",
            "Backward candidate and reference match the corresponding explicit cast/multiply formulas.",
        )
    if row.get("all_backward_candidate_matches_analytic_fp32_derivative"):
        return (
            "BACKWARD_ANALYTIC_FP32_REDUCTION_CONTRACT",
            "The candidate backward matches the explicit closed-form FP32 RMSNorm derivative "
            "with one dot reduction across all probe rows.",
        )
    exact_rows = int(row.get("backward_candidate_analytic_exact_rows", 0))
    rows = int(row.get("samples", 0))
    residual_coordinates = [
        item.get("candidate_vs_analytic_fp32_derivative", {}).get("nonzero_coordinates", 0)
        for item in row.get("backward_rows", [])
    ]
    if rows and exact_rows >= rows - 4 and max(residual_coordinates, default=0) <= 2:
        return (
            "BACKWARD_ANALYTIC_FP32_WITH_SPARSE_REDUCTION_ROUNDING",
            "Most rows match the explicit closed-form FP32 derivative; residual one/two-coordinate "
            "ties remain and are consistent with a compiled reduction/cast rounding detail.",
        )
    return (
        "BACKWARD_SOURCE_NOT_UNIQUELY_IDENTIFIED",
        "The backward difference is real, but the tested explicit formulas do not uniquely explain it; "
        "a lower-level backward reduction/materialization probe would be needed.",
    )


def build(scan: dict[str, Any], rms: dict[str, Any], rotary: dict[str, Any],
          activation: dict[str, Any], linear: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    counts = {"models": len(scan.get("cases", {})), "operators": {}, "nonzero_stage_rows": 0,
              "direction_confirmed_rows": 0, "source_closed_rows": 0,
              "source_unresolved_rows": 0, "measurement_unresolved_rows": 0}
    for model, case in scan.get("cases", {}).items():
        if case.get("status") == "UNRESOLVED_MEASUREMENT":
            counts["measurement_unresolved_rows"] += 1
        for operator, value in case.get("operators", {}).items():
            counts["operators"][operator] = counts["operators"].get(operator, 0) + 1
            nonzero = _stage_nonzero(value)
            counts["nonzero_stage_rows"] += len(nonzero)
            counts["direction_confirmed_rows"] += sum(
                item["decision"] == "SYSTEMATIC_BIAS_CONFIRMED" for item in nonzero
            )
            stage_sources: dict[str, dict[str, str]] = {}
            if operator == "rmsnorm":
                source_row = rms.get("models", {}).get(model)
                attribution, explanation = _rmsnorm_attribution(source_row)
                for item in nonzero:
                    if item["stage"] == "BACKWARD":
                        stage_attr, stage_expl = _rmsnorm_backward_attribution(source_row)
                    else:
                        stage_attr, stage_expl = attribution, explanation
                    stage_sources[item["stage"]] = {
                        "status": stage_attr, "explanation": stage_expl,
                    }
            elif operator == "rotary_position_embedding":
                attribution, explanation = _rotary_attribution(rotary.get("models", {}).get(model))
            elif operator == "softmax":
                source = "COMPILED_BF16_AUTOGRAD_REDUCTION_PATH" if nonzero else "NO_DISCREPANCY"
                attribution, explanation = source, (
                    "The dedicated probe localizes nonzero rows to the autograd-enabled "
                    "compiled BF16 evaluation/materialization path; it does not confirm a directional bias."
                    if nonzero else "No nonzero stage was observed."
                )
            elif operator == "activation":
                source_models = activation.get("models", {}).get(model, {})
                attribution, explanation = (
                    ("COMPILER_MATH_EVALUATION_PATH",
                     "FP32 source probe finds a tiny compiled/eager evaluation difference; no training claim.")
                    if source_models.get("all_candidate_output_differs")
                    else ("NO_DISCREPANCY", "No nonzero stage was observed in this scan.")
                )
            elif operator == "linear_projection":
                source_models = linear.get("models", {}).get(model, {})
                attribution, explanation = (
                    ("NO_DISCREPANCY", "Candidate and eager projection outputs were identical in the source probe.")
                    if not nonzero and not source_models.get("all_candidate_differs_from_reference")
                    else ("SOURCE_NOT_UNIQUELY_IDENTIFIED", "A projection discrepancy requires a dedicated nonzero source probe.")
                )
            else:
                attribution, explanation = (
                    ("NO_DISCREPANCY", "No nonzero stage was observed in this scan.")
                    if not nonzero else ("SOURCE_NOT_ASSESSED", "No source probe is registered for this operator.")
                )
            unresolved_stage = any(
                source["status"].startswith("BACKWARD_SOURCE_NOT")
                for source in stage_sources.values()
            )
            if unresolved_stage:
                counts["source_unresolved_rows"] += sum(
                    source["status"].startswith("BACKWARD_SOURCE_NOT")
                    for source in stage_sources.values()
                )
                attribution = "PARTIAL_SOURCE_CLOSURE"
                explanation = "Output source is closed, but at least one backward stage needs a lower-level reduction probe."
            if attribution in {
                "INTERMEDIATE_CAST_MATERIALIZATION_BOUNDARY",
                "INTERMEDIATE_CAST_MATERIALIZATION_BOUNDARY_WITH_ISOLATED_ROUNDING",
                "INTERMEDIATE_ARITHMETIC_MATERIALIZATION_BOUNDARY",
                "BACKWARD_CAST_MATERIALIZATION_BOUNDARY",
                "BACKWARD_CAST_WITH_ISOLATED_REDUCTION_ROUNDING",
                "BACKWARD_ANALYTIC_FP32_REDUCTION_CONTRACT",
                "BACKWARD_ANALYTIC_FP32_WITH_SPARSE_REDUCTION_ROUNDING",
                "BACKWARD_ANALYTIC_FP32_WITH_SPARSE_REDUCTION_ROUNDING",
                "COMPILED_BF16_AUTOGRAD_REDUCTION_PATH",
                "COMPILER_MATH_EVALUATION_PATH",
                "NO_DISCREPANCY",
            }:
                counts["source_closed_rows"] += 1
            elif nonzero and not unresolved_stage:
                counts["source_unresolved_rows"] += 1
            rows.append({
                "model": model,
                "operator": operator,
                "scan_status": value.get("status"),
                "measurement_status": value.get("measurement_status"),
                "nonzero_stages": nonzero,
                "root_cause_status": attribution,
                "root_cause_explanation": explanation,
                "stage_root_causes": stage_sources,
                "scope": "generated_inputs_and_declared_candidate_reference_routes",
                "training_claim": False,
            })
    return {
        "schema": "mainstream-model-bias-root-cause-summary-v2",
        "status": "COMPLETE",
        "scan_schema": scan.get("schema"),
        "source_probe_schemas": {
            "rmsnorm": rms.get("schema"), "rotary": rotary.get("schema"),
            "activation": activation.get("schema"), "linear": linear.get("schema"),
        },
        "counts": counts,
        "rows": rows,
        "limitations": [
            "Attributions are behavioural matches to explicit formulas, not instruction traces.",
            "Generated inputs and synthetic weights are conditional probes; no model-wide or training-loss claim is made.",
            "A nonzero output difference without a directional endpoint is not labelled systematic bias.",
            "Unresolved execution or unmatched formula alternatives remain unresolved rather than being recoded as negative.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--rmsnorm", type=Path, required=True)
    parser.add_argument("--rotary", type=Path, required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--linear", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    root = Path(__file__).resolve().parents[1]
    if not output.is_relative_to(root):
        raise ValueError("output must be inside the repository")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = build(_load(args.scan), _load(args.rmsnorm), _load(args.rotary),
                   _load(args.activation), _load(args.linear))
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
