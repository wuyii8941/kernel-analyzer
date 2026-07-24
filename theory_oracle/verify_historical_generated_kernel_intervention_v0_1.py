#!/usr/bin/env python
"""Independent audit for the generated-kernel intervention certificate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    first, second = load(args.first), load(args.second)
    errors: list[str] = []
    for report in (first, second):
        if report.get("schema_version") != "forkcert.historical_generated_kernel_intervention.v0.1":
            errors.append("unexpected intervention schema")
        if report.get("case_id") != "case_001":
            errors.append("unexpected case id")
        if report["intervention"]["target_kernel"] != "triton_red_fused_sum_view_1":
            errors.append("unexpected target kernel")
        if report["intervention"]["old_expression"] == report["intervention"]["new_expression"]:
            errors.append("intervention did not change target expression")
        if not report["context"]["summary_equal_except_target_text"]:
            errors.append("non-target context summary is not invariant")
        if not report["context"]["non_target_text_signature_equal"]:
            errors.append("non-target generated wrapper text is not invariant")
        if report["outputs"]["reference_vs_original"]["max_abs"] <= 1.0:
            errors.append("original wrong-result witness is not large enough for this case")
        residual_ok = report["outputs"].get("intervention_within_control_residual_tolerance")
        if residual_ok is None:
            residual_ok = (
                report["outputs"]["reference_vs_intervention"]["max_abs"]
                <= report["outputs"].get("residual_tolerance", float("inf"))
            )
        if not residual_ok:
            errors.append("intervention residual is above the declared control tolerance")
        if report["claim"]["allowed_claim_level"] != "INTERVENTION_DEPENDENT_ATTRIBUTION":
            errors.append("claim level was upgraded or downgraded unexpectedly")
        if not report.get("limitations"):
            errors.append("limitations are missing")
    for key in ("intervention", "outputs", "context", "claim"):
        if first[key] != second[key]:
            errors.append(f"independent intervention reports differ in {key}")
    audit = {
        "schema_version": "forkcert.historical_generated_kernel_intervention_audit.v0.1",
        "inputs": [str(Path(args.first).resolve()), str(Path(args.second).resolve())],
        "valid": not errors,
        "errors": errors,
        "evidence_level": "INTERVENTION_DEPENDENT_ATTRIBUTION" if not errors else "INVALID",
        "allowed_claim_level": "INTERVENTION_DEPENDENT_ATTRIBUTION" if not errors else "INVALID",
        "target_kernel": first["intervention"]["target_kernel"],
        "original_max_abs_error": first["outputs"]["reference_vs_original"]["max_abs"],
        "intervention_residual_max_abs": first["outputs"]["reference_vs_intervention"]["max_abs"],
        "non_target_context_invariant": first["context"]["summary_equal_except_target_text"],
        "limitations": [
            "this is a generated-code hypothesis intervention, not the hidden historical patch",
            "no root-cause or correctness-proof claim is licensed",
            "runtime/autotuning identity is not fully observed",
        ],
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
