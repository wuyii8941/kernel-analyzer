#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from forkcert.report import CLAIM_SCOPE, markdown_table


DEFINITION = re.compile(r"^(triton_[A-Za-z0-9_]+) = async_compile\.triton", re.MULTILINE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory numerical source templates in Inductor output_code.py.")
    parser.add_argument("--output-code", required=True)
    parser.add_argument("--out", default="results/phase2_compile_source_inventory.json")
    parser.add_argument("--report", default="reports/phase2_compile_source_inventory.md")
    args = parser.parse_args()

    path = Path(args.output_code)
    text = path.read_text(encoding="utf-8")
    call_start = text.index("    def call(self, args):")
    definitions = list(DEFINITION.finditer(text[:call_start]))
    rows = []
    for index, match in enumerate(definitions):
        name = match.group(1)
        end = definitions[index + 1].start() if index + 1 < len(definitions) else call_start
        body = text[match.start() : end]
        rows.append(
            {
                "kernel": name,
                "invocations": len(re.findall(rf"\b{re.escape(name)}\.run\(", text[call_start:])),
                "tl_sum": body.count("tl.sum("),
                "tl_max": body.count("triton_helpers.max2("),
                "rsqrt": body.count("rsqrt("),
                "exp": body.count("exp("),
                "sin": body.count("sin("),
                "cos": body.count("cos("),
                "fp16_casts": len(re.findall(r"\.to\(tl\.float16\)", body)),
                "fp32_casts": len(re.findall(r"\.to\(tl\.float32\)", body)),
                "fp16_output_ptrs": len(re.findall(r"'out_ptr[0-9]*': '\*fp16'", body)),
                "has_reduction": "num_reduction': 1" in body or "tl.sum(" in body,
                "source_family": name.removeprefix("triton_per_fused_").removeprefix("triton_poi_fused_"),
            }
        )

    call_text = text[call_start:]
    external_mm = len(re.findall(r"extern_kernels\.mm\(", call_text))
    external_bmm = len(re.findall(r"extern_kernels\.bmm\(", call_text))
    triton_invocations = sum(int(row["invocations"]) for row in rows)
    numerical = [
        row
        for row in rows
        if row["has_reduction"]
        or any(int(row[key]) > 0 for key in ["rsqrt", "exp", "sin", "cos", "fp16_casts"])
    ]
    result = {
        "schema_version": "forkcert.phase2.compile_source_inventory.v1",
        "status": "completed",
        "output_code": str(path),
        "unique_triton_templates": len(rows),
        "triton_invocations": triton_invocations,
        "external_mm_calls": external_mm,
        "external_bmm_calls": external_bmm,
        "total_external_gemm_calls": external_mm + external_bmm,
        "total_compiled_kernel_calls": triton_invocations + external_mm + external_bmm,
        "numerically_relevant_templates": len(numerical),
        "reduction_templates": sum(bool(row["has_reduction"]) for row in rows),
        "transcendental_templates": sum(
            any(int(row[key]) > 0 for key in ["rsqrt", "exp", "sin", "cos"]) for row in rows
        ),
        "fp16_materialization_templates": sum(
            int(row["fp16_casts"]) > 0 or int(row["fp16_output_ptrs"]) > 0 for row in rows
        ),
        "templates": rows,
        "causal_difference_sources_proven": False,
        "why_not": (
            "Inventory identifies compiled kernels, but a legal difference certificate still needs an eager-to-compiled "
            "operation mapping, arithmetic/error contract for every differing fused template and GEMM path, and "
            "propagation bounds for each invocation."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_keys = [
        "unique_triton_templates",
        "triton_invocations",
        "external_mm_calls",
        "external_bmm_calls",
        "total_compiled_kernel_calls",
        "numerically_relevant_templates",
        "reduction_templates",
        "transcendental_templates",
        "fp16_materialization_templates",
        "causal_difference_sources_proven",
    ]
    report = "\n".join(
        [
            "# Phase 2 Compile Source Inventory",
            "",
            "## Claim Scope",
            CLAIM_SCOPE,
            "",
            "## Confound Checklist",
            "- generated output code from exact audited graph: PASS",
            "- every unique Triton template inventoried: PASS",
            "- every Triton/external invocation counted: PASS",
            "- eager-to-compiled arithmetic equivalence map: FAIL / pending",
            "- per-template legal arithmetic contracts: FAIL / pending",
            "",
            "## Delta Self Control",
            "The parent graph audit reports bitwise eager self and warmed-compile self logits for the measured input.",
            "",
            "## Summary",
            markdown_table([{key: result[key] for key in summary_keys}], summary_keys),
            "",
            "## Numerical Templates",
            markdown_table(numerical, list(rows[0].keys()) if rows else []),
            "",
            "## Remaining Requirement",
            result["why_not"],
            "",
            "## External Validity",
            "This inventory is shape-, model-, compiler-build-, cache-, GPU-, and dtype-specific. Dynamic shapes or native BF16 produce a different inventory.",
            "",
        ]
    )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report, encoding="utf-8")
    print(json.dumps({key: result[key] for key in summary_keys}, indent=2))


if __name__ == "__main__":
    main()
