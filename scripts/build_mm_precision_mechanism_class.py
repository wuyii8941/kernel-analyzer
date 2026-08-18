#!/usr/bin/env python3
"""Aggregate independently proved MM precision-bias units into one bounded class."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "results/coverage/cases"
MEMBERS = (
    {
        "architecture": "qwen3_1p7b", "phase": "FORWARD", "role": "layer-0 v_proj",
        "case": "qwen128_vproj.json",
        "decomposition": "qwen128_vproj_precision_decomposition.json",
    },
    {
        "architecture": "mamba_130m", "phase": "FORWARD", "role": "layer-0 mixer in_proj",
        "case": "mamba_seq64_input_proj.json",
        "decomposition": "mamba_seq64_input_proj_precision_decomposition.json",
    },
    {
        "architecture": "phi4_mini_3p8b", "phase": "BACKWARD", "role": "lm_head input VJP",
        "case": "phi4_seq64_lmhead_dx.json",
        "decomposition": "phi4_seq64_lmhead_dx_precision_decomposition.json",
    },
)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def main() -> None:
    rows = []
    for declared in MEMBERS:
        case_path = CASES / declared["case"]
        decomposition_path = CASES / declared["decomposition"]
        case = json.loads(case_path.read_text())
        decomposition = json.loads(decomposition_path.read_text())
        assert case["candidate_id"] == decomposition["candidate_id"]
        assert case["cause_axis"] == "PRECISION"
        assert all(value is True for key, value in case["concrete_program_proof"].items()
                   if key.endswith("_exact"))
        assert all(decomposition["gates"].values())
        assert decomposition["direction"]["total"]["status"] == "PASS"
        rows.append({
            **{key: declared[key] for key in ("architecture", "phase", "role")},
            "operator_family": "MM",
            "candidate_id": case["candidate_id"],
            "case_status": case["status"],
            "case_result_sha256": case["result_sha256"],
            "case_file_sha256": hashlib.sha256(case_path.read_bytes()).hexdigest(),
            "decomposition_result_sha256": decomposition["result_sha256"],
            "decomposition_file_sha256": hashlib.sha256(decomposition_path.read_bytes()).hexdigest(),
            "coherent_sources": decomposition["coherent_sources"],
            "total": {
                "u_statistic": decomposition["direction"]["total"]["cross_state_inner_product_u"],
                "cluster_bootstrap_95": decomposition["direction"]["total"]["cluster_bootstrap_95"],
            },
            "sources": {
                name: {
                    "status": decomposition["direction"][name]["status"],
                    "u_statistic": decomposition["direction"][name]["cross_state_inner_product_u"],
                    "cluster_bootstrap_95": decomposition["direction"][name]["cluster_bootstrap_95"],
                }
                for name in ("kernel", "output_rounding")
            },
        })

    support = {
        source: [row["candidate_id"] for row in rows if source in row["coherent_sources"]]
        for source in ("kernel", "output_rounding")
    }
    gates = {
        "three_distinct_concrete_fb_units": len({row["candidate_id"] for row in rows}) == 3,
        "three_distinct_models": len({row["architecture"] for row in rows}) == 3,
        "forward_and_backward_endpoints_represented": {row["phase"] for row in rows} == {"FORWARD", "BACKWARD"},
        "every_total_direction_ci_lower_positive": all(
            row["total"]["cluster_bootstrap_95"]["lower_95"] > 0.0 for row in rows
        ),
        "kernel_source_replicated_in_two_units": len(support["kernel"]) >= 2,
        "output_rounding_source_replicated_in_two_units": len(support["output_rounding"]) >= 2,
        "cross_operator_family_generalization": False,
    }
    payload = {
        "schema": "kernel-analyzer-mm-precision-mechanism-class-v1",
        "status": "COMPLETE_WITHIN_MM_FAMILY_MECHANISM_CLASS",
        "class_name": "low-precision MM directional-bias mechanisms",
        "mathematical_unit": "Y=XW^T with actual VJP dX=QW and dW=Q^TX",
        "error_identity": (
            "actual_low - fp32_same_operands = local_kernel_difference + "
            "deterministic_output_rounding"
        ),
        "members": rows,
        "mechanism_support": support,
        "gates": gates,
        "conclusion": (
            "This is a replicated mechanism class within the MM operator family. Local kernel "
            "arithmetic and deterministic output rounding are separate, additive directional-bias "
            "sources; their presence is invocation-dependent."
        ),
        "claim_boundary": (
            "Three concrete F+B units across three models and both forward/backward endpoints. "
            "All are MM, so this does not establish a property that generalizes across operator "
            "families. Only Phi currently closes downstream carrier and T4 accumulation."
        ),
    }
    payload["result_sha256"] = digest(payload)
    output = CASES / "mm_precision_mechanism_class.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output.relative_to(ROOT)), "status": payload["status"],
                      "support": support}, sort_keys=True))


if __name__ == "__main__":
    main()
