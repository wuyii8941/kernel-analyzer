#!/usr/bin/env python
"""Build a fail-closed causal-coverage ledger from the live Qwen3 backward census."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-summary", required=True)
    parser.add_argument("--runtime-census", required=True)
    parser.add_argument("--runtime-audit", required=True)
    parser.add_argument("--signature-clusters")
    parser.add_argument("--repair-evaluation")
    parser.add_argument("--nonidentifiable-family", action="append", default=[])
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    static = json.loads(Path(args.static_summary).read_text())
    census = json.loads(Path(args.runtime_census).read_text())
    audit = json.loads(Path(args.runtime_audit).read_text())
    if static["status"] != "VALID_BACKWARD_GENERATED_KERNEL_SUMMARY":
        raise ValueError("invalid static backward summary")
    if census["status"] != "VALID_BACKWARD_RUNTIME_CENSUS":
        raise ValueError("invalid backward runtime census")
    if audit["status"] != "VALID_BACKWARD_RUNTIME_CENSUS_AUDIT":
        raise ValueError("invalid backward runtime census audit")

    signatures = None
    signature_rows: dict[str, dict[str, Any]] = {}
    if args.signature_clusters:
        signatures = json.loads(Path(args.signature_clusters).read_text())
        if signatures["status"] != "VALID_SIGNATURE_PARTITION_NOT_SEMANTIC_EQUIVALENCE":
            raise ValueError("invalid backward argument-signature clusters")
        signature_rows = {row["treatment_id"]: row for row in signatures["families"]}
    repair_evaluation = None
    repair_rows: dict[str, dict[str, Any]] = {}
    if args.repair_evaluation:
        repair_evaluation = json.loads(Path(args.repair_evaluation).read_text())
        if repair_evaluation["status"] not in {
            "VALID_BACKWARD_SINGLETON_REPAIR_EVALUATION",
            "VALID_BACKWARD_SINGLETON_REPAIR_EVALUATION_V0_2",
        }:
            raise ValueError("invalid backward singleton-repair evaluation")
        repair_rows = {
            row["kernel_family"]: {"treatment": name, **row}
            for name, row in repair_evaluation["treatments"].items()
        }

    runtime_counts = census["family_call_counts"]
    rows: list[dict[str, Any]] = []
    for family in static["kernel_families"]:
        name = family["name"]
        calls = int(runtime_counts[name])
        if calls == 1:
            mapping = "SINGLETON"
            representatives: list[int] | None = [0]
        elif calls in (27, 28):
            mapping = "CANDIDATE_SINGLE_ROLE_REPEATED_UNVALIDATED"
            representatives = [0, calls // 2, calls - 1]
        else:
            mapping = "MULTI_ROLE_OR_ORDERING_UNRESOLVED"
            representatives = None
        signature = signature_rows.get(name)
        repair = repair_rows.get(name)
        nonidentifiable = name in set(args.nonidentifiable_family)
        if nonidentifiable and calls != 1:
            raise ValueError("only singleton families may be marked nonidentifiable here")
        rows.append(
            {
                "treatment_id": name,
                "kind": "generated_triton_family",
                "runtime_calls": calls,
                "constituent_aten": family["original_aten"],
                "source_nodes": family["source_nodes"],
                "role_mapping_state": mapping,
                "candidate_representatives": representatives,
                "argument_signature_clusters": None
                if signature is None
                else signature["signature_cluster_count"],
                "signature_role_status": None
                if signature is None
                else signature["semantic_role_status"],
                "coverage_state": (
                    "VALID_SELECTED_STATE_REPAIR"
                    if repair is not None
                    else "UNINSTANTIATED_NONIDENTIFIABLE_INTERNAL_ABI"
                    if nonidentifiable
                    else "RUNTIME_OBSERVED_ONLY"
                ),
                "valid_repairs": int(repair is not None),
                "valid_injections": 0,
                "repair_endpoint_profiles": None if repair is None else repair["profiles"],
                "fully_covered": False,
            }
        )

    for operation, calls in sorted(census["external_call_counts_during_backward"].items()):
        treatment_id = f"extern:{operation}"
        signature = signature_rows.get(treatment_id)
        rows.append(
            {
                "treatment_id": treatment_id,
                "kind": "external_kernel_family",
                "runtime_calls": int(calls),
                "constituent_aten": [f"aten.{operation}"],
                "source_nodes": [],
                "role_mapping_state": "MULTI_ROLE_OR_ORDERING_UNRESOLVED",
                "candidate_representatives": None,
                "argument_signature_clusters": None
                if signature is None
                else signature["signature_cluster_count"],
                "signature_role_status": None
                if signature is None
                else signature["semantic_role_status"],
                "coverage_state": "RUNTIME_OBSERVED_ONLY",
                "valid_repairs": 0,
                "valid_injections": 0,
                "repair_endpoint_profiles": None,
                "fully_covered": False,
            }
        )

    calls = sum(row["runtime_calls"] for row in rows)
    if len(rows) != 41 or calls != 1857:
        raise ValueError(f"backward denominator mismatch: {len(rows)} families, {calls} calls")
    payload = {
        "schema_version": "forkcert.qwen3-backward-causal-coverage-ledger.v0.1",
        "scope": "Qwen3-0.6B GRPO heldout-transport-B step29 compiled natural backward",
        "denominator": {
            "treatment_families": len(rows),
            "runtime_calls": calls,
            "triton_families": 39,
            "external_families": 2,
        },
        "metrics": {
            "runtime_observed_families": len(rows),
            "singleton_families": sum(row["role_mapping_state"] == "SINGLETON" for row in rows),
            "candidate_single_role_repeated_families": sum(
                row["role_mapping_state"] == "CANDIDATE_SINGLE_ROLE_REPEATED_UNVALIDATED"
                for row in rows
            ),
            "unresolved_multi_role_families": sum(
                row["role_mapping_state"] == "MULTI_ROLE_OR_ORDERING_UNRESOLVED"
                for row in rows
            ),
            "validly_intervened_families": sum(row["valid_repairs"] > 0 for row in rows),
            "selected_state_repair_families": sum(row["valid_repairs"] > 0 for row in rows),
            "repair_exact_null_families": sum(
                row["repair_endpoint_profiles"] is not None
                and all(
                    endpoint["repair_exactly_null"]
                    for endpoint in row["repair_endpoint_profiles"].values()
                )
                for row in rows
            ),
            "repair_nonnull_families": sum(
                row["repair_endpoint_profiles"] is not None
                and any(
                    not endpoint["repair_exactly_null"]
                    for endpoint in row["repair_endpoint_profiles"].values()
                )
                for row in rows
            ),
            "operator_level_nonidentifiable_families": sum(
                row["coverage_state"] == "UNINSTANTIATED_NONIDENTIFIABLE_INTERNAL_ABI"
                for row in rows
            ),
            "fully_covered_families": 0,
        },
        "treatment_families": rows,
        "coverage_rules": [
            "runtime observation grants no causal credit",
            "multi-role family ordering must be mapped before representative selection",
            "same-name calls require role, shape, fusion and state transport before equivalence credit",
            "repair/injection must preserve the original compiled treatment context",
            "gradient/update and semantic endpoints must be reported separately",
            "population and correctness ledgers remain independent",
        ],
        "verdict": (
            "BACKWARD_RUNTIME_DENOMINATOR_COMPLETE_FOR_SELECTED_STATE; "
            f"SELECTED_STATE_REPAIR_PARTIAL_{sum(row['valid_repairs'] > 0 for row in rows)}_OF_41; "
            "INJECTION_AND_TRANSPORT_UNINSTANTIATED"
        ),
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Qwen3 backward causal coverage ledger v0.1",
        "",
        "## Verdict",
        "",
        payload["verdict"],
        "",
        "All rows are dynamically observed treatment candidates, not causal evidence.",
        "",
        "| Treatment | Kind | Runtime calls | Signatures | Role mapping | Representatives | State |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for row in rows:
        reps = "UNDECLARED" if row["candidate_representatives"] is None else ",".join(
            map(str, row["candidate_representatives"])
        )
        lines.append(
            f"| `{row['treatment_id']}` | {row['kind']} | {row['runtime_calls']} | "
            f"{row['argument_signature_clusters'] if row['argument_signature_clusters'] is not None else 'NA'} | "
            f"{row['role_mapping_state']} | {reps} | {row['coverage_state']} |"
        )
    lines += [
        "",
        "The selected-state runtime denominator is 41 family names and 1,857 calls. "
        f"{sum(row['valid_repairs'] > 0 for row in rows)} families have selected-state repair credit. "
        "No family has injection, population-transport, long-run, root-cause or correctness credit.",
    ]
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
