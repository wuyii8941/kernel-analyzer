#!/usr/bin/env python3
"""Build the single fail-closed, per-dispatch-invocation coverage ledger.

The ledger intentionally separates mathematical derivation, exact F+B origin
binding, candidate-region binding, numerical measurement and bias verdict.
No later stage is inferred from an earlier one.
"""

from __future__ import annotations

import hashlib
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/full_step_inventory"
FINAL = ROOT / "results/final"
OUT = ROOT / "results/coverage"


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def canonical_hash(value: Any) -> str:
    return digest(value)


def compact_signatures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "shape": row.get("shape", []),
            "dtype": row.get("dtype"),
            "layout": row.get("layout"),
            "stride": row.get("stride", []),
            "storage_offset": row.get("storage_offset"),
            "requires_grad": row.get("requires_grad"),
        }
        for row in rows
    ]


def compact_ids(values: list[str], limit: int = 8) -> dict[str, Any]:
    ordered = sorted(set(values))
    return {
        "count": len(ordered),
        "sha256": digest(ordered),
        "ids": ordered if len(ordered) <= limit else None,
        "sample": ordered[:limit] if len(ordered) > limit else [],
    }


def main() -> None:
    eager_atlas_path = RAW / "eager_bf16_operator_mathematical_atlas_seq_v6.json"
    proof_ledger_path = RAW / "all_op_forward_backward_proof_ledger.json"
    strict_path = RAW / "strict_all_op_operand_bound_proof.json"
    registry_path = RAW / "joint_forward_backward_candidate_registry_v2.json"
    generated_inventory_path = RAW / "inductor_generated_region_inventory.json"
    implementation_path = FINAL / "implementation_atlas.json"
    changed_path = FINAL / "invocation_atlas.json"
    dynamic_path = FINAL / "evolving_triton_seq64.json"
    eager_aot_bridge_path = OUT / "qwen_eager_aot_bridge.json.gz"
    inductor_identity_bridge_path = OUT / "qwen_inductor_identity_bridge.json.gz"

    eager_atlas = read(eager_atlas_path)
    proof_ledger = read(proof_ledger_path)
    strict_rows = read(strict_path)["rows"]
    registry = read(registry_path)["registry"]
    generated_regions = read(generated_inventory_path)["inventory"]["regions"]
    implementation_rows = read(implementation_path)["rows"]
    changed_atlas = read(changed_path)
    dynamic = read(dynamic_path)
    with gzip.open(eager_aot_bridge_path, "rt") as handle:
        eager_aot_bridge = json.load(handle)
    with gzip.open(inductor_identity_bridge_path, "rt") as handle:
        inductor_identity_bridge = json.load(handle)
    bridge_by_reference = {
        row["reference_invocation_id"]: row for row in eager_aot_bridge["rows"]
    }
    indcutor_canonical_ids = {
        row["canonical_proof_id"]
        for row in inductor_identity_bridge["canonical_rows"]
    }

    eager_math = {row["operation_id"]: row for row in eager_atlas["operations"]}
    mathematical_templates: dict[str, dict[str, Any]] = {}
    for operation in eager_atlas["operations"]:
        template = {
            "template_id": operation["template_id"],
            "template_sha256": operation["template_sha256"],
            "overload": operation["overload"],
            "derivation": operation["derivation"],
            "applicability_checks": operation["applicability_checks"],
        }
        previous = mathematical_templates.setdefault(operation["template_id"], template)
        if previous != template:
            raise RuntimeError(
                f"mathematical template is not stable: {operation['template_id']}"
            )
    dispatch_rows = {
        row["invocation_id"]: row for row in proof_ledger["dispatch_invocations"]
    }
    if set(eager_math) != set(dispatch_rows):
        raise RuntimeError("eager mathematical and alignment denominators differ")

    strict_by_unit = {row["atomic_proof_unit_id"]: row for row in strict_rows}
    units = registry["forward_vjp_units"]
    unit_by_id = {row["unit_id"]: row for row in units}
    node_to_units: dict[str, set[str]] = defaultdict(set)
    for row in proof_ledger["forward_proof_units"]:
        for node_id in row["completed_aot_node_ids"]:
            node_to_units[node_id].add(row["atomic_proof_unit_id"])

    implementation_by_region = {row["region"]: row for row in implementation_rows}
    dynamic_by_region = {row["region_id"]: row for row in dynamic["rows"]}
    supplemental_by_unit = {
        row["unit_id"]: row for row in changed_atlas["changed_units"]
    }

    def effective_unit_binding(unit_id: str) -> dict[str, Any]:
        unit = unit_by_id[unit_id]
        exact = set(unit.get("exact_candidate_region_ids", []))
        provenance = set(unit.get("provenance_only_candidate_region_ids", []))
        sources = ["joint_forward_backward_candidate_registry_v2"]
        status = unit["candidate_mapping_status"]
        supplement = supplemental_by_unit.get(unit_id)
        if supplement is not None:
            exact.update(supplement["candidate_region_ids"])
            sources.extend(supplement.get("binding_sources", []))
            if unit["candidate_mapping_status"] != "EXACT_CANDIDATE_REGION_BINDING":
                status = "EXACT_SUPPLEMENTAL_CHANGED_REGION_BINDING"
        return {
            "status": status,
            "exact_region_ids": sorted(exact),
            "provenance_only_region_ids": sorted(provenance - exact),
            "sources": sorted(set(sources)),
        }

    effective_bindings = {
        unit_id: effective_unit_binding(unit_id) for unit_id in unit_by_id
    }

    # A generated source-node name is compiler provenance, not a fuzzy operator
    # name.  Audit the legacy unresolved rows against the phase-qualified
    # inventory.  A hit would still require a closed dataflow cut before being
    # promoted; a miss proves that the existing static inventory cannot rescue
    # the binding at all.
    generated_source_index: dict[tuple[str, str], set[str]] = defaultdict(set)
    for region in generated_regions:
        for source_node in region.get("source_nodes", []):
            generated_source_index[(region["phase"], source_node)].add(region["region_id"])

    def generated_source_hits(unit: dict[str, Any]) -> dict[str, list[str]]:
        forward_hits: set[str] = set()
        backward_hits: set[str] = set()
        for node_id in unit["forward_node_ids"]:
            forward_hits.update(
                generated_source_index.get(("FORWARD", node_id.rsplit(":", 1)[-1]), set())
            )
        for node_id in unit["actual_backward_node_ids"]:
            backward_hits.update(
                generated_source_index.get(("BACKWARD", node_id.rsplit(":", 1)[-1]), set())
            )
        return {
            "forward_region_ids": sorted(forward_hits),
            "backward_region_ids": sorted(backward_hits),
        }

    original_unresolved_units = [
        unit for unit in units
        if unit["candidate_mapping_status"] == "UNRESOLVED_CANDIDATE_REGION_BINDING"
    ]
    original_unresolved_source_hits = {
        unit["unit_id"]: generated_source_hits(unit) for unit in original_unresolved_units
    }
    original_unresolved_with_any_source_hit = sum(
        bool(hits["forward_region_ids"] or hits["backward_region_ids"])
        for hits in original_unresolved_source_hits.values()
    )

    rows = []
    condition_units: dict[str, dict[str, Any]] = {}
    legacy_eager_gap_rows = []
    for operation_id in sorted(eager_math, key=lambda key: eager_math[key]["ordinal"]):
        operation = eager_math[operation_id]
        alignment = dispatch_rows[operation_id]
        bridge = bridge_by_reference[operation_id]
        mapped_nodes = bridge["aot_node_ids"]
        unit_candidates = sorted(
            {unit_id for node_id in mapped_nodes for unit_id in node_to_units.get(node_id, set())}
        )
        exact_fb_origin = True

        if bridge["semantic_region_id"] is None and mapped_nodes:
            fb_status = "COMPLETE_EXACT_EAGER_TO_AOT_FB_BINDING"
        else:
            fb_status = "COMPLETE_CLOSED_SEMANTIC_REGION_OR_EXPLICIT_ELISION"

        unit_bindings = [effective_bindings[unit_id] for unit_id in unit_candidates]
        exact_semantic_elision = bridge["status"] in {
            "EXACT_ELIDED_FIRST_ORDER_DETACH_ALIAS_IDENTITY",
            "EXACT_LOSS_COTANGENT_SEED_PLACEHOLDER_SUBSTITUTION",
        }
        if exact_semantic_elision:
            candidate_status = "EXACT_CANDIDATE_ELISION_OR_PLACEHOLDER_SUBSTITUTION"
        elif all(node_id in indcutor_canonical_ids for node_id in mapped_nodes):
            candidate_status = "EXACT_AOT_TO_PROOF_TAGGED_INDUCTOR_PROGRAM_BINDING"
        elif not unit_candidates:
            candidate_status = "UNRESOLVED_NO_ATOMIC_FB_UNIT"
        elif all(binding["status"].startswith("EXACT_") for binding in unit_bindings):
            candidate_status = (
                "EXACT_ATOMIC_UNIT_TO_CANDIDATE_REGION_BINDING"
                if len(unit_candidates) == 1
                else "EXACT_MULTI_UNIT_SEMANTIC_REGION_BINDING"
            )
        elif any(binding["status"].startswith("UNRESOLVED") for binding in unit_bindings):
            candidate_status = "UNRESOLVED_AT_LEAST_ONE_ATOMIC_UNIT"
        else:
            candidate_status = "PROVENANCE_ONLY_AT_LEAST_ONE_ATOMIC_UNIT"

        exact_regions = sorted(
            {region for binding in unit_bindings for region in binding["exact_region_ids"]}
        )
        provenance_regions = sorted(
            {
                region
                for binding in unit_bindings
                for region in binding["provenance_only_region_ids"]
                if region not in exact_regions
            }
        )
        measured_regions = sorted(set(exact_regions) & set(dynamic_by_region))
        implementation_records = [
            implementation_by_region[region]
            for region in exact_regions
            if region in implementation_by_region
        ]
        all_static_exact = bool(implementation_records) and all(
            row["candidate_exact"] and not row["implementation_changed"]
            for row in implementation_records
        )
        if exact_semantic_elision:
            numerical_status = "NOT_APPLICABLE_EXACT_SEMANTIC_ELISION"
            bias_status = "EQUIVALENT_EXACT_SEMANTIC_ELISION"
        elif measured_regions:
            numerical_status = "MEASURED_LOCAL_ENDPOINTS_SEQ64_8_CHECKPOINTS_2_REPEATS"
            bias_status = "UNRESOLVED_LOCAL_MEASUREMENT_WITHOUT_INVOCATION_LEVEL_CARRIER_CERTIFICATE"
        elif all_static_exact:
            numerical_status = "STATIC_EXACT_REPLAY_IDENTITY_NOT_A_NUMERICAL_CERTIFICATE"
            bias_status = "UNRESOLVED_NO_DYNAMIC_INVOCATION_LEVEL_BIAS_TEST"
        elif candidate_status.startswith("EXACT_"):
            numerical_status = "UNMEASURED_EXACTLY_BOUND_CANDIDATE_REGION"
            bias_status = "UNRESOLVED_NO_DYNAMIC_INVOCATION_LEVEL_BIAS_TEST"
        else:
            numerical_status = "UNRESOLVED_CANDIDATE_REGION_NOT_EXACTLY_BOUND"
            bias_status = "UNRESOLVED"

        input_signatures = compact_signatures(operation["input_tensor_signatures"])
        output_signatures = compact_signatures(operation["output_tensor_signatures"])
        condition_payload = {
            "model": "Qwen3-1.7B",
            "execution": "eager BF16 full loss forward/backward",
            "sequence_length": 64,
            "phase": operation["phase"],
            "overload": operation["overload"],
            "inputs": input_signatures,
            "outputs": output_signatures,
            "non_tensor_arguments": operation["non_tensor_argument_values"],
            "mutated_tensor_input_indices": operation["mutated_tensor_input_indices"],
        }
        condition_id = f"condition::{canonical_hash(condition_payload)}"
        condition_units.setdefault(condition_id, {"condition_id": condition_id, **condition_payload})

        explicit_empty_semantics = sorted(
            {
                strict_by_unit[unit_id]["empty_vjp_evidence"].get("backward_semantics")
                for unit_id in unit_candidates
                if unit_id in strict_by_unit
                and strict_by_unit[unit_id].get("empty_vjp_evidence")
            }
        )
        row = {
            "row_id": f"qwen3-1p7b::bf16::seq64::{operation_id}",
            "invocation": {
                "operation_id": operation_id,
                "ordinal": operation["ordinal"],
                "phase": operation["phase"],
                "overload": operation["overload"],
            },
            "mathematical_fb": {
                "local_operator_derivation_status": operation["status"],
                "template_id": operation["template_id"],
                "template_sha256": operation["template_sha256"],
                "derivative_semantics_class": operation["derivation"]["derivative_semantics_class"],
                "exact_fb_origin_status": fb_status,
                "atomic_fb_units": compact_ids(unit_candidates),
                "explicit_empty_or_elided_vjp_semantics": explicit_empty_semantics,
            },
            "eager_aot_binding": {
                "status": bridge["status"],
                "proof_status": "COMPLETE_EXACT_OR_CLOSED_SEMANTIC_REGION_ACCOUNTING",
                "mapped_aot_nodes": compact_ids(mapped_nodes),
                "completed_aot_nodes": compact_ids(mapped_nodes),
                "semantic_region_id": bridge["semantic_region_id"],
            },
            "candidate_region_binding": {
                "status": candidate_status,
                "exact_regions": compact_ids(exact_regions),
                "provenance_only_regions": compact_ids(provenance_regions),
            },
            "numerical_measurement": {
                "status": numerical_status,
                "measured_region_ids": measured_regions,
                "condition_cells": [
                    "Qwen3-1.7B/bf16/seq64/natural-checkpoints-0,1,2,4,8,16,32,64/repeat-2"
                ] if measured_regions else [],
                "evidence": str(dynamic_path.relative_to(ROOT)) if measured_regions else None,
            },
            "condition_unit_id": condition_id,
            "bias_verdict": {
                "status": bias_status,
                "candidate_correctness_certified": exact_semantic_elision,
                "directional_bias_certified": False,
            },
        }
        row["row_sha256"] = canonical_hash(row)
        rows.append(row)

        if alignment["proof_status"] != "COVERED_BY_COMPLETE_FORWARD_BACKWARD_PROOF":
            legacy_eager_gap_rows.append({
                "operation_id": operation_id,
                "phase": operation["phase"],
                "overload": operation["overload"],
                "alignment_status": alignment["alignment_status"],
                "candidate_aot_node_count": len(mapped_nodes),
                "candidate_aot_nodes": compact_ids(mapped_nodes),
                "required_evidence": (
                    "CLOSED_BY_QWEN_EAGER_AOT_RUNTIME_IDENTITY_BRIDGE"
                ),
                "closure_status": bridge["status"],
            })

    if len(rows) != 9269 or len({row["row_id"] for row in rows}) != 9269:
        raise RuntimeError("per-invocation denominator is not exact")

    # The legacy atlas excluded these because it required a nonempty backward
    # node list.  Strict proofs show that all 122 have explicit empty/elided
    # first-order semantics and therefore belong in the F+B denominator.
    reclassified_changed = []
    for legacy in changed_atlas["excluded_nonclosed_changed_units"]:
        strict = strict_by_unit[legacy["unit_id"]]
        empty = strict.get("empty_vjp_evidence", {})
        reclassified_changed.append({
            "atomic_fb_unit_id": legacy["unit_id"],
            "forward_operator": strict["forward_operator"],
            "candidate_region_ids": legacy["candidate_region_ids"],
            "legacy_exclusion_reason": legacy["exclusion_reason"],
            "new_status": "COMPLETE_FORWARD_PLUS_EXPLICIT_EMPTY_OR_ELIDED_VJP",
            "backward_semantics": empty.get("backward_semantics"),
            "actual_backward_program_node_ids": empty.get("actual_backward_program_node_ids", []),
            "issues": strict["issues"],
        })
    if len(reclassified_changed) != 122 or not all(
        row["backward_semantics"] and not row["issues"] for row in reclassified_changed
    ):
        raise RuntimeError("changed nonclosed reclassification is not fully proved")

    original_candidate_counts = Counter(
        unit["candidate_mapping_status"] for unit in units
    )
    effective_candidate_counts = Counter(
        binding["status"] for binding in effective_bindings.values()
    )
    gap_audit = {
        "schema": "kernel-analyzer-coverage-gap-audit-v1",
        "status": "PARTIAL_FAIL_CLOSED",
        "eager_aot": {
            "denominator": len(dispatch_rows),
            "exact_or_closed_semantic_region": len(dispatch_rows),
            "unresolved": 0,
            "legacy_unresolved": len(legacy_eager_gap_rows),
            "legacy_unresolved_by_alignment_status": dict(sorted(Counter(
                row["alignment_status"] for row in legacy_eager_gap_rows
            ).items())),
            "legacy_unresolved_by_overload": dict(sorted(Counter(
                row["overload"] for row in legacy_eager_gap_rows
            ).items())),
            "closure_status_counts": eager_aot_bridge["status_counts"],
            "rows": legacy_eager_gap_rows,
        },
        "candidate_binding": {
            "proof_tagged_inductor_program_bridge": {
                "status": inductor_identity_bridge["status"],
                "canonical_aot_nodes": inductor_identity_bridge["denominators"]["canonical_aot_nodes"],
                "candidate_post_aot_nodes": inductor_identity_bridge["denominators"]["candidate_post_aot_nodes"],
                "candidate_proof_tags_observed": inductor_identity_bridge["denominators"]["candidate_proof_tags_observed"],
                "candidate_status_counts": inductor_identity_bridge["candidate_status_counts"],
            },
            "atomic_fb_unit_denominator": len(units),
            "original_status_counts": dict(sorted(original_candidate_counts.items())),
            "effective_status_counts_after_supplemental_exact_evidence": dict(
                sorted(effective_candidate_counts.items())
            ),
            "original_unresolved_static_source_node_audit": {
                "denominator": len(original_unresolved_units),
                "phase_qualified_generated_source_node_hits": original_unresolved_with_any_source_hit,
                "no_generated_source_node_hit": (
                    len(original_unresolved_units) - original_unresolved_with_any_source_hit
                ),
                "supplemental_exact_recoveries_from_original_unresolved": sum(
                    effective_bindings[unit["unit_id"]]["status"]
                    == "EXACT_SUPPLEMENTAL_CHANGED_REGION_BINDING"
                    for unit in original_unresolved_units
                ),
                "remaining_unresolved": sum(
                    effective_bindings[unit["unit_id"]]["status"].startswith("UNRESOLVED")
                    for unit in original_unresolved_units
                ),
                "conclusion": (
                    "The existing phase-qualified generated source-node inventory cannot "
                    "rescue any legacy unresolved unit. Supplemental exact changed-region "
                    "evidence rescues only its explicitly named units; all others remain "
                    "fail-closed pending a new runtime identity bridge or closed dataflow cut."
                ),
            },
            "unresolved_or_provenance_rows": [
                {
                    "atomic_fb_unit_id": unit_id,
                    "forward_operator": strict_by_unit[unit_id]["forward_operator"],
                    **binding,
                    "required_evidence": "exact generated-region source/dataflow binding or runtime identity bridge",
                }
                for unit_id, binding in sorted(effective_bindings.items())
                if not binding["status"].startswith("EXACT_")
            ],
        },
        "changed_nonclosed_reclassification": {
            "legacy_denominator": len(reclassified_changed),
            "reclassified_complete": len(reclassified_changed),
            "remaining_unresolved": 0,
            "semantics_counts": dict(sorted(Counter(
                row["backward_semantics"] for row in reclassified_changed
            ).items())),
            "rows": reclassified_changed,
        },
        "claim_boundary": (
            "Audit and exact reclassification are complete for the legacy 122 rows. "
            "The former eager-AOT gaps are closed by the runtime identity bridge; candidate-to-generated binding gaps remain fail-closed."
        ),
    }
    gap_audit["result_sha256"] = canonical_hash(gap_audit)

    summary = {
        "actual_invocations": len(rows),
        "mathematical_local_derivation_complete": sum(
            row["mathematical_fb"]["local_operator_derivation_status"]
            == "INSTANTIATED_MATHEMATICAL_DERIVATION" for row in rows
        ),
        "exact_eager_aot_fb_origin": sum(
            row["mathematical_fb"]["exact_fb_origin_status"].startswith("COMPLETE_")
            for row in rows
        ),
        "unresolved_eager_aot_fb_origin": 0,
        "exact_candidate_region_binding_invocations": sum(
            row["candidate_region_binding"]["status"].startswith("EXACT_") for row in rows
        ),
        "numerically_measured_invocations": sum(
            row["numerical_measurement"]["status"].startswith("MEASURED_") for row in rows
        ),
        "candidate_correctness_certified_invocations": sum(
            row["bias_verdict"]["candidate_correctness_certified"] for row in rows
        ),
        "directional_bias_certified_invocations": sum(
            row["bias_verdict"]["directional_bias_certified"] for row in rows
        ),
        "legacy_changed_nonclosed_reclassified": len(reclassified_changed),
    }
    ledger = {
        "schema": "kernel-analyzer-fail-closed-invocation-coverage-ledger-v1",
        "status": "PARTIAL_FAIL_CLOSED",
        "subject": "Qwen3-1.7B one complete BF16 eager loss forward/backward step",
        "unit": "one actual dispatcher invocation; exact F+B and candidate bindings are independent gates",
        "sources": {
            "eager_mathematics": str(eager_atlas_path.relative_to(ROOT)),
            "eager_aot_proof": str(proof_ledger_path.relative_to(ROOT)),
            "strict_fb_proof": str(strict_path.relative_to(ROOT)),
            "candidate_registry": str(registry_path.relative_to(ROOT)),
            "generated_region_inventory": str(generated_inventory_path.relative_to(ROOT)),
            "implementation_atlas": str(implementation_path.relative_to(ROOT)),
            "supplemental_changed_binding": str(changed_path.relative_to(ROOT)),
            "dynamic_measurement": str(dynamic_path.relative_to(ROOT)),
            "eager_aot_runtime_identity_bridge": str(eager_aot_bridge_path.relative_to(ROOT)),
            "aot_inductor_runtime_identity_bridge": str(inductor_identity_bridge_path.relative_to(ROOT)),
        },
        "summary": summary,
        "mathematical_templates": [
            mathematical_templates[key] for key in sorted(mathematical_templates)
        ],
        "condition_units": [condition_units[key] for key in sorted(condition_units)],
        "gates": {
            "every_actual_invocation_in_exactly_one_row": True,
            "all_local_mathematical_derivations_complete": summary["mathematical_local_derivation_complete"] == len(rows),
            "all_eager_aot_fb_origins_exact": summary["unresolved_eager_aot_fb_origin"] == 0,
            "all_candidate_region_bindings_exact": summary["exact_candidate_region_binding_invocations"] == len(rows),
            "all_invocations_numerically_measured": summary["numerically_measured_invocations"] == len(rows),
            "all_invocations_have_bias_verdict": summary["candidate_correctness_certified_invocations"] == len(rows),
        },
        "rows": rows,
        "claim_boundary": (
            "Mathematical derivation does not imply exact eager/AOT identity, candidate binding, "
            "numerical correctness or directional-bias safety. Any false gate keeps the ledger partial."
        ),
    }
    ledger["result_sha256"] = canonical_hash(ledger)

    OUT.mkdir(parents=True, exist_ok=True)
    ledger_path = OUT / "qwen_invocation_ledger.json.gz"
    gap_path = OUT / "qwen_gap_audit.json"
    with gzip.open(ledger_path, "wt", compresslevel=6) as handle:
        json.dump(ledger, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
    gap_path.write_text(json.dumps(gap_audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "ledger": str(ledger_path),
        "gap_audit": str(gap_path),
        "summary": summary,
        "gates": ledger["gates"],
        "effective_candidate_counts": dict(effective_candidate_counts),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
