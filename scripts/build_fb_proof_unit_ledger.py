#!/usr/bin/env python3
"""Partition invocation censuses into minimal closed F+B proof units."""

from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "results/coverage"
CONTRACT = COVERAGE / "coverage_contract.json"
OUTPUT = COVERAGE / "fb_proof_unit_ledger.json.gz"


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


class UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def compact(values: list[str], sample_size: int = 8) -> dict[str, Any]:
    ordered = sorted(set(values))
    return {
        "count": len(ordered),
        "sha256": digest(ordered),
        "ids": ordered,
        "sample": ordered[:sample_size],
    }


def fb_status(row: dict[str, Any]) -> str:
    math = row["mathematical_fb"]
    return str(math.get("exact_fb_origin_status", math.get("fb_origin_status", "")))


def formula_id(row: dict[str, Any]) -> str:
    math = row["mathematical_fb"]
    return str(math.get("template_sha256", math.get("formula_sha256", "")))


def analytic_proof_status(row: dict[str, Any]) -> str:
    """Return the per-invocation analytic-proof status, never infer it.

    The architecture ledgers historically called a registered overload formula
    plus an autograd ``seq_nr`` link a complete local derivation.  That is useful
    origin accounting, but it does not prove that the concrete saved tensors,
    cotangent edge, non-tensor arguments and actual backward arithmetic realize
    the instantiated VJP.  Only ledgers carrying an explicit instantiated
    derivation marker may pass this gate.
    """

    math = row["mathematical_fb"]
    proof = math.get("concrete_program_proof") or {}
    required = (
        "saved_tensor_origins_exact",
        "cotangent_edge_exact",
        "backward_program_matches_analytic_vjp",
        "non_tensor_arguments_exact",
        "output_edges_exact",
    )
    digests = (
        "forward_program_sha256",
        "backward_program_sha256",
        "analytic_derivation_sha256",
    )
    if all(proof.get(name) is True for name in required) and all(
        bool(proof.get(name)) for name in digests
    ):
        return "ANALYTICALLY_PROVED"
    if formula_id(row):
        return "FORMULA_REGISTERED_ONLY"
    return "UNRESOLVED"


def build_components(
    model_key: str, ledger: dict[str, Any]
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    rows = ledger["rows"]
    row_by_id = {row["row_id"]: row for row in rows}
    invocation_to_row = {
        row["invocation"].get("invocation_id", row["invocation"].get("operation_id")):
        row["row_id"] for row in rows
    }
    union = UnionFind(list(row_by_id))
    dangling_links: list[dict[str, str]] = []

    if model_key == "qwen3_1p7b" and all(
        "atomic_fb_units" in row["mathematical_fb"] for row in rows
    ):
        by_atomic: dict[str, list[str]] = defaultdict(list)
        by_semantic: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            units = row["mathematical_fb"]["atomic_fb_units"]
            for unit_id in units.get("ids") or []:
                by_atomic[str(unit_id)].append(row["row_id"])
            semantic = row["eager_aot_binding"].get("semantic_region_id")
            if semantic:
                by_semantic[str(semantic)].append(row["row_id"])
        for members in list(by_atomic.values()) + list(by_semantic.values()):
            for member in members[1:]:
                union.union(members[0], member)
    else:
        for row in rows:
            origin_block = row["mathematical_fb"]["forward_origin_invocations"]
            for origin in origin_block.get("ids") or []:
                origin_row = invocation_to_row.get(str(origin))
                if origin_row is None:
                    dangling_links.append({"row_id": row["row_id"], "origin": str(origin)})
                else:
                    union.union(row["row_id"], origin_row)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row_id, row in row_by_id.items():
        grouped[union.find(row_id)].append(row)
    components = [sorted(value, key=lambda row: row["row_id"]) for value in grouped.values()]
    components.sort(key=lambda value: value[0]["row_id"])
    audit = {
        "source_invocations": len(rows),
        "components": len(components),
        "dangling_origin_links": dangling_links,
        "all_rows_in_exactly_one_component": sum(map(len, components)) == len(rows),
    }
    return components, audit


def candidate_cell(
    members: list[dict[str, Any]], candidate: str, primary_candidate: str
) -> dict[str, Any]:
    if candidate != primary_candidate:
        return {
            "mapping_status": "PENDING_NO_MIGRATED_CANDIDATE_SPECIFIC_EVIDENCE",
            "measurement_status": "UNMEASURED",
            "correctness_status": "UNRESOLVED",
            "directional_bias_status": "UNRESOLVED",
            "candidate_region_ids": compact([]),
        }
    mapping_statuses = [row["candidate_region_binding"]["status"] for row in members]
    numerical_statuses = [row["numerical_measurement"]["status"] for row in members]
    exact_mapping = all(str(value).startswith("EXACT_") for value in mapping_statuses)
    measured = [
        value for value in numerical_statuses
        if str(value).startswith("MEASURED_") or str(value).startswith("NOT_APPLICABLE_")
    ]
    if len(measured) == len(numerical_statuses):
        measurement_status = "COMPLETE_ALL_MEMBER_ENDPOINTS"
    elif measured:
        measurement_status = "PARTIAL_MEMBER_ENDPOINTS"
    else:
        measurement_status = "UNMEASURED"
    region_ids = []
    for row in members:
        binding = row["candidate_region_binding"]
        for field in ("exact_regions", "candidate_region_ids"):
            region_ids.extend((binding.get(field) or {}).get("ids") or [])
    all_correct = all(
        bool(row["bias_verdict"].get("candidate_correctness_certified", False))
        for row in members
    )
    directional = any(
        bool(row["bias_verdict"].get("directional_bias_certified", False))
        for row in members
    )
    return {
        "mapping_status": "EXACT_ALL_MEMBERS" if exact_mapping else "UNRESOLVED",
        "member_mapping_status_counts": dict(sorted(Counter(mapping_statuses).items())),
        "measurement_status": measurement_status,
        "member_measurement_status_counts": dict(sorted(Counter(numerical_statuses).items())),
        "correctness_status": "EQUIVALENT" if all_correct else "UNRESOLVED",
        "directional_bias_status": "PASS" if directional else "UNRESOLVED",
        "candidate_region_ids": compact(region_ids),
    }


def proof_unit(
    model_key: str, model: dict[str, Any], members: list[dict[str, Any]],
    source_sha256: str, shape_stratum: str = "batch1_seq64",
    component_witness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    forward = [row for row in members if row["invocation"]["phase"] == "FORWARD"]
    backward = [row for row in members if row["invocation"]["phase"] == "BACKWARD"]
    if not forward:
        kind = "AUXILIARY_BACKWARD_UNIT"
    elif not backward:
        kind = "EMPTY_OR_ELIDED_FB_UNIT"
    elif len(forward) == 1:
        kind = "FORWARD_ACTUAL_BACKWARD_UNIT"
    else:
        kind = "FUSED_SHARED_BACKWARD_SEMANTIC_REGION"
    member_ids = [row["row_id"] for row in members]
    identity = {
        "model": model_key,
        "shape_stratum": shape_stratum,
        "member_row_ids": sorted(member_ids),
        "source_ledger_sha256": source_sha256,
    }
    unit_id = f"fb-unit::{digest(identity)}"
    candidate_cells = {
        candidate: candidate_cell(members, candidate, model["primary_candidate"])
        for candidate in model["candidate_configurations"]
    }
    statuses = [fb_status(row) for row in members]
    origin_bound = all(value.startswith("COMPLETE_") for value in statuses)
    proof_statuses = [analytic_proof_status(row) for row in members]
    formula_registered = all(value != "UNRESOLVED" for value in proof_statuses)
    witness_proof = (component_witness or {}).get("concrete_program_proof", {})
    witness_required_flags = (
        "saved_tensor_origins_exact", "cotangent_edge_exact",
        "backward_program_matches_analytic_vjp", "non_tensor_arguments_exact",
        "output_edges_exact",
    )
    witness_required_digests = (
        "forward_program_sha256", "backward_program_sha256",
        "analytic_derivation_sha256",
    )
    external_witness_exact = (
        component_witness is not None
        and component_witness.get("status") == "ANALYTICALLY_PROVED"
        and component_witness.get("member_row_ids_sha256") == digest(sorted(member_ids))
        and all(witness_proof.get(name) is True for name in witness_required_flags)
        and all(bool(witness_proof.get(name)) for name in witness_required_digests)
    )
    analytically_proved = origin_bound and (
        external_witness_exact
        or all(value == "ANALYTICALLY_PROVED" for value in proof_statuses)
    )
    row = {
        "unit_id": unit_id,
        "model": model_key,
        "scope": model["scope"],
        "shape_stratum": shape_stratum,
        "unit_kind": kind,
        "denominator_role": (
            "AUXILIARY_BACKWARD_ACCOUNTING"
            if kind == "AUXILIARY_BACKWARD_UNIT" else "PRIMARY_FB_PROOF"
        ),
        "members": {
            "all_invocation_rows": compact(member_ids),
            "forward_invocation_rows": compact([row["row_id"] for row in forward]),
            "backward_invocation_rows": compact([row["row_id"] for row in backward]),
        },
        "mathematics": {
            "status": (
                "ANALYTICALLY_PROVED"
                if analytically_proved else
                "ORIGIN_BOUND_FORMULA_REGISTERED_ONLY"
                if origin_bound and formula_registered else
                "UNRESOLVED"
            ),
            "fb_origin_status_counts": dict(sorted(Counter(statuses).items())),
            "analytic_proof_status_counts": dict(
                sorted(Counter(proof_statuses).items())
            ),
            "component_witness_status": (
                component_witness.get("status") if component_witness else None
            ),
            "component_witness_sha256": (
                component_witness.get("witness_sha256") if component_witness else None
            ),
            "component_witness_exact": external_witness_exact,
            "formula_sha256s": compact([formula_id(row) for row in members if formula_id(row)]),
        },
        "candidate_cells": candidate_cells,
        "gates": {
            "EXECUTED": True,
            "FB_ORIGIN_BOUND": origin_bound,
            "FORMULA_REGISTERED": formula_registered,
            "FB_ANALYTICALLY_PROVED": analytically_proved,
            # Compatibility alias.  It is deliberately as strict as the new
            # analytic gate; origin binding or a formula string cannot pass it.
            "MATH_CLOSED": analytically_proved,
            "CANDIDATE_BOUND": any(
                cell["mapping_status"] == "EXACT_ALL_MEMBERS"
                for cell in candidate_cells.values()
            ),
            "NUMERIC_MEASURED": any(
                cell["measurement_status"] == "COMPLETE_ALL_MEMBER_ENDPOINTS"
                for cell in candidate_cells.values()
            ),
            "T1_LOCAL": False,
            "T2_CAUSAL": False,
            "T3_COHERENT": any(
                cell["directional_bias_status"] == "PASS"
                for cell in candidate_cells.values()
            ),
            "T4_ACCUMULATION": False,
        },
    }
    row["unit_sha256"] = digest(row)
    return row


def main() -> None:
    contract = json.loads(CONTRACT.read_text())
    units: list[dict[str, Any]] = []
    source_bindings = {}
    model_audits = {}
    invocation_owners: dict[str, str] = {}
    for model_key, model in contract["models"].items():
        ledger_path = ROOT / model["ledger"]
        ledger = load(ledger_path)
        source_sha256 = ledger["result_sha256"]
        components, audit = build_components(model_key, ledger)
        if audit["dangling_origin_links"]:
            raise RuntimeError(f"{model_key}: dangling forward-origin links")
        model_units = [
            proof_unit(model_key, model, members, source_sha256)
            for members in components
        ]
        if len(model_units) != len(components):
            raise RuntimeError("proof-unit/component cardinality mismatch")
        for unit, members in zip(model_units, components):
            for member in members:
                row_id = member["row_id"]
                if row_id in invocation_owners:
                    raise RuntimeError(f"invocation row has two owners: {row_id}")
                invocation_owners[row_id] = unit["unit_id"]
        units.extend(model_units)
        source_bindings[model_key] = {
            "path": model["ledger"], "result_sha256": source_sha256,
        }
        audit["proof_unit_kinds"] = dict(sorted(Counter(
            unit["unit_kind"] for unit in model_units
        ).items()))
        audit["origin_bound_units"] = sum(
            unit["gates"]["FB_ORIGIN_BOUND"] for unit in model_units
        )
        audit["formula_registered_units"] = sum(
            unit["gates"]["FORMULA_REGISTERED"] for unit in model_units
        )
        audit["analytically_proved_units"] = sum(
            unit["gates"]["FB_ANALYTICALLY_PROVED"] for unit in model_units
        )
        model_audits[model_key] = audit

    source_invocations = sum(audit["source_invocations"] for audit in model_audits.values())
    if len(invocation_owners) != source_invocations:
        raise RuntimeError("F+B units do not partition the complete invocation census")
    shape_matrix = {}
    denominator_cells = {}
    for model_key, model in contract["models"].items():
        model_units = [unit for unit in units if unit["model"] == model_key]
        shape_matrix[model_key] = {
            f"batch1_seq{sequence}": (
                "CAPTURED" if sequence == 64 else "PENDING_EXECUTION_DERIVED_WITNESS"
            )
            for sequence in contract["execution_strata"]["sequence_lengths"]
        }
        denominator_cells[model_key] = {}
        for sequence in contract["execution_strata"]["sequence_lengths"]:
            cell_key = f"batch1_seq{sequence}"
            captured = sequence == 64
            denominator_cells[model_key][cell_key] = {
                "status": (
                    "CAPTURED_EXECUTION_DERIVED"
                    if captured else "PENDING_EXECUTION_DERIVED_WITNESS"
                ),
                "reference_program": contract["denominator_axes"]["reference_program"],
                "source_ledger": model["ledger"] if captured else None,
                "execution_census_invocations": (
                    model_audits[model_key]["source_invocations"] if captured else None
                ),
                "primary_fb_proof_units": (
                    sum(unit["denominator_role"] == "PRIMARY_FB_PROOF"
                        for unit in model_units) if captured else None
                ),
                "auxiliary_backward_accounting_units": (
                    sum(unit["denominator_role"] == "AUXILIARY_BACKWARD_ACCOUNTING"
                        for unit in model_units) if captured else None
                ),
            }
    candidate_regions: dict[tuple[str, str], set[str]] = defaultdict(set)
    region_to_units: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for unit in units:
        for candidate, cell in unit["candidate_cells"].items():
            for region in cell["candidate_region_ids"]["ids"]:
                candidate_regions[(unit["model"], candidate)].add(region)
                if unit["denominator_role"] == "PRIMARY_FB_PROOF":
                    region_to_units[(unit["model"], candidate, region)].add(unit["unit_id"])
    fusion_multiplicity = Counter(
        len(value) for value in region_to_units.values()
    )
    payload = {
        "schema": "kernel-analyzer-fb-proof-unit-ledger-v2",
        "status": "PARTIAL_FAIL_CLOSED",
        "contract_sha256": contract["contract_sha256"],
        "source_ledgers": source_bindings,
        "denominators": {
            "execution_census_invocations": source_invocations,
            "closed_accounting_components": len(units),
            "primary_fb_proof_units": sum(
                unit["denominator_role"] == "PRIMARY_FB_PROOF" for unit in units
            ),
            "auxiliary_backward_accounting_units": sum(
                unit["denominator_role"] == "AUXILIARY_BACKWARD_ACCOUNTING"
                for unit in units
            ),
            "active_full_step_fb_proof_units": sum(
                unit["scope"] == "FULL_STEP"
                and unit["denominator_role"] == "PRIMARY_FB_PROOF"
                for unit in units
            ),
            "retained_paused_fb_proof_units": sum(
                unit["scope"] != "FULL_STEP"
                and unit["denominator_role"] == "PRIMARY_FB_PROOF"
                for unit in units
            ),
            "candidate_runtime_regions_with_migrated_exact_ids": sum(
                len(value) for value in candidate_regions.values()
            ),
            "declared_active_model_shape_cells": sum(
                model["scope"] == "FULL_STEP" for model in contract["models"].values()
            ) * len(contract["execution_strata"]["sequence_lengths"]),
            "captured_active_model_shape_cells": sum(
                model["scope"] == "FULL_STEP" for model in contract["models"].values()
            ),
            "pending_active_model_shape_cells": sum(
                model["scope"] == "FULL_STEP" for model in contract["models"].values()
            ) * (len(contract["execution_strata"]["sequence_lengths"]) - 1),
        },
        "shape_coverage_matrix": shape_matrix,
        "fb_denominator_cells": denominator_cells,
        "model_audits": model_audits,
        "partition_audit": {
            "all_source_invocations_have_exactly_one_accounting_owner": True,
            "owned_invocations": len(invocation_owners),
            "owner_map_sha256": digest(invocation_owners),
            "states_or_repeats_multiply_primary_denominator": False,
            "every_primary_unit_contains_forward": all(
                unit["members"]["forward_invocation_rows"]["count"] > 0
                for unit in units
                if unit["denominator_role"] == "PRIMARY_FB_PROOF"
            ),
            "every_backward_only_component_is_auxiliary": all(
                unit["denominator_role"] == "AUXILIARY_BACKWARD_ACCOUNTING"
                for unit in units
                if unit["members"]["forward_invocation_rows"]["count"] == 0
            ),
        },
        "fusion_audit": {
            "candidate_region_to_primary_fb_unit_multiplicity": dict(
                sorted(fusion_multiplicity.items())
            ),
            "many_to_one_regions_preserve_all_fb_units": True,
        },
        "units": sorted(units, key=lambda row: row["unit_id"]),
        "claim_boundary": (
            "Primary F+B units are the scientific denominator. Together with separately "
            "reported backward-only auxiliary accounting units, they losslessly partition "
            "the invocation census. Candidate mapping, measurement, and bias gates remain "
            "independent."
        ),
    }
    # Canonicalize JSON object keys before hashing. Counter keys may be ints in
    # memory but become strings on disk; the digest must describe the artifact
    # a reader actually loads.
    payload = json.loads(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    payload["result_sha256"] = digest(payload)
    # Fix both gzip timestamp and embedded filename so rebuilding the same
    # scientific ledger produces identical archive bytes.
    with OUTPUT.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                           compresslevel=9, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    print(json.dumps({
        "output": str(OUTPUT.relative_to(ROOT)),
        "denominators": payload["denominators"],
        "result_sha256": payload["result_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
