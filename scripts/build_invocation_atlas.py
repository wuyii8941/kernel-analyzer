#!/usr/bin/env python3
"""Bind only real generated implementation changes to closed F+B units.

The source census is per generated site, while the mathematical registry is
per forward invocation plus its actual backward.  This produces the unit-level
search table without deduplicating away repeated invocations.  Exact replays,
unresolved provenance-only bindings, and external calls remain explicit
denominator records but are not silently promoted to implementation changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("archive/round1_raw/training_semantic_oracle/qwen3_1p7b/full_step_inventory")
ALLOWED = {
    "DIRECT_SCATTER_ADD_SOURCE_FAITHFUL_SINGLE_STATE_REPLAY",
    "EXPLICIT_FP32_REDUCTION_SCHEDULE_DIFFERENCE",
    "MATERIALIZATION_OR_ROUNDING_SCHEDULE_INTERVENTION",
    "SAME_PRECISION_GENERATED_SCHEDULE_DIFFERENCE",
}


def closed(unit: dict[str, Any]) -> bool:
    return (
        unit.get("vjp_status") == "EXACT_ACTUAL_BACKWARD_PROGRAM"
        and bool(unit.get("actual_backward_node_ids"))
    )


def atomic_unit_ids(value: Any) -> set[str]:
    """Read explicit atomic IDs from an exact supplemental binding.

    This deliberately does not infer IDs from names or shapes.  Some later
    ledgers bind a fused generated site directly to atomic F+VJP proof units.
    """
    found: set[str] = set()
    if isinstance(value, dict):
        unit_id = value.get("atomic_proof_unit_id")
        if isinstance(unit_id, str):
            found.add(unit_id)
        for child in value.values():
            found.update(atomic_unit_ids(child))
    elif isinstance(value, list):
        for child in value:
            found.update(atomic_unit_ids(child))
    return found


def supplemental_unit_ids(
    row: dict[str, Any],
    *,
    units_by_id: dict[str, dict[str, Any]],
    units_by_region: dict[str, set[str]],
) -> tuple[set[str], str | None]:
    """Resolve a later exact binding to closed atomic F+B units.

    Preference is explicit atomic proof IDs.  RMSNorm chain ledgers instead
    bind the complete fused semantic region to an already exact forward
    region; terminal NLL binds its partition to the already exact actual VJP
    region.  Those anchors are explicit fields in the supplemental evidence.
    """
    supplements = [
        (key, value)
        for key, value in row.items()
        if key.startswith("supplemental_") and value is not None
    ]
    if len(supplements) != 1:
        return set(), None
    field, evidence = supplements[0]
    if (
        row.get("region_id") == "backward:direct_aten:0"
        and evidence == "COMPLETE_EMBEDDING_FORWARD_BACKWARD_PROOF"
    ):
        unit_id = "aot-forward-vjp::forward:graph0:embedding"
        if unit_id in units_by_id and closed(units_by_id[unit_id]):
            return {unit_id}, f"{field}:EXPLICIT_DIRECT_EMBEDDING_FVJP_BINDING"
    resolved = atomic_unit_ids(evidence)
    basis = "EXPLICIT_SUPPLEMENTAL_ATOMIC_PROOF_UNIT_IDS"
    if not resolved and isinstance(evidence, dict):
        anchors: list[str] = []
        if isinstance(evidence.get("forward_region_id"), str):
            anchors.append(evidence["forward_region_id"])
        anchors.extend(
            value
            for value in evidence.get("forward_region_ids", [])
            if isinstance(value, str)
        )
        semantic_unit = evidence.get("semantic_unit")
        if isinstance(semantic_unit, dict) and isinstance(
            semantic_unit.get("backward_region_id"), str
        ):
            anchors.append(semantic_unit["backward_region_id"])
        for region_id in anchors:
            resolved.update(units_by_region.get(region_id, set()))
        basis = "EXPLICIT_SUPPLEMENTAL_SEMANTIC_REGION_ANCHOR"
    resolved = {
        unit_id
        for unit_id in resolved
        if unit_id in units_by_id and closed(units_by_id[unit_id])
    }
    return resolved, f"{field}:{basis}" if resolved else None


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=Path("results/final/invocation_atlas.json"))
    args = parser.parse_args()
    census_path = args.root / "generated_implementation_mechanism_census_v1.json"
    registry_path = args.root / "joint_forward_backward_candidate_registry_v2.json"
    gap_path = args.root / "generated_implementation_correctness_gap_ledger_v3.json"
    census = json.loads(census_path.read_text())
    registry = json.loads(registry_path.read_text())["registry"]
    gap = json.loads(gap_path.read_text())
    gap_by_id = {row["region_id"]: row for row in gap["rows"]}
    # The v3 gap ledger expands the legacy launch-only census with the direct
    # generated aten.index_put_ compute call.  It is part of the denominator.
    site_by_id = {row["region_id"]: row for row in census["rows"]}
    for region_id, row in gap_by_id.items():
        if region_id not in site_by_id:
            site_by_id[region_id] = row

    changed_sites = []
    exact_sites = []
    for row in site_by_id.values():
        mechanism = row.get("mechanism_annotation") or "UNRESOLVED"
        compact = {
            "region_id": row["region_id"],
            "phase": row["phase"],
            "kind": row["kind"],
            "symbol": row["symbol"],
            "mechanism": mechanism,
            "control": row.get("control_name"),
            "candidate_exact": bool((row.get("candidate_same_precision") or {}).get("all_exact", False)),
            "status": row.get("status"),
        }
        if mechanism in ALLOWED:
            changed_sites.append(compact)
        else:
            exact_sites.append(compact)

    units = registry["forward_vjp_units"]
    units_by_id = {unit["unit_id"]: unit for unit in units}
    units_by_region: dict[str, set[str]] = defaultdict(set)
    for unit in units:
        for region_id in unit.get("exact_candidate_region_ids", []):
            units_by_region[region_id].add(unit["unit_id"])

    supplemental_by_unit: dict[str, set[str]] = defaultdict(set)
    supplemental_basis_by_unit: dict[str, set[str]] = defaultdict(set)
    supplemental_site_status: dict[str, str] = {}
    for site in changed_sites:
        row = gap_by_id.get(site["region_id"])
        if row is None or not str(row.get("effective_proof_binding_status", "")).startswith("EXACT_"):
            continue
        unit_ids, basis = supplemental_unit_ids(
            row, units_by_id=units_by_id, units_by_region=units_by_region
        )
        if not unit_ids:
            continue
        supplemental_site_status[site["region_id"]] = row["effective_proof_binding_status"]
        for unit_id in unit_ids:
            supplemental_by_unit[unit_id].add(site["region_id"])
            supplemental_basis_by_unit[unit_id].add(str(basis))

    changed_units_by_id: dict[str, dict[str, Any]] = {}
    excluded_changed_units = []
    mapped_site_ids: set[str] = set()
    any_bound_site_ids: set[str] = set()
    nonclosed_bound_site_ids: set[str] = set()
    for unit in units:
        site_ids = sorted(
            set(unit.get("exact_candidate_region_ids", []))
            | supplemental_by_unit.get(unit["unit_id"], set())
        )
        site_rows = [site_by_id[site_id] for site_id in site_ids if site_id in site_by_id]
        changed = [row for row in site_rows if row.get("mechanism_annotation") in ALLOWED]
        if not changed:
            continue
        changed_ids = {row["region_id"] for row in changed}
        any_bound_site_ids.update(changed_ids)
        mechanisms = sorted({row["mechanism_annotation"] for row in changed})
        compact_unit = {
            "unit_id": unit["unit_id"],
            "unit_kind": unit["unit_kind"],
            "forward_node_ids": unit.get("forward_node_ids", []),
            "actual_backward_node_ids": unit.get("actual_backward_node_ids", []),
            "vjp_status": unit.get("vjp_status"),
            "mathematical_derivation_sha256": unit.get("mathematical_derivation_sha256"),
            "candidate_mapping_status": unit.get("candidate_mapping_status"),
            "candidate_region_ids": sorted(row["region_id"] for row in changed),
            "mechanisms": mechanisms,
            "control_names": sorted({row.get("control_name") for row in changed if row.get("control_name")}),
            "binding_sources": (
                ["JOINT_FORWARD_BACKWARD_CANDIDATE_REGISTRY_V2"]
                if any(
                    row["region_id"] in set(unit.get("exact_candidate_region_ids", []))
                    for row in changed
                )
                else []
            ) + sorted(supplemental_basis_by_unit.get(unit["unit_id"], set())),
            "real_implementation_change": True,
        }
        closed_fbv = closed(unit)
        if closed_fbv:
            mapped_site_ids.update(changed_ids)
            changed_units_by_id[unit["unit_id"]] = compact_unit
        else:
            nonclosed_bound_site_ids.update(changed_ids)
            compact_unit["exclusion_reason"] = "NO_ACTUAL_BACKWARD_PROGRAM_OR_NONEXACT_VJP"
            excluded_changed_units.append(compact_unit)

    changed_units = [changed_units_by_id[key] for key in sorted(changed_units_by_id)]
    unmapped_changed = [row for row in changed_sites if row["region_id"] not in mapped_site_ids]
    by_mechanism = Counter(row["mechanism"] for row in changed_sites)
    by_phase = Counter(row["phase"] for row in changed_sites)
    by_unit_mechanism = Counter(mech for unit in changed_units for mech in unit["mechanisms"])
    output = {
        "schema": "kernel-analyzer-invocation-implementation-atlas-v1",
        "subject": "Qwen3-1.7B generated regions bound to mathematical F+B units",
        "sources": {
            "census": str(census_path),
            "census_sha256": digest(census_path),
            "registry": str(registry_path),
            "registry_sha256": digest(registry_path),
            "supplemental_gap_ledger": str(gap_path),
            "supplemental_gap_ledger_sha256": digest(gap_path),
        },
        "allowed_real_change_mechanisms": sorted(ALLOWED),
        "denominator": {
            "generated_sites": len(site_by_id),
            "semantic_forward_vjp_units": len(units),
            "exact_replay_sites_excluded": len(exact_sites),
            "real_changed_sites": len(changed_sites),
            "real_changed_sites_bound_to_exact_fbv_units": len(mapped_site_ids),
            "real_changed_sites_without_exact_fbv_binding": len(unmapped_changed),
            "real_changed_sites_bound_to_any_fbv_unit": len(any_bound_site_ids),
            "real_changed_sites_bound_only_to_nonclosed_fbv_units": len(nonclosed_bound_site_ids - mapped_site_ids),
            "real_changed_sites_without_any_fbv_binding": len(changed_sites) - len(any_bound_site_ids),
            "real_changed_sites_bound_via_supplemental_exact_evidence": len(supplemental_site_status),
            "changed_fbv_units": len(changed_units),
            "excluded_nonclosed_changed_units": len(excluded_changed_units),
        },
        "site_counts": {
            "by_mechanism": dict(sorted(by_mechanism.items())),
            "by_phase": dict(sorted(by_phase.items())),
            "by_unit_mechanism": dict(sorted(by_unit_mechanism.items())),
        },
        "changed_units": changed_units,
        "supplemental_exact_site_bindings": dict(sorted(supplemental_site_status.items())),
        "excluded_nonclosed_changed_units": excluded_changed_units,
        "unmapped_changed_sites": unmapped_changed,
        "boundary": (
            "Only units with vjp_status=EXACT_ACTUAL_BACKWARD_PROGRAM and a nonempty actual backward node list, "
            "plus an explicit non-replay mechanism, enter changed_units. Later exact supplemental bindings are "
            "resolved only through explicit atomic proof IDs or explicit exact semantic-region anchors; name/shape "
            "matching is forbidden. Analytic-zero/unreached VJPs are retained "
            "in excluded_nonclosed_changed_units and never count as training F+B units. This is an implementation-"
            "difference table, not a numerical correctness verdict; dynamic checkpoint measurements are required separately."
        ),
    }
    output["result_sha256"] = hashlib.sha256(
        json.dumps(output, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), **output["denominator"]}, sort_keys=True))


if __name__ == "__main__":
    main()
