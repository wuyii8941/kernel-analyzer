#!/usr/bin/env python3
"""Build a value-blind implementation census from all-op screen artifacts.

Every runtime invocation and exact callsite/ABI identity remains in the coverage
denominator.  Deep follow-up is deduplicated by implementation pattern so that
the same generated/library operation repeated across layers is measured once.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from kernel_analyzer.implementation_identity import build_implementation_identity


def _read(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _records(document: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    for state_id, state in document.get("states", {}).items():
        repeats = state.get("repeats", [])
        if not repeats:
            continue
        # Repeats validate runtime stability; they do not create new coordinates.
        for record in repeats[0].get("summary", {}).get("records", []):
            yield state_id, record


def _campaign_rows(screen_path: Path, screen: dict[str, Any]) -> dict[str, dict[str, Any]]:
    campaign = screen.get("campaign")
    if not campaign:
        return {}
    path = Path(campaign)
    if not path.is_absolute():
        path = screen_path.parents[4] / path if str(path).startswith("results/") else screen_path.parent / path
    if not path.exists():
        return {}
    return {row["region_id"]: row for row in _read(path).get("rows", [])}


def _identity(record: dict[str, Any], campaign: dict[str, Any]) -> dict[str, Any] | None:
    if "runtime_pointer_contracts" in record:
        contracts = record["runtime_pointer_contracts"]
    elif "runtime_operand_contracts" in record:
        contracts = record["runtime_operand_contracts"]
    else:
        return None
    phase = str(record.get("phase", campaign.get("phase", "UNRESOLVED")))
    symbol = str(
        record.get("symbol") or record.get("operation") or record.get("function")
        or campaign.get("symbol") or "UNRESOLVED"
    )
    semantic = campaign.get("original_aten") or record.get("semantic_operations") or []
    program_digest = (
        campaign.get("embedded_program_sha256")
        or record.get("implementation_sha256")
        or record.get("typed_reference_program_sha256")
    )
    # A direct op with no tensor ABI still needs a stable call identity.  For
    # library calls with tensor contracts, the generated callsite line is not
    # the implementation and would falsely count every layer as novel.
    if not contracts and program_digest is None:
        program_digest = record.get("source_line_sha256")
    kind = "TRITON_GENERATED" if "runtime_pointer_contracts" in record else "EXTERNAL_OR_LIBRARY"
    return build_implementation_identity(
        backend="inductor",
        implementation_kind=kind,
        phase=phase,
        operation=symbol,
        operand_contracts=contracts,
        program_digest=program_digest,
        # A generated source line is a callsite, not an implementation class;
        # using it here would relabel every transformer layer as a new kernel.
        structural_program_digest=None,
        semantic_operations=semantic,
        fusion_boundary=campaign.get("source_nodes", []),
        launch_contract=record.get("launch_contract", {}),
    )


def build(paths: list[Path]) -> dict[str, Any]:
    invocations = 0
    unresolved = 0
    identities: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    states: defaultdict[str, set[str]] = defaultdict(set)
    sources: defaultdict[str, set[str]] = defaultdict(set)
    for path in paths:
        doc = _read(path)
        campaign_rows = _campaign_rows(path, doc)
        for state_id, record in _records(doc):
            invocations += 1
            identity = _identity(record, campaign_rows.get(record.get("region_id"), {}))
            if identity is None:
                unresolved += 1
                continue
            exact_id = identity["exact_implementation_id"]
            identities.setdefault(exact_id, identity)
            counts[exact_id] += 1
            states[exact_id].add(state_id)
            sources[exact_id].add(str(path))
    rows = []
    for exact_id in sorted(identities):
        identity = identities[exact_id]
        rows.append({
            "exact_implementation_id": exact_id,
            "implementation_pattern_id": identity["implementation_pattern_id"],
            "semantic_family_id": identity["semantic_family_id"],
            "invocation_count": counts[exact_id],
            "state_count": len(states[exact_id]),
            "source_artifacts": sorted(sources[exact_id]),
            "identity": identity,
        })
    return {
        "schema": "kernel-analyzer-implementation-census-v1",
        "status": "COMPLETE" if unresolved == 0 else "PARTIAL_LEGACY_ABI_UNRESOLVED",
        "counting_rule": "Every invocation and exact identity remains in coverage; deep measurement selects one representative per implementation pattern, with exact variants reopened only for a declared ABI or reduction-topology contrast.",
        "denominator": {
            "runtime_invocations": invocations,
            "identity_resolved_invocations": invocations - unresolved,
            "identity_unresolved_invocations": unresolved,
            "unique_exact_implementations": len(rows),
            "unique_implementation_patterns": len({row["implementation_pattern_id"] for row in rows}),
            "unique_semantic_families": len({row["semantic_family_id"] for row in rows}),
        },
        "implementations": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screens", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.screens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["denominator"], sort_keys=True))


if __name__ == "__main__":
    main()
