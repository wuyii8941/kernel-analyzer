#!/usr/bin/env python3
"""Inventory every saved Triton definition in registered execution releases.

The pool is release-qualified and reads no numerical result.  Repeated shapes,
protocols, or source bodies remain visible; they are not counted as new
operator families.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path


def triton_symbols(source):
    tree = ast.parse(source)
    symbols = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "triton"):
            symbols.extend(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
    return sorted(set(symbols))


def build(inventory):
    releases = sorted({Path(row["release"]).resolve() for row in inventory["records"]})
    records = []
    release_rows = []
    source_digests = Counter()
    canonical_sources = {}
    for release in releases:
        sources = sorted((release / "trace").glob("**/output_code.py"))
        if not sources:
            release_rows.append({
                "release": str(release),
                "status": "NO_SAVED_TRACE_SOURCE",
                "source_path_occurrences": 0,
            })
            continue
        for source in sources:
            canonical = source.resolve()
            canonical_sources.setdefault(canonical, set()).add(str(release))
        release_rows.append({
            "release": str(release),
            "status": "SOURCE_INVENTORIED" if sources else "NO_SAVED_TRACE_SOURCE",
            "source_path_occurrences": len(sources),
        })

    source_failures = []
    for source, release_aliases in sorted(canonical_sources.items()):
        raw = source.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        source_digests[digest] += 1
        try:
            symbols = triton_symbols(raw.decode())
        except (UnicodeDecodeError, SyntaxError) as exc:
            source_failures.append({
                "source": str(source), "release_aliases": sorted(release_aliases),
                "reason": str(exc),
            })
            continue
        records.extend({
            "release": sorted(release_aliases)[0],
            "release_aliases": sorted(release_aliases),
            "source": str(source),
            "source_sha256": digest,
            "symbol": symbol,
            "source_inventory_status": "SAVED_TRITON_DEFINITION",
            "reference_status": "NOT_ASSESSED_BY_SOURCE_POOL",
            "runtime_measurement_status": "NOT_ASSESSED_BY_SOURCE_POOL",
        } for symbol in symbols)
    if source_failures:
        # Parsing failures are global source facts; do not mutate the release
        # occurrence accounting or silently omit them.
        for row in release_rows:
            row["status"] = (
                "PARTIAL_SOURCE_INVENTORY"
                if any(row["release"] in failure.get("release_aliases", [])
                       for failure in source_failures)
                else row["status"]
            )
    return {
        "schema": "all-registered-release-source-pool-v1",
        "scope": "All saved output_code.py files below releases in the input inventory",
        "selection_uses_numerical_results": False,
        "release_count": len(releases),
        "release_status_counts": dict(Counter(row["status"] for row in release_rows)),
        "source_path_occurrences": sum(
            row["source_path_occurrences"] for row in release_rows),
        "source_count": len(canonical_sources),
        "source_with_triton_count": len({row["source"] for row in records}),
        "unique_source_digests": len(source_digests),
        "duplicate_source_digest_groups": sum(count > 1 for count in source_digests.values()),
        "definition_records": len(records),
        "release_records": release_rows,
        "source_failures": source_failures,
        "records": records,
        "all_kernel_support_established": False,
        "warning": (
            "Definitions are release-qualified occurrences, not unique mathematical "
            "operators, operator families, bias mechanisms, or valid measurements."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("Choose a new output under /data1/tzh")
    raw = args.inventory.read_bytes()
    inventory = json.loads(raw)
    result = build(inventory)
    result["input_sha256"] = hashlib.sha256(raw).hexdigest()
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({
        key: result[key] for key in (
            "release_count", "source_count", "source_with_triton_count",
            "unique_source_digests", "definition_records"
        )
    }))


if __name__ == "__main__":
    main()
