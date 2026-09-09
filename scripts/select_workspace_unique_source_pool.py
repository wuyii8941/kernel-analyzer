#!/usr/bin/env python3
"""Select one provenance-rich representative for every saved source digest."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


PROVENANCE_ORDER = {
    "REGISTERED_RELEASE": 0,
    "UNREGISTERED_RUNTIME_RELEASE": 1,
    "OTHER_SAVED_COMPILED_SOURCE": 2,
}


def select(census):
    by_digest = {}
    for row in census["source_records"]:
        digest = row["source_sha256"]
        by_digest.setdefault(digest, []).append(row)
    selected_sources = {}
    for digest, rows in by_digest.items():
        selected_sources[digest] = min(
            rows,
            key=lambda row: (
                PROVENANCE_ORDER.get(row["provenance_status"], 99), row["source"]
            ),
        )
    definitions = {}
    for row in census["definition_records"]:
        selected = selected_sources[row["source_sha256"]]
        if row["source"] != selected["source"]:
            continue
        key = (row["source_sha256"], row.get("symbol"))
        definitions.setdefault(key, row)
    records = [{
        "source": row["source"],
        "source_sha256": row["source_sha256"],
        "symbol": row.get("symbol"),
        "source_provenance_status": row["provenance_status"],
        "runtime_package": row.get("runtime_package"),
        "source_definition_status": row.get("status"),
        "function_semantic_ast_sha256": row.get("function_semantic_ast_sha256"),
        "reference_status": "NOT_ASSESSED_BY_UNIQUE_SOURCE_SELECTION",
        "runtime_measurement_status": "NOT_ASSESSED_BY_UNIQUE_SOURCE_SELECTION",
    } for row in definitions.values() if row.get("symbol")]
    records.sort(key=lambda row: (row["source"], row["symbol"]))
    return {
        "schema": "workspace-unique-source-pool-v1",
        "scope": "One representative path for every byte-distinct saved output_code.py",
        "selection_uses_numerical_results": False,
        "source_count": len(selected_sources),
        "definition_records": len(records),
        "selected_provenance_counts": dict(Counter(
            row["provenance_status"] for row in selected_sources.values())),
        "records": records,
        "all_kernel_support_established": False,
        "warning": (
            "Byte-distinct sources and function ASTs are not mathematical operator "
            "families; this pool only avoids scanning exact source copies repeatedly."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("Choose a new output under /data1/tzh")
    raw = args.census.read_bytes()
    census = json.loads(raw)
    result = select(census)
    result["input_sha256"] = hashlib.sha256(raw).hexdigest()
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({key: result[key] for key in (
        "source_count", "definition_records", "selected_provenance_counts"
    )}))


if __name__ == "__main__":
    main()
