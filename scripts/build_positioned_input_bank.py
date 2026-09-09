#!/usr/bin/env python3
"""Derive a text input bank with explicit absolute position IDs.

This is a data-construction step, not a numerical selection step.  Token IDs,
state order, roles, and identifiers are preserved exactly; only the declared
position sequence is added so long-context computations can be exercised
without allocating an equally long attention window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive(
    document: dict, *, start: int, row_offset: int = 0, row_limit: int | None = None,
) -> dict:
    if type(start) is not int or start < 0:
        raise ValueError("position start must be a nonnegative integer")
    if type(row_offset) is not int or row_offset < 0:
        raise ValueError("row offset must be a nonnegative integer")
    if row_limit is not None and (type(row_limit) is not int or row_limit <= 0):
        raise ValueError("row limit must be a positive integer")
    key = "states" if isinstance(document.get("states"), list) else "records"
    rows = document.get(key)
    if not isinstance(rows, list) or not rows:
        raise ValueError("input bank must contain nonempty states or records")
    result = deepcopy(document)
    stop = None if row_limit is None else row_offset + row_limit
    result[key] = result[key][row_offset:stop]
    if not result[key]:
        raise ValueError("declared source row slice is empty")
    for row in result[key]:
        tokens = row.get("token_ids", row.get("input_ids"))
        if not isinstance(tokens, list) or not tokens:
            raise ValueError("every state must contain nonempty token IDs")
        row["position_ids"] = list(range(start, start + len(tokens)))
    result["position_ids_protocol"] = {
        "kind": "CONTIGUOUS_ABSOLUTE_POSITIONS",
        "start": start,
        "source_row_offset": row_offset,
        "source_row_limit": row_limit,
        "token_values_or_numerical_results_used_for_selection": False,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--row-offset", type=int, default=0)
    parser.add_argument("--row-limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("Choose a new output under /data1/tzh")
    source = json.loads(args.source.read_text())
    result = derive(
        source, start=args.start, row_offset=args.row_offset, row_limit=args.row_limit,
    )
    result["position_ids_protocol"]["source_path"] = str(args.source.resolve())
    result["position_ids_protocol"]["source_sha256"] = sha(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "states": len(result.get("states", result.get("records", []))),
        "start": args.start,
    }))


if __name__ == "__main__":
    main()
