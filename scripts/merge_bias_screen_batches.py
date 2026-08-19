#!/usr/bin/env python3
"""Merge resource-bounded screening batches without changing measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob("batch_*/screening_gram.json"))
    if not paths:
        raise ValueError("no completed screening batches")
    payloads = [json.loads(path.read_text()) for path in paths]
    fixed = ("schema", "architecture", "state_count", "status", "selection_rule")
    for key in fixed:
        if len({json.dumps(row.get(key), sort_keys=True) for row in payloads}) != 1:
            raise ValueError(f"batch metadata differs for {key}")
    cases = [case for payload in payloads for case in payload["cases"]]
    case_ids = [case["case_id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("screen batches overlap")
    merged = {key: payloads[0][key] for key in fixed}
    merged.update({
        "batch_count": len(paths),
        "cases": cases,
        "resource_bounded_batches": True,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "batches": len(paths), "cases": len(cases)}))


if __name__ == "__main__":
    main()
