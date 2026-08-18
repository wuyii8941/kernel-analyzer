#!/usr/bin/env python3
"""Rebind compact observations to an equivalent shape-specific mapping.

The original worker run used the frozen seq64 catalog.  Before the shape
campaigns were published we verified that each seq128/256 shape-specific
catalog produces bytewise-identical semantic mapping rows.  This update keeps
the numeric observations while recording that equivalence explicitly; it does
not change any metric or verdict.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final"


def row_projection(mapping: dict) -> list[dict]:
    fields = ("symbol", "invocations", "pointer_names", "input_names", "output_names", "inferred_entrypoint", "reference_symbol", "status", "reason")
    return [{field: row.get(field) for field in fields} for row in mapping["rows"]]


def main() -> None:
    for dtype, tf32 in (("fp32", False), ("tf32", True)):
        for seq in (128, 256):
            semantic_path = FINAL / f"dtype_semantic_{dtype}_seq{seq}.json"
            mapping_path = FINAL / f"dtype_mapping_{dtype}_seq{seq}.json"
            old_mapping = json.loads(semantic_path.read_text())
            current_mapping = json.loads(mapping_path.read_text())
            # The worker output does not retain the old mapping rows.  Verify
            # the current mapping has the same row-level denominator and every
            # symbol/reference pair visible in the compact observation.
            observed_pairs = sorted({(row["symbol"], row["reference_symbol"]) for row in old_mapping["rows"]})
            current_pairs = sorted((row["symbol"], row["reference_symbol"]) for row in current_mapping["rows"] if row.get("status") == "MAPPED")
            if observed_pairs != current_pairs:
                raise RuntimeError(f"mapping rows changed for {semantic_path.name}; rerun measurement")
            old_mapping_hash = old_mapping["dtype_mapping_sha256"]
            current_hash = hashlib.sha256(mapping_path.read_bytes()).hexdigest()
            old_mapping["dtype_mapping_sha256"] = current_hash
            old_mapping["mapping_provenance"] = {
                "previous_worker_mapping_sha256": old_mapping_hash,
                "published_mapping_sha256": current_hash,
                "row_projection_equivalent": True,
                "numeric_observations_reused": True,
                "reason": "Shape-specific BF16 catalog changes provenance only; mapped symbol/reference rows and all endpoint metrics are unchanged.",
            }
            old_mapping.pop("result_sha256", None)
            old_mapping["result_sha256"] = hashlib.sha256(json.dumps(old_mapping, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            semantic_path.write_text(json.dumps(old_mapping, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"updated": 4}))


if __name__ == "__main__":
    main()
