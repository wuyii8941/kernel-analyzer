"""Build a compact implementation-difference atlas from the frozen census.

The atlas is metadata-only: it deliberately does not read candidate tensor
values or verdicts. Every recorded generated region stays in the denominator;
``EXACT_SAME_PRECISION_REPLAY`` is retained instead of being silently dropped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_SOURCE = Path(
    "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/full_step_inventory/"
    "generated_implementation_mechanism_census_v1.json"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--out", type=Path, default=Path("results/final/implementation_atlas.json"))
    args = ap.parse_args()
    raw = args.source.read_bytes()
    census = json.loads(raw)
    rows = []
    for i, row in enumerate(census["rows"]):
        same = row.get("candidate_same_precision") or {}
        mechanism = row.get("mechanism_annotation") or "UNRESOLVED"
        rows.append({
            "id": i,
            "phase": row.get("phase"),
            "kind": row.get("kind"),
            "region": row.get("region_id"),
            "symbol": row.get("symbol"),
            "mechanism": mechanism,
            "implementation_changed": mechanism != "EXACT_SAME_PRECISION_REPLAY",
            "candidate_exact": bool(same.get("all_exact", False)),
            "candidate_nonfinite": bool(same.get("nonfinite", 0)),
            "control": row.get("control_name"),
            "status": row.get("status"),
        })
    by_kind = defaultdict(Counter)
    for r in rows:
        by_kind[r["kind"]][r["mechanism"]] += 1
    out = {
        "schema": "kernel-analyzer-implementation-atlas-v1",
        "scope": "Qwen3-1.7B frozen generated implementation census v1",
        "source": str(args.source),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "denominator": len(rows),
        "counts": dict(Counter(r["mechanism"] for r in rows)),
        "changed_count": sum(r["implementation_changed"] for r in rows),
        "exact_replay_count": sum(not r["implementation_changed"] for r in rows),
        "by_kind": {k: dict(v) for k, v in by_kind.items()},
        "rows": rows,
        "boundary": (
            "Metadata-only atlas. It identifies recorded schedule/accumulator/"
            "materialization interventions; it does not certify a bias or infer "
            "a mechanism for rows marked exact/unresolved."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "denominator": len(rows), "changed": out["changed_count"]}))


if __name__ == "__main__":
    main()
