#!/usr/bin/env python3
"""Mechanically select a held-out candidate pool without reading outcomes.

The existing backward atlas contains screening rows but not all bindings needed
for a full v4 run.  This command makes that gap explicit instead of silently
turning a screen row into an eligible held-out case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ATLAS = ROOT / "results/property/bias_formation/hotspot_search/backward_rescreen_atlas.json"
DEFAULT_OUT = ROOT / "results/property/direct_persistence_v4/heldout_pool.json"


def display_path(path: Path) -> str:
    """Return a stable repository-relative path on Python 3.8+.

    ``Path.is_relative_to`` was added in Python 3.9, while the repository's
    CPU environment still includes Python 3.8.
    """

    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()
    atlas = load(args.atlas)
    source_rows = [row for row in atlas.get("rows", []) if isinstance(row, dict)]
    ranked = sorted(
        source_rows,
        key=lambda row: hashlib.sha256(str(row.get("task_id", row.get("case_id", ""))).encode()).hexdigest(),
    )
    selected: list[dict[str, Any]] = []
    seen_endpoints: set[str] = set()
    for row in ranked:
        if len(selected) >= args.count:
            break
        case_id = str(row.get("case_id", ""))
        endpoint = str(row.get("task_id", case_id))
        if not case_id or case_id in {"liger_fused_ce_t128", "phi4_seq64_lmhead_dx", "qwen_seq128_lmhead_dx"}:
            continue
        # The atlas contains both the old ``backward-cell`` and the newer
        # ``multishape-backward-cell`` label for some exact endpoints.  They
        # are not two experiments; keep one row per endpoint so the held-out
        # pool cannot silently double count an operator.
        if endpoint in seen_endpoints:
            continue
        seen_endpoints.add(endpoint)
        # These values are intentionally null when the atlas does not provide
        # a complete v4 binding. The validator will block the row; no value is
        # guessed from a neighboring endpoint.
        selected.append({
            "case_id": case_id,
            "model": row.get("model"),
            "implementation_class": row.get("family"),
            "endpoint": endpoint,
            "sequence_length": row.get("sequence_length"),
            "state_order": None,
            "state_bank_digest": None,
            "parameter_coordinate_digest": None,
            "repair": None,
            "role": "SEEN_IMPL_NEW_OPERANDS",
            "source_atlas": display_path(args.atlas),
            "selection_key": hashlib.sha256(str(row.get("task_id", case_id)).encode()).hexdigest(),
        })
    pool = {
        "schema": "kernel-analyzer-direct-persistence-v4-heldout-pool-v1",
        "status": "FROZEN_BEFORE_REVEAL",
        "selection_rule": "sort source task_id by SHA256 and take the first count after excluding predeclared cases; no labels or persistence results are read",
        "source": display_path(args.atlas),
        "implementation_scope": "The current atlas rows are SEEN_IMPL_NEW_OPERANDS; no NEW_IMPL row is declared ready without a separate exact binding.",
        "rows": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(pool, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": pool["status"], "rows": len(selected), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
