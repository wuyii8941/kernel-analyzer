#!/usr/bin/env python3
"""Audit candidate-blind F/B phase assignment for the six source maps."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final"
CANONICAL = ROOT / "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/full_step_inventory/triton_online_reference_campaign_v1.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.evolving_triton_observation import _dtype_mapping_phase


def main() -> None:
    canonical_rows = json.loads(CANONICAL.read_text())["rows"]
    phases: dict[str, set[str]] = {}
    for row in canonical_rows:
        symbol = str(row.get("reference_symbol", row["symbol"]))
        phases.setdefault(symbol, set()).add(str(row["phase"]))
    static = json.loads((FINAL / "source_matrix_static.json").read_text())
    replay = json.loads((FINAL / "source_replay_matrix.json").read_text()) if (FINAL / "source_replay_matrix.json").exists() else {}
    cells = []
    for cell in static["cells"]:
        mapping_path = FINAL / str(cell["mapping_file"])
        mapping = json.loads(mapping_path.read_text())
        counts = Counter()
        unresolved = []
        for row in mapping["rows"]:
            if row["status"] != "MAPPED":
                continue
            try:
                phase = _dtype_mapping_phase(str(row["symbol"]), str(row["reference_symbol"]), phases)
            except ValueError as exc:
                unresolved.append({"symbol": row["symbol"], "reference_symbol": row["reference_symbol"], "error": str(exc)})
                continue
            counts[phase] += int(row["invocations"])
        cells.append({
            "dtype": mapping["dtype"],
            "tf32": bool(mapping["tf32"]),
            "seq_len": int(mapping["seq_len"]),
            "mapping_file": str(cell["mapping_file"]),
            "mapped_invocations": int(mapping["denominator"]["mapped_invocations"]),
            "forward_invocations": counts["FORWARD"],
            "backward_invocations": counts["BACKWARD"],
            "phase_unresolved_rows": unresolved,
            "all_phase_resolved": not unresolved and counts["FORWARD"] + counts["BACKWARD"] == int(mapping["denominator"]["mapped_invocations"]),
        })
    output = {
        "schema": "kernel-analyzer-source-phase-audit-v1",
        "subject": "candidate-blind source-mapped F+B phase audit",
        "candidate_values_used_to_select_or_classify": False,
        "cells": cells,
        "all_cells_phase_resolved": all(row["all_phase_resolved"] for row in cells),
        "numeric_replay": "COMPLETE" if replay.get("numeric_replay") == "COMPLETE" else "PENDING_GPU_REMEASUREMENT",
        "natural_bias_case_added": False,
        "property_claim": False,
        "boundary": "Phase resolution validates dispatch routing only; it does not observe candidate tensors or assign numerical verdicts.",
    }
    output["result_sha256"] = hashlib.sha256(json.dumps(output, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path = FINAL / "source_phase_audit.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(path), "all_cells_phase_resolved": output["all_cells_phase_resolved"]}, sort_keys=True))


if __name__ == "__main__":
    main()
