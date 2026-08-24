#!/usr/bin/env python3
"""Add only severity quantities derivable from preserved raw replay vectors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


OUT = Path("results/property/direct_persistence_v4/severity.json")
RAW = {
    "phi4_seq64_lmhead_dx": Path(
        "/data1/tzh/cache/kernel_analyzer/direct_persistence_v4/phi_seq64_raw_stage.json"
    ),
    "qwen_seq128_lmhead_dx": Path(
        "/data1/tzh/cache/kernel_analyzer/direct_persistence_v4/qwen_seq128_raw_stage.json"
    ),
}


def row(case_id: str, path: Path) -> dict:
    if not path.is_file():
        return {"case_id": case_id, "status": "ABSTAIN_MISSING_RAW_REPLAY"}
    d = json.loads(path.read_text(encoding="utf-8"))
    v = d.get("vectors", {})
    required = ("candidate_update", "repair_update")
    if any(k not in v for k in required):
        return {"case_id": case_id, "status": "ABSTAIN_MISSING_UPDATE_VECTORS"}
    candidate = np.asarray(v["candidate_update"], dtype=np.float64)
    repair = np.asarray(v["repair_update"], dtype=np.float64)
    direct = candidate - repair
    direct_sum = np.sum(direct, axis=0)
    normal_energy = float(np.sqrt(np.sum(candidate * candidate)))
    direct_resultant = float(np.linalg.norm(direct_sum))
    return {
        "case_id": case_id,
        "status": "PARTIAL_RAW_SEVERITY_DERIVED",
        "steps": int(direct.shape[0]),
        "direct_resultant_l2": direct_resultant,
        "normal_update_path_l2": normal_energy,
        "direct_resultant_over_normal_update_path": direct_resultant / max(normal_energy, 1e-30),
        "parameter_norm": None,
        "loss_gradient_projection": None,
        "claim_boundary": "The ratio uses the captured candidate update path as a normal-update scale. Parameter norm, loss projection and observed loss remain ABSTAIN.",
    }


def main() -> None:
    d = json.loads(OUT.read_text(encoding="utf-8"))
    existing = {str(r.get("case_id")): r for r in d.get("rows", [])}
    for case_id, path in RAW.items():
        existing[case_id] = row(case_id, path)
    d["rows"] = [existing[k] for k in sorted(existing)]
    d["raw_stage_reanalysis"] = {
        "status": "PARTIAL_RAW_SEVERITY_DERIVED",
        "rows": sorted(RAW),
        "claim_boundary": "Only direct resultant relative to captured normal update path is available for these two raw replays; all other severity quantities remain fail-closed.",
    }
    OUT.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": d["raw_stage_reanalysis"]["status"], "rows": len(d["rows"])}, sort_keys=True))


if __name__ == "__main__":
    main()
