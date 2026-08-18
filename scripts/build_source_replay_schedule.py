#!/usr/bin/env python3
"""Build the explicit six-cell by checkpoint replay schedule.

This is a plan/provenance artifact only.  It does not assign numerical
verdicts; the runner must replace each pending row after real CUDA execution.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final"


def main() -> None:
    static = json.loads((FINAL / "source_matrix_static.json").read_text())
    bank = json.loads((FINAL / "natural_bank.json").read_text())
    steps = [int(row["step"]) for row in bank["checkpoints"]]
    rows = []
    replay_dir = Path("/data1/tzh/cache/kernel_analyzer/source_replay")
    for cell in static["cells"]:
        for step in steps:
            tag = "tf32" if cell["tf32"] else "fp32"
            output_file = f"{tag}_seq{cell['seq_len']}_step{step}.json"
            worker = replay_dir / output_file
            status = "PENDING_GPU_REMEASUREMENT"
            if worker.exists():
                try:
                    observed = json.loads(worker.read_text())
                    gates = observed.get("gates", {})
                    valid = (
                        observed.get("checkpoint_step") == step
                        and observed.get("seq_len") == int(cell["seq_len"])
                        and observed.get("dtype") == ("fp32" if bool(cell["tf32"]) else str(cell["dtype"]))
                        and bool(observed.get("tf32")) == bool(cell["tf32"])
                        and observed.get("dtype_mapping_sha256") == hashlib.sha256(
                            (FINAL / str(cell["mapping_file"])).read_bytes()
                        ).hexdigest()
                        and gates.get("all_expected_ordinary_regions_observed_twice")
                        and gates.get("all_changed_region_ids_retained_twice")
                        and gates.get("all_observation_repeats_stable")
                        and gates.get("candidate_values_used_to_select_regions") is False
                    )
                    if valid:
                        status = "COMPLETE"
                except (OSError, ValueError, json.JSONDecodeError):
                    status = "PENDING_GPU_REMEASUREMENT"
            rows.append({
                "dtype": cell["dtype"],
                "tf32": bool(cell["tf32"]),
                "seq_len": int(cell["seq_len"]),
                "step": step,
                "mapping_file": cell["mapping_file"],
                "expected_invocations": int(cell["runtime_invocations"]),
                "repeat_count": 2,
                "output_file": output_file,
                "status": status,
            })
    result = {
        "schema": "kernel-analyzer-source-replay-schedule-v1",
        "subject": "Qwen3-1.7B natural checkpoint source-mapped replay",
        "candidate_values_used_to_select_or_classify": False,
        "checkpoint_steps": steps,
        "cell_count": len(static["cells"]),
        "planned_runs": len(rows),
        "repeat_count": 2,
        "rows": rows,
        "numeric_verdicts": "NOT_ASSIGNED",
        "boundary": "Each row remains pending until the real generated kernel is observed twice on the same checkpoint and all mapped F+B invocations are present.",
    }
    result["result_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = FINAL / "source_replay_schedule.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(path), "planned_runs": len(rows)}))


if __name__ == "__main__":
    main()
