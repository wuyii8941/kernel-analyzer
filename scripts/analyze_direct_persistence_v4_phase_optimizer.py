#!/usr/bin/env python3
"""Compare real early/middle/late optimizer states without cross-phase mixing.

Every input must be a complete raw-stage capture from its own natural training
phase.  The command refuses missing phases, mixed case identities, or reused
state IDs.  It never installs one phase's moments on another phase's vectors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.analyze_direct_persistence_v4_optimizer_state import analyze


PHASES = ("early", "middle", "late")


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    for phase in PHASES:
        parser.add_argument(f"--{phase}", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {phase: getattr(args, phase) for phase in PHASES}
    missing = [phase for phase, path in paths.items() if path is None]
    if missing:
        result = {
            "schema": "kernel-analyzer-direct-persistence-v4-phase-optimizer-result-v1",
            "status": "ABSTAIN_MISSING_PHASE_CAPTURE",
            "missing_phases": missing,
            "claim_boundary": "Natural optimizer phase dependence cannot be inferred from incomplete phase captures.",
        }
    else:
        captures = {phase: load(path) for phase, path in paths.items()}
        errors: list[str] = []
        case_ids = {str(payload.get("case_id")) for payload in captures.values()}
        if len(case_ids) != 1:
            errors.append("phase captures have different case_id values")
        all_states: list[str] = []
        phase_results: dict[str, Any] = {}
        for phase, payload in captures.items():
            if payload.get("status") != "COMPLETE":
                errors.append(f"{phase}: raw capture status is not COMPLETE")
            state_ids = [str(value) for value in payload.get("state_ids", [])]
            if not state_ids:
                errors.append(f"{phase}: state_ids are missing")
            all_states.extend(state_ids)
            phase_result = analyze(payload)
            if phase_result.get("status") != "COMPLETE_SAME_STATE_OPTIMIZER_ABLATION":
                errors.append(f"{phase}: raw capture failed optimizer validation")
            phase_results[phase] = phase_result
        if len(all_states) != len(set(all_states)):
            errors.append("state IDs are reused across natural phases")
        if errors:
            result = {
                "schema": "kernel-analyzer-direct-persistence-v4-phase-optimizer-result-v1",
                "status": "ABSTAIN_INVALID_PHASE_CAPTURE",
                "errors": errors,
                "claim_boundary": "No natural phase conclusion is emitted from mixed or incomplete captures.",
            }
        else:
            result = {
                "schema": "kernel-analyzer-direct-persistence-v4-phase-optimizer-result-v1",
                "status": "COMPLETE_PHASE_CONDITIONED_OPTIMIZER_COMPARISON",
                "case_id": next(iter(case_ids)),
                "phases": {
                    phase: {
                        "state_count": len(captures[phase]["state_ids"]),
                        "arms": phase_results[phase]["arms"],
                        "input": str(paths[phase]),
                    }
                    for phase in PHASES
                },
                "claim_boundary": "Each phase uses its own captured weights, inputs, gradients and moments; no cross-phase mixing was performed.",
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
