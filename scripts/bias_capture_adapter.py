#!/usr/bin/env python3
"""Shared, side-effect-safe capture-adapter skeleton for the v2.1 cases.

The adapters currently perform only engineering preflight.  They never load a
model or start a GPU campaign unless a future implementation explicitly
replaces the guarded execution hook.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/property/bias_formation_v2_1/pilot"


def _load_roster() -> dict[str, Any]:
    return json.loads((ROOT / "results/property/bias_formation_v2_1/roster_bound.json").read_text())


def _source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_adapter(case_id: str, adapter_name: str, argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(prog=adapter_name)
    parser.add_argument("--dry-run-states", type=int, default=2)
    parser.add_argument("--formation-only", action="store_true")
    parser.add_argument("--consequence-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--execute", action="store_true", help="reserved; guarded until v2.1 pilot is approved")
    args = parser.parse_args(argv)
    if args.dry_run_states < 1:
        raise SystemExit("--dry-run-states must be positive")
    if args.formation_only and args.consequence_only:
        raise SystemExit("formation-only and consequence-only are mutually exclusive")
    mode = "formation" if args.formation_only else "consequence" if args.consequence_only else "formation+consequence"
    roster = _load_roster()
    case = next((row for row in roster["cases"] if row["case_id"] == case_id), None)
    if case is None:
        raise SystemExit("case is absent from frozen roster: " + case_id)
    state_ids = list(case.get("state_ids", ()))
    selected = state_ids[:args.dry_run_states]
    output = args.output or (OUT / f"{case_id}.preflight.json")
    existing = output.exists()
    checks = {
        "case_in_frozen_roster": True,
        "exact_endpoint_bound": bool(case.get("endpoint_or_region_id")),
        "state_ids_available_for_dry_run": len(selected) == args.dry_run_states,
        "state_ids_are_unique": len(set(selected)) == len(selected),
        "no_scientific_verdict": True,
        # These are protocol invariants, not mode-dependent pass/fail gates:
        # formation never accepts trajectory data, and consequence always
        # declares the four-arm requirement even when this invocation is
        # formation-only.
        "formation_has_no_trajectory": True,
        "consequence_requires_four_arms": True,
    }
    # Source and sham are recorded separately; a name such as NOT_AVAILABLE is
    # never accepted as a successful binding.
    checks["repair_source_bound"] = bool(case.get("repair_binding", {}).get("bound"))
    checks["sham_source_bound"] = bool(case.get("sham_binding", {}).get("bound"))
    status = "PREFLIGHT_READY" if all(checks.values()) and not args.execute else (
        "EXECUTION_GUARDED_NO_GPU" if args.execute else "PREFLIGHT_BLOCKED"
    )
    result = {
        "schema": "kernel-analyzer-bias-capture-preflight-v2_1",
        "case_id": case_id,
        "adapter": adapter_name,
        "mode": mode,
        "dry_run_states": args.dry_run_states,
        "selected_state_ids": selected,
        "device_requested": args.device,
        "resume_requested": args.resume,
        "existing_output": existing,
        "execute_requested": args.execute,
        "gpu_execution_started": False,
        "scientific_verdict": False,
        "status": status,
        "checks": checks,
        "frozen_endpoint": case.get("endpoint_or_region_id"),
        "runner_source_sha256": _source_hash(Path(__file__)),
        "claim_boundary": "engineering_preflight_only; no 16+16 measurement",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name("." + output.name + ".tmp")
    temp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(output)
    print(json.dumps({"case_id": case_id, "status": status, "output": str(output), "gpu_execution_started": False}, sort_keys=True))
    return result
