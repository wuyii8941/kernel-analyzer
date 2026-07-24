#!/usr/bin/env python
"""Independent artifact and result audit for the two Qwen3 operator pilots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EAGER = "742468b7b182ea8e70fec4733f702dbcc71ebb64fa3f4aec5e9fbc2450a29806"
CANDIDATE = "1107b4ac9c2662b34572cee3b4b4e1bf454a4b6d0a6def0c427d84f9944a09f2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_one(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, observed: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "observed": observed})

    for name, record in manifest["artifacts"].items():
        path = ROOT / record["path"]
        observed = sha256(path) if path.is_file() else None
        check(f"artifact:{name}", observed == record["sha256"], observed)

    result_path = ROOT / manifest["output"]
    check("result_exists", result_path.is_file(), str(result_path))
    if not result_path.is_file():
        return {"manifest": str(manifest_path), "valid": False, "checks": checks}
    result = json.loads(result_path.read_text())
    check("result_status", result["status"] == "VALID_SELECTED_STATE_OPERATOR_ATTRIBUTION", result["status"])
    check("all_gates_true", all(result["gates"].values()), result["gates"])
    check("eager_anchor", result["anchors"]["eager"] == EAGER, result["anchors"]["eager"])
    check("candidate_anchor", result["anchors"]["candidate"] == CANDIDATE, result["anchors"]["candidate"])

    arm_names = set(result["arms"])
    eager_arm = "split_EE" if "split_EE" in arm_names else "split_EEE"
    candidate_arm = "split_CC" if "split_CC" in arm_names else "split_CCC"
    check("eager_arm_hash", result["arms"][eager_arm]["sha256"] == [EAGER, EAGER], result["arms"][eager_arm]["sha256"])
    check("candidate_arm_hash", result["arms"][candidate_arm]["sha256"] == [CANDIDATE, CANDIDATE], result["arms"][candidate_arm]["sha256"])
    check("all_arm_repeats_exact", all(x["repeat_exact"] for x in result["arms"].values()))

    repair_key = next(key for key in result["contrasts"] if key.startswith("repair_") and "residual" not in key)
    injection_key = next(key for key in result["contrasts"] if key.startswith("injection_"))
    residual_key = next(key for key in result["contrasts"] if key.startswith("repair_residual"))
    total_key = next(key for key in result["contrasts"] if key.startswith("total_"))
    check("repair_exact_zero", result["contrasts"][repair_key]["l2"] == 0.0, result["contrasts"][repair_key])
    check("injection_exact_zero", result["contrasts"][injection_key]["l2"] == 0.0, result["contrasts"][injection_key])
    check(
        "residual_equals_total",
        result["contrasts"][residual_key] == result["contrasts"][total_key],
        {"residual": result["contrasts"][residual_key], "total": result["contrasts"][total_key]},
    )
    return {
        "manifest": str(manifest_path.relative_to(ROOT)),
        "result": str(result_path.relative_to(ROOT)),
        "valid": all(row["passed"] for row in checks),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifests = [
        ROOT / "theory_oracle/QWEN3_LM_HEAD_OPERATOR_PILOT_MANIFEST_V0_1.json",
        ROOT / "theory_oracle/QWEN3_FINAL_RMSNORM_OPERATOR_PILOT_MANIFEST_V0_1.json",
    ]
    subjects = [audit_one(path) for path in manifests]
    payload = {
        "schema_version": "forkcert.qwen3-operator-pilots-audit.v0.1",
        "verdict": "VALID" if all(row["valid"] for row in subjects) else "INVALID",
        "subjects": subjects,
    }
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["verdict"] == "VALID" else 1)


if __name__ == "__main__":
    main()
