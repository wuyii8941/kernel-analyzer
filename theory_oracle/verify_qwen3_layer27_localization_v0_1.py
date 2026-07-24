#!/usr/bin/env python
"""Independent fail-closed audit of the Qwen3 layer-27 localization result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_WHOLE = "1107b4ac9c2662b34572cee3b4b4e1bf454a4b6d0a6def0c427d84f9944a09f2"
EXPECTED_SPLIT = "0caf1a4c5e3f18e4fb918ea9e3d571a6874946d24d033d26db2f8d8338cdba57"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest_path = ROOT / args.manifest
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
    result = json.loads(result_path.read_text()) if result_path.is_file() else {}
    if result:
        check("invalid_status_preserved", result["status"] == "INVALID_FOR_ORIGINAL_CANDIDATE_LOCALIZATION", result["status"])
        expected_gates = {
            "whole_eager_anchor_exact": True,
            "split_reference_preserved": True,
            "split_candidate_anchor_exact": False,
            "all_repeats_exact": True,
        }
        check("gates_exact", result["gates"] == expected_gates, result["gates"])
        whole = result["anchors"]["candidate"]
        split = result["arms"]["split_CCC"]["sha256"]
        check("whole_candidate_identity", whole == EXPECTED_WHOLE, whole)
        check("split_candidate_identity", split == [EXPECTED_SPLIT, EXPECTED_SPLIT], split)
        check("candidate_mismatch_real", whole != split[0], {"whole": whole, "split": split[0]})
        check("all_repeats_exact", all(row["repeat_exact"] for row in result["arms"].values()))
        check("subject_not_operator", result["subject_is_operator"] is False, result["subject_is_operator"])
    payload = {
        "schema_version": "forkcert.qwen3-layer27-localization-audit.v0.1",
        "verdict": "VALID_INVALIDATION" if all(row["passed"] for row in checks) else "AUDIT_FAILED",
        "checks": checks,
        "interpretation": "mixed-arm contrasts are not operator or layer causal evidence",
    }
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["verdict"] == "VALID_INVALIDATION" else 1)


if __name__ == "__main__":
    main()
