#!/usr/bin/env python
"""Independent audit of the materialized original Qwen3 op/kernel inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


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
    manifest = json.loads((ROOT / args.manifest).read_text())
    inventory = ROOT / manifest["output_dir"]
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, observed: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "observed": observed})

    for name, record in manifest["artifacts"].items():
        path = ROOT / record["path"]
        observed = sha256(path) if path.is_file() else None
        check(f"artifact:{name}", observed == record["sha256"], observed)
    result_path = inventory / "result.json"
    check("result_exists", result_path.is_file(), str(result_path))
    result = json.loads(result_path.read_text()) if result_path.is_file() else {}
    if result:
        check("result_status", result["status"] == "VALID_ORIGINAL_CANDIDATE_KERNEL_INVENTORY", result["status"])
        check("all_gates_true", all(result["gates"].values()), result["gates"])
        required = result["trace"]["required_forward_artifact_counts"]
        check("two_complete_forward_traces", all(count == 2 for count in required.values()), required)
        check("trace_nonempty", result["trace"]["file_count"] > 0, result["trace"]["file_count"])
        check("real_tensors_disabled", result["trace"]["save_real_tensors"] is False)
        missing_or_changed = []
        for record in result["trace"]["files"]:
            path = inventory / record["path"]
            if not path.is_file() or sha256(path) != record["sha256"] or path.stat().st_size != record["size"]:
                missing_or_changed.append(record["path"])
        check("trace_file_manifest_exact", not missing_or_changed, missing_or_changed)
    summary_path = inventory / "kernel_summary.json"
    check("summary_exists", summary_path.is_file(), str(summary_path))
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        check("summary_valid", summary["status"] == "VALID_DESCRIPTIVE_SUMMARY", summary["status"])
        boundary = summary["boundary_kernel_family"]
        check("boundary_family_27_calls", boundary["call_count"] == 27, boundary["call_count"])
        check(
            "boundary_family_ops",
            {"aten.add", "aten.pow", "aten.mean", "aten.rsqrt", "aten.mul"}.issubset(boundary["original_aten"]),
            boundary["original_aten"],
        )
    payload = {
        "schema_version": "forkcert.qwen3-original-compiled-inventory-audit.v0.2",
        "verdict": "VALID" if all(row["passed"] for row in checks) else "INVALID",
        "checks": checks,
        "causal_claim": "NONE",
    }
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["verdict"] == "VALID" else 1)


if __name__ == "__main__":
    main()
