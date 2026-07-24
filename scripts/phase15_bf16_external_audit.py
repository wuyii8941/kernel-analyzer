#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from forkcert.io import read_jsonl
from forkcert.report import markdown_table


def audit_payload(
    preflight: dict,
    training: dict,
    online_rows: list[dict],
    certificates: list[dict],
    expected_rows: int,
) -> dict:
    capability = ((preflight.get("device") or {}).get("capability") or [0, 0])
    checks = {
        "preflight_passed": preflight.get("passed") is True,
        "native_bf16_hardware": int(capability[0]) >= 8 and (preflight.get("device") or {}).get("bf16_supported") is True,
        "training_dtype_bf16": training.get("training_compute_dtype") == "bf16",
        "online_row_count": len(online_rows) == expected_rows,
        "online_dtype_bf16": bool(online_rows) and all(row.get("training_compute_dtype") == "bf16" for row in online_rows),
        "path_names_bf16": bool(online_rows)
        and all("-bf16-" in str(row.get("path_ref")) and "-bf16-" in str(row.get("path_alt")) for row in online_rows),
        "online_repeats_exact": bool(online_rows)
        and all(float(row.get("delta_self_ref", -1)) == 0.0 and float(row.get("delta_self_alt", -1)) == 0.0 for row in online_rows),
        "certificate_coverage": len(certificates) == expected_rows,
    }
    applicable = [row for row in certificates if int(row.get("advantage_sign", 0)) != 0]
    return {
        "schema_version": "forkcert.bf16-external-audit.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "hardware": preflight.get("device"),
        "training_compute_dtype": training.get("training_compute_dtype"),
        "online_rows": len(online_rows),
        "certificates": len(certificates),
        "applicable_clipping_decisions": len(applicable),
        "actual_clipping_forks": sum(bool(row.get("actual_fork")) for row in applicable),
        "region_counts": {
            region: sum(row.get("region") == region for row in applicable)
            for region in sorted({str(row.get("region")) for row in applicable})
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a native BF16 ForkCert external replay.")
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--training-metadata", required=True)
    parser.add_argument("--online-jsonl", required=True)
    parser.add_argument("--certificates", required=True)
    parser.add_argument("--expected-rows", type=int, default=51200)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    payload = audit_payload(
        json.loads(Path(args.preflight).read_text(encoding="utf-8")),
        json.loads(Path(args.training_metadata).read_text(encoding="utf-8")),
        read_jsonl(args.online_jsonl),
        read_jsonl(args.certificates),
        args.expected_rows,
    )
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    check_rows = [{"check": key, "passed": value} for key, value in payload["checks"].items()]
    report = "\n".join(
        [
            "# Native BF16 External Replay Audit",
            "",
            "## Claim Scope",
            "This audit proves that the replay used native BF16-capable hardware and BF16 training compute. It does not convert the empirical envelope into a legal analytic error bound.",
            "",
            "## Checks",
            markdown_table(check_rows, ["check", "passed"]),
            "",
            "## Result",
            f"Overall: {'PASS' if payload['passed'] else 'FAIL'}.",
            f"Applicable clipping decisions: {payload['applicable_clipping_decisions']}; actual forks: {payload['actual_clipping_forks']}.",
            "",
        ]
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
