#!/usr/bin/env python
"""Independent audit for the Accelerate-native Qwen repair v0.8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import theory_oracle.verify_qwen3_grpo_grad_branch_repair_v0_5 as base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--result")
    parser.add_argument("--out-audit")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    audit = base.Audit()
    base.preflight(manifest, audit)
    metrics = None
    mode = "preflight"
    if args.result:
        mode = "result"
        path = Path(args.result)
        audit.add("result_exists", path.is_file(), str(path))
        if path.is_file():
            result = json.loads(path.read_text(encoding="utf-8"))
            audit.add(
                "result_schema_v08",
                result.get("schema_version")
                == "forkcert.qwen3-grpo-grad-branch-repair.v0.8",
                result.get("schema_version"),
            )
            audit.add("native_hash_shape", result.get("hash_canonical_shape") == [4, 128])
            audit.add(
                "accelerate_native_realization",
                result.get("accelerate_native_realization_exact") is True,
            )
            audit.add(
                "specialization_history_exact",
                result.get("specialization_history_exact") is True,
            )
            audit.add(
                "prior_invalid_versions_preserved",
                result.get("prior_invalid_versions_preserved")
                == ["v0.5", "v0.6", "v0.7"],
            )
            compatibility_view = dict(result)
            compatibility_view["schema_version"] = (
                "forkcert.qwen3-grpo-grad-branch-repair.v0.5"
            )
            metrics = base.result_audit(manifest, compatibility_view, audit)
    payload = {
        "schema_version": "forkcert.qwen3-grpo-grad-branch-repair-audit.v0.8",
        "mode": mode,
        "verdict": "VALID" if audit.valid else "INVALID",
        "checks": audit.checks,
        "metrics": metrics,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out_audit:
        Path(args.out_audit).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(0 if audit.valid else 1)


if __name__ == "__main__":
    main()
