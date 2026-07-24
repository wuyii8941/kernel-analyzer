#!/usr/bin/env python
"""Verify the repeated-family proxy sham against the compiled baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from theory_oracle.evaluate_qwen3_backward_singleton_repairs_v0_1 import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "FROZEN_PRE_EXECUTION":
        raise ValueError("manifest is not frozen")
    artifact_gates = {
        name: sha256_file(Path(row["path"]).resolve()) == row["sha256"]
        for name, row in manifest["artifacts"].items()
    }
    sham = json.loads((Path(manifest["output"]).resolve() / "result.json").read_text())
    baseline = json.loads(
        Path(
            "results/training_step_oracle/qwen3_grpo_natural_transition_v0_2/compiled_1/result.json"
        ).read_text()
    )
    control = sham.get("backward_repeated_family_proxy_sham", {})
    continuous_hashes_exact = {
        key: sham["continuous"][key]["tensor_hashes_sha256"]
        == baseline["continuous"][key]["tensor_hashes_sha256"]
        for key in (
            "scaled_gradient",
            "unscaled_gradient",
            "clipped_gradient",
            "parameter_update",
        )
    }
    gates = {
        "manifest_artifacts_exact": all(artifact_gates.values()),
        "sham_status_valid": control.get("status")
        == "VALID_BACKWARD_REPEATED_FAMILY_PROXY_SHAM",
        "all_embedded_sham_gates_true": bool(control.get("gates"))
        and all(control["gates"].values()),
        "all_continuous_tensor_hashes_exact": all(continuous_hashes_exact.values()),
        "semantic_exact": sham["semantic"] == baseline["semantic"],
        "post_state_exact": sham["post_state"] == baseline["post_state"],
    }
    payload = {
        "schema_version": "forkcert.qwen3-backward-repeated-family-proxy-sham-verification.v0.1",
        "status": "VALID_EXACT_NULL_PROXY_SHAM" if all(gates.values()) else "INVALID_OR_NON_NULL_SHAM",
        "manifest": str(manifest_path),
        "artifact_gates": artifact_gates,
        "continuous_hashes_exact": continuous_hashes_exact,
        "gates": gates,
        "interpretation_limit": (
            "exact-null excludes proxy dispatch itself but not extra allocation, kernel-launch, "
            "fusion-boundary or synchronization effects of eager replacement"
        ),
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "gates": gates}, indent=2))
    if payload["status"] != "VALID_EXACT_NULL_PROXY_SHAM":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
