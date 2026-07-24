#!/usr/bin/env python
"""Gate a traced whole-model transition against an uninstrumented baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_transition(baseline: dict[str, Any], traced: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "baseline_valid": baseline.get("valid") is True,
        "traced_valid": traced.get("valid") is True,
        "pre_state_exact": baseline.get("pre_state") == traced.get("pre_state"),
        "scorer_exact": baseline.get("anchors", {}).get("observed_scorer_sha256")
        == traced.get("anchors", {}).get("observed_scorer_sha256"),
        "scorer_values_exact": baseline.get("continuous", {}).get("scorer_logps")
        == traced.get("continuous", {}).get("scorer_logps"),
        "loss_exact": baseline.get("continuous", {}).get("loss")
        == traced.get("continuous", {}).get("loss"),
        "semantic_exact": baseline.get("semantic") == traced.get("semantic"),
        "scaled_gradient_exact": baseline.get("continuous", {})
        .get("scaled_gradient", {})
        .get("tensor_hashes_sha256")
        == traced.get("continuous", {})
        .get("scaled_gradient", {})
        .get("tensor_hashes_sha256"),
        "unscaled_gradient_exact": baseline.get("continuous", {})
        .get("unscaled_gradient", {})
        .get("tensor_hashes_sha256")
        == traced.get("continuous", {})
        .get("unscaled_gradient", {})
        .get("tensor_hashes_sha256"),
        "clipped_gradient_exact": baseline.get("continuous", {})
        .get("clipped_gradient", {})
        .get("tensor_hashes_sha256")
        == traced.get("continuous", {})
        .get("clipped_gradient", {})
        .get("tensor_hashes_sha256"),
        "parameter_update_exact": baseline.get("continuous", {})
        .get("parameter_update", {})
        .get("tensor_hashes_sha256")
        == traced.get("continuous", {})
        .get("parameter_update", {})
        .get("tensor_hashes_sha256"),
        "post_state_exact": baseline.get("post_state") == traced.get("post_state"),
        "graph_family_exact": baseline.get("realization", {}).get("graph_family_digest")
        == traced.get("realization", {}).get("graph_family_digest"),
        "compiler_protocol_exact": baseline.get("realization", {}).get(
            "compiler_config_digest"
        )
        == traced.get("realization", {}).get("compiler_config_digest"),
    }
    observability = traced.get("observability", {})
    trace_files = observability.get("trace_files", [])
    artifact_hashes_valid = bool(trace_files) and all(
        Path(row["path"]).is_file() and sha256_file(row["path"]) == row["sha256"]
        for row in trace_files
    )
    manifests = traced.get("compiler", {}).get("graph_manifests", [])
    manifest_hashes_valid = bool(manifests) and all(
        Path(row["path"]).is_file() and sha256_file(row["path"]) == row["sha256"]
        for row in manifests
    )
    artifacts = {
        "trace_files_present_and_hashed": artifact_hashes_valid,
        "graph_manifests_present_and_hashed": manifest_hashes_valid,
        "generated_code_count": int(observability.get("generated_code_count", 0)),
        "provenance_mapping_count": int(
            observability.get("provenance_mapping_count", 0)
        ),
    }
    artifacts["forward_backward_code_present"] = artifacts["generated_code_count"] >= 2
    artifacts["provenance_present"] = artifacts["provenance_mapping_count"] >= 2
    forward_check_names = {
        "baseline_valid",
        "traced_valid",
        "pre_state_exact",
        "scorer_exact",
        "scorer_values_exact",
        "loss_exact",
        "semantic_exact",
        "graph_family_exact",
        "compiler_protocol_exact",
    }
    forward_equivalence = all(checks[name] for name in forward_check_names)
    training_equivalence = all(checks.values())
    auditable = all(
        value
        for key, value in artifacts.items()
        if key not in {"generated_code_count", "provenance_mapping_count"}
    )
    return {
        "schema_version": "forkcert.whole-model-observability-gate.v0.1",
        "instrumented_pipeline_equivalent": training_equivalence,
        "forward_observability_equivalent": forward_equivalence,
        "training_transition_equivalent": training_equivalence,
        "artifacts_auditable": auditable,
        "forward_kernel_inventory_eligible": forward_equivalence and auditable,
        "training_kernel_inventory_eligible": training_equivalence and auditable,
        "operator_kernel_inventory_eligible": training_equivalence and auditable,
        "checks": checks,
        "artifacts": artifacts,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "claim": (
            "traced whole-model realization is eligible for provenance inventory"
            if training_equivalence and auditable
            else (
                "forward kernel inventory only; no backward/update kernel claim"
                if forward_equivalence and auditable
                else "no operator/kernel claim; observability or equivalence gate failed"
            )
        ),
        "nonclaims": [
            "equivalent endpoints do not prove eager or compiled correctness",
            "trace provenance does not by itself localize discrepancy production",
            "kernel inventory does not by itself establish endpoint mediation",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--traced", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--require",
        choices=["forward-equivalence", "forward-inventory", "training"],
        default="forward-inventory",
    )
    args = parser.parse_args()
    baseline_path = Path(args.baseline).resolve()
    traced_path = Path(args.traced).resolve()
    result = compare_transition(
        json.loads(baseline_path.read_text()), json.loads(traced_path.read_text())
    )
    result["baseline"] = {"path": str(baseline_path), "sha256": sha256_file(baseline_path)}
    result["traced"] = {"path": str(traced_path), "sha256": sha256_file(traced_path)}
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    required = {
        "forward-equivalence": result["forward_observability_equivalent"],
        "forward-inventory": result["forward_kernel_inventory_eligible"],
        "training": result["training_kernel_inventory_eligible"],
    }[args.require]
    raise SystemExit(0 if required else 2)


if __name__ == "__main__":
    main()
