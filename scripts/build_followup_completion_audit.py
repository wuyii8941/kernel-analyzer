#!/usr/bin/env python3
"""Build a fail-closed audit for the current training-bias follow-up."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> tuple[Path, dict[str, Any] | None]:
    path = ROOT / relative
    if not path.is_file():
        return path, None
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return path, value


def one(name: str, relative: str, accepted: set[str]) -> dict[str, Any]:
    path, payload = read(relative)
    observed = None if payload is None else str(payload.get("status"))
    return {
        "requirement": name,
        "artifact": str(path.relative_to(ROOT)),
        "observed_status": observed,
        "accepted_statuses": sorted(accepted),
        "complete": observed in accepted,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checks = [
        one(
            "Phi loss-direction check",
            "results/property/loss_direction_stress_v1/phi_lmhead_summary.json",
            {"COMPLETE"},
        ),
        one(
            "Liger loss-direction check",
            "results/property/loss_direction_stress_v1/liger_fused_ce_summary.json",
            {"COMPLETE"},
        ),
        one(
            "training-equivalence synthetic calibration",
            "results/property/training_equivalence_v1/synthetic_validation.json",
            {"GO"},
        ),
        one(
            "pure-FP32 summation-order prediction",
            "results/property/liger_fp32_chunk_order_v1/summary.json",
            {"COMPLETE"},
        ),
        one(
            "original frozen Qwen recovery",
            "results/property/training_bias_profile_v2/prospective_batch_1/rerun_qwen_recovered/raw/qwen_seq64_backward_1000_output_0.json",
            {"COMPLETE"},
        ),
        one(
            "optimizer and checkpoint comparison",
            "results/property/optimizer_condition_benchmark_v1/summary.json",
            {"COMPLETE"},
        ),
        one(
            "repeated paired training consequences",
            "results/property/independent_consequence_v1/summary.json",
            {"COMPLETE"},
        ),
        one(
            "frozen generalization benchmark",
            "results/property/generalization_benchmark_v1/summary.json",
            {"COMPLETE", "COMPLETE_WITH_EXPLICIT_ABSTENTIONS"},
        ),
        one(
            "Gemma new-model/new-implementation method bridge",
            "results/property/generalization_benchmark_v1/gemma4_method_bridge_result.json",
            {"COMPLETE_METHOD_BRIDGE_NOT_PROSPECTIVE_CASE_SELECTION"},
        ),
        one(
            "original frozen Mamba recovery",
            "results/property/training_bias_profile_v2/prospective_batch_1/rerun_mamba_recovered/status.json",
            {"COMPLETE", "ABSTAIN_RUNTIME_ENVIRONMENT_VERIFIED"},
        ),
        one(
            "long-horizon case accounting",
            "results/property/declared_persistent_4096/summary.json",
            {"COMPLETE", "COMPLETE_WITH_UNRESOLVED_REPLAYS"},
        ),
    ]
    incomplete = [row["requirement"] for row in checks if not row["complete"]]
    payload = {
        "schema": "kernel-analyzer-followup-completion-audit-v1",
        "status": "COMPLETE" if not incomplete else "INCOMPLETE",
        "checks": checks,
        "incomplete_requirements": incomplete,
        "claim_boundary": (
            "An explicit, reproducible abstention closes execution accounting but "
            "does not become a scientific negative. Documentation, repository tests, "
            "and remote push are verified separately after these checks pass."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "incomplete": len(incomplete)}))


if __name__ == "__main__":
    main()
