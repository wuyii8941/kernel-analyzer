#!/usr/bin/env python3
"""Summarize the frozen Liger paired training validation without overclaiming."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/property/training_numerical_analysis_v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    candidate_path = BASE / "training_utility/candidate_stream111.json"
    reference_path = BASE / "training_utility/reference_stream111.json"
    candidate = json.loads(candidate_path.read_text())
    reference = json.loads(reference_path.read_text())
    if candidate["stream"] != 111 or reference["stream"] != 111:
        raise RuntimeError("unexpected input stream")
    if candidate["steps_completed"] != 4096 or reference["steps_completed"] != 4096:
        raise RuntimeError("paired run is incomplete")
    if candidate["mode"] != "NATURAL_CANDIDATE" or reference["mode"] != "NATURAL_REPAIR":
        raise RuntimeError("unexpected implementation labels")
    candidate_losses = [row["training_loss"] for row in candidate["rows"]]
    reference_losses = [row["training_loss"] for row in reference["rows"]]
    differing = sum(left != right for left, right in zip(candidate_losses, reference_losses))
    historical_pairs = []
    for name, candidate_name, reference_name in (
        ("historical_stream_1", "natural_candidate.json", "natural_repair.json"),
        ("historical_stream_2", "natural_candidate_stream2.json", "natural_repair_stream2.json"),
    ):
        earlier = ROOT / "results/property/single_point_collapse_v2"
        old_candidate = json.loads((earlier / candidate_name).read_text())
        old_reference = json.loads((earlier / reference_name).read_text())
        historical_pairs.append({
            "name": name,
            "candidate_minus_reference_validation_loss": (
                old_candidate["validation_loss_mean"] - old_reference["validation_loss_mean"]
            ),
            "data_use": "HISTORICAL_CONTEXT_NOT_NEW_CONFIRMATION",
        })
    gaps = [
        row["candidate_minus_reference_validation_loss"] for row in historical_pairs
    ] + [candidate["validation_loss_mean"] - reference["validation_loss_mean"]]
    gap_mean = sum(gaps) / len(gaps)
    gap_sd = math.sqrt(sum((value - gap_mean) ** 2 for value in gaps) / (len(gaps) - 1))
    descriptive_half_width = 4.3026527297 * gap_sd / math.sqrt(len(gaps))
    summary = {
        "schema": "kernel-analyzer-training-utility-summary-v1",
        "status": "COMPLETE",
        "protocol": "results/property/training_numerical_analysis_v1/training_utility_protocol.json",
        "case_id": "liger_fused_ce_t128_small_gpt2",
        "stream": 111,
        "steps": 4096,
        "validation_loss": {
            "candidate": candidate["validation_loss_mean"],
            "reference": reference["validation_loss_mean"],
            "candidate_minus_reference": (
                candidate["validation_loss_mean"] - reference["validation_loss_mean"]
            ),
        },
        "training_loss_nonidentity": {
            "differing_steps": differing,
            "total_steps": len(candidate_losses),
            "maximum_absolute_gap": max(
                abs(left - right) for left, right in zip(candidate_losses, reference_losses)
            ),
        },
        "historical_context": {
            "paired_streams": historical_pairs,
            "three_stream_gap_mean": gap_mean,
            "descriptive_t_interval_95": [
                gap_mean - descriptive_half_width,
                gap_mean + descriptive_half_width,
            ],
            "interpretation": (
                "The two historical gaps are positive and the newly frozen gap is "
                "negative. The three-pair descriptive interval crosses zero; this does "
                "not support a repeatable validation-quality direction."
            ),
        },
        "source_sha256": {
            str(candidate_path.relative_to(ROOT)): _sha(candidate_path),
            str(reference_path.relative_to(ROOT)): _sha(reference_path),
        },
        "claim_boundary": (
            "This is one newly frozen paired run. It establishes the observed paired "
            "trajectory and validation-loss difference, not a population-level quality "
            "benefit or degradation. Historical streams remain separate observations."
        ),
    }
    output = BASE / "training_utility_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
