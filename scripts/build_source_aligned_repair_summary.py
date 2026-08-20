#!/usr/bin/env python3
"""Summarize source-aligned repairs for the three previously misrepaired MMs."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "results/coverage/cases"


DECLARED = (
    {
        "case_id": "qwen64_vproj_mm",
        "candidate_id": "qwen_seq64_forward_8_output",
        "decomposition": "qwen64_vproj_precision_decomposition.json",
        "repair": "qwen64_vproj_source_aligned_repair.json.gz",
        "full_arm": "JOINT",
        "old_arm": "KERNEL_ONLY",
        "expected_states": 32,
    },
    {
        "case_id": "qwen128_vproj_mm",
        "candidate_id": "qwen_seq128_forward_8_output",
        "decomposition": "qwen128_vproj_precision_decomposition.json",
        "repair": "qwen128_vproj_source_aligned_repair.json.gz",
        "full_arm": "ROUNDING_ONLY",
        "old_arm": "KERNEL_ONLY",
        "expected_states": 32,
    },
    {
        "case_id": "mamba_seq64_input_proj",
        "candidate_id": "mamba_seq64_forward_1_output",
        "decomposition": "mamba_seq64_input_proj_precision_decomposition.json",
        "repair": "mamba_seq64_input_proj_source_aligned_repair.json.gz",
        "full_arm": "JOINT",
        "old_arm": "KERNEL_ONLY",
        "expected_states": 4,
    },
)


def load(name: str) -> dict[str, Any]:
    path = CASES / name
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(payload)
    temporary.replace(path)


def build() -> dict[str, Any]:
    rows = []
    for declared in DECLARED:
        decomposition = load(declared["decomposition"])
        repair = load(declared["repair"])
        if decomposition["candidate_id"] != declared["candidate_id"]:
            raise RuntimeError("decomposition candidate mismatch")
        if repair["candidate_id"] != declared["candidate_id"]:
            raise RuntimeError("repair candidate mismatch")
        if repair["schema"] != "kernel-analyzer-mm-source-aligned-repair-v1":
            raise RuntimeError("unexpected source-repair schema")
        allowed_status = (
            "COMPLETE_SOURCE_ALIGNED_REPAIR_POPULATION"
            if declared["expected_states"] == 32
            else "COMPLETE_SOURCE_ALIGNED_REPAIR_PILOT"
        )
        if repair["status"] != allowed_status:
            raise RuntimeError("source-repair population is incomplete")
        if len(repair["states"]) != declared["expected_states"] or not all(
            state["sham_exact"] for state in repair["states"]
        ):
            raise RuntimeError("source-repair population or sham is incomplete")
        full_arm = declared["full_arm"]
        if full_arm not in repair["arm_verdicts"]:
            raise RuntimeError("declared full-source arm is absent")
        verdict = dict(repair["arm_verdicts"][full_arm])
        verdict.setdefault("repair_status", "SOURCE_DEBIASED_IN_EXPECTATION")
        if verdict["verdict"] == "REPAIR_UNRESOLVED":
            verdict["verdict"] = "SOURCE_DEBIASED_IN_EXPECTATION_DOWNSTREAM_UNRESOLVED"
        carrier_status = repair["direction"][full_arm]["carrier_removed"]["status"]
        if declared["expected_states"] < 32 and carrier_status != "PASS":
            carrier_status = "UNRESOLVED_4_STATE_PILOT"
        rows.append({
            **declared,
            "coherent_sources": decomposition["coherent_sources"],
            "arms_run": repair["arms"],
            "evidence_scope": "POPULATION" if declared["expected_states"] == 32 else "CAUSAL_PILOT",
            "full_source_repair": verdict,
            "finite_repeat_repaired_source": {
                "status": repair["direction"][full_arm]["repaired_source"]["status"],
                "u_statistic": repair["direction"][full_arm]["repaired_source"]["cross_state_inner_product_u"],
                "confidence": repair["direction"][full_arm]["repaired_source"]["cluster_bootstrap_95"],
            },
            "removed_source": {
                "status": repair["direction"][full_arm]["source_removed"]["status"],
                "u_statistic": repair["direction"][full_arm]["source_removed"]["cross_state_inner_product_u"],
                "confidence": repair["direction"][full_arm]["source_removed"]["cluster_bootstrap_95"],
            },
            "downstream_carrier_effect": {
                "status": carrier_status,
                "u_statistic": repair["direction"][full_arm]["carrier_removed"]["cross_state_inner_product_u"],
                "confidence": repair["direction"][full_arm]["carrier_removed"]["cluster_bootstrap_95"],
            },
            "old_intervention_was_full_repair": (
                declared["old_arm"] == full_arm
                and set(decomposition["coherent_sources"]) == {"kernel"}
            ),
            "full_fp32_reference_equivalence": "NOT_MEASURED",
        })
    return {
        "schema": "kernel-analyzer-source-aligned-repair-summary-v1",
        "status": "COMPLETE_THREE_CASE_SOURCE_ALIGNED_REPAIR_AUDIT",
        "cases": rows,
        "definitions": {
            "KERNEL_ONLY": "FP32 MM followed by the original deterministic low-dtype cast",
            "ROUNDING_ONLY": "retain measured kernel residual and replace nearest rounding by coordinate-wise unbiased stochastic rounding",
            "JOINT": "FP32 MM plus coordinate-wise unbiased stochastic rounding",
        },
        "claim_boundary": (
            "These arms repair or debias the locally identified source. A candidate-repair "
            "difference proves that source affects real F+B. It does not prove that the "
            "repair arm equals an end-to-end FP32 training step; that comparison remains "
            "NOT_MEASURED. Rounding repair is exact in conditional expectation, not in each "
            "single low-precision materialization."
        ),
    }


def markdown(payload: dict[str, Any]) -> str:
    table = []
    for row in payload["cases"]:
        verdict = row["full_source_repair"]
        table.append(
            f"| {row['case_id']} | {', '.join(row['coherent_sources'])} | "
            f"{row['evidence_scope']} | {row['full_arm']} | {verdict['verdict']} | "
            f"{row['downstream_carrier_effect']['status']} |"
        )
    return """# Source-aligned repair audit

## Correction

The historical `FP32 MM -> BF16 cast` arm is only a kernel-arithmetic repair.
It retains deterministic output rounding.  Therefore its trajectory separation
cannot be cited as repair success for an output-rounding source.

## Results

| Case | Proven directional sources | Scope | Full-source arm | Repair verdict | Downstream carrier |
|---|---|---|---|---|---|
""" + "\n".join(table) + """

## Interpretation

`ROUNDING_ONLY` and `JOINT` preserve the declared low-precision ABI and make
the quantizer coordinate-wise centered in conditional expectation.  The saved
finite-repeat certificate reports Monte Carlo residual separately.  Thus a
successful result is a **source-debiasing causal repair**, not a claim that one
random BF16 realization equals FP32.

Candidate--repair trajectory drift means the repaired source affects training.
It does not mean the repair remains wrong, and it also does not establish that
the repair equals a full high-precision reference.  End-to-end FP32-reference
equivalence is explicitly `NOT_MEASURED` for all three cases.
"""


def main() -> None:
    payload = build()
    write(CASES / "source_aligned_repair_summary.json", json.dumps(
        payload, indent=2, sort_keys=True,
    ) + "\n")
    write(CASES / "source_aligned_repair_summary.md", markdown(payload))
    print(json.dumps({
        "status": payload["status"],
        "cases": len(payload["cases"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
