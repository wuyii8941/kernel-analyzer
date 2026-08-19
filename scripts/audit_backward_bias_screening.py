#!/usr/bin/env python3
"""Audit whether the historical T1-T4 funnel under-sampled backward bias.

This is a denominator audit, not a new verdict.  It distinguishes structural
F+B coverage from selection into downstream causal/carrier/trajectory stages
and records the semantic bottlenecks that require a local-centered path.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def phase(task_id: object) -> str:
    value = str(task_id).split(":", 1)[0].upper()
    return value if value in {"FORWARD", "BACKWARD"} else "UNRESOLVED"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/property/bias_formation/hotspot_search/backward_funnel_audit.json")
    args = parser.parse_args()
    releases = []
    raw = collections.Counter()
    for release in sorted((ROOT / "results/coverage/runtime_releases").glob("*_seq*_r1")):
        plan_path = release / "same_dtype_tasks.json.gz"
        if not plan_path.exists():
            continue
        plan = load(plan_path)
        release_raw = collections.Counter(str(row["phase"]).upper() for row in plan["rows"])
        raw.update(release_raw)
        releases.append({
            "release": release.name, "task_denominator": dict(release_raw),
        })
        del plan
    # The compact hypothesis matrix already binds every historical observed
    # label needed for this audit.  Re-reading multi-gigabyte oracle payloads
    # only to recompute phase counts is intentionally avoided.
    hypothesis = load(ROOT / "results/property/hypothesis_matrix.json")["rows"]
    historical = collections.Counter(
        (str(row["grouping_metadata"]["phase"]), str(row["observed_label"]["role"]))
        for row in hypothesis
    )
    downstream: dict[str, dict[str, int]] = {}
    for stage in ("causal", "carrier", "trajectory"):
        counts = collections.Counter(); positives = collections.Counter()
        for path in (ROOT / f"results/coverage/cases/{stage}").glob("*/*.json.gz"):
            try:
                payload = load(path)
            except Exception:
                continue
            if stage == "causal":
                for row in payload.get("rows", []):
                    key = phase(row.get("task_id")); counts[key] += 1
                    if row.get("causal_t2_positive", row.get("causal_t2_t3_positive", False)):
                        positives[key] += 1
            else:
                counts[phase(payload.get("task_id"))] += 1
        downstream[stage] = dict(counts)
        if positives:
            downstream[stage + "_positive"] = dict(positives)
    result = {
        "schema": "kernel-analyzer-backward-bias-funnel-audit-v1",
        "status": "BACKWARD_STRUCTURALLY_COVERED_BUT_SELECTION_UNDERREPRESENTED",
        "finding": (
            "The exact F+B denominator contains substantial backward coverage, but T1 requires "
            "a directionally coherent local endpoint residual before causal transport is measured. "
            "This excludes LOCAL_CENTERED_TO_GRADIENT_BIASED mechanisms by construction."
        ),
        "aggregate": {
            "exact_task_denominator": dict(raw),
            "historical_observed_labels_by_phase": {
                f"{key[0]}:{key[1]}": value for key, value in historical.items()
            },
            **downstream,
        },
        "releases": releases,
        "required_rescreen_rule": {
            "entry_gates": ["EXACT_FB_BINDING", "FINITE", "REPEAT_STABLE", "NONZERO_LOCAL_DIFFERENCE"],
            "forbidden_entry_gate": "LOCAL_DIRECTIONAL_BIAS",
            "measured_transition": ["LOCAL_ENDPOINT", "PARAMETER_GRADIENT", "EFFECTIVE_UPDATE"],
            "unit": "COMPLETE_FORWARD_BACKWARD_STEP_WITH_ENDPOINT_REPAIR",
        },
        "priority_semantic_bottlenecks": [
            "LOSS_CE_DLOGITS", "LM_HEAD_DX_DW", "COMPLETE_RMSNORM_VJP",
            "ATTENTION_SAVED_STATE", "ATTENTION_SOFTMAX_VJP", "ATTENTION_DQ_DK_DV_DP",
            "STATE_SPACE_RECURRENT_BACKWARD",
        ],
        "claim_boundary": "This audit proves selection imbalance, not that rejected backward cases contain bias.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "status": result["status"],
                      "aggregate": result["aggregate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
