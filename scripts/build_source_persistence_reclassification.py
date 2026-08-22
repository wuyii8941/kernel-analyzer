#!/usr/bin/env python3
"""Separate persistent local-source evidence from generic feedback drift."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/property/joint_bias_formation_v1/source_persistence_reclassification.json"


ROWS = [
    ("qwen_seq128_lm_head_dx", "SOURCE_OR_TRANSPORT_PERSISTENT_HEADLINE",
     "Complete F+B, matched root-arithmetic repair, carrier and 32-step paired trajectory."),
    ("liger_fused_ce", "SOURCE_OR_TRANSPORT_PERSISTENT_HEADLINE",
     "Aligned FP32 dW-accumulator repair and 32-step full-weight trajectory."),
    ("phi4_seq64_lmhead_dx", "SOURCE_OR_TRANSPORT_PERSISTENT_HEADLINE",
     "Aligned analytic dX-MM repair; A=4.488 on the evolving final-norm carrier."),
    ("qwen_saved_p_seq128", "UNRESOLVED_LOCAL_VS_FEEDBACK_ATTRIBUTION",
     "Aligned semantic repair and directional separation exist, but no four-counterfactual local/feedback recurrence is bound."),
    ("qwen_layer23_attention_state", "UNRESOLVED_LOCAL_VS_FEEDBACK_ATTRIBUTION",
     "Closed semantic-region trajectory exists; source fidelity/parity and local-feedback attribution remain bounded."),
    ("mamba_seq64_input_proj", "MISMATCHED_FORMATION_AND_TRAJECTORY_CONTRAST",
     "Formation uses JOINT kernel+rounding while the trajectory uses KERNEL_ONLY."),
    ("qwen64_vproj", "MISMATCHED_FORMATION_AND_TRAJECTORY_CONTRAST",
     "Conditional JOINT formation is not aligned with the historical KERNEL_ONLY trajectory."),
    ("qwen128_vproj", "DIFFUSIVE_OR_CANCELING_LOCAL_SOURCE",
     "Aligned rounding-only local A=0.999; directional persistence is not confirmed."),
    ("qwen3vl_silu", "FEEDBACK_SUSTAINED_BACKGROUND_CONTROL",
     "Local A=1.001 while feedback/actual A=3.968/3.949; even response is a two-step cold-start transient."),
    ("gemma4_norm", "FEEDBACK_SUSTAINED_BACKGROUND_CONTROL",
     "Source-negative held-out result with Adam feedback; SGD/moment reset collapses it."),
]


def main() -> None:
    rows = [{"case_id": case, "classification": status, "reason": reason}
            for case, status, reason in ROWS]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["classification"]] = counts.get(row["classification"], 0) + 1
    payload = {
        "schema": "kernel-analyzer-source-persistence-reclassification-v1",
        "status": "COMPLETE_EVIDENCE_PRESERVING_RECLASSIFICATION",
        "rows": rows,
        "counts": counts,
        "headline_source_or_transport_persistent_count": counts.get(
            "SOURCE_OR_TRANSPORT_PERSISTENT_HEADLINE", 0
        ),
        "claim_boundary": (
            "Feedback-sustained trajectory separation is retained as a real training-dynamics "
            "effect but is not counted as persistent operator-local source bias. Unresolved and "
            "mismatched rows are not silently promoted or rejected. The 11/12 feedback rate "
            "applies only to the frozen residual-nonzero, parameter-reachable screen-negative sample."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "counts": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
