#!/usr/bin/env python3
"""Build the compact final evidence summary for the paper mainline."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/property/joint_bias_formation_v1"


def load(relative: str) -> dict:
    return json.loads((BASE / relative).read_text())


def main() -> None:
    rms = load("rms_persistence/rms_persistence.json")["formation_population"]["correlation"]
    cases = load("source_persistence_reclassification.json")
    final_norm = load("four_scale_arms/phi_lmhead.json")["arms"]
    layer26 = load("four_scale_arms/phi_layer26_post_attention_norm.json")["arms"]
    random_null = load("four_scale_arms/phi_repeated_random_null.json")
    payload = {
        "schema": "kernel-analyzer-mainline-closure-v1",
        "status": "COMPLETE_MAINLINE_EVIDENCE_WITH_SCOPE",
        "source_persistence_headline_cases": cases[
            "headline_source_or_transport_persistent_count"
        ],
        "rms_baseline": rms,
        "phi_final_norm": {
            "operator_A": final_norm["A_operator"]["coherence_amplification"],
            "precision_A": final_norm["D_precision"]["coherence_amplification"],
            "data_order_A": final_norm["C_data_order"]["coherence_amplification"],
        },
        "phi_repeated_random_null": {
            "natural_local_A": random_null["natural_common_state_local_effect"][
                "coherence_amplification"
            ],
            **random_null["null_amplification"],
            "seeds": len(random_null["repeated_random_nulls"]),
        },
        "phi_layer26_carrier": {
            "operator_A": layer26["A_operator"]["coherence_amplification"],
            "precision_A": layer26["D_precision"]["coherence_amplification"],
            "data_order_A": layer26["C_data_order"]["coherence_amplification"],
        },
        "conclusion": (
            "Local RMS does not separate directional formation in the 32-row sample. "
            "Phi final-norm source persistence greatly exceeds a five-seed repeated matched "
            "random null, but the same endpoint is diffusive on a second reachable carrier. "
            "The defensible property is therefore temporal persistence of an endpoint-mediated "
            "effective update on a declared carrier, not model-wide directionality."
        ),
        "claim_boundary": (
            "Three cases currently support the persistent-local-source headline. The result "
            "does not establish an all-operator predictor or full-parameter precision comparison."
        ),
    }
    output = BASE / "mainline_closure.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
