#!/usr/bin/env python
"""Re-execute selected candidate external MM calls through eager ATen."""

from qwen3_candidate_extern_reexecution_common_v0_1 import run_experiment


ROLES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
SELECTIONS = [
    {"index": 7 * layer + offset, "role": f"layer{layer}.{role}"}
    for layer in (0, 14, 27)
    for offset, role in enumerate(ROLES)
]
SELECTIONS.append({"index": 196, "role": "lm_head"})


if __name__ == "__main__":
    run_experiment(
        {
            "description": __doc__,
            "schema_version": "forkcert.qwen3-candidate-extern-mm-reexecution.v0.1",
            "valid_status": "VALID_ORIGINAL_CANDIDATE_EXTERN_MM_REEXECUTION",
            "operation": "mm",
            "expected_family_calls": 197,
            "selections": SELECTIONS,
            "claim_limits": [
                "candidate external wrapper re-executed through eager ATen on identical candidate inputs",
                "shared-library path evidence, not an independent arithmetic implementation",
                "null effect is not proof that upstream linear inputs agree",
                "repair/reexecution only; no injection, population or correctness claim",
            ],
        }
    )
