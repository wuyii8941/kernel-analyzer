#!/usr/bin/env python
"""Re-execute selected candidate external BMM calls through eager ATen."""

from qwen3_candidate_extern_reexecution_common_v0_1 import run_experiment


if __name__ == "__main__":
    run_experiment(
        {
            "description": __doc__,
            "schema_version": "forkcert.qwen3-candidate-extern-bmm-reexecution.v0.1",
            "valid_status": "VALID_ORIGINAL_CANDIDATE_EXTERN_BMM_REEXECUTION",
            "operation": "bmm",
            "expected_family_calls": 56,
            "selections": [
                {"index": 0, "role": "layer0.query_key_scores"},
                {"index": 1, "role": "layer0.probability_value"},
                {"index": 28, "role": "layer14.query_key_scores"},
                {"index": 29, "role": "layer14.probability_value"},
                {"index": 54, "role": "layer27.query_key_scores"},
                {"index": 55, "role": "layer27.probability_value"},
            ],
            "claim_limits": [
                "candidate external wrapper re-executed through eager ATen on identical candidate inputs",
                "shared-library path evidence, not an independent arithmetic implementation",
                "null effect is not proof that upstream BMM inputs agree",
                "repair/reexecution only; no injection, population or correctness claim",
            ],
        }
    )
