from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from theory_oracle.freeze_qwen3_boundary_condition_family_v0_1 import (
    SCHEMA_VERSION as FAMILY_SCHEMA_VERSION,
)
from theory_oracle.plan_qwen3_boundary_confirmation_resource_v0_1 import (
    plan_resource,
)
from theory_oracle.aggregate_qwen3_boundary_conditioned_calibration_v0_1 import (
    WEIGHTING_CONTRACT_ID,
)


class BoundaryResourcePlanTests(unittest.TestCase):
    def test_twelve_comparisons_raise_signflip_minimum_above_eight(self) -> None:
        endpoints = [f"endpoint-{index}" for index in range(12)]
        family = {
            "schema_version": FAMILY_SCHEMA_VERSION,
            "valid": True,
            "status": "FROZEN_SUPPORT_COMPLETE_BOUNDARY_FAMILY",
            "weighting_contract_id": WEIGHTING_CONTRACT_ID,
            "endpoint_family": endpoints,
            "confirmatory_comparisons": len(endpoints),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "family.json"
            path.write_text(json.dumps(family), encoding="utf-8")
            result = plan_resource(
                family,
                path,
                family_alpha=0.05,
                minimum_trajectories=8,
                resource_cap=100,
            )
        self.assertTrue(result["valid"], result["errors"])
        self.assertGreater(
            result["minimum_trajectories_for_signflip_resolution"], 8
        )
        self.assertFalse(result["candidate_effect_mean_sign_or_variance_used"])


if __name__ == "__main__":
    unittest.main()
