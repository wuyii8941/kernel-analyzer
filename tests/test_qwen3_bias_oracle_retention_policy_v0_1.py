from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Qwen3BiasOracleRetentionPolicyTests(unittest.TestCase):
    def test_retained_snapshots_are_prospective_sha_choices(self) -> None:
        policy = json.loads(
            (
                ROOT
                / "theory_oracle"
                / "QWEN3_BIAS_ORACLE_RETENTION_POLICY_V0_1.json"
            ).read_text(encoding="utf-8")
        )
        key = policy["selection_key_sha256"]
        expected = []
        for index in range(4):
            plan = json.loads(
                (
                    ROOT
                    / "theory_oracle"
                    / f"QWEN3_BIAS_ORACLE_CALIBRATION_{index}_CAPTURE_PLAN_V0_1.json"
                ).read_text(encoding="utf-8")
            )
            for phase in ("early", "middle", "late"):
                rows = [row for row in plan["targets"] if row["phase"] == phase]
                selected = min(
                    rows,
                    key=lambda row: hashlib.sha256(
                        f"{key}/{row['state_id']}".encode("utf-8")
                    ).hexdigest(),
                )
                expected.append(selected["state_id"])
        self.assertEqual(policy["calibration_full_snapshot_retention"], expected)

    def test_policy_cannot_itself_trigger_deletion(self) -> None:
        policy = json.loads(
            (
                ROOT
                / "theory_oracle"
                / "QWEN3_BIAS_ORACLE_RETENTION_POLICY_V0_1.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            policy["status"],
            "FROZEN_POLICY_NO_DELETION_AUTHORIZED_BY_POLICY_ALONE",
        )
        self.assertGreaterEqual(len(policy["deletion_gate"]), 6)
        self.assertGreaterEqual(policy["storage_safety_floor_bytes"], 2 * 1024**4)


if __name__ == "__main__":
    unittest.main()
