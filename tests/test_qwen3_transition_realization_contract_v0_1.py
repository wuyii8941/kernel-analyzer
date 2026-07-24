from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from theory_oracle.qwen3_grpo_natural_transition_v0_2 import (
    json_sha256,
    load_realization_contract,
)


class TransitionRealizationContractTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        snapshot = root / "snapshot"
        snapshot.mkdir()
        metadata = snapshot / "forkcert_transition_snapshot.json"
        metadata.write_text('{"optimizer_step":10}\n', encoding="utf-8")
        metadata_digest = hashlib.sha256(metadata.read_bytes()).hexdigest()
        contract = {
            "schema_version": "forkcert.qwen3-transition-realization-contract.v0.1",
            "status": "FROZEN_BEFORE_TRANSITION_ENDPOINT_EXECUTION",
            "snapshot_metadata_sha256": metadata_digest,
            "optimizer_step": 10,
            "target_minibatch_sha256": "target",
            "history_state_preserved": True,
            "contract_state_preserved": True,
            "compiler_config_digest": "config",
            "graph_family_digest": "family",
            "candidate_ordered_unique_graph_family": [],
            "reference_scorer_sha256": "reference",
            "candidate_scorer_sha256": "candidate",
        }
        contract["contract_sha256"] = json_sha256(contract)
        path = root / "contract.json"
        path.write_text(json.dumps(contract), encoding="utf-8")
        return snapshot, path

    def test_valid_contract_loads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot, path = self._fixture(Path(directory))
            value = load_realization_contract(path, snapshot, 10)
            self.assertEqual(value["candidate_scorer_sha256"], "candidate")

    def test_contract_is_snapshot_specific(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot, path = self._fixture(Path(directory))
            (snapshot / "forkcert_transition_snapshot.json").write_text(
                '{"optimizer_step":11}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "snapshot identity"):
                load_realization_contract(path, snapshot, 10)

    def test_invalid_status_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot, path = self._fixture(Path(directory))
            value = json.loads(path.read_text(encoding="utf-8"))
            value["status"] = "INVALID"
            value["contract_sha256"] = json_sha256(
                {key: item for key, item in value.items() if key != "contract_sha256"}
            )
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not frozen and valid"):
                load_realization_contract(path, snapshot, 10)

    def test_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot, path = self._fixture(Path(directory))
            value = json.loads(path.read_text(encoding="utf-8"))
            value["candidate_scorer_sha256"] = "changed"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                load_realization_contract(path, snapshot, 10)


if __name__ == "__main__":
    unittest.main()
