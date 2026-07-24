from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from theory_oracle.qwen3_u2_direction_projection_v0_1 import (
    load_frozen_direction,
    project_delta_artifact,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class U2DirectionProjectionTests(unittest.TestCase):
    def frozen_direction(self, root: Path) -> tuple[dict, dict]:
        shards = []
        for key, tensor in (
            ("w", torch.tensor([1.0, 0.0], dtype=torch.float64)),
            ("b", torch.tensor([0.0, 2.0], dtype=torch.float64)),
        ):
            path = root / f"{key}.safetensors"
            save_file({key: tensor}, path)
            shards.append({"path": str(path), "sha256": sha(path), "tensor_key": key})
        manifest = {
            "schema_version": "forkcert.qwen3-u2-frozen-direction.v0.1",
            "valid": True,
            "verdict": "VALID_FROZEN_U2_CALIBRATION_DIRECTION",
            "status": "FROZEN_BEFORE_CONFIRMATION",
            "endpoint_name": "U2_calibration_direction_shift",
            "direction": {"normalization_l2": math.sqrt(5.0), "shards": shards},
        }
        path = root / "direction.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        link = {"path": str(path), "sha256": sha(path)}
        return load_frozen_direction(link), link

    def test_projection_preserves_sign_and_uses_all_parameter_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            direction, _ = self.frozen_direction(root)
            positive = root / "positive.safetensors"
            negative = root / "negative.safetensors"
            save_file(
                {"w": torch.tensor([1.0, 0.0]), "b": torch.tensor([0.0, 1.0])},
                positive,
            )
            save_file(
                {"w": torch.tensor([-1.0, 0.0]), "b": torch.tensor([0.0, -1.0])},
                negative,
            )
            first = project_delta_artifact(
                {"path": str(positive), "sha256": sha(positive)}, direction
            )
            second = project_delta_artifact(
                {"path": str(negative), "sha256": sha(negative)}, direction
            )
            self.assertAlmostEqual(first, 3.0 / math.sqrt(5.0))
            self.assertAlmostEqual(second, -first)

    def test_equal_l2_deltas_can_have_opposite_directional_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            direction, _ = self.frozen_direction(root)
            values = []
            for name, sign in (("up", 1.0), ("down", -1.0)):
                path = root / f"{name}.safetensors"
                save_file(
                    {"w": torch.tensor([sign, 0.0]), "b": torch.tensor([0.0, 0.0])},
                    path,
                )
                values.append(
                    project_delta_artifact(
                        {"path": str(path), "sha256": sha(path)}, direction
                    )
                )
            self.assertAlmostEqual(abs(values[0]), abs(values[1]))
            self.assertEqual(values[0], -values[1])


if __name__ == "__main__":
    unittest.main()
