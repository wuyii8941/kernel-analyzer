import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch

SPEC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SPEC), str(SPEC / "corpus"), str(SPEC.parent / "2_tool/src")]
from kernel_spec.num import NotEstablished
from region_device import check


class DeviceCorrespondence(unittest.TestCase):
    def run_case(self, *, replay_value=1, equal=True, write_outcome="checked"):
        launch = NS(index=12, executed=True, grid=(1, 1, 1),
                    args=[NS(kind="tensor", storage_ptr=7, after=torch.tensor([1]))])
        with ExitStack() as stack:
            def mock(name, **kwargs):
                return stack.enter_context(patch(name, **kwargs))
            mock("kernel_analyzer.reference_eval.capture.load_compiled", return_value=object())
            replay = mock("kernel_analyzer.reference_eval.capture.replay",
                          return_value={7: torch.tensor([replay_value])})
            emulator = mock("kernel_analyzer.reference_eval.emulate.emulate", return_value=(object(), None))
            mock("region_device.D.write_parameters", return_value={"out"})
            mock("region_device.D.inspect_reference_writes", return_value={"outcome": write_outcome})
            mock("kernel_analyzer.reference_eval.emulate._verification_outputs",
                 return_value=[("out", np.array([True]), {"emulated": np.array([True]),
                               "bit_equal": np.array([equal]), "nan_both": np.array([True])})])
            result = check([launch], [Path("saved")])
            self.assertEqual(replay.call_count, 2)
            emulator.assert_called_once_with(launch, ungrouped=(), swapped=())
            return result

    def test_every_required_check_can_refuse(self):
        self.assertEqual(self.run_case()["status"], "checked")
        for kwargs in ({"replay_value": 2}, {"equal": False}, {"write_outcome": "not_established"}):
            with self.subTest(kwargs=kwargs):
                self.assertEqual(self.run_case(**kwargs)["status"], "not_established")

    def test_budget_refuses_before_gpu_work(self):
        with patch("kernel_analyzer.reference_eval.capture.load_compiled") as load:
            with self.assertRaises(NotEstablished):
                check([NS(grid=(3, 1, 1))], [Path("saved")], max_programs=2)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
