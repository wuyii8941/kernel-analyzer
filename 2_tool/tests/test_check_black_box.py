"""check.run_black_box: implementations without TTIR are compared with the specification only (K - f)."""
import numpy as np
import torch

from kernel_analyzer.check import Case, f64_point_spec, run_black_box


class _Shifted(Case):
    name = "shifted"

    def __init__(self, bias):
        self.bias = bias

    def inputs(self, seed):
        return {"x": np.random.default_rng(seed).normal(size=64)}

    def launch(self, inp):
        y = np.float32(np.tanh(inp["x"])) + np.float32(self.bias)
        return {"y": torch.from_numpy(np.asarray(y, dtype=np.float32))}

    def spec(self, inp):
        return {"y": f64_point_spec(np.tanh(inp["x"]))}


def _verdicts(report):
    return {r["rule"]: r["verdict"] for r in report["outputs"]["y"]["total_black_box"]["rules"]}


def test_black_box_measures_the_total_and_finds_a_uniform_shift():
    rep = run_black_box(_Shifted(1e-4), dev=range(0, 8), conf=range(8, 40))
    assert rep["mode"] == "black-box" and "no decomposition" in rep["comparison"]
    assert _verdicts(rep)["R1"] == "DETECTED_NEGATIVE"  # R1 projects on -1/sqrt(n): a positive shift
    assert rep["outputs"]["y"]["special_values"]["k_vs_f_class_mismatch"] == 0


def test_black_box_without_a_shift_does_not_detect_a_mean():
    rep = run_black_box(_Shifted(0.0), dev=range(0, 8), conf=range(8, 40))
    assert _verdicts(rep)["R1"] != "DETECTED_NEGATIVE"
    assert rep["outputs"]["y"]["total_black_box"]["scale"]["relative_rms"] < 1e-6
