"""Default detector on synthetic units (CPU)."""

import numpy as np
import pytest

pytest.importorskip("scipy")

from kernel_analyzer.reference_eval.detect import detect  # noqa: E402


def test_detector_sees_a_cancelling_mean_and_stays_quiet_on_noise():
    rng = np.random.default_rng(0)
    n, d = 128, 256
    k = rng.standard_normal((n, d))
    quiet = detect(rng.standard_normal((n, d)), k, 64, shape=(16, 16), seed=1)
    assert quiet["vector_mean"]["verdict"] == "NOT_CONFIRMED"
    mean = np.zeros(d)
    mean[0::2], mean[1::2] = 1.0, -1.0  # zero coordinate sum
    e = rng.standard_normal((n, d)) + 3.0 * mean / np.linalg.norm(mean)
    found = detect(e, k, 64, shape=(16, 16), seed=1)
    tests = {t["test"]: t for t in found["vector_mean"]["tests"]}
    assert found["vector_mean"]["verdict"] == "DETECTED"
    assert tests["omnibus"]["verdict"] == "DETECTED" and tests["total"]["verdict"] == "NOT_CONFIRMED"


def test_scaling_shows_in_the_alignment_family():
    rng = np.random.default_rng(2)
    k = rng.standard_normal((128, 256))
    e = rng.standard_normal((128, 256)) + 0.05 * k
    r = detect(e, k, 64, seed=3)
    assert r["alignment"]["verdict"] == "DETECTED"
