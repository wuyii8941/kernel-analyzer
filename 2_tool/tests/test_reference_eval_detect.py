"""Default detector on synthetic units (CPU)."""

import math

import numpy as np
import pytest

pytest.importorskip("scipy")

from kernel_analyzer.reference_eval.detect import detect, omnibus  # noqa: E402


def _verdicts(r):
    return r["vector_mean"]["verdict"], r["alignment"]["verdict"]


def test_detector_sees_a_cancelling_mean_and_stays_quiet_on_noise():
    rng = np.random.default_rng(0)
    n, d = 128, 256
    k = rng.standard_normal((n, d))
    e0 = rng.standard_normal((n, d))
    quiet = detect(e0, e0, k, 64, shape=(16, 16), seed=1)
    assert quiet["vector_mean"]["verdict"] == "NOT_CONFIRMED"
    mean = np.zeros(d)
    mean[0::2], mean[1::2] = 1.0, -1.0  # zero coordinate sum
    e = rng.standard_normal((n, d)) + 3.0 * mean / np.linalg.norm(mean)
    found = detect(e, e, k, 64, shape=(16, 16), seed=1)
    tests = {t["test"]: t for t in found["vector_mean"]["tests"]}
    assert found["vector_mean"]["verdict"] == "DETECTED"
    assert tests["omnibus"]["verdict"] == "DETECTED" and tests["total"]["verdict"] == "NOT_CONFIRMED"


def test_scaling_shows_in_the_alignment_family():
    rng = np.random.default_rng(2)
    k = rng.standard_normal((128, 256))
    e = rng.standard_normal((128, 256)) + 0.05 * k
    r = detect(e, e, k, 64, seed=3)
    assert r["alignment"]["verdict"] == "DETECTED"


def test_zero_width_omnibus_is_the_plain_sign_flip_test():
    rng = np.random.default_rng(4)
    e = rng.standard_normal((40, 30)) + 0.3
    r = omnibus(e, e, np.random.default_rng(9), flips=499)
    # plain U-statistic sign-flip test with the same flips
    n = e.shape[0]
    g = e @ e.T
    off = g - np.diag(np.diag(g))
    stat = off.sum() / (n * (n - 1))
    signs = np.random.default_rng(9).choice([-1.0, 1.0], size=(499, n))
    flipped = np.einsum("bi,ij,bj->b", signs, off, signs) / (n * (n - 1))
    assert math.isclose(r["lower_bound_of_statistic"], stat, rel_tol=1e-12)
    assert r["p"] == (1 + (flipped >= stat * (1 - 1e-12)).sum()) / 500


# --- reference intervals: the cases that version 1 (midpoints) got wrong, and their neighbours ---------

def _zero_residual_boxes(rng, n, d, k, asymmetric=False):
    """True residual 0; the enclosure's midpoint is a stable function of k (a deterministic evaluation
    artefact, like two interval computations of the same exact value), so midpoints correlate with k."""

    mid = 3e-15 * k + 1e-15 * np.tanh(k)
    rad = np.abs(mid) * 1.5 + 1e-16
    lo, hi = mid - rad, mid + rad
    if asymmetric:  # zero near one end of every box
        lo, hi = -0.05 * rad, 1.9 * rad + mid
    assert np.all(lo <= 0) and np.all(hi >= 0)
    return lo, hi


@pytest.mark.parametrize("asymmetric", [False, True])
def test_boxes_containing_zero_never_detect(asymmetric):
    rng = np.random.default_rng(6)
    n, d = 96, 512
    k = rng.standard_normal((n, d)) * (1 + rng.random(d))
    lo, hi = _zero_residual_boxes(rng, n, d, k, asymmetric)
    mid = 0.5 * (lo + hi)
    # the midpoints alone look like a strong aligned "bias" (what version 1 tested)
    s = (mid * k).sum(axis=1) / np.linalg.norm(k, axis=1)
    assert abs(s.mean()) / (s.std(ddof=1) / math.sqrt(n)) > 20
    r = detect(lo, hi, k, 32, shape=(16, 32), seed=7)
    assert _verdicts(r) == ("NOT_CONFIRMED", "NOT_CONFIRMED")
    omni = r["vector_mean"]["tests"][0]
    assert omni["test"] == "omnibus" and omni["p"] == 1.0


def test_real_small_bias_with_narrow_boxes_is_detected_with_its_sign():
    rng = np.random.default_rng(8)
    n, d = 96, 512
    k = rng.standard_normal((n, d))
    e = -2e-9 + 1e-8 * rng.standard_normal((n, d))  # negative mean, far below the noise per coordinate
    lo, hi = e - 1e-15, e + 1e-15
    r = detect(lo, hi, k, 32, seed=9)
    tests = {t["test"]: t for t in r["vector_mean"]["tests"]}
    assert r["vector_mean"]["verdict"] == "DETECTED"
    assert tests["total"]["verdict"] == "DETECTED" and tests["total"]["interval_sign"] == "-"


def test_wide_boxes_hide_a_bias_conservatively_and_narrow_ones_reveal_it():
    rng = np.random.default_rng(10)
    n, d = 96, 256
    k = rng.standard_normal((n, d))
    e = 1e-9 + 1e-10 * rng.standard_normal((n, d))
    wide = detect(e - 5e-9, e + 5e-9, k, 32, seed=11)  # boxes much wider than the bias
    narrow = detect(e - 1e-13, e + 1e-13, k, 32, seed=11)
    assert wide["vector_mean"]["verdict"] == "NOT_CONFIRMED"
    assert narrow["vector_mean"]["verdict"] == "DETECTED"
    for t in wide["vector_mean"]["tests"] + wide["alignment"]["tests"]:
        assert t["verdict"] == "NOT_CONFIRMED"


def test_directions_actually_used_are_returned():
    rng = np.random.default_rng(12)
    e = rng.standard_normal((64, 48))
    r = detect(e, e, rng.standard_normal((64, 48)), 32, shape=(6, 8), seed=1, return_directions=True)
    dirs = r["directions"]
    assert set(dirs) >= {"total", "learned", "rows", "cols", "aligned", "relative"}
    assert math.isclose(np.linalg.norm(dirs["rows"]), 1.0, rel_tol=1e-12)
    assert dirs["aligned"].shape == (64, 48)
