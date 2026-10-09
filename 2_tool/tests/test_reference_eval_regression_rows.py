"""Real-data regression for the default detector (CPU).

The fixture holds, for every output where blind_test_v1 phase 2's semantic residual intervals all contained
zero while the version 1 detector (midpoints) reported a detection, the e_sem intervals and K_R midpoints of
seeds 0-95 on the output's first coordinates.  The rows are selected from the frozen report by that
property (scripts/blind_test_v1_regression.py), not by program name.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

from kernel_analyzer.reference_eval.detect import detect  # noqa: E402

FIXTURE = Path(__file__).parent / "data" / "blind_v1_zero_semantic_residual_rows.npz"


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture written by scripts/blind_test_v1_regression.py")
def test_zero_semantic_residual_rows_are_not_detected():
    data = np.load(FIXTURE)
    keys = sorted({k.rsplit("__", 1)[0] for k in data.files})
    assert keys, "empty fixture"
    for key in keys:
        lo, hi, kr = data[f"{key}__lo"], data[f"{key}__hi"], data[f"{key}__kr"]
        assert np.all(lo <= 0) and np.all(hi >= 0), key  # K_R = f within the enclosures
        r = detect(lo, hi, kr, 32, seed=0)
        assert r["vector_mean"]["verdict"] == "NOT_CONFIRMED", key
        assert r["alignment"]["verdict"] == "NOT_CONFIRMED", key
        for fam in ("vector_mean", "alignment"):
            for t in r[fam]["tests"]:
                assert t["p"] >= 0.05, (key, t["test"], t["p"])
