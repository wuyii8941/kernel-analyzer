import importlib.util
from pathlib import Path

import numpy as np


_SPEC = importlib.util.spec_from_file_location(
    "analyze_tcmp_pattern_screen",
    Path(__file__).parents[1] / "scripts" / "analyze_tcmp_pattern_screen.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


def test_exact_sign_flip_distinguishes_aligned_from_canceling_states():
    aligned = np.asarray([[1.0, 0.0]] * 8)
    canceling = np.asarray([[1.0, 0.0], [-1.0, 0.0]] * 4)
    p_aligned, amplification_aligned = _MODULE._sign_flip_p(aligned)
    p_canceling, amplification_canceling = _MODULE._sign_flip_p(canceling)
    assert p_aligned < p_canceling
    assert amplification_aligned > amplification_canceling


def test_bh_marks_nothing_when_no_p_value_exists():
    rows = [{
        "p_value": None,
        "implementation_pattern_id": "p",
        "endpoint": "out",
        "screen_positive_bh_q_0_10": False,
    }]
    _MODULE._bh(rows, 0.10)
    assert rows[0]["screen_positive_bh_q_0_10"] is False
