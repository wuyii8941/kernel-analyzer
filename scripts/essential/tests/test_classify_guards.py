"""A2 (docs/protocol_essential_bugs_phase2_20261007.md): classification-layer guards.

    PYTHONPATH=src /data1/tzh/envs/ka_main/bin/python -m pytest -q scripts/essential/tests/test_classify_guards.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import classify as Q  # noqa: E402


def _cond():
    return {"op": "index_add", "id": "t0", "shape": "1d", "m": 3, "dup": "some", "alpha": 1.0, "values": "ints"}


def test_fr_with_no_usable_element_is_not_established():
    keep = [{"r_lo": np.zeros(3), "r_hi": np.zeros(3), "k": np.zeros(3), "ok": np.zeros(3, dtype=bool), "shape": (3,)}] * 3
    fr = {"status": "ok", "keep": {"out": keep}, "notes": {}, "mixed": {}}
    spec = {"status": "ok", "readings": {"main": {"status": "ok", "out": (np.ones(3), np.ones(3))}}, "grad": None}
    res = Q.fr_assess("index", _cond(), "inductor_cuda32", fr, {0: spec, 1: spec, 2: spec})
    o = res["outputs"]["out"]
    assert o["semantic_vs_main"] is None and o["not_established"]


def test_fr_with_usable_elements_still_decides():
    keep = [{"r_lo": np.ones(3), "r_hi": np.ones(3), "k": np.ones(3), "ok": np.ones(3, dtype=bool), "shape": (3,)}] * 3
    fr = {"status": "ok", "keep": {"out": keep}, "notes": {}, "mixed": {}}
    spec = {"status": "ok", "readings": {"main": {"status": "ok", "out": (np.ones(3), np.ones(3))}}, "grad": None}
    res = Q.fr_assess("index", _cond(), "inductor_cuda32", fr, {0: spec, 1: spec, 2: spec})
    assert res["outputs"]["out"]["semantic_vs_main"] is False


def test_shared_relation_needs_a_comparable_pair():
    assert Q.combine("deviates", {"deviating_elements": 0, "compared_outputs": 0}, "deviates") == "shared_relation_not_established"
    assert Q.combine("deviates", {"deviating_elements": 0, "compared_outputs": 2}, "deviates") == "shared_deviation"
    assert Q.combine("deviates", {"deviating_elements": 5, "compared_outputs": 2}, "deviates") == "both_deviate_differently"
    assert Q.combine("compatible", {"deviating_elements": 0, "compared_outputs": 0}, "compatible") == "compatible"
