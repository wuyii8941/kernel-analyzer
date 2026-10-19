"""Bitwise verification must include every written integer and boolean output."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from kernel_analyzer.reference_eval import emulate as E


def buffer(name, value, actual, kind="i", **updates):
    dtype = np.float64 if kind == "f" else np.int64
    lo = np.array(value, dtype=dtype)
    after = np.array(actual, dtype=dtype)
    fields = dict(name=name, kind=kind, elem="f64" if kind == "f" else "i64",
                  lo=lo, hi=lo.copy() if kind == "f" else None,
                  st=np.zeros(lo.shape, dtype=np.int8), written=np.ones(lo.shape, dtype=bool),
                  cond=np.zeros(lo.shape, dtype=bool), premise=None, iset=None,
                  actual_after=after, actual_after_st=np.zeros(lo.shape, dtype=np.int8),
                  after_raw=after.view(np.uint8))
    return SimpleNamespace(**(fields | updates))


def verify(*buffers):
    result = SimpleNamespace(buffers=dict(enumerate(buffers)), aborted={})
    emulator = SimpleNamespace(oracle=None, trace={}, swappable=[], ptx_choices={}, sass_choices={})
    with patch.object(E, "emulate", return_value=(result, emulator)):
        return E.verify(SimpleNamespace(asm={"ttir": "test"}), ungrouped=[], swapped=[])


def test_integer_only_outputs_are_compared_without_float_conversion():
    result = verify(buffer("out", [2**53+1, -100], [2**53+1, -100]))
    assert result["status"] == "bit_identical"
    assert result["buffers"]["out"]["bit_identical"] == 2
    result = verify(buffer("out", [2**53+1], [2**53]))
    assert result["status"] == "mismatch"


def test_matching_float_does_not_hide_wrong_integer_output():
    result = verify(buffer("float", [1], [1], "f"), buffer("index", [2], [3]))
    assert result["status"] == "mismatch"
    assert result["buffers"]["index"]["mismatch"] == 1


@pytest.mark.parametrize("updates", [dict(st=np.array([4])), dict(cond=np.array([True])),
                                    dict(premise=np.array([True])), dict(hi=np.array([3])),
                                    dict(iset=np.array([True])), dict(actual_after=None)])
def test_unestablished_integer_output_cannot_be_skipped(updates):
    result = verify(buffer("float", [1], [1], "f"), buffer("index", [2], [2], **updates))
    assert result["status"] == "partial"
    assert result["buffers"]["index"]["not_emulated"] == 1


def test_boolean_unknown_is_not_a_verified_byte():
    result = verify(buffer("mask", [0, 1], [0, 1], "b"))
    assert result["status"] == "bit_identical"
    result = verify(buffer("mask", [-1], [-1], "b"))
    assert result["status"] == "not_established"
