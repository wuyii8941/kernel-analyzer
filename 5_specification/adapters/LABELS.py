"""Adapter for contracts/LABELS.yaml. The reference is kernel_spec.e2_train.shift_and_count; the mutants are an
independent implementation of the misreadings named in the contract."""
from __future__ import annotations

from kernel_spec import e2_train
from kernel_spec.num import Result


def reference(profile, inp):
    return Result("value", e2_train.shift_and_count(profile, inp))


READINGS = {}


def _independent(profile, inp, mis):
    ii = -100 if mis == "hard_ignore" else int(inp.get("ignore_index", -100))
    rows = [list(map(int, r)) for r in inp["labels"]]
    if mis == "no_shift":
        sh = [list(r) for r in rows]
    elif mis == "right_shift":
        sh = [[ii] + r[:-1] for r in rows]
    else:
        sh = [r[1:] + [ii] for r in rows]
    if mis == "count_before":
        cnt = sum(1 for r in rows for t in r if t != ii)
    elif mis == "count_all":
        cnt = sum(len(r) for r in sh)
    else:
        cnt = sum(1 for r in sh for t in r if t != ii)
    return Result("value", {"shifted": sh, "count": cnt})


def _m(kind):
    return lambda profile, inp: _independent(profile, inp, kind)


MUTANTS = {"LB.M1": _m("no_shift"), "LB.M2": _m("right_shift"), "LB.M3": _m("count_before"), "LB.M4": _m("count_all"),
           "LB.M5": _m("hard_ignore")}
