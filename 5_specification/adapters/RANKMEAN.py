"""Adapter for contracts/RANKMEAN.yaml: global token mean over ranks from per-rank loss sums and exact counts.
Inputs are checked before any arithmetic: equal lengths, unique participants, nonnegative integral counts (never
truncated), and no loss mass on a rank without tokens."""
from __future__ import annotations

from fractions import Fraction

from kernel_spec import v24
from kernel_spec.num import Invalid, Result, Undefined, Unsupported, integer, rational, require_fields, vector

tp = v24.module("spec_training_program")


def _inputs(profile, inp, count=integer):
    if profile != "base":
        raise Unsupported(f"profile {profile!r}")
    required = {"S", "n", "participant_ids"}
    require_fields(inp, required, "RANKMEAN input")
    if set(inp) - required:
        raise Unsupported("RANKMEAN input contains options outside the equal-valid-token profile")
    ids = inp["participant_ids"]
    S = vector(inp["S"], "S")
    n = vector(inp["n"], "n", item=count)
    if not isinstance(ids, (list, tuple)) or not (len(S) == len(n) == len(ids)) or not S:
        raise Invalid("S, n and participant_ids must have the same nonzero length")
    try:
        distinct_ids = set(ids)
    except TypeError as exc:
        raise Unsupported("participant IDs must have a hashable representation") from exc
    if len(distinct_ids) != len(ids):
        raise Invalid("duplicate participant IDs")
    if any(c < 0 for c in n):
        raise Invalid("negative count")
    if any(c == 0 and s != 0 for s, c in zip(S, n)):
        raise Invalid("loss mass on a rank without tokens")
    return S, n


def reference(profile, inp):
    S, n = _inputs(profile, inp)
    if sum(n) == 0:
        raise Undefined("no valid tokens on any rank")
    return Result("value", {"loss": v24.value(tp.cross_rank_mean(S, n))})


def check_output(profile, inp, observed):
    """Exact membership for the explicitly selected equal-token task (RM.C1)."""
    S, n = _inputs(profile, inp)
    total = sum(n)
    if total == 0:
        raise Undefined("no valid tokens on any rank")
    if not isinstance(observed, dict) or set(observed) != {"loss"}:
        return False
    return rational(observed["loss"], "observed loss") * total == sum(S)


# ------------------------------------------------------------------ independent mutants
def _local_means(S, n):
    return [s / c for s, c in zip(S, n) if c > 0]


def m1(profile, inp):
    S, n = _inputs(profile, inp)
    means = _local_means(S, n)
    return Result("value", {"loss": sum(means) / len(means)})


def m2(profile, inp):
    S, n = _inputs(profile, inp)
    return Result("value", {"loss": sum(_local_means(S, n))})


def m3(profile, inp):
    S, n = _inputs(profile, inp)
    return Result("value", {"loss": sum(S) / len(S)})


def m4(profile, inp):
    S, n = _inputs(profile, inp, count=lambda x, what: int(v24.value(Fraction(x) if not isinstance(x, str) else
                                                                     Fraction(x))))
    return Result("value", {"loss": sum(S) / sum(n)})


def m5(profile, inp):
    S = vector(inp["S"], "S")
    n = vector(inp["n"], "n", item=integer)
    pairs = list(zip(S, n))
    return Result("value", {"loss": sum(s for s, _ in pairs) / sum(c for _, c in pairs)})


MUTANTS = {"RM.M1": m1, "RM.M2": m2, "RM.M3": m3, "RM.M4": m4, "RM.M5": m5}
READINGS = {}
