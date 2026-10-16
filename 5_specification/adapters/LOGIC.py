"""Adapter for contracts/LOGIC.yaml. Reference: kernel_spec.aten; isnan reads the raw IEEE values (profile ieee_values).
Mutants replace one operation by an independent implementation of the misreading."""
from __future__ import annotations

import yaml

from kernel_spec import aten as A
from kernel_spec.aten_adapter import make_reference
from kernel_spec.num import Invalid, Result, Unsupported
from kernel_spec.registry import CONTRACTS

COVERS = yaml.safe_load((CONTRACTS / "LOGIC.yaml").read_text())["covers"]


def _isnan_values(inp, nan_on_inf=False):
    (t,) = inp["args"]
    vals = t["v"]

    def walk(v):
        if isinstance(v, list):
            return [walk(x) for x in v]
        s = str(v).strip().lower()
        return s == "nan" or (nan_on_inf and s in ("inf", "+inf", "-inf"))
    return Result("value", {"out": walk(vals)})


def _wrap(base, nan_on_inf=False):
    def reference(profile, inp):
        if inp.get("op") == "aten.isnan":
            if profile != "ieee_values":
                raise Unsupported("isnan is defined in the ieee_values profile")
            return _isnan_values(inp, nan_on_inf)
        if profile != "real":
            raise Unsupported(f"profile {profile} covers isnan only")
        return base(profile, inp)
    return reference


reference = _wrap(make_reference(COVERS))


def _bitwise_logical_and(a, b):
    return A.binary(lambda x, y: bool(int(x) & int(y)), a, b, out_dtype="b8")


def _lt_as_le(a, b):
    return A.le(a, b)


def _all_empty_false(a, dim=None, keepdim=False):
    t = A.as_ten(a)
    if t.a.size == 0:
        return A.Ten(A.np.array(False, dtype=object), "b8")
    return A.all_(a, dim, keepdim)


def _bnot_int(a):
    t = A.as_ten(a)
    return A.unary(lambda x: bool(~int(x)), t, out_dtype="b8")


def _inf_as_nan_lt(a, b):
    def lt(x, y):
        if not getattr(x, "finite", True) or not getattr(y, "finite", True):
            return False
        return x < y
    return A.binary(lambda x, y: bool(lt(A.R(x) if not isinstance(x, A.R) else x, A.R(y) if not isinstance(y, A.R) else y)),
                    a, b, out_dtype="b8")


MUTANTS = {
    "LG.M1": _wrap(make_reference(COVERS, {"aten.logical_and": _bitwise_logical_and})),
    "LG.M2": _wrap(make_reference(COVERS, {"aten.lt": _lt_as_le})),
    "LG.M3": _wrap(make_reference(COVERS, {"aten.all": _all_empty_false})),
    "LG.M4": _wrap(make_reference(COVERS, {"aten.bitwise_not": _bnot_int})),
    "LG.M5": _wrap(make_reference(COVERS), nan_on_inf=True),
    "LG.M6": _wrap(make_reference(COVERS, {"aten.lt": _inf_as_nan_lt})),
}
READINGS = {}
