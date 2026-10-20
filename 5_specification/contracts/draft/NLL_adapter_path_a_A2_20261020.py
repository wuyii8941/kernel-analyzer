"""Adapter for contracts/NLL.yaml: aten.nll_loss_forward / aten.nll_loss_backward through the ATen-call interface
(kernel_spec.aten_adapter.make_reference; semantics in kernel_spec.norm_emb_nll_aten).

The mutants and the diagnostic reading are independent compact implementations below (not switches of the
reference).
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import yaml

from kernel_spec import aten as A
from kernel_spec import norm_emb_nll_aten as NA
from kernel_spec.aten_adapter import make_output_checker, make_reference
from kernel_spec.num import Undefined
from kernel_spec.real import R
from kernel_spec.registry import CONTRACTS

PROFILE = "aten_real"
COVERS = yaml.safe_load((CONTRACTS / "NLL.yaml").read_text())["covers"]
OPS = {k: NA.OPS[k] for k in ("aten.nll_loss_forward", "aten.nll_loss_backward")}
assert set(OPS) == set(COVERS)
# Harness convention for tuple-returning overloads: (output, total_weight), both defined by the task.
ATEN_RETURNS = {"aten.nll_loss_forward": {"positions": 2, "not_compared": {}}}


def _ref(extra=None):
    return make_reference(COVERS, extra={**OPS, **(extra or {})}, profiles=(PROFILE,),
                          argument_decoder=NA.decode_argument)


reference = _ref()
check_output = make_output_checker(COVERS, extra=OPS, profiles=(PROFILE,), argument_decoder=NA.decode_argument)


# ------------------------------------------------------------------ independent implementation
def _matrix(t):
    a = t.a if t.a.ndim == 2 else t.a.reshape(1, -1)
    return [[v if isinstance(v, float) else v.v for v in row] for row in a]


def _targets(t):
    return [int(v) for v in t.a.reshape(-1)]


def _scalar(v, dtype):
    return A.Ten(np.array(R(v), dtype=object), dtype)


def _forward(mis):
    def op(self, target, weight, reduction, ignore_index):
        x, ts = _matrix(self), _targets(target)
        w = None if weight is None else [v.v for v in weight.a.reshape(-1)]
        if mis == "enum_swapped":
            reduction = {1: 2, 2: 1}.get(reduction, reduction)
        loss, tw, count = [], Fraction(0), 0
        for n, c in enumerate(ts):
            if c == ignore_index:
                if mis == "ignored_contribute":
                    loss.append(-x[n][0] * (w[0] if w else 1))
                else:
                    loss.append(Fraction(0))
                continue
            wc = w[c] if w else Fraction(1)
            loss.append(-wc * x[n][c] if mis != "sign" else wc * x[n][c])
            tw += wc
            count += 1
        if mis in ("all_targets", "total_weight_all"):
            tw = Fraction(len(ts))
        if reduction == 0 and self.a.ndim == 2:
            total = len(ts) if mis == "none_total_weight_count" else 0
            return [A.Ten(np.array([R(v) for v in loss], dtype=object), self.dtype), _scalar(total, self.dtype)]
        s = sum(loss, Fraction(0))
        if reduction == 1:
            denom = tw
            if mis == "total_weight_all":
                denom = sum((w[c] if w else 1) for c in ts if c != ignore_index)
            if mis == "weighted_mean_by_count":
                denom = count
            if denom == 0:
                if mis == "empty_mean_zero":
                    return [_scalar(0, self.dtype), _scalar(0, self.dtype)]
                raise Undefined("0/0")
            s = s / denom
        return [_scalar(s, self.dtype), _scalar(tw, self.dtype)]
    return op


def _backward(mis):
    def op(grad_output, self, target, weight, reduction, ignore_index, total_weight):
        ts = _targets(target)
        C = self.shape[-1]
        w = None if weight is None or mis == "weight_ignored" else [v.v for v in weight.a.reshape(-1)]
        go = [v.v for v in grad_output.a.reshape(-1)]
        if mis == "recomputed_total_weight":
            tw = sum(((w[c] if w else Fraction(1)) for c in ts if c != ignore_index), Fraction(0))
        else:
            tw = total_weight.a.reshape(-1)[0].v
        rows = len(ts)
        out = [[Fraction(0)] * C for _ in range(rows)]
        for n, c in enumerate(ts):
            if c == ignore_index:
                if mis != "ignored_get_gradient":
                    continue
                c = 0
            g = go[n] if (reduction == 0 and self.a.ndim == 2) else go[0]
            if reduction == 1 and mis != "no_division":
                if tw == 0:
                    raise Undefined("division by total_weight 0")
                g = g / tw
            out[n][c] = -(w[c] if w else 1) * g
        flat = [R(v) for row in out for v in row]
        return A.Ten(np.array(flat, dtype=object).reshape(self.shape), self.dtype)
    return op


def _mf(mis):
    return _ref({"aten.nll_loss_forward": _forward(mis)})


def _mb(mis):
    return _ref({"aten.nll_loss_backward": _backward(mis)})


MUTANTS = {
    "NL.M1": _mf("all_targets"),
    "NL.M2": _mf("ignored_contribute"),
    "NL.M3": _mf("total_weight_all"),
    "NL.M4": _mf("sign"),
    "NL.M5": _mb("no_division"),
    "NL.M6": _mb("ignored_get_gradient"),
    "NL.M7": _mf("weighted_mean_by_count"),
    "NL.M8": _mf("enum_swapped"),
    "NL.M9": _mf("empty_mean_zero"),
    "NL.M10": _mf("none_total_weight_count"),
    "NL.M11": _mb("weight_ignored"),
}
READINGS = {"NL.R1": _mb("recomputed_total_weight")}
