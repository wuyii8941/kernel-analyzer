"""PyTorch Inductor, third batch: special functions (Inductor's own approximations), ordering ops with ties,
integer-like edge semantics.  Same contract as tool_spec_cases_inductor; the specification is the eager function
in float64.  For special functions e_sem therefore measures the kernel's approximation algorithm (evaluated exactly
by K_R) against the function; eager float32 has its own approximation error, reported by
scripts/tool_spec_eager_baseline.py for comparison.
"""

from __future__ import annotations

import torch

from tool_spec_cases_inductor import _case, rn

R, D = 64, 256
CASES = []


def _pos(g, lo=0.05, hi=8.0):
    return {"x": torch.rand(R, D, generator=g) * (hi - lo) + lo}


def _wide(g):
    return {"x": rn(g, R, D, scale=4)}


# special functions
CASES += _case("lgamma", lambda x: torch.lgamma(x), lambda g: {"x": rn(g, R, D, scale=5)}, "lgamma (negative x too)", bwd=["x"])
CASES += _case("digamma", lambda x: torch.digamma(x), _pos, "digamma on (0.05, 8)", bwd=["x"])
CASES += _case("digamma_neg", lambda x: torch.digamma(x), lambda g: {"x": -torch.rand(R, D, generator=g) * 5 - 0.01},
               "digamma on negative non-integers", bwd=None)
CASES += _case("polygamma1", lambda x: torch.polygamma(1, x), _pos, "polygamma(1, x)", bwd=None)
CASES += _case("erfinv", lambda x: torch.erfinv(x), lambda g: {"x": torch.rand(R, D, generator=g) * 1.98 - 0.99},
               "erfinv on (-0.99, 0.99)", bwd=["x"])
CASES += _case("erfcx", lambda x: torch.special.erfcx(x), _wide, "erfcx", bwd=["x"])
CASES += _case("i0", lambda x: torch.i0(x), _wide, "i0", bwd=["x"])
CASES += _case("i0e", lambda x: torch.special.i0e(x), _wide, "i0e", bwd=["x"])
CASES += _case("i1", lambda x: torch.special.i1(x), _wide, "i1", bwd=["x"])
CASES += _case("i1e", lambda x: torch.special.i1e(x), _wide, "i1e", bwd=["x"])
CASES += _case("ndtri", lambda x: torch.special.ndtri(x), lambda g: {"x": torch.rand(R, D, generator=g) * 0.998 + 0.001},
               "ndtri", bwd=None)
CASES += _case("log_ndtr", lambda x: torch.special.log_ndtr(x), lambda g: {"x": rn(g, R, D, scale=6)}, "log_ndtr", bwd=["x"])
CASES += _case("xlogy", lambda x, y: torch.xlogy(x, y), lambda g: {"x": torch.cat([torch.zeros(4, D), rn(g, R - 4, D)]),
                                                                      "y": torch.rand(R, D, generator=g) + 0.01},
               "xlogy with x = 0 rows", bwd=["x", "y"])
CASES += _case("xlog1py", lambda x, y: torch.special.xlog1py(x, y), lambda g: {"x": rn(g, R, D), "y": torch.rand(R, D, generator=g) - 0.5},
               "xlog1py", bwd=["x", "y"])
CASES += _case("sinc", lambda x: torch.sinc(x), lambda g: {"x": torch.cat([torch.zeros(2, D), rn(g, R - 2, D, scale=3)])},
               "sinc with x = 0 rows", bwd=["x"])
CASES += _case("logaddexp", lambda x, y: torch.logaddexp(x, y), lambda g: {"x": rn(g, R, D, scale=10), "y": rn(g, R, D, scale=10)},
               "logaddexp", bwd=["x", "y"])
CASES += _case("logaddexp2", lambda x, y: torch.logaddexp2(x, y), lambda g: {"x": rn(g, R, D, scale=10), "y": rn(g, R, D, scale=10)},
               "logaddexp2", bwd=["x", "y"])
CASES += _case("hypot", lambda x, y: torch.hypot(x, y), lambda g: {"x": rn(g, R, D), "y": rn(g, R, D)}, "hypot", bwd=["x", "y"])
CASES += _case("atan2", lambda x, y: torch.atan2(x, y), lambda g: {"x": rn(g, R, D), "y": rn(g, R, D)}, "atan2", bwd=["x", "y"])
CASES += _case("expm1_small", lambda x: torch.expm1(x), lambda g: {"x": rn(g, R, D, scale=1e-4)}, "expm1 near 0", bwd=["x"])
CASES += _case("log1p_small", lambda x: torch.log1p(x), lambda g: {"x": rn(g, R, D, scale=1e-4)}, "log1p near 0", bwd=["x"])
CASES += _case("sigmoid_tail", lambda x: torch.sigmoid(x), lambda g: {"x": rn(g, R, D, scale=30)}, "sigmoid with large |x|", bwd=["x"])
CASES += _case("tanh_tail", lambda x: torch.tanh(x), lambda g: {"x": rn(g, R, D, scale=10)}, "tanh", bwd=["x"])
CASES += _case("pow_frac", lambda x: torch.pow(x.abs() + 0.1, 2.7), lambda g: {"x": rn(g, R, D)}, "pow(|x| + 0.1, 2.7)", bwd=["x"])
CASES += _case("rsqrt", lambda x: torch.rsqrt(x), _pos, "rsqrt", bwd=["x"])
CASES += _case("cosh_sinh", lambda x: torch.cosh(x) - torch.sinh(x), lambda g: {"x": rn(g, R, D, scale=4)}, "cosh - sinh", bwd=["x"])
# integer-like edge semantics on floats
CASES += _case("fmod_neg", lambda x, y: torch.fmod(x, y), lambda g: {"x": rn(g, R, D, scale=10), "y": rn(g, R, D) + 0.0},
               "fmod (sign of dividend)", bwd=None)
CASES += _case("remainder_neg", lambda x, y: torch.remainder(x, y), lambda g: {"x": rn(g, R, D, scale=10), "y": rn(g, R, D)},
               "remainder (sign of divisor)", bwd=None)
CASES += _case("floor_divide_neg", lambda x, y: torch.floor_divide(x, y), lambda g: {"x": rn(g, R, D, scale=10), "y": rn(g, R, D)},
               "floor_divide", bwd=None)
CASES += _case("round_half_even", lambda x: torch.round(x), lambda g: {"x": torch.round(rn(g, R, D, scale=6) * 2) / 2},
               "round half to even (exact .5 inputs)", bwd=None)
CASES += _case("round_decimals", lambda x: torch.round(x, decimals=2), lambda g: {"x": rn(g, R, D, scale=6)},
               "round decimals 2", bwd=None)
CASES += _case("frac_trunc", lambda x: torch.frac(x) + torch.trunc(x) * 0.5, lambda g: {"x": rn(g, R, D, scale=6)},
               "frac + 0.5 trunc", bwd=None)
CASES += _case("nan_to_num", lambda x: torch.nan_to_num(x, nan=0.5, posinf=7.0, neginf=-7.0),
               lambda g: {"x": torch.where(torch.rand(R, D, generator=g) < 0.05, torch.tensor(float("inf")), rn(g, R, D))},
               "nan_to_num posinf", bwd=None)


# ordering with ties: values and (as float) indices
def _tied(g):
    return {"x": torch.round(rn(g, R, D) * 2) / 2}


CASES += _case("sort_stable_idx", lambda x: torch.sort(x, dim=-1, stable=True).indices.to(x.dtype), _tied,
               "sort stable=True indices with ties", bwd=None)
CASES += _case("sort_desc_stable_idx", lambda x: torch.sort(x, dim=-1, descending=True, stable=True).indices.to(x.dtype),
               _tied, "sort descending stable indices with ties", bwd=None)
CASES += _case("sort_values", lambda x: torch.sort(x, dim=-1).values, lambda g: {"x": rn(g, R, D)}, "sort values", bwd=["x"])
CASES += _case("topk_values", lambda x: torch.topk(x, 8, dim=-1).values, lambda g: {"x": rn(g, R, D)}, "topk 8 values", bwd=["x"])
CASES += _case("cummax_vals", lambda x: torch.cummax(x, dim=-1).values, lambda g: {"x": rn(g, R, D)}, "cummax values", bwd=["x"])
CASES += _case("cummax_idx_ties", lambda x: torch.cummax(x, dim=-1).indices.to(x.dtype), _tied,
               "cummax indices with ties (last occurrence)", bwd=None)
CASES += _case("argmax_ties", lambda x: torch.argmax(x, dim=-1).to(x.dtype), _tied, "argmax first occurrence", bwd=None)
CASES += _case("searchsorted_right", lambda s, v: torch.searchsorted(s, v, right=True).to(v.dtype),
               lambda g: {"s": torch.sort(torch.round(rn(g, R, 64) * 4) / 4, dim=-1).values,
                          "v": torch.round(rn(g, R, 32) * 4) / 4}, "searchsorted right with exact hits", bwd=None)
CASES += _case("bucketize_left", lambda v, b: torch.bucketize(v, b, right=False).to(v.dtype),
               lambda g: {"v": torch.round(rn(g, R, D) * 4) / 4, "b": torch.linspace(-2, 2, 17)},
               "bucketize left with exact boundaries", bwd=None)
