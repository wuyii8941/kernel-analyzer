"""Adapter for contracts/ROPE.yaml: rotary position embedding (rotate-half and interleaved layouts, partial rotary
dimension, theta base, linear position scaling, position ids; precomputed cos/sin tables in the duplicated (HF / Liger)
and half-width (GPT-OSS) layouts) with the output and the VJP against an explicit upstream.

The reference uses kernel_spec.real (exact rationals, rigorous enclosures of cos / sin / powers). The mutants are an
independent mpmath-interval implementation with the misreading as a parameter.
"""
from __future__ import annotations

from fractions import Fraction

from mpmath import iv

from kernel_spec import d1_attention as A
from kernel_spec import real as rl
from kernel_spec.num import Invalid, Result, Unsupported, integer, rational
from kernel_spec.real import R

ANGLE = {"rotate_half": "half", "interleaved": "pairs"}
TABLE = {"tables_duplicated": "dup", "tables_half": "half"}
KEYS_ANGLE = {"x", "positions", "rotary_dim", "base", "scaling_factor", "upstream"}
KEYS_TABLE = {"x", "cos", "sin", "upstream"}


def _rows(x, rank):
    """Iterate over the token rows [L][d] of a rank-2 or rank-3 tensor."""
    return [x] if rank == 2 else x


def parse(profile, inp):
    if profile not in ANGLE and profile not in TABLE:
        raise Unsupported(f"profile {profile!r}")
    keys = KEYS_ANGLE if profile in ANGLE else KEYS_TABLE
    if set(inp) - keys or keys - set(inp) - {"scaling_factor"}:
        raise Invalid(f"input fields must be {sorted(keys)}")
    rank = len(A.shape_of(inp["x"], "x"))
    if rank not in (2, 3):
        raise Invalid("x must be [L][d] or [H][L][d]")
    x, xs = A.tensor(inp["x"], rank, "x")
    g, gs = A.tensor(inp["upstream"], rank, "upstream")
    if gs != xs:
        raise Invalid("upstream must have the shape of x")
    L, d = xs[-2], xs[-1]
    if profile in ANGLE:
        pos = inp["positions"]
        if not isinstance(pos, list) or len(pos) != L:
            raise Invalid("positions must list one integer per token")
        pos = [integer(p, "positions") for p in pos]
        r = integer(inp["rotary_dim"], "rotary_dim")
        if r < 2 or r % 2 or r > d:
            raise Invalid("rotary_dim must be even, >= 2 and <= head_dim")
        base = rational(inp["base"], "base")
        fac = rational(inp.get("scaling_factor", 1), "scaling_factor")
        if base <= 0 or fac <= 0:
            raise Invalid("base and scaling_factor must be positive")
        return dict(kind="angle", layout=ANGLE[profile], x=x, g=g, rank=rank, L=L, d=d, r=r, pos=pos, base=base,
                    fac=fac)
    cos, cs = A.tensor(inp["cos"], 2, "cos")
    sin, ss = A.tensor(inp["sin"], 2, "sin")
    if cs != ss or cs[0] != L:
        raise Invalid("cos and sin must be [L][w] tables")
    w = cs[1]
    layout = TABLE[profile]
    r = w if layout == "dup" else 2 * w
    if r < 2 or r % 2 or r > d:
        raise Invalid(f"table width {w} gives rotary width {r}, outside [2, head_dim] or odd")
    if layout == "dup":
        h = r // 2
        for t in range(L):
            if cos[t][:h] != cos[t][h:] or sin[t][:h] != sin[t][h:]:
                raise Invalid("duplicated-layout tables must have equal halves (ROPE.TABLE_HALVES)")
    return dict(kind="table", layout="half", x=x, g=g, rank=rank, L=L, d=d, r=r, cos=cos, sin=sin,
                tw=w, tlayout=layout)


def _pairs(layout, r):
    return [(i, i + r // 2) for i in range(r // 2)] if layout == "half" else [(2 * i, 2 * i + 1) for i in range(r // 2)]


def _cs_angle(a, t, i):
    theta = R(a["base"]) ** R(Fraction(-2 * i, a["r"]))
    ang = R(a["pos"][t]) * theta / R(a["fac"])
    return rl.cos(ang), rl.sin(ang)


def _cs_table(a, t, i):
    return R(a["cos"][t][i]), R(a["sin"][t][i])


def reference(profile, inp):
    a = parse(profile, inp)
    cs = _cs_angle if a["kind"] == "angle" else _cs_table
    pairs = _pairs(a["layout"], a["r"])
    tables = [[cs(a, t, i) for i in range(len(pairs))] for t in range(a["L"])]
    outs, grads = [], []
    for X, G in zip(_rows(a["x"], a["rank"]), _rows(a["g"], a["rank"])):
        orow, grow = [], []
        for t in range(a["L"]):
            o = [R(v) for v in X[t]]
            gr = [R(v) for v in G[t]]
            for n, (p, s) in enumerate(pairs):
                c, si = tables[t][n]
                o[p] = R(X[t][p]) * c - R(X[t][s]) * si
                o[s] = R(X[t][s]) * c + R(X[t][p]) * si
                gr[p] = R(G[t][p]) * c + R(G[t][s]) * si
                gr[s] = R(G[t][s]) * c - R(G[t][p]) * si
            orow.append(A.values(o))
            grow.append(A.values(gr))
        outs.append(orow)
        grads.append(grow)
    if a["rank"] == 2:
        outs, grads = outs[0], grads[0]
    return Result("value", {"out": outs, "grad": grads})


# ------------------------------------------------------------------ mutants: independent interval implementation
def _I(q):
    q = Fraction(q)
    return iv.mpf(q.numerator) / q.denominator


def _indep(profile, inp, m=None):
    a = parse(profile, inp)
    d, r, L = a["d"], a["r"], a["L"]
    if a["kind"] == "angle":
        layout = a["layout"]
        if m == "layout_swapped":
            layout = "pairs" if layout == "half" else "half"
        expo_den = d if m == "theta_with_head_dim" else r
        fac = 1 if m == "scaling_ignored" else a["fac"]
        pos = list(range(L)) if m == "positions_from_zero" else a["pos"]

        def cs(t, i):
            th = iv.exp(-_I(Fraction(2 * i, expo_den)) * iv.log(_I(a["base"])))
            ang = _I(pos[t]) * th / _I(fac)
            return iv.cos(ang), iv.sin(ang)
        pairs = [(i, i + r // 2) for i in range(r // 2)] if layout == "half" else \
            [(2 * i, 2 * i + 1) for i in range(r // 2)]
        idx = list(range(r // 2))
    else:
        if m == "partial_ignored_full_head":         # Liger <= 0.8.4 (bug B002): rotate the whole head, read cos[i]
            pairs = [(i, i + d // 2) for i in range(d // 2)]
            idx = list(range(d // 2))
        elif m == "half_table_read_as_duplicated":   # Liger main (bug B001): rotary width = table width
            rr = a["tw"]
            pairs = [(i, i + rr // 2) for i in range(rr // 2)]
            idx = list(range(rr // 2))
        else:
            pairs = [(i, i + r // 2) for i in range(r // 2)]
            idx = list(range(r // 2))

        def cs(t, i):
            if i >= a["tw"]:
                raise Invalid("mutant reads beyond the table")
            return _I(a["cos"][t][i]), _I(a["sin"][t][i])
    outs, grads = [], []
    for X, G in zip(_rows(a["x"], a["rank"]), _rows(a["g"], a["rank"])):
        orow, grow = [], []
        for t in range(L):
            o = [_I(v) for v in X[t]]
            gr = [_I(v) for v in G[t]]
            for (p, s), i in zip(pairs, idx):
                c, si = cs(t, i)
                xp, xs, gp, gs = _I(X[t][p]), _I(X[t][s]), _I(G[t][p]), _I(G[t][s])
                o[p], o[s] = xp * c - xs * si, xs * c + xp * si
                if m == "backward_forward_rotation":
                    gr[p], gr[s] = gp * c - gs * si, gs * c + gp * si
                else:
                    gr[p], gr[s] = gp * c + gs * si, gs * c - gp * si
            orow.append(o)
            grow.append(gr)
        outs.append(orow)
        grads.append(grow)
    if a["rank"] == 2:
        outs, grads = outs[0], grads[0]
    return Result("value", {"out": outs, "grad": grads})


def _m(kind):
    return lambda profile, inp: _indep(profile, inp, kind)


MUTANTS = {
    "RO.M1": _m("layout_swapped"),
    "RO.M2": _m("layout_swapped"),
    "RO.M3": _m("partial_ignored_full_head"),
    "RO.M4": _m("half_table_read_as_duplicated"),
    "RO.M5": _m("theta_with_head_dim"),
    "RO.M6": _m("scaling_ignored"),
    "RO.M7": _m("backward_forward_rotation"),
    "RO.M8": _m("positions_from_zero"),
}
READINGS = {}
