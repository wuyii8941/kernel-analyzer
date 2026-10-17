"""Adapter for contracts/BLIND_V2.yaml: the blind test v2 issuer's family specifications f (G1..G8).

One profile per family; input is a dict of the family's tensors (nested lists of exact rationals / integers) and runtime
scalars; output y with the program's output shape. Reference on kernel_spec.real (exact rationals, rigorous enclosures of
the square roots). The mutants are an independent Fraction / mpmath-interval implementation of the answer key's planted
semantic variants and of natural misreadings.
"""
from __future__ import annotations

from fractions import Fraction

from mpmath import iv

from kernel_spec import f_issuer as I
from kernel_spec import real as rl
from kernel_spec.num import Invalid, Result, Undefined, to_iv
from kernel_spec.real import R

PROFILES = ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8")
KEYS = {"G1": ("x", "gamma", "beta", "res", "eps"), "G2": ("a", "b"), "G3": ("h",), "G4": ("theta", "v", "g", "lr", "mu", "wd"),
        "G5": ("packed", "sc", "a", "G"), "G6": ("a", "b"), "G7": ("q", "k", "v", "pos", "scale"), "G8": ("x",)}


def _same(shapes, what):
    if any(s != shapes[0] for s in shapes):
        raise Invalid(f"{what}: operand shapes {shapes} differ")


def _byte(x, what):
    k = I.count(x, what)
    if not -128 <= k <= 255:
        raise Invalid(f"{what}: {k} is not a byte (int8 or uint8 value)")
    return k


def _shape(m):
    return (len(m), len(m[0]))


def _inputs(profile, inp):
    I.profile(profile, PROFILES)
    I.keys(inp, KEYS[profile], profile)
    p = profile
    if p == "G1":
        x, res = I.mat(inp["x"], "x"), I.mat(inp["res"], "res")
        _same([_shape(x), _shape(res)], "G1 x, res")
        D = len(x[0])
        return dict(x=x, res=res, gamma=I.vec(inp["gamma"], "gamma", D), beta=I.vec(inp["beta"], "beta", D),
                    eps=I.scalar(inp["eps"], "eps"))
    if p in ("G2", "G6"):
        if p == "G6":
            a, b = I.mat(inp["a"], "a"), I.mat(inp["b"], "b")
            _same([_shape(a), _shape(b)], "G6 a, b")
            return dict(a=a, b=b)
        sa, a = I.tensor(inp["a"], "a")
        sb, b = I.tensor(inp["b"], "b")
        _same([sa, sb], "G2 a, b")
        return dict(shape=sa, a=a, b=b)
    if p == "G3":
        return dict(h=I.mat(inp["h"], "h"))
    if p == "G4":
        st, th = I.tensor(inp["theta"], "theta")
        sv, v = I.tensor(inp["v"], "v")
        sg, g = I.tensor(inp["g"], "g")
        _same([st, sv, sg], "G4 theta, v, g")
        return dict(shape=st, theta=th, v=v, g=g, lr=I.scalar(inp["lr"], "lr"), mu=I.scalar(inp["mu"], "mu"),
                    wd=I.scalar(inp["wd"], "wd"))
    if p == "G5":
        pk = I.mat(inp["packed"], "packed", item=_byte)
        K = 2 * len(pk[0])
        a = I.vec(inp["a"], "a", K)
        G = I.count(inp["G"], "G", 1)
        sc = I.mat(inp["sc"], "sc")
        _same([(len(pk), -(-K // G)), _shape(sc)], "G5 sc (R x ceil(K/G))")
        return dict(packed=pk, a=a, G=G, sc=sc)
    if p == "G7":
        q = I.mat(inp["q"], "q")
        sk, kf = I.rank(inp["k"], "k", 3)
        sv, vf = I.rank(inp["v"], "v", 3)
        Rr, D = _shape(q)
        if sk[0] != Rr or sk[2] != D or sv[:2] != sk[:2]:
            raise Invalid(f"G7: q {(Rr, D)}, k {sk}, v {sv} do not fit")
        L, Dv = sk[1], sv[2]
        k = [[kf[(r * L + j) * D:(r * L + j + 1) * D] for j in range(L)] for r in range(Rr)]
        v = [[vf[(r * L + j) * Dv:(r * L + j + 1) * Dv] for j in range(L)] for r in range(Rr)]
        pos = I.vec(inp["pos"], "pos", Rr, item=I.count)
        return dict(q=q, k=k, v=v, pos=pos, scale=I.scalar(inp["scale"], "scale"), L=L, Dv=Dv)
    return dict(x=I.mat(inp["x"], "x"))


def _nibbles(byte):
    """q_{2j}, q_{2j+1} of the issuer's formula; Python's & and >> act on the two's-complement bits, so an int8 byte and
    the uint8 byte with the same bits give the same nibbles."""
    return (byte & 15) - 8, ((byte >> 4) & 15) - 8


# ------------------------------------------------------------------------------------------------ reference
def reference(profile, inp):
    z = _inputs(profile, inp)
    V = I.v
    if profile == "G1":
        y = []
        for xr, rr in zip(z["x"], z["res"]):
            if not xr:
                raise Undefined("mean over an empty row")
            mu = sum((R(v) for v in xr), R(0)) / len(xr)
            var = sum(((R(v) - mu) * (R(v) - mu) for v in xr), R(0)) / len(xr)
            q = var + z["eps"]
            if q.sign() <= 0:
                raise Undefined("(var + eps)^(-1/2) of a non-positive number")
            rstd = R(1) / rl.sqrt(q)
            y.append([V((R(v) - mu) * rstd * gm + bt + rv) for v, gm, bt, rv in zip(xr, z["gamma"], z["beta"], rr)])
        return Result("value", {"y": y})
    if profile == "G2":
        out = [V(rl.max_(R(a), R(0)) * rl.max_(R(a), R(0)) * b) for a, b in zip(z["a"], z["b"])]
        return Result("value", {"y": I.nest(out, z["shape"])})
    if profile == "G3":
        y = []
        for hr in z["h"]:
            t = [R(1) + R(h) / (R(1) + rl.abs_(R(h))) for h in hr]
            s = sum(t, R(0))
            y.append([V(ti / s) for ti in t])
        return Result("value", {"y": y})
    if profile == "G4":
        out = []
        for th, v, g in zip(z["theta"], z["v"], z["g"]):
            vn = R(z["mu"]) * v + g
            out.append(V(R(th) - R(z["lr"]) * (vn + R(z["wd"]) * th)))
        return Result("value", {"y": I.nest(out, z["shape"])})
    if profile == "G5":
        G, y = z["G"], []
        for pr, scr in zip(z["packed"], z["sc"]):
            qs = [qq for byte in pr for qq in _nibbles(byte)]
            y.append(V(sum((R(qk) * scr[k // G] * z["a"][k] for k, qk in enumerate(qs)), R(0))))
        return Result("value", {"y": y})
    if profile == "G6":
        y = []
        for ar, br in zip(z["a"], z["b"]):
            dot = sum((R(x) * yv for x, yv in zip(ar, br)), R(0))
            na, nb = rl.sqrt(sum((R(x) * x for x in ar), R(0))), rl.sqrt(sum((R(x) * x for x in br), R(0)))
            den = na * nb
            if den.sign() == 0:
                raise Undefined("zero norm")
            y.append(V(dot / den))
        return Result("value", {"y": y})
    if profile == "G7":
        y = []
        for qr, kr, vr, pos in zip(z["q"], z["k"], z["v"], z["pos"]):
            s = [rl.max_(sum((R(a) * b for a, b in zip(qr, kr[j])), R(0)) * z["scale"], R(0)) if j <= pos else R(0)
                 for j in range(z["L"])]
            den = sum(s, R(0)) + 1
            y.append([V(sum((s[j] * vr[j][d] for j in range(z["L"])), R(0)) / den) for d in range(z["Dv"])])
        return Result("value", {"y": y})
    y = []
    for xr in z["x"]:
        D, acc, row = len(xr), R(0), []
        for xv in xr:
            acc = acc + xv
            row.append(V(acc / D))
        y.append(row)
    return Result("value", {"y": y})


# ------------------------------------------------------------------------------------------------ mutants
def _indep(profile, inp, mis):
    z = _inputs(profile, inp)
    if profile == "G1":
        y = []
        for xr, rr in zip(z["x"], z["res"]):
            n = len(xr)
            mu = sum(xr) / n
            ss = sum((v - mu) ** 2 for v in xr)
            if mis == "dm1":
                rstd = 1 / iv.sqrt(to_iv(ss / (n - 1) + z["eps"]))
            elif mis == "eps_out":
                rstd = 1 / iv.sqrt(to_iv(ss / n)) + to_iv(z["eps"])
            else:
                rstd = 1 / iv.sqrt(to_iv(ss / n + z["eps"]))
            y.append([to_iv(v - mu) * rstd * to_iv(gm) + to_iv(bt + rv)
                      for v, gm, bt, rv in zip(xr, z["gamma"], z["beta"], rr)])
        return Result("value", {"y": y})
    if profile == "G2":
        a, b = (z["b"], z["a"]) if mis == "swap" else (z["a"], z["b"])
        out = [(abs(u) if mis == "abs" else max(u, Fraction(0))) ** 2 * w for u, w in zip(a, b)]
        return Result("value", {"y": I.nest(out, z["shape"])})
    if profile == "G3":
        y = []
        for hr in z["h"]:
            u = [h / (1 + abs(h)) for h in hr]
            den = sum(u) if mis == "noone" else sum(1 + e for e in u)
            if den == 0:
                raise Undefined("zero denominator")
            y.append([(1 + e) / den for e in u])
        return Result("value", {"y": y})
    if profile == "G4":
        lr, mu, wd = z["lr"], z["mu"], z["wd"]
        out = []
        for th, v, g in zip(z["theta"], z["v"], z["g"]):
            vn = mu * (v + g) if mis == "mom_after" else mu * v + g
            out.append(th - lr * vn - wd * th if mis == "decay_without_lr" else th - lr * (vn + wd * th))
        return Result("value", {"y": I.nest(out, z["shape"])})
    if profile == "G5":
        G, y = z["G"], []
        for pr, scr in zip(z["packed"], z["sc"]):
            acc = Fraction(0)
            for j, byte in enumerate(pr):
                lo, hi = byte % 16, (byte % 256) // 16
                if mis != "no_offset":
                    lo, hi = lo - 8, hi - 8
                if mis == "nibble_swap":
                    lo, hi = hi, lo
                acc += lo * scr[(2 * j) // G] * z["a"][2 * j] + hi * scr[(2 * j + 1) // G] * z["a"][2 * j + 1]
            y.append(acc)
        return Result("value", {"y": y})
    if profile == "G6":
        y = []
        for ar, br in zip(z["a"], z["b"]):
            dot = sum(x * w for x, w in zip(ar, br))
            saa, sbb = sum(x * x for x in ar), sum(w * w for w in br)
            if saa == 0 or sbb == 0:
                raise Undefined("zero norm")
            na = to_iv(saa) if mis == "nosqrt" else iv.sqrt(to_iv(saa))
            y.append(to_iv(dot) / (na * iv.sqrt(to_iv(sbb))))
        return Result("value", {"y": y})
    if profile == "G7":
        y = []
        for qr, kr, vr, pos in zip(z["q"], z["k"], z["v"], z["pos"]):
            s = []
            for j in range(z["L"]):
                inside = j < pos if mis == "strict_pos" else j <= pos
                d = sum(a * b for a, b in zip(qr, kr[j])) * z["scale"]
                s.append((d if mis == "no_relu" else max(d, Fraction(0))) if inside else Fraction(0))
            den = sum(s) + (0 if mis == "no_plus_one" else 1)
            if den == 0:
                raise Undefined("zero denominator")
            y.append([sum(s[j] * vr[j][d] for j in range(z["L"])) / den for d in range(z["Dv"])])
        return Result("value", {"y": y})
    y = []
    for xr in z["x"]:
        D = len(xr)
        if mis == "exclusive":
            y.append([sum(xr[:d], Fraction(0)) / D for d in range(D)])
        elif mis == "running_mean":
            y.append([sum(xr[:d + 1]) / (d + 1) for d in range(D)])
        else:
            y.append([sum(xr[:d + 1]) / D for d in range(D)])
    return Result("value", {"y": y})


def _m(mis):
    return lambda profile, inp: _indep(profile, inp, mis)


MUTANTS = {
    "BV2.M1": _m("dm1"),
    "BV2.M2": _m("eps_out"),
    "BV2.M3": _m("swap"),
    "BV2.M4": _m("abs"),
    "BV2.M5": _m("nibble_swap"),
    "BV2.M6": _m("noone"),
    "BV2.M7": _m("nosqrt"),
    "BV2.M8": _m("mom_after"),
    "BV2.M9": _m("decay_without_lr"),
    "BV2.M10": _m("no_offset"),
    "BV2.M11": _m("no_plus_one"),
    "BV2.M12": _m("strict_pos"),
    "BV2.M13": _m("no_relu"),
    "BV2.M14": _m("exclusive"),
    "BV2.M15": _m("running_mean"),
}
READINGS = {}
