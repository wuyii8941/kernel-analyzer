"""Adapter for contracts/BLIND_V1.yaml: the blind test v1 issuer's family specifications f (F1..F8, F2S).

One profile per family; input is a dict of the family's tensors (nested lists of exact rationals) and runtime scalars,
output y with the shape of the program's output. The reference is written on kernel_spec.real (exact rationals, rigorous
enclosures of (.)^(-1/2), exp and log). The mutants are an independent Fraction / mpmath-interval implementation of the
planted semantic variants of the issuer's answer key and of natural misreadings of the formulas.
"""
from __future__ import annotations

from fractions import Fraction

from mpmath import iv

from kernel_spec import f_issuer as I
from kernel_spec import real as rl
from kernel_spec.num import Invalid, Result, Undefined, to_iv
from kernel_spec.real import R

PROFILES = ("F1", "F2", "F2S", "F3", "F4", "F5", "F6", "F7", "F8")
KEYS = {"F1": ("x", "s", "b", "eps"), "F2": ("a", "b"), "F2S": ("a", "b", "t"), "F3": ("x", "eps"),
        "F4": ("x", "cos", "sin", "t"), "F5": ("q", "sc", "a", "G"), "F6": ("h", "m", "x"), "F7": ("p", "q"),
        "F8": ("pred", "target", "m", "delta", "nvalid")}


def _same(shapes, what):
    if any(s != shapes[0] for s in shapes):
        raise Invalid(f"{what}: operand shapes {shapes} differ")


def _int8(x, what):
    k = I.count(x, what)
    if not -128 <= k <= 127:
        raise Invalid(f"{what}: {k} outside the int8 range")
    return k


def _inputs(profile, inp):
    """Validated inputs per family (shared by the reference and the independent mutants)."""
    I.profile(profile, PROFILES)
    I.keys(inp, KEYS[profile], profile)
    p = profile
    if p == "F1":
        x, s, b = (I.mat(inp[k], k) for k in ("x", "s", "b"))
        _same([(len(z), len(z[0])) for z in (x, s, b)], "F1 x, s, b")
        return dict(x=x, s=s, b=b, eps=I.scalar(inp["eps"], "eps"))
    if p in ("F2", "F2S"):
        sa, a = I.tensor(inp["a"], "a")
        sb, b = I.tensor(inp["b"], "b")
        _same([sa, sb], "a, b")
        out = dict(shape=sa, a=a, b=b)
        if p == "F2S":
            out["t"] = I.scalar(inp["t"], "t")
        return out
    if p == "F3":
        return dict(x=I.mat(inp["x"], "x"), eps=I.scalar(inp["eps"], "eps"))
    if p == "F4":
        x = I.mat(inp["x"], "x")
        D = len(x[0])
        if D % 4:
            raise Invalid(f"F4: D = {D} is not a multiple of 4 (P = D/4)")
        P = D // 4
        c, sn = I.mat(inp["cos"], "cos"), I.mat(inp["sin"], "sin")
        _same([(len(x), P), (len(c), len(c[0])), (len(sn), len(sn[0]))], "F4 x1, cos, sin")
        return dict(x=x, c=c, s=sn, t=I.scalar(inp["t"], "t"), P=P)
    if p == "F5":
        q = I.mat(inp["q"], "q", item=_int8)
        K = len(q[0])
        a = I.vec(inp["a"], "a", K)
        G = I.count(inp["G"], "G", 1)
        sc = I.mat(inp["sc"], "sc")
        NG = -(-K // G)
        _same([(len(q), NG), (len(sc), len(sc[0]))], "F5 sc (R x ceil(K/G))")
        return dict(q=q, a=a, G=G, sc=sc)
    if p == "F6":
        h, m = I.mat(inp["h"], "h"), I.mat(inp["m"], "m")
        sx, xf = I.rank(inp["x"], "x", 3)
        _same([(len(h), len(h[0])), (len(m), len(m[0])), sx[:2]], "F6 h, m, x[:, :, 0]")
        Rr, L, D = sx
        x = [[xf[(r * L + l) * D:(r * L + l + 1) * D] for l in range(L)] for r in range(Rr)]
        return dict(h=h, m=m, x=x, D=D)
    if p == "F7":
        P_, Q = I.mat(inp["p"], "p"), I.mat(inp["q"], "q")
        _same([(len(P_), len(P_[0])), (len(Q), len(Q[0]))], "F7 p, q")
        return dict(p=P_, q=Q)
    sp, pr = I.tensor(inp["pred"], "pred")
    st, tg = I.tensor(inp["target"], "target")
    sm, m = I.tensor(inp["m"], "m")
    _same([sp, st, sm], "F8 pred, target, m")
    return dict(shape=sp, pred=pr, target=tg, m=m, delta=I.scalar(inp["delta"], "delta"),
                nvalid=I.count(inp["nvalid"], "nvalid", 0))


# ------------------------------------------------------------------------------------------------ reference
def _gate(a):
    return R(Fraction(1, 2)) * (R(1) + R(a) / (R(1) + rl.abs_(R(a))))


def _mean(row):
    if not row:
        raise Undefined("mean over an empty row")
    return sum((R(v) for v in row), R(0)) / len(row)


def _rsqrt_pos(q):
    if q.sign() <= 0:
        raise Undefined("(.)^(-1/2) of a non-positive number")
    return R(1) / rl.sqrt(q)


def reference(profile, inp):
    z = _inputs(profile, inp)
    V = I.v
    if profile == "F1":
        y = []
        for xr, sr, br in zip(z["x"], z["s"], z["b"]):
            ms = _mean([v * v for v in xr])
            r = _rsqrt_pos(ms + z["eps"])
            y.append([V(R(xv) * r * (R(1) + sv) + bv) for xv, sv, bv in zip(xr, sr, br)])
        return Result("value", {"y": y})
    if profile in ("F2", "F2S"):
        t = R(z["t"]) if profile == "F2S" else None
        out = []
        for a, b in zip(z["a"], z["b"]):
            y = (R(a) * _gate(a)) * b
            out.append(V(y * t if t is not None else y))
        return Result("value", {"y": I.nest(out, z["shape"])})
    if profile == "F3":
        y = []
        for xr in z["x"]:
            mu = _mean(xr)
            var = _mean([V((R(v) - mu) * (R(v) - mu)) for v in xr])
            y.append([V(mu), V(_rsqrt_pos(var + z["eps"]))])
        return Result("value", {"y": y})
    if profile == "F4":
        P, t = z["P"], R(z["t"])
        y = []
        for xr, cr, sr in zip(z["x"], z["c"], z["s"]):
            x1, x2, xp = xr[:P], xr[P:2 * P], xr[2 * P:]
            y1 = [V(t * (R(a) * c - R(b) * s)) for a, b, c, s in zip(x1, x2, cr, sr)]
            y2 = [V(t * (R(a) * s + R(b) * c)) for a, b, c, s in zip(x1, x2, cr, sr)]
            y.append(y1 + y2 + list(xp))
        return Result("value", {"y": y})
    if profile == "F5":
        G = z["G"]
        y = [V(sum((R(qv) * scr[k // G] * z["a"][k] for k, qv in enumerate(qr)), R(0))) for qr, scr in zip(z["q"], z["sc"])]
        return Result("value", {"y": y})
    if profile == "F6":
        y = []
        for hr, mr, xr in zip(z["h"], z["m"], z["x"]):
            w = [R(mv) * _gate(hv) for hv, mv in zip(hr, mr)]
            den = sum(w, R(0))
            if den.sign() == 0:
                raise Undefined("sum_l w[r,l] = 0")
            y.append([V(sum((w[l] * xr[l][d] for l in range(len(w))), R(0)) / den) for d in range(z["D"])])
        return Result("value", {"y": y})
    if profile == "F7":
        y = []
        for pr, qr in zip(z["p"], z["q"]):
            if not pr:
                raise Undefined("softmax over an empty row")
            lsp = rl.log(sum((rl.exp(R(v)) for v in pr), R(0)))
            lsq = rl.log(sum((rl.exp(R(v)) for v in qr), R(0)))
            lp = [R(v) - lsp for v in pr]
            lq = [R(v) - lsq for v in qr]
            y.append(V(sum((rl.exp(a) * (a - b) for a, b in zip(lp, lq)), R(0))))
        return Result("value", {"y": y})
    # F8
    if z["nvalid"] == 0:
        raise Undefined("nvalid = 0")
    out = []
    for p_, t_, m_ in zip(z["pred"], z["target"], z["m"]):
        d = p_ - t_
        g = d if abs(d) <= z["delta"] else z["delta"] * (1 if d >= 0 else -1)   # sign(0) = +1 (issuer section 2)
        out.append(g * m_ / z["nvalid"])
    return Result("value", {"y": I.nest(out, z["shape"])})


# ------------------------------------------------------------------------------------------------ mutants
# Independent implementation (Fractions; mpmath intervals for sqrt / exp / log), one misreading at a time.
def _q(x):
    return Fraction(x)


def _gq(a):
    return Fraction(1, 2) * (1 + a / (1 + abs(a)))


def _isqrt_inv(q):
    if q <= 0:
        raise Undefined("non-positive")
    return 1 / iv.sqrt(to_iv(q))


def _indep(profile, inp, mis):
    z = _inputs(profile, inp)
    if profile == "F1":
        y = []
        for xr, sr, br in zip(z["x"], z["s"], z["b"]):
            if not xr:
                raise Undefined("empty")
            ms = sum(_q(v) ** 2 for v in xr) / len(xr)
            r = _isqrt_inv(ms + (0 if mis == "noeps" else z["eps"]))
            y.append([to_iv(xv) * r * to_iv(sv if mis == "s_without_one" else 1 + sv) + to_iv(bv)
                      for xv, sv, bv in zip(xr, sr, br)])
        return Result("value", {"y": y})
    if profile in ("F2", "F2S"):
        a, b = (z["b"], z["a"]) if mis == "swap" else (z["a"], z["b"])
        t = z.get("t", 1) if mis != "no_t" else 1
        return Result("value", {"y": I.nest([u * _gq(u) * w * t for u, w in zip(a, b)], z["shape"])})
    if profile == "F3":
        y = []
        for xr in z["x"]:
            n = len(xr)
            if n == 0:
                raise Undefined("empty")
            mu = sum(map(_q, xr)) / n
            ss = sum((v - mu) ** 2 for v in xr)
            if mis == "dm1":
                if n < 2:
                    raise Undefined("D - 1 = 0")
                y.append([mu, _isqrt_inv(ss / (n - 1) + z["eps"])])
            elif mis == "eps_outside":
                y.append([mu, _isqrt_inv(ss / n) + to_iv(z["eps"])])
            else:
                y.append([mu, _isqrt_inv(ss / n + z["eps"])])
        return Result("value", {"y": y})
    if profile == "F4":
        P, t = z["P"], z["t"]
        y = []
        for xr, cr, sr in zip(z["x"], z["c"], z["s"]):
            row = [None] * len(xr)
            for i in range(P):
                x1, x2, c, s = xr[i], xr[P + i], cr[i], sr[i]
                row[i] = t * (x1 * c - x2 * s)
                row[P + i] = t * ((x2 * c - x1 * s) if mis == "y2_sign" else (x1 * s + x2 * c))
            for j in range(2 * P, len(xr)):
                row[j] = t * xr[j] if mis == "t_on_passthrough" else xr[j]
            y.append(row)
        return Result("value", {"y": y})
    if profile == "F5":
        G, out = z["G"], []
        for qr, scr in zip(z["q"], z["sc"]):
            NG = len(scr)
            acc = Fraction(0)
            for k, qv in enumerate(qr):
                gi = min((k + 1) // G, NG - 1) if mis == "group_off_by_one" else k // G
                acc += qv * scr[gi] * z["a"][k]
            out.append(acc)
        return Result("value", {"y": out})
    if profile == "F6":
        y = []
        for hr, mr, xr in zip(z["h"], z["m"], z["x"]):
            g = [_gq(hv) for hv in hr]
            w = [mv * gv for mv, gv in zip(mr, g)]
            den = sum(g) if mis == "unmasked_den" else sum(w)
            if den == 0:
                raise Undefined("zero denominator")
            y.append([sum(w[l] * xr[l][d] for l in range(len(w))) / den for d in range(z["D"])])
        return Result("value", {"y": y})
    if profile == "F7":
        y = []
        for pr, qr in zip(z["p"], z["q"]):
            ep = [iv.exp(to_iv(v)) for v in pr]
            eq = [iv.exp(to_iv(v)) for v in qr]
            sp_, sq_ = sum(ep[1:], ep[0]), sum(eq[1:], eq[0])
            P_ = [e / sp_ for e in ep]
            Q_ = [e / sq_ for e in eq]
            if mis == "reversed":
                P_, Q_ = Q_, P_
            terms = [a * (iv.log(a) - iv.log(b)) for a, b in zip(P_, Q_)]
            y.append(sum(terms[1:], terms[0]))
        return Result("value", {"y": y})
    if z["nvalid"] == 0:
        raise Undefined("nvalid = 0")
    out = []
    for p_, t_, m_ in zip(z["pred"], z["target"], z["m"]):
        d = p_ - t_
        if abs(d) <= z["delta"]:
            g = d
        else:
            sg = (d > 0) - (d < 0) if mis == "sign0_zero" else (1 if d >= 0 else -1)
            g = z["delta"] * sg
        out.append(g / z["nvalid"] if mis == "mask_ignored" else g * m_ / z["nvalid"])
    return Result("value", {"y": I.nest(out, z["shape"])})


def _m(mis):
    return lambda profile, inp: _indep(profile, inp, mis)


MUTANTS = {
    "BV1.M1": _m("noeps"),
    "BV1.M2": _m("swap"),
    "BV1.M3": _m("dm1"),
    "BV1.M4": _m("y2_sign"),
    "BV1.M5": _m("group_off_by_one"),
    "BV1.M6": _m("unmasked_den"),
    "BV1.M7": _m("no_t"),
    "BV1.M8": _m("s_without_one"),
    "BV1.M9": _m("eps_outside"),
    "BV1.M10": _m("t_on_passthrough"),
    "BV1.M11": _m("reversed"),
    "BV1.M12": _m("mask_ignored"),
    "BV1.M13": _m("sign0_zero"),
}
READINGS = {}
