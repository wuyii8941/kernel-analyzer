import json, os, sys, time
import numpy as np
from fractions import Fraction
from decimal import Decimal, getcontext, ROUND_FLOOR, ROUND_CEILING, ROUND_HALF_EVEN

getcontext().prec = 60
ROOT = "/home/claude/verif/blind_v1_verification"


def D(x):
    return Decimal(float(x))  # exact for any float32/float64 value


def f32(x):
    return float(np.float32(x))


def sqrt_interval(v: Decimal):
    """[lo, hi] enclosing sqrt(v) using directed rounding at the context precision."""
    c = getcontext()
    c.rounding = ROUND_FLOOR
    lo = v.sqrt()
    c.rounding = ROUND_CEILING
    hi = v.sqrt()
    c.rounding = ROUND_HALF_EVEN
    return lo, hi


def recip_interval(lo, hi):
    """1/[lo,hi] for lo>0."""
    c = getcontext()
    c.rounding = ROUND_FLOOR
    rlo = Decimal(1) / hi
    c.rounding = ROUND_CEILING
    rhi = Decimal(1) / lo
    c.rounding = ROUND_HALF_EVEN
    return rlo, rhi


def mul_interval(a: Decimal, lo, hi):
    """a * [lo,hi] for a exact Decimal of either sign (directed rounding)."""
    c = getcontext()
    if a >= 0:
        c.rounding = ROUND_FLOOR; l = a * lo
        c.rounding = ROUND_CEILING; h = a * hi
    else:
        c.rounding = ROUND_FLOOR; l = a * hi
        c.rounding = ROUND_CEILING; h = a * lo
    c.rounding = ROUND_HALF_EVEN
    return l, h


# ----------------------------------------------------------------------------- family evaluators
# each returns (lo, hi) as float64 numpy arrays over the row-major flattened output, plus optional spec arrays

def eval_F1(inp, sc, spec=False):
    x = inp["x"].astype(np.float64); s = inp["s"].astype(np.float64); b = inp["b"].astype(np.float64)
    eps = D(f32(sc["eps"]))
    R, Dn = x.shape
    lo = np.empty(R * Dn); hi = np.empty(R * Dn)
    for r in range(R):
        xs = [D(v) for v in x[r]]
        ms = sum((v * v for v in xs), Decimal(0)) / Dn   # exact (dyadic) at prec 60: products exact up to ~48 bits, sums exact
        slo, shi = sqrt_interval(ms + eps) if not spec else sqrt_interval(ms + eps)
        rlo, rhi = recip_interval(slo, shi)
        for d in range(Dn):
            a = xs[d] * (Decimal(1) + D(s[r, d]))
            l, h = mul_interval(a, rlo, rhi)
            bb = D(b[r, d])
            lo[r * Dn + d] = float(l + bb); hi[r * Dn + d] = float(h + bb)
    return lo, hi


def eval_F1_noeps(inp, sc):
    # declared semantics of a program without eps (used only for the semantic check)
    sc2 = dict(sc); sc2["eps"] = 0.0
    return eval_F1(inp, sc2)


def eval_F2(inp, sc, scale=None):
    a = inp["a"].astype(np.float64); b = inp["b"].astype(np.float64)
    R, Dn = a.shape
    out = np.empty(R * Dn)
    t = D(f32(scale)) if scale is not None else None
    for r in range(R):
        for d in range(Dn):
            av = D(a[r, d]); bv = D(b[r, d])
            g = Decimal("0.5") * (Decimal(1) + av / (Decimal(1) + abs(av)))
            y = (av * g) * bv
            if t is not None:
                y = y * t
            out[r * Dn + d] = float(y)
    return out, out


def eval_F3(inp, sc, exact_spec=False, dm1=False):
    x = inp["x"].astype(np.float64)
    eps = D(f32(sc["eps"]))
    R, Dn = x.shape
    CH = Dn // 8
    lo = np.empty(R * 2); hi = np.empty(R * 2)
    for r in range(R):
        xs = [D(v) for v in x[r]]
        if exact_spec:
            mean = sum(xs, Decimal(0)) / Dn
            m2 = sum(((v - mean) * (v - mean) for v in xs), Decimal(0))
        else:
            ch = xs[0:CH]
            mean = sum(ch, Decimal(0)) / CH
            m2 = sum(((v - mean) * (v - mean) for v in ch), Decimal(0))
            for k in range(CH, Dn, CH):
                chb = xs[k:k + CH]
                mb = sum(chb, Decimal(0)) / CH
                m2b = sum(((v - mb) * (v - mb) for v in chb), Decimal(0))
                delta = mb - mean
                c1 = D(f32(CH / (k + CH)))        # compile-time f32 constant as declared in TTIR
                c2 = D(f32(k * CH / (k + CH)))
                mean = mean + delta * c1
                m2 = m2 + m2b + delta * delta * c2
        var = m2 / (Dn - 1 if dm1 else Dn)
        slo, shi = sqrt_interval(var + eps)
        rlo, rhi = recip_interval(slo, shi)
        lo[2 * r] = hi[2 * r] = float(mean)
        lo[2 * r + 1] = float(rlo); hi[2 * r + 1] = float(rhi)
    return lo, hi


def eval_F4(inp, sc, sign_bug=False):
    x = inp["x"].astype(np.float64); c = inp["cos"].astype(np.float64); s = inp["sin"].astype(np.float64)
    t = D(f32(sc["t"]))
    R, Dn = x.shape
    P = Dn // 4; HALF = Dn // 2
    out = np.empty(R * Dn)
    for r in range(R):
        for i in range(P):
            x1 = D(x[r, i]); x2 = D(x[r, P + i]); cc = D(c[r, i]); ss = D(s[r, i])
            y1 = (x1 * cc - x2 * ss) * t
            y2 = ((x2 * cc - x1 * ss) if sign_bug else (x1 * ss + x2 * cc)) * t
            out[r * Dn + i] = float(y1); out[r * Dn + P + i] = float(y2)
        for j in range(HALF):
            out[r * Dn + HALF + j] = float(x[r, HALF + j])
    return out, out


def eval_F5(inp, sc, group_bug=False):
    q = inp["q"].astype(np.int64); scl = inp["sc"].astype(np.float64); a = inp["a"].astype(np.float64)
    R, K = q.shape
    NG = scl.shape[1]; G = K // NG
    out = np.empty(R)
    af = [Fraction(float(v)) for v in a]
    for r in range(R):
        scf = [Fraction(float(v)) for v in scl[r]]
        acc = Fraction(0)
        for k in range(K):
            gi = min((k + 1) // G, NG - 1) if group_bug else k // G
            acc += Fraction(int(q[r, k])) * scf[gi] * af[k]
        out[r] = float(acc)
    return out, out


def eval_F6(inp, sc, unmasked_den=False):
    x = inp["x"].astype(np.float64); h = inp["h"].astype(np.float64); m = inp["m"].astype(np.float64)
    R, L, Dn = x.shape
    out = np.empty(R * Dn)
    for r in range(R):
        g = [Decimal("0.5") * (Decimal(1) + D(hv) / (Decimal(1) + abs(D(hv)))) for hv in h[r]]
        w = [D(m[r, l]) * g[l] for l in range(L)]
        den = sum(g, Decimal(0)) if unmasked_den else sum(w, Decimal(0))
        for d in range(Dn):
            num = Decimal(0)
            col = x[r, :, d]
            for l in range(L):
                if m[r, l] != 0.0:
                    num += w[l] * D(col[l])
            out[r * Dn + d] = float(num / den)
    return out, out


def eval_F7(inp, sc):
    p = inp["p"].astype(np.float64); q = inp["q"].astype(np.float64)
    R, V = p.shape
    out = np.empty(R)
    for r in range(R):
        pr = [D(v) for v in p[r]]; qr = [D(v) for v in q[r]]
        mp = max(pr); mq = max(qr)
        ep = [(v - mp).exp() for v in pr]; eq = [(v - mq).exp() for v in qr]
        lsep = sum(ep, Decimal(0)).ln(); lseq = sum(eq, Decimal(0)).ln()
        kl = Decimal(0)
        for i in range(V):
            lp = pr[i] - mp - lsep; lq = qr[i] - mq - lseq
            kl += lp.exp() * (lp - lq)
        out[r] = float(kl)
    return out, out


FAMILY_EVAL = {
    "F1": lambda inp, sc: eval_F1(inp, sc),
    "F2": lambda inp, sc: eval_F2(inp, sc),
    "F2S": lambda inp, sc: eval_F2(inp, sc, scale=sc["t"]),
    "F3": lambda inp, sc: eval_F3(inp, sc),
    "F4": lambda inp, sc: eval_F4(inp, sc),
    "F5": lambda inp, sc: eval_F5(inp, sc),
    "F6": lambda inp, sc: eval_F6(inp, sc),
    "F7": lambda inp, sc: eval_F7(inp, sc),
}

# programs whose declared semantics differ from the family clean formula (from the sealed key; used only to pick the evaluator)
PROGRAM_SEMANTICS = {
    "prog_27": lambda inp, sc: eval_F3(inp, sc, dm1=True),       # var / (D-1)
    "prog_18": lambda inp, sc: eval_F1_noeps(inp, sc),           # eps omitted (not in package but kept for completeness)
}


def projections(e, kr, family, r5=None):
    n = e.size
    res = {}
    res["R1"] = float(np.sum(e) * (-1.0 / np.sqrt(n)))
    res["R2"] = float(np.sum(e * (-np.sign(kr))) / np.sqrt(n))
    nrm = np.linalg.norm(kr)
    res["R3"] = float(-np.sum(e * kr) / nrm) if nrm > 0 else float("nan")
    if family == "F4":
        Dn = 256; R = n // Dn
        w = np.zeros(Dn); w[:64] = -1.0; w[64:128] = 1.0
        w = np.tile(w, R); w = w / np.linalg.norm(w)
        res["R4"] = float(np.sum(e * w))
    if r5 is not None:
        r5 = r5.reshape(-1).astype(np.float64)
        res["R5"] = float(np.sum(e * r5) / np.linalg.norm(r5))
    return res


def main(progs):
    report = {}
    for prog in progs:
        pdir = os.path.join(ROOT, prog)
        for seed_dir in sorted(d for d in os.listdir(pdir) if d.startswith("seed")):
            d = os.path.join(pdir, seed_dir)
            tv = json.load(open(os.path.join(d, "tool_values.json")))
            fam = tv["family"]
            inp = np.load(os.path.join(d, "inputs.npz"))
            sc = json.load(open(os.path.join(d, "scalars.json")))
            K = np.load(os.path.join(d, "K.npy")).astype(np.float64).reshape(-1)
            tlo = np.load(os.path.join(d, "KR_lo.npy")); thi = np.load(os.path.join(d, "KR_hi.npy"))
            idx = np.load(os.path.join(d, "output_flat_index.npy"))
            assert np.array_equal(idx, np.arange(idx.size)), "non-trivial output index"
            t0 = time.time()
            ev = PROGRAM_SEMANTICS.get(prog, FAMILY_EVAL[fam])
            mlo, mhi = ev(inp, sc)
            dt = time.time() - t0
            mid = 0.5 * (mlo + mhi)
            # containment: tool interval must contain the true value (my interval is tight; use its midpoint, widened by my width)
            tol = np.maximum(mhi - mlo, 1e-300)
            viol = np.sum((mid < tlo - tol) | (mid > thi + tol))
            inside_strict = np.sum((mlo >= tlo) & (mhi <= thi))
            e = K - mid
            entry = {"family": fam, "n": int(K.size), "seconds": round(dt, 1),
                     "tool_interval_violations": int(viol), "my_interval_inside_tool": int(inside_strict),
                     "my_max_width": float(np.max(mhi - mlo)), "tool_max_width": float(np.max(thi - tlo)),
                     "residual_mean_mine": float(np.mean(e)), "residual_mean_tool_mid": tv["residual"]["mean_mid"],
                     "projections": {}}
            if fam == "F3":
                for col, name in ((0, "mean"), (1, "rstd")):
                    sel = slice(col, None, 2)
                    r5p = os.path.join(pdir, f"R5_direction_{name}.npy")
                    r5 = np.load(r5p) if os.path.exists(r5p) else None
                    mine = projections(e[sel], mid[sel], fam, r5)
                    entry["projections"][name] = {k: {"mine": v, "tool": tv["projections"][name].get(k)} for k, v in mine.items()}
            else:
                r5p = os.path.join(pdir, "R5_direction_output.npy")
                r5 = np.load(r5p) if os.path.exists(r5p) else None
                mine = projections(e, mid, fam, r5)
                entry["projections"]["output"] = {k: {"mine": v, "tool": tv["projections"]["output"].get(k)} for k, v in mine.items()}
            report[f"{prog}/{seed_dir}"] = entry
            print(prog, seed_dir, fam, f"{dt:.1f}s", "viol", viol, "inside", inside_strict, "/", K.size,
                  "mywidth", f"{entry['my_max_width']:.2e}", "toolwidth", f"{entry['tool_max_width']:.2e}", flush=True)
            for grp, dd in entry["projections"].items():
                for rule, v in dd.items():
                    print(f"   {grp} {rule}: mine={v['mine']:.6e} tool={v['tool']}")
    json.dump(report, open("/home/claude/verif/verification_report.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
