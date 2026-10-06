import json, os, sys, time, math
import numpy as np
from decimal import Decimal, getcontext, ROUND_HALF_EVEN

getcontext().prec = 50
ROOT = "/home/claude/v2ver/blind_test_v2_verification_20261004"
KEY = json.load(open("/home/claude/blind/blind_test_v2/answer_key_SEALED.json"))["programs"]
P1 = json.load(open("/home/claude/v2sub/blind_test_v2_phase1_submission_20261004/seeds_0_95/phase1_report.json"))
P3 = json.load(open("/home/claude/v2p3/blind_test_v2_phase3_submission_20261004/phase3_report.json"))

D = lambda v: Decimal(float(v))
ONE = Decimal(1)


def rsqrt(v):
    return ONE / v.sqrt()


# ------------------------------------------------------------------ declared semantics / spec per family and variant
def g1(inp, sc, variant):
    x = inp["x"].astype(np.float64); res = inp["res"].astype(np.float64)
    gam = [D(v) for v in inp["gamma"]]; bet = [D(v) for v in inp["beta"]]
    eps = D(sc["eps"])
    R, Dn = x.shape
    out = [None] * (R * Dn)
    for r in range(R):
        xs = [D(v) for v in x[r]]
        mean = sum(xs, Decimal(0)) / Dn
        ds = [v - mean for v in xs]
        ss = sum((d * d for d in ds), Decimal(0))
        if variant == "S-dm1":
            var = ss / (Dn - 1); rs = rsqrt(var + eps)
        elif variant == "S-eps-out":
            var = ss / Dn; rs = rsqrt(var) + eps
        else:
            var = ss / Dn; rs = rsqrt(var + eps)
        for d in range(Dn):
            out[r * Dn + d] = ds[d] * rs * gam[d] + bet[d] + D(res[r, d])
    return out


def g2(inp, sc, variant):
    a = inp["a"].astype(np.float64); b = inp["b"].astype(np.float64)
    if variant == "S-swap":
        a, b = b, a
    av = a.reshape(-1); bv = b.reshape(-1)
    out = [None] * (av.size)
    for i in range(av.size):
        x = D(av[i])
        r = abs(x) if variant == "S-abs" else (x if x > 0 else Decimal(0))
        out[i] = r * r * D(bv[i])
    return out


def g3(inp, sc, variant):
    h = inp["h"].astype(np.float64)
    R, Dn = h.shape
    out = [None] * (R * Dn)
    for r in range(R):
        hs = [D(v) for v in h[r]]
        q = [v / (ONE + abs(v)) for v in hs]
        t = [ONE + v for v in q]
        den = sum(q, Decimal(0)) if variant == "S-noone" else sum(t, Decimal(0))
        for d in range(Dn):
            out[r * Dn + d] = t[d] / den
    return out


def g4(inp, sc, variant):
    th = inp["theta"].astype(np.float64).reshape(-1); v = inp["v"].astype(np.float64).reshape(-1); g = inp["g"].astype(np.float64).reshape(-1)
    lr = D(sc["lr"]); mu = D(sc["mu"]); wd = D(sc["wd"])
    out = [None] * (th.size)
    for i in range(th.size):
        T = D(th[i]); V = D(v[i]); G = D(g[i])
        vn = mu * (V + G) if variant == "S-mom-after" else mu * V + G
        out[i] = T - lr * (vn + wd * T)
    return out


def g5(inp, sc, variant):
    pk = inp["packed"].astype(np.int64); scl = inp["sc"].astype(np.float64); a = inp["a"].astype(np.float64)
    R, KH = pk.shape; K = 2 * KH; NG = scl.shape[1]; G = K // NG
    av = [D(x) for x in a]
    out = [None] * (R)
    for r in range(R):
        s = [D(x) for x in scl[r]]
        acc = Decimal(0)
        for j in range(KH):
            byte = int(pk[r, j])
            lo = (byte & 15) - 8; hi = ((byte >> 4) & 15) - 8
            if variant == "S-nibble-swap":
                lo, hi = hi, lo
            acc += Decimal(lo) * s[(2 * j) // G] * av[2 * j] + Decimal(hi) * s[(2 * j + 1) // G] * av[2 * j + 1]
        out[r] = acc
    return out


def g6(inp, sc, variant):
    a = inp["a"].astype(np.float64); b = inp["b"].astype(np.float64)
    R, Dn = a.shape
    out = [None] * (R)
    for r in range(R):
        A = [D(v) for v in a[r]]; B = [D(v) for v in b[r]]
        dot = sum((x * y for x, y in zip(A, B)), Decimal(0))
        saa = sum((x * x for x in A), Decimal(0)); sbb = sum((y * y for y in B), Decimal(0))
        na = saa if variant == "S-nosqrt" else saa.sqrt()
        nb = sbb.sqrt()
        out[r] = dot / (na * nb)
    return out


def g7(inp, sc, variant):
    q = inp["q"].astype(np.float64); k = inp["k"].astype(np.float64); v = inp["v"].astype(np.float64); pos = inp["pos"].astype(np.int64)
    scale = D(sc["scale"])
    R, L, Dn = k.shape
    out = [None] * (R * Dn)
    for r in range(R):
        Q = [D(x) for x in q[r]]
        s = []
        for j in range(L):
            if j <= pos[r]:
                val = sum((Q[d] * D(k[r, j, d]) for d in range(Dn)), Decimal(0)) * scale
                s.append(val if val > 0 else Decimal(0))
            else:
                s.append(Decimal(0))
        den = sum(s, Decimal(0)) + ONE
        for d in range(Dn):
            num = sum((s[j] * D(v[r, j, d]) for j in range(L) if s[j] != 0), Decimal(0))
            out[r * Dn + d] = num / den
    return out


def g8(inp, sc, variant):
    x = inp["x"].astype(np.float64)
    R, Dn = x.shape
    out = [None] * (R * Dn)
    for r in range(R):
        c = Decimal(0)
        for d in range(Dn):
            c += D(x[r, d]); out[r * Dn + d] = c / Dn
    return out


FAM = {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5, "G6": g6, "G7": g7, "G8": g8}
SEMANTIC_VARIANTS = {"S-dm1", "S-eps-out", "S-abs", "S-swap", "S-noone", "S-mom-after", "S-nibble-swap", "S-nosqrt"}

# ------------------------------------------------------------------ optimizer steps (real semantics) in Decimal
LR = Decimal(2) ** -10; EPS = Decimal(2) ** -27
B1 = Decimal("0.9"); B2 = Decimal("0.999")


def step_sgd(g):
    return -LR * g


def make_adam(m, v, t):
    bc1 = ONE - B1 ** t; bc2 = ONE - B2 ** t
    sb2 = bc2.sqrt()
    def f(g, mi, vi):
        mp = B1 * mi + (ONE - B1) * g
        vp = B2 * vi + (ONE - B2) * g * g
        return -LR / bc1 * mp / (vp.sqrt() / sb2 + EPS)
    return f


def run(progs):
    cache = {}
    report = {}
    for prog in progs:
        k = KEY[prog]; fam = k["family"]; variant = k["variant"]
        sem = variant if variant in SEMANTIC_VARIANTS else "clean"
        for seed in ("seed000", "seed032"):
            fdir = os.path.join(ROOT, "family", fam, seed)
            pdir = os.path.join(ROOT, "programs", prog, seed)
            t0 = time.time()
            ck = (fam, seed, sem)
            if ck not in cache:
                inp = np.load(os.path.join(fdir, "inputs.npz")); sc = json.load(open(os.path.join(fdir, "scalars.json")))
                krD = FAM[fam](inp, sc, sem)
                if sem == "clean":
                    fD = krD
                elif (fam, seed, "clean") in cache:
                    fD = cache[(fam, seed, "clean")][0]
                else:
                    fD = FAM[fam](inp, sc, "clean")
                    cache[(fam, seed, "clean")] = (fD, fD)
                cache[ck] = (krD, fD)
            krD, fD = cache[ck]
            kr = np.array([float(x) for x in krD]); fval = np.array([float(x) for x in fD])
            K = np.load(os.path.join(pdir, "K.npy")).astype(np.float64)
            klo = np.load(os.path.join(pdir, "KR_lo.npy")); khi = np.load(os.path.join(pdir, "KR_hi.npy"))
            flo = np.load(os.path.join(fdir, "f_lo.npy")); fhi = np.load(os.path.join(fdir, "f_hi.npy"))
            n = K.size
            # relative tolerance for my float() conversion of a 50-digit value: half ulp of double
            tol = np.abs(kr) * 1.2e-16 + 1e-300
            kr_viol = int(np.sum((kr < klo - tol) | (kr > khi + tol)))
            tolf = np.abs(fval) * 1.2e-16 + 1e-300
            f_viol = int(np.sum((fval < flo - tolf) | (fval > fhi + tolf)))
            entry = {"family": fam, "n": int(n), "KR_violations": kr_viol, "f_violations": f_viol,
                     "KR_tool_max_width": float(np.max(khi - klo)), "f_tool_max_width": float(np.max(fhi - flo)), "u": {}}
            # u for the three optimizers, computed in Decimal; compared to the tool interval
            st = np.load(os.path.join(fdir, "adamw_history_state.npz"))
            m = st["exp_avg"].astype(np.float64); v = st["exp_avg_sq"].astype(np.float64)
            adam_h = make_adam(None, None, 17); adam_z = make_adam(None, None, 1)
            for opt in ("sgd", "adamw_history", "adamw_zero"):
                ulo = np.load(os.path.join(pdir, f"u_{opt}_lo.npy")); uhi = np.load(os.path.join(pdir, f"u_{opt}_hi.npy"))
                u = np.empty(n); r = np.empty(n)
                for i in range(n):
                    gK = D(K[i]); gR = krD[i]
                    if opt == "sgd":
                        sK = step_sgd(gK); sR = step_sgd(gR)
                    elif opt == "adamw_history":
                        mi = D(m[i]); vi = D(v[i]); sK = adam_h(gK, mi, vi); sR = adam_h(gR, mi, vi)
                    else:
                        sK = adam_z(gK, Decimal(0), Decimal(0)); sR = adam_z(gR, Decimal(0), Decimal(0))
                    u[i] = float(sK - sR); r[i] = float(sR)
                utol = np.abs(u) * 1.2e-16 + 1e-300
                viol = int(np.sum((u < ulo - utol) | (u > uhi + utol)))
                entry["u"][opt] = {"violations": viol, "tool_max_width": float(np.max(uhi - ulo)), "r": None}
                entry["u"][opt]["_u"] = u; entry["u"][opt]["_r"] = r
            entry["_e"] = np.array([float(D(K[i]) - krD[i]) for i in range(n)]); entry["_kr"] = kr
            entry["seconds"] = round(time.time() - t0, 1)
            report[(prog, seed)] = entry
            print(prog, seed, fam, variant, f"{entry['seconds']}s", "KRviol", kr_viol, "fviol", f_viol,
                  "u viol", {o: entry['u'][o]['violations'] for o in entry['u']}, flush=True)
    return report


def projections(e, ref, family):
    n = e.size
    res = {"R1": math.fsum(e) * (-1.0 / math.sqrt(n)),
           "R2": math.fsum(e * (-np.sign(ref))) / math.sqrt(n)}
    nr = math.sqrt(math.fsum(ref * ref))
    res["R3"] = -math.fsum(e * ref) / nr if nr > 0 else float("nan")
    return res


def check_projections(report):
    out = []
    for (prog, seed), e in report.items():
        if seed != "seed032":
            continue
        # phase 1 output layer: first confirmation unit = seed 32
        rec = P1[prog]["outputs"][0]["record"]["rules"]
        mine = projections(e["_e"], e["_kr"], e["family"])
        for rr in rec:
            if rr["rule"] in mine and rr.get("per_unit_bounds"):
                lo, hi = rr["per_unit_bounds"][0]
                v = mine[rr["rule"]]; tol = max(abs(lo), abs(hi)) * 1e-9 + 1e-300
                out.append(("phase1", prog, rr["rule"], v, lo, hi, lo - tol <= v <= hi + tol))
        for opt in ("sgd", "adamw_history", "adamw_zero"):
            u = e["u"][opt]["_u"]; r = e["u"][opt]["_r"]
            mine = projections(u, r, e["family"])
            rec3 = P3[prog][opt]["ideal_response"]["seeds_0_95"]["record"]["rules"]
            for rr in rec3:
                if rr["rule"] in mine and rr.get("per_unit_bounds"):
                    lo, hi = rr["per_unit_bounds"][0]
                    v = mine[rr["rule"]]; tol = max(abs(lo), abs(hi)) * 1e-6 + 1e-300
                    out.append((opt, prog, rr["rule"], v, lo, hi, lo - tol <= v <= hi + tol))
    return out


if __name__ == "__main__":
    progs = sys.argv[1:] or sorted(KEY)
    rep = run(progs)
    pr = check_projections(rep)
    bad = [p for p in pr if not p[6]]
    print("projection checks", len(pr), "outside", len(bad))
    for b in bad[:20]:
        print("  OUTSIDE", b)
    summ = {f"{p}/{s}": {k: v for k, v in e.items() if not k.startswith("_")} for (p, s), e in rep.items()}
    for kk in summ:
        for o in summ[kk]["u"]:
            summ[kk]["u"][o] = {a: b for a, b in summ[kk]["u"][o].items() if not a.startswith("_")}
    json.dump({"entries": summ, "projection_checks": [list(map(lambda z: z if not isinstance(z, np.bool_) else bool(z), p)) for p in pr]},
              open(os.environ.get("OUTJSON", "/home/claude/v2ver/report_tmp.json"), "w"), indent=1, default=float)
