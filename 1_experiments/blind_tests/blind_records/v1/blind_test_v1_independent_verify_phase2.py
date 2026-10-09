import json, os, sys, time, math
import numpy as np
from decimal import Decimal
sys.path.insert(0, "/home/claude/verif")
import verify as V

ROOT = "/home/claude/verif2/blind_v1_phase2_verification"

# --- spec f per family (exact formula; scalars given either as python double or float32 value)
def spec_eval(fam, inp, scalars):
    if fam == "F1":
        return V.eval_F1(inp, {"eps": scalars["eps"]})          # eval_F1 applies f32() to eps -> override below
    raise NotImplementedError


def f_eval(fam, inp, sc_vals):
    """sc_vals: dict scalar->python float already in the chosen mode (given double or float32 value)."""
    if fam == "F1":
        return eval_F1_exact(inp, sc_vals["eps"])
    if fam == "F2":
        return V.eval_F2(inp, {})
    if fam == "F2S":
        return eval_F2_scale_exact(inp, sc_vals["t"])
    if fam == "F3":
        return eval_F3_exact(inp, sc_vals["eps"])
    if fam == "F4":
        return eval_F4_exact(inp, sc_vals["t"])
    if fam == "F5":
        return V.eval_F5(inp, {})
    if fam == "F6":
        return V.eval_F6(inp, {})
    raise KeyError(fam)


D = V.D

def eval_F1_exact(inp, eps_val, noeps=False):
    x = inp["x"].astype(np.float64); s = inp["s"].astype(np.float64); b = inp["b"].astype(np.float64)
    eps = Decimal(0) if noeps else D(eps_val)
    R, Dn = x.shape
    lo = np.empty(R * Dn); hi = np.empty(R * Dn)
    for r in range(R):
        xs = [D(v) for v in x[r]]
        ms = sum((v * v for v in xs), Decimal(0)) / Dn
        slo, shi = V.sqrt_interval(ms + eps)
        rlo, rhi = V.recip_interval(slo, shi)
        for d in range(Dn):
            a = xs[d] * (Decimal(1) + D(s[r, d]))
            l, h = V.mul_interval(a, rlo, rhi)
            bb = D(b[r, d])
            lo[r * Dn + d] = float(l + bb); hi[r * Dn + d] = float(h + bb)
    return lo, hi


def eval_F2_scale_exact(inp, t_val):
    a = inp["a"].astype(np.float64); b = inp["b"].astype(np.float64)
    R, Dn = a.shape
    out = np.empty(R * Dn); t = D(t_val)
    for r in range(R):
        for d in range(Dn):
            av = D(a[r, d]); bv = D(b[r, d])
            g = Decimal("0.5") * (Decimal(1) + av / (Decimal(1) + abs(av)))
            out[r * Dn + d] = float(((av * g) * bv) * t)
    return out, out


def eval_F3_exact(inp, eps_val):
    x = inp["x"].astype(np.float64)
    eps = D(eps_val)
    R, Dn = x.shape
    lo = np.empty(R * 2); hi = np.empty(R * 2)
    for r in range(R):
        xs = [D(v) for v in x[r]]
        mean = sum(xs, Decimal(0)) / Dn
        m2 = sum(((v - mean) * (v - mean) for v in xs), Decimal(0))
        var = m2 / Dn
        slo, shi = V.sqrt_interval(var + eps)
        rlo, rhi = V.recip_interval(slo, shi)
        lo[2 * r] = hi[2 * r] = float(mean)
        lo[2 * r + 1] = float(rlo); hi[2 * r + 1] = float(rhi)
    return lo, hi


def eval_F3_program(inp, eps_val, dm1=False, naive=False):
    """declared semantics of the F3 programs: chunked merge with f32 compile-time constants, or naive E[x^2]-E[x]^2."""
    x = inp["x"].astype(np.float64)
    eps = D(eps_val)
    R, Dn = x.shape
    CH = Dn // 8
    lo = np.empty(R * 2); hi = np.empty(R * 2)
    for r in range(R):
        xs = [D(v) for v in x[r]]
        if naive:
            mean = sum(xs, Decimal(0)) / Dn
            ex2 = sum((v * v for v in xs), Decimal(0)) / Dn
            var = ex2 - mean * mean
        else:
            ch = xs[0:CH]
            mean = sum(ch, Decimal(0)) / CH
            m2 = sum(((v - mean) * (v - mean) for v in ch), Decimal(0))
            for k in range(CH, Dn, CH):
                chb = xs[k:k + CH]
                mb = sum(chb, Decimal(0)) / CH
                m2b = sum(((v - mb) * (v - mb) for v in chb), Decimal(0))
                delta = mb - mean
                c1 = D(V.f32(CH / (k + CH))); c2 = D(V.f32(k * CH / (k + CH)))
                mean = mean + delta * c1
                m2 = m2 + m2b + delta * delta * c2
            var = m2 / (Dn - 1 if dm1 else Dn)
        slo, shi = V.sqrt_interval(var + eps)
        rlo, rhi = V.recip_interval(slo, shi)
        lo[2 * r] = hi[2 * r] = float(mean)
        lo[2 * r + 1] = float(rlo); hi[2 * r + 1] = float(rhi)
    return lo, hi


def eval_F4_exact(inp, t_val, sign_bug=False):
    x = inp["x"].astype(np.float64); c = inp["cos"].astype(np.float64); s = inp["sin"].astype(np.float64)
    t = D(t_val)
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


# declared semantics (K_R) per program; kernel scalars are what the kernel received: float32 values
def kr_eval(prog, fam, inp, sc32):
    if fam == "F1":
        return eval_F1_exact(inp, sc32["eps"], noeps=(prog == "prog_18"))
    if fam == "F2":
        if prog == "prog_33":   # swapped arguments
            inp2 = {"a": inp["b"], "b": inp["a"]}
            return V.eval_F2(inp2, {})
        return V.eval_F2(inp, {})
    if fam == "F3":
        return eval_F3_program(inp, sc32["eps"], dm1=(prog == "prog_27"), naive=(prog == "prog_25"))
    if fam == "F4":
        return eval_F4_exact(inp, sc32["t"], sign_bug=(prog == "prog_35"))
    if fam == "F5":
        return V.eval_F5(inp, {}, group_bug=(prog == "prog_20"))
    if fam == "F6":
        return V.eval_F6(inp, {}, unmasked_den=(prog == "prog_10"))
    raise KeyError(fam)


def proj(e, kr_mid, fam, valid, r5):
    e = e[valid]; k = kr_mid[valid]; n = e.size
    res = {"R1": math.fsum(e) * (-1.0 / math.sqrt(n)),
           "R2": math.fsum(e * (-np.sign(k))) / math.sqrt(n)}
    nrm = math.sqrt(math.fsum(k * k))
    res["R3"] = -math.fsum(e * k) / nrm if nrm > 0 else float("nan")
    if fam == "F4":
        Dn = 256; R = kr_mid.size // Dn
        w = np.zeros(Dn); w[:64] = -1.0; w[64:128] = 1.0
        w = np.tile(w, R); w = w / np.linalg.norm(w)
        res["R4"] = math.fsum(e * w[valid])
    if r5 is not None:
        r5v = r5[valid]; nr = math.sqrt(math.fsum(r5v * r5v))
        res["R5"] = math.fsum(e * r5v) / nr if nr > 0 else None
    return res


def inside(v, iv, rel=1e-9):
    if v is None or iv is None:
        return None
    lo, hi = iv
    tol = max(abs(lo), abs(hi)) * rel + 1e-300
    return (lo - tol) <= v <= (hi + tol)


def main():
    index = json.load(open(os.path.join(ROOT, "index.json")))
    report = {}
    f_cache = {}
    for ent in index["programs"]:
        prog, fam = ent["program"], ent["family"]
        pdir = os.path.join(ROOT, "programs", prog)
        for seed in ("seed000", "seed032"):
            fdir = os.path.join(ROOT, "family_inputs_and_f", fam, seed)
            inp = np.load(os.path.join(fdir, "inputs.npz"))
            scal = json.load(open(os.path.join(fdir, "scalars.json")))
            sc_given = {k: v["value"] for k, v in scal.items()}
            sc_fp32 = {k: v["float32"] for k, v in scal.items()}
            t0 = time.time()
            # spec f, two scalar modes (cache per family/seed)
            for mode, scv in (("given", sc_given), ("fp32", sc_fp32)):
                key = (fam, seed, mode)
                if key not in f_cache:
                    f_cache[key] = f_eval(fam, inp, scv)
            # declared semantics K_R (kernel scalars are float32)
            kr_lo, kr_hi = kr_eval(prog, fam, inp, sc_fp32)
            kr_mid = 0.5 * (kr_lo + kr_hi)
            t_kr = np.load(os.path.join(pdir, seed, "KR_lo.npy")), np.load(os.path.join(pdir, seed, "KR_hi.npy"))
            tolk = np.maximum(kr_hi - kr_lo, 1e-300)
            kr_viol = int(np.sum((kr_mid < t_kr[0] - tolk) | (kr_mid > t_kr[1] + tolk)))
            tv = json.load(open(os.path.join(pdir, seed, "tool_values.json")))
            entry = {"family": fam, "n": int(kr_mid.size), "KR_tool_interval_violations": kr_viol,
                     "KR_tool_max_width": float(np.max(t_kr[1] - t_kr[0])), "modes": {}, "seconds": None}
            for mode in ("given", "fp32"):
                f_lo, f_hi = f_cache[(fam, seed, mode)]
                f_mid = 0.5 * (f_lo + f_hi)
                tf_lo = np.load(os.path.join(fdir, f"f_{mode}_lo.npy")); tf_hi = np.load(os.path.join(fdir, f"f_{mode}_hi.npy"))
                tolf = np.maximum(f_hi - f_lo, 1e-300)
                f_viol = int(np.sum((f_mid < tf_lo - tolf) | (f_mid > tf_hi + tolf)))
                e_sem = kr_mid - f_mid
                te_lo = np.load(os.path.join(pdir, seed, f"e_sem_{mode}_lo.npy")); te_hi = np.load(os.path.join(pdir, seed, f"e_sem_{mode}_hi.npy"))
                tole = np.maximum((kr_hi - kr_lo) + (f_hi - f_lo), 1e-300)
                e_viol = int(np.sum((e_sem < te_lo - tole) | (e_sem > te_hi + tole)))
                mres = {"f_tool_interval_violations": f_viol, "f_tool_max_width": float(np.max(tf_hi - tf_lo)),
                        "e_sem_tool_interval_violations": e_viol, "e_sem_tool_max_width": float(np.max(te_hi - te_lo)),
                        "projections": {}}
                groups = [("output", slice(None))] if fam != "F3" else [("mean", slice(0, None, 2)), ("rstd", slice(1, None, 2))]
                for gname, sl in groups:
                    valid = np.load(os.path.join(pdir, f"valid_coordinates_{gname}.npy"))
                    r5p = os.path.join(pdir, f"R5_direction_{mode}_{gname}.npy")
                    r5 = np.load(r5p) if os.path.exists(r5p) else None
                    mine = proj(e_sem[sl], kr_mid[sl], fam, valid, r5)
                    tproj = tv["projections"][mode][gname]
                    cmp = {}
                    for rule, val in mine.items():
                        tp = tproj.get(rule)
                        ivd = tp["directed"] if tp else None
                        ivp = tp["pipeline"] if tp else None
                        cmp[rule] = {"mine": val, "tool_directed": ivd, "tool_pipeline": ivp,
                                     "inside_directed": inside(val, ivd), "inside_pipeline": inside(val, ivp)}
                    mres["projections"][gname] = cmp
                entry["modes"][mode] = mres
            entry["seconds"] = round(time.time() - t0, 1)
            report[f"{prog}/{seed}"] = entry
            print(prog, seed, fam, f"{entry['seconds']}s", "KRviol", kr_viol,
                  "| given: fviol", entry["modes"]["given"]["f_tool_interval_violations"], "eviol", entry["modes"]["given"]["e_sem_tool_interval_violations"],
                  "| fp32: fviol", entry["modes"]["fp32"]["f_tool_interval_violations"], "eviol", entry["modes"]["fp32"]["e_sem_tool_interval_violations"], flush=True)
            for mode in ("given", "fp32"):
                for g, cmp in entry["modes"][mode]["projections"].items():
                    bad = [r for r, c in cmp.items() if c["inside_directed"] is False]
                    if bad:
                        print("   OUTSIDE directed interval:", mode, g, [(r, cmp[r]["mine"], cmp[r]["tool_directed"]) for r in bad], flush=True)
    json.dump(report, open("/home/claude/verif2/phase2_verification_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
