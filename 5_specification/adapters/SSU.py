"""Adapter for contracts/SSU.yaml: one step of the Mamba selective state update (mamba_ssm selective_state_update).

reference: validate shapes, dt <- dt + dt_bias, softplus if requested, then per (row, head, channel p, state n)
    profile mamba_ref:  state' = exp(dt A) state + dt B x
    profile paper_zoh:  state' = exp(dt A) state + (exp(dt A) - 1) / (dt A) * dt B x   (dt A = 0: domain_undefined)
and y = sum_n C_n state'_n (+ D x) (* silu(z)). Rows whose state_batch_indices entry is negative are padding: y = 0,
the state pool is untouched. MUTANTS are independent element loops (`_loop`).
"""
from __future__ import annotations

from kernel_spec import d2_common as U
from kernel_spec import real as rl
from kernel_spec.num import Invalid, Result, Undefined, Unsupported, integer
from kernel_spec.real import R

KEYS = {"state", "x", "dt", "A", "B", "C", "D", "z", "dt_bias", "dt_softplus", "state_batch_indices"}


def parse(profile, inp):
    if profile not in ("mamba_ref", "paper_zoh"):
        raise Unsupported(f"profile {profile!r}")
    U.check_keys(inp, KEYS)
    st_rank = U.shape_of(inp["state"], "state")
    heads = len(st_rank) == 4
    if len(st_rank) not in (3, 4):
        raise Invalid("state must be (batch, dim, dstate) or (batch, nheads, dim, dstate)")

    def get(key, rank4, rank3, unsq):
        if inp.get(key) is None:
            return None, None
        if heads:
            return U.tensor(inp[key], rank4, key)
        t, s = U.tensor(inp[key], rank3, key)
        return unsq(t), unsq_shape(s, key)

    def unsq_shape(s, key):
        if key in ("state",):
            return (s[0], 1) + s[1:]
        if key in ("x", "dt", "z", "B", "C"):
            return (s[0], 1, s[1])
        return (1,) + s                                   # A, D, dt_bias gain the head dimension in front

    uns = {"state": lambda t: [[r] for r in t], "x": lambda t: [[r] for r in t], "dt": lambda t: [[r] for r in t],
           "z": lambda t: [[r] for r in t], "A": lambda t: [t], "B": lambda t: [[r] for r in t],
           "C": lambda t: [[r] for r in t], "D": lambda t: [t], "dt_bias": lambda t: [t]}
    r4 = {"state": 4, "x": 3, "dt": 3, "z": 3, "A": 3, "B": 3, "C": 3, "D": 2, "dt_bias": 2}
    vals, shp = {}, {}
    for key in r4:
        vals[key], shp[key] = get(key, r4[key], r4[key] - 1, uns[key])
    for key in ("state", "x", "dt", "A", "B", "C"):
        if vals[key] is None:
            raise Invalid(f"{key} is required")
    S, H, P, N = shp["state"]
    Bt = shp["x"][0]
    if shp["x"] != (Bt, H, P) or shp["dt"] != (Bt, H, P):
        raise Invalid("x and dt must be (batch, nheads, dim) matching the state")
    if shp["A"] != (H, P, N):
        raise Invalid("A must be (nheads, dim, dstate)")
    G = shp["B"][1]
    if G < 1 or H % G:
        raise Invalid("nheads must be divisible by ngroups")
    if shp["B"] != (Bt, G, N) or shp["C"] != shp["B"]:
        raise Invalid("B and C must be (batch, ngroups, dstate)")
    for key, want in (("D", (H, P)), ("dt_bias", (H, P)), ("z", (Bt, H, P))):
        if vals[key] is not None and shp[key] != want:
            raise Invalid(f"{key} has shape {shp[key]}, expected {want}")
    idx = inp.get("state_batch_indices")
    if idx is not None:
        idx = [integer(i, "state_batch_indices") for i in idx]
        if len(idx) != Bt or any(i >= S for i in idx):
            raise Invalid("state_batch_indices: one entry per row, below the number of state slots")
    elif S != Bt:
        raise Invalid("without state_batch_indices the state batch must equal the input batch")
    soft = U.flag(inp, "dt_softplus")
    return dict(v=vals, S=S, H=H, P=P, N=N, Bt=Bt, G=G, idx=idx, soft=soft, heads=heads, zoh=profile == "paper_zoh")


def _finish(a, y, st):
    if not a["heads"]:
        y = [row[0] for row in y]
        st = [s[0] for s in st]
    return Result("value", {"out": {"y": U.plain(y), "state": U.plain(st)}})


def reference(profile, inp):
    a = parse(profile, inp)
    v, H, P, N = a["v"], a["H"], a["P"], a["N"]
    rep = H // a["G"]
    st = [[[row[:] for row in hd] for hd in slot] for slot in v["state"]]
    y = [[[None] * P for _ in range(H)] for _ in range(a["Bt"])]
    for b in range(a["Bt"]):
        slot = a["idx"][b] if a["idx"] is not None else b
        if slot < 0:
            y[b] = [[R(0)] * P for _ in range(H)]
            continue
        for h in range(H):
            g = h // rep
            for p in range(P):
                dt = v["dt"][b][h][p] + (v["dt_bias"][h][p] if v["dt_bias"] is not None else R(0))
                if a["soft"]:
                    dt = rl.softplus(dt)
                acc = R(0)
                for n in range(N):
                    dA = rl.exp(dt * v["A"][h][p][n])
                    if a["zoh"]:
                        z = dt * v["A"][h][p][n]
                        if z.sign() == 0:
                            raise Undefined("ZOH B_bar = (dt A)^-1 (exp(dt A) - 1) dt B with dt A = 0")
                        dB = (dA - 1) / z * dt * v["B"][b][g][n]
                    else:
                        dB = dt * v["B"][b][g][n]
                    st[slot][h][p][n] = st[slot][h][p][n] * dA + dB * v["x"][b][h][p]
                    acc = acc + st[slot][h][p][n] * v["C"][b][g][n]
                if v["D"] is not None:
                    acc = acc + v["x"][b][h][p] * v["D"][h][p]
                if v["z"] is not None:
                    zz = v["z"][b][h][p]
                    acc = acc * (zz * rl.sigmoid(zz))
                y[b][h][p] = acc
    return _finish(a, y, st)


# ------------------------------------------------------------------ mutants: independent loop with one misreading
def _loop(profile, inp, kind):
    a = parse(profile, inp)
    v, H, P, N = a["v"], a["H"], a["P"], a["N"]
    old = {(s, h, p, n): v["state"][s][h][p][n] for s in range(a["S"]) for h in range(H) for p in range(P)
           for n in range(N)}
    new = dict(old)
    y = {}
    for b in range(a["Bt"]):
        s = a["idx"][b] if a["idx"] is not None else b
        if s < 0 and kind != "padding_processed":
            for h in range(H):
                for p in range(P):
                    y[(b, h, p)] = R(0)
            continue
        s = max(s, 0) if kind == "padding_processed" else s
        for h in range(H):
            g = (h % a["G"]) if kind == "groups_modulo" else h // (H // a["G"])
            for p in range(P):
                dt = v["dt"][b][h][p]
                if v["dt_bias"] is not None and kind != "dt_bias_ignored":
                    dt = dt + v["dt_bias"][h][p]
                if a["soft"] and kind != "softplus_ignored":
                    dt = rl.log1p(rl.exp(dt)) if dt <= R(20) else dt
                out = R(0)
                for n in range(N):
                    An = v["A"][h][p][n]
                    dA = rl.exp(An) if kind == "decay_without_dt" else rl.exp(An * dt)
                    if a["zoh"] and kind != "zoh_as_euler":
                        if (An * dt).sign() == 0:
                            raise Undefined("dt A = 0")
                        bb = (dA - 1) / An * v["B"][b][g][n]
                    else:
                        bb = v["B"][b][g][n] if kind == "input_without_dt" else v["B"][b][g][n] * dt
                    new[(s, h, p, n)] = old[(s, h, p, n)] * dA + bb * v["x"][b][h][p]
                    src = old if kind == "output_from_old_state" else new
                    out = out + v["C"][b][g][n] * src[(s, h, p, n)]
                if v["D"] is not None and kind not in ("skip_D_omitted", "gate_before_D"):
                    out = out + v["D"][h][p] * v["x"][b][h][p]
                if v["z"] is not None:
                    zz = v["z"][b][h][p]
                    out = out * zz if kind == "gate_without_sigmoid" else out * zz / (1 + rl.exp(-zz))
                if kind == "gate_before_D" and v["D"] is not None:
                    out = out + v["D"][h][p] * v["x"][b][h][p]
                y[(b, h, p)] = out
        if kind == "state_not_written":
            new = dict(old)
    ys = [[[y[(b, h, p)] for p in range(P)] for h in range(H)] for b in range(a["Bt"])]
    sts = [[[[new[(s, h, p, n)] for n in range(N)] for p in range(P)] for h in range(H)] for s in range(a["S"])]
    return _finish(a, ys, sts)


def _m(kind):
    return lambda profile, inp: _loop(profile, inp, kind)


MUTANTS = {
    "SS.M1": _m("dt_bias_ignored"),
    "SS.M2": _m("skip_D_omitted"),
    "SS.M3": _m("output_from_old_state"),
    "SS.M4": _m("decay_without_dt"),
    "SS.M5": _m("input_without_dt"),
    "SS.M6": _m("softplus_ignored"),
    "SS.M7": _m("gate_without_sigmoid"),
    "SS.M8": _m("gate_before_D"),
    "SS.M9": _m("groups_modulo"),
    "SS.M10": _m("padding_processed"),
    "SS.M11": _m("zoh_as_euler"),
    "SS.M12": _m("state_not_written"),
}
READINGS = {}
