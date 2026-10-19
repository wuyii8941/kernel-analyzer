"""Adapter for contracts/GDR.yaml: gated delta rule as computed by FLA chunk_gated_delta_rule (profile fla_chunk) and
the paper's normalised token-mixer recurrence (profile paper_block).

reference: parse and validate the call, derive alpha_t = exp(g_t) (or the in-kernel gate), beta_t (or its sigmoid),
the optional L2 normalisation of q / k, then run kernel_spec.d2_linear.recurrent per sequence and value head.
READINGS['chunked']: the same call evaluated by the chunkwise WY algorithm (kernel_spec.d2_linear.chunked) with the
call's chunk size; it must agree with the recurrence exactly (clause GD.C11).
MUTANTS: independent code (the scalar loop `_loop` below, or the chunkwise routine for the two chunk misreadings).
"""
from __future__ import annotations

import math
from fractions import Fraction

from mpmath import iv

from kernel_spec import d2_common as U
from kernel_spec import d2_linear as L
from kernel_spec import real as rl
from kernel_spec.num import Invalid, Result, Undefined, Unsupported, integer, rational
from kernel_spec.real import R

FLA_KEYS = {"q", "k", "v", "g", "beta", "scale", "initial_state", "output_final_state", "use_qk_l2norm_in_kernel",
            "use_beta_sigmoid_in_kernel", "allow_neg_eigval", "use_gate_in_kernel", "A_log", "dt_bias",
            "state_v_first", "cu_seqlens", "chunk_size"}
PAPER_KEYS = {"q", "k", "v", "g", "beta", "initial_state", "output_final_state", "cu_seqlens"}
EPS = Fraction(1, 10 ** 6)


def default_scale(K):
    r = math.isqrt(K)
    if r * r == K:
        return R(Fraction(1, r))
    return R(1 / iv.sqrt(iv.mpf(K)))


def _l2(x, eps):
    den = rl.sqrt(U.rsum([e * e for e in x]) + R(eps))
    if den.sign() == 0:
        raise Undefined("L2 normalisation of a zero vector")
    return [e / den for e in x]


def parse(profile, inp):
    if profile not in ("fla_chunk", "paper_block"):
        raise Unsupported(f"profile {profile!r}")
    U.check_keys(inp, FLA_KEYS, required={"q", "k", "v", "g", "beta"})
    if profile == "paper_block":
        extra = set(inp) - PAPER_KEYS
        if extra & FLA_KEYS:
            raise Unsupported(f"options {sorted(extra & FLA_KEYS)} are not part of the paper_block profile")
        U.check_keys(inp, PAPER_KEYS)
    q, qs = U.tensor(inp["q"], 4, "q")
    B, T, H, K = qs
    k, _ = U.tensor(inp["k"], 4, "k", shape=qs)
    v, vs = U.tensor(inp["v"], 4, "v", shape=(B, T, None, None))
    HV, V = vs[2], vs[3]
    if H < 1 or K < 1 or V < 1 or HV < 1:
        raise Invalid("empty head or feature dimension")
    if HV % H:
        raise Invalid(f"HV={HV} must be divisible by H={H}")
    g, _ = U.tensor(inp["g"], 3, "g", shape=(B, T, HV))
    beta, _ = U.tensor(inp["beta"], 3, "beta", shape=(B, T, HV))
    final = U.flag(inp, "output_final_state")
    vfirst = U.flag(inp, "state_v_first")
    l2 = U.flag(inp, "use_qk_l2norm_in_kernel")
    bsig = U.flag(inp, "use_beta_sigmoid_in_kernel")
    neg = U.flag(inp, "allow_neg_eigval")
    gate = U.flag(inp, "use_gate_in_kernel")
    if neg and not bsig:
        raise Invalid("allow_neg_eigval=True requires use_beta_sigmoid_in_kernel=True")
    chunk = integer(inp.get("chunk_size", 64), "chunk_size")
    if chunk not in (16, 32, 64):
        raise Invalid("chunk_size must be 16, 32 or 64")
    A_log = dt_bias = None
    if gate:
        if "A_log" not in inp:
            raise Invalid("use_gate_in_kernel=True requires A_log")
        A_log, _ = U.tensor(inp["A_log"], 1, "A_log", shape=(HV,))
        if inp.get("dt_bias") is not None:
            dt_bias, _ = U.tensor(inp["dt_bias"], 1, "dt_bias", shape=(HV,))
    # The retained FLA entry accepts these keyword arguments with gate=False;
    # its forward path reads them only inside the gate=True branch.
    cu = inp.get("cu_seqlens")
    if cu is not None:
        cu = [integer(c, "cu_seqlens") for c in cu]
        if B != 1:
            raise Invalid("with cu_seqlens the batch size must be 1")
        if len(cu) < 2 or cu[0] != 0 or cu[-1] != T or any(b < a for a, b in zip(cu, cu[1:])):
            raise Invalid("cu_seqlens must start at 0, be non-decreasing and end at T")
        segs = [(0, cu[i], cu[i + 1]) for i in range(len(cu) - 1)]
    else:
        segs = [(b, 0, T) for b in range(B)]
    N = len(segs)
    h0 = None
    if inp.get("initial_state") is not None:
        st, _ = U.tensor(inp["initial_state"], 4, "initial_state",
                         shape=(N, HV, V, K) if vfirst else (N, HV, K, V))
        h0 = [[[[st[n][h][j][i] for j in range(V)] for i in range(K)] if vfirst else st[n][h] for h in range(HV)]
              for n in range(N)]
    if profile == "fla_chunk":
        scale = R(rational(inp["scale"], "scale")) if inp.get("scale") is not None else default_scale(K)
    else:
        scale = R(1)
    # derived per-token quantities
    if gate:
        alpha = [[[rl.exp(-rl.exp(A_log[h]) * rl.softplus(g[b][t][h] + (dt_bias[h] if dt_bias else R(0))))
                   for h in range(HV)] for t in range(T)] for b in range(B)]
    else:
        alpha = [[[rl.exp(g[b][t][h]) for h in range(HV)] for t in range(T)] for b in range(B)]
    if bsig:
        f = R(2 if neg else 1)
        beta = [[[f * rl.sigmoid(beta[b][t][h]) for h in range(HV)] for t in range(T)] for b in range(B)]
    if profile == "paper_block":
        q = [[[_l2(q[b][t][h], 0) for h in range(H)] for t in range(T)] for b in range(B)]
        k = [[[_l2(k[b][t][h], 0) for h in range(H)] for t in range(T)] for b in range(B)]
    elif l2:
        q = [[[_l2(q[b][t][h], EPS) for h in range(H)] for t in range(T)] for b in range(B)]
        k = [[[_l2(k[b][t][h], EPS) for h in range(H)] for t in range(T)] for b in range(B)]
    return dict(q=q, k=k, v=v, alpha=alpha, beta=beta, B=B, T=T, H=H, HV=HV, K=K, V=V, segs=segs, h0=h0,
                final=final, vfirst=vfirst, scale=scale, chunk=chunk, explicit_scale=inp.get("scale") is not None)


def _assemble(a, runner, head_of=None, carry=False):
    B, T, HV, K, V = a["B"], a["T"], a["HV"], a["K"], a["V"]
    rep = HV // a["H"]
    head_of = head_of or (lambda hv: hv // rep)
    o = [[[None] * HV for _ in range(T)] for _ in range(B)]
    finals = []
    carried = {}
    for n, (b, s0, s1) in enumerate(a["segs"]):
        fin = []
        for hv in range(HV):
            hq = head_of(hv)
            h0 = a["h0"][n][hv] if a["h0"] is not None else U.zeros(K, V)
            if carry and hv in carried:
                h0 = carried[hv]
            qs = [a["q"][b][t][hq] for t in range(s0, s1)]
            ks = [a["k"][b][t][hq] for t in range(s0, s1)]
            vs = [a["v"][b][t][hv] for t in range(s0, s1)]
            al = [a["alpha"][b][t][hv] for t in range(s0, s1)]
            be = [a["beta"][b][t][hv] for t in range(s0, s1)]
            outs, hT = runner(qs, ks, vs, al, be, a["scale"], h0)
            carried[hv] = hT
            for i, t in enumerate(range(s0, s1)):
                o[b][t][hv] = outs[i]
            fin.append([[hT[i][j] for i in range(K)] for j in range(V)] if a["vfirst"] else hT)
        finals.append(fin)
    out = {"o": U.plain(o)}
    if a["final"]:
        out["final_state"] = U.plain(finals)
    return Result("value", {"out": out})


def reference(profile, inp):
    return _assemble(parse(profile, inp), L.recurrent)


def _chunked_reading(profile, inp):
    a = parse(profile, inp)
    return _assemble(a, lambda q, k, v, al, be, sc, h0: L.chunked(q, k, v, al, be, sc, h0, a["chunk"]))


READINGS = {"chunked": _chunked_reading}


# ------------------------------------------------------------------ mutants (independent scalar loop)
def _loop(q, k, v, al, be, sc, h0, kind=None):
    """Element-wise scalar loop of the recurrence; `kind` selects one misreading."""
    K, V = len(h0), len(h0[0])
    S = {(i, j): h0[i][j] for i in range(K) for j in range(V)}
    out = []
    for t in range(len(q)):
        old = dict(S)
        if kind != "gate_after_write":
            S = {key: al[t] * x for key, x in S.items()}
        src = old if kind == "delta_undecayed" else S
        for j in range(V):
            pred = R(0)
            for i in range(K):
                pred = pred + k[t][i] * src[(i, j)]
            d = be[t] * (v[t][j] - pred)
            for i in range(K):
                S[(i, j)] = S[(i, j)] + k[t][i] * d
        if kind == "gate_after_write":
            S = {key: al[t] * x for key, x in S.items()}
        read = old if kind == "output_before_update" else S
        if kind == "output_before_update" and t == 0:
            read = {key: x for key, x in old.items()}
        row = []
        for j in range(V):
            acc = R(0)
            for i in range(K):
                acc = acc + q[t][i] * read[(i, j)]
            row.append(sc * acc)
        out.append(row)
    return out, [[S[(i, j)] for j in range(V)] for i in range(K)]


def _mut(kind):
    def run(profile, inp):
        if kind == "layout_ignored" and inp.get("state_v_first"):
            inp = dict(inp, state_v_first=False)
            a = parse(profile, inp)
            a["vfirst"] = True                      # output layout still V-first, input read as [K, V]
            return _assemble(a, _loop)
        if kind == "neg_eig_ignored" and inp.get("allow_neg_eigval"):
            inp = dict(inp, allow_neg_eigval=False)
        if kind == "l2_no_eps" and inp.get("use_qk_l2norm_in_kernel"):
            inp = dict(inp, use_qk_l2norm_in_kernel=False)
            a = parse(profile, inp)
            for key in ("q", "k"):
                a[key] = [[[_l2(x, 0) for x in row] for row in seq] for seq in a[key]]
            return _assemble(a, _loop)
        if kind == "dt_bias_ignored" and inp.get("dt_bias") is not None:
            inp = dict(inp)
            inp.pop("dt_bias")
        a = parse(profile, inp)
        if kind == "default_scale_one" and not a["explicit_scale"]:
            a["scale"] = R(1)
            return _assemble(a, lambda q, k, v, al, be, sc, h0: L.chunked(q, k, v, al, be, sc, h0, a["chunk"]))
        if kind == "chunk_no_decay_carry":
            return _assemble(a, lambda q, k, v, al, be, sc, h0: L.chunked(q, k, v, al, be, sc, h0, a["chunk"],
                                                                         carry_decay=False))
        if kind == "gva_modulo":
            return _assemble(a, _loop, head_of=lambda hv: hv % a["H"])
        if kind == "varlen_carry":
            return _assemble(a, _loop, carry=True)
        if kind in ("gate_after_write", "delta_undecayed", "output_before_update"):
            return _assemble(a, lambda *x: _loop(*x, kind=kind))
        return _assemble(a, _loop)
    return run


MUTANTS = {
    "GD.M1": _mut("gate_after_write"),
    "GD.M2": _mut("delta_undecayed"),
    "GD.M3": _mut("output_before_update"),
    "GD.M4": _mut("default_scale_one"),
    "GD.M5": _mut("gva_modulo"),
    "GD.M6": _mut("layout_ignored"),
    "GD.M7": _mut("varlen_carry"),
    "GD.M8": _mut("neg_eig_ignored"),
    "GD.M9": _mut("l2_no_eps"),
    "GD.M10": _mut("dt_bias_ignored"),
    "GD.M11": _mut("chunk_no_decay_carry"),
}
