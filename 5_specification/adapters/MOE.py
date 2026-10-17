"""Adapter for contracts/MOE.yaml: mixture-of-experts routing (softmax router, top-k with ties as allowed sets, gate
renormalisation or raw gates), token-order capacity dropping (top-1), combination of the expert outputs and the
load-balancing auxiliary loss (top-1 assignments or dispatched experts).

Expert computations are inputs (expert_outputs[t][e]) or linear experts (x @ expert_weights[e]); the contract judges
routing, dropping, combination and the auxiliary loss. Top-k ties are decided on the exact router logits (softmax is
strictly increasing). When the implementation's routing choice is supplied (selection), the other outputs are values
for that choice; otherwise ties give allowed sets over every joint choice (exact values only).
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations, product

from mpmath import iv

from kernel_spec import d1_attention as A
from kernel_spec import real as rl
from kernel_spec.num import Invalid, NotEstablished, Result, Undefined, Unsupported, integer, rational
from kernel_spec.real import NEG, R

PROFILES = {"renorm_aux_top1": ("renorm", "top1"), "renorm_aux_dispatched": ("renorm", "dispatched"),
            "raw_aux_top1": ("raw", "top1"), "raw_aux_dispatched": ("raw", "dispatched")}
KEYS = {"router_logits", "expert_outputs", "x", "expert_weights", "k", "capacity", "alpha", "selection"}
BUDGET = 4096


def parse(profile, inp):
    if profile not in PROFILES:
        raise Unsupported(f"profile {profile!r}")
    extra = set(inp) - KEYS
    if extra:
        raise Invalid(f"unknown input fields {sorted(extra)}")
    lg, ls = A.tensor(inp["router_logits"], 2, "router_logits", allow_inf=True)
    T, E = ls
    if T < 1 or E < 1:
        raise Invalid("at least one token and one expert")
    for t in range(T):
        if any(v == rl.POS for v in lg[t]):
            raise Invalid("+inf router logit")
        if all(v == NEG for v in lg[t]):
            raise Undefined(f"token {t}: every router logit is -inf (softmax undefined)")
    if inp.get("expert_outputs") is not None:
        if inp.get("expert_weights") is not None or inp.get("x") is not None:
            raise Invalid("give expert_outputs or (x, expert_weights), not both")
        y, ys = A.tensor(inp["expert_outputs"], 3, "expert_outputs")
        if ys[:2] != (T, E):
            raise Invalid("expert_outputs must be [T][E][Dout]")
    else:
        x, xs = A.tensor(inp["x"], 2, "x")
        Wt, ws = A.tensor(inp["expert_weights"], 3, "expert_weights")
        if xs[0] != T or ws[0] != E or ws[1] != xs[1]:
            raise Invalid("x [T][D] and expert_weights [E][D][Dout] do not match")
        y = [[[sum((x[t][d] * Wt[e][d][o] for d in range(xs[1])), Fraction(0)) for o in range(ws[2])]
              for e in range(E)] for t in range(T)]
    k = integer(inp["k"], "k")
    if not 1 <= k <= E:
        raise Invalid("k must be in [1, E]")
    cap = inp.get("capacity")
    if cap is not None:
        cap = integer(cap, "capacity", lo=0)
        if k > 1:
            raise Unsupported("capacity with k > 1: token-major and slot-major drop orders differ (MO.DROP_ORDER)")
    alpha = rational(inp["alpha"], "alpha")
    sel = inp.get("selection")
    if sel is not None:
        sel = [[integer(e, "selection") for e in row] for row in sel]
        if len(sel) != T:
            raise Invalid("selection needs one row per token")
    gate, auxk = PROFILES[profile]
    return dict(T=T, E=E, lg=lg, y=y, k=k, cap=cap, alpha=alpha, sel=sel, gate=gate, auxk=auxk)


def _gt(a, b):
    """a > b for exact logits (Fractions or -inf)."""
    if a == NEG:
        return False
    if b == NEG:
        return True
    return a > b


def valid_sets(lg, k):
    E = len(lg)
    out = []
    for S in combinations(range(E), k):
        rest = [u for u in range(E) if u not in S]
        if all(not _gt(lg[u], lg[s]) for s in S for u in rest):
            out.append(S)
    return out


def argmax_set(lg):
    return [e for e in range(len(lg)) if not any(_gt(lg[u], lg[e]) for u in range(len(lg)))]


def probs(lg):
    fin = [v for v in lg if v != NEG]
    m = max(fin)
    ex = [R(0) if v == NEG else rl.exp(R(v - m)) for v in lg]
    tot = R(0)
    for v in ex:
        tot = tot + v
    return [v / tot for v in ex]


def evaluate(a, P, choice):
    """choice[t] = (S sorted tuple, top-1 expert). Returns weights, dropped, out, aux as R / bool nested lists."""
    T, E = a["T"], a["E"]
    Dout = len(a["y"][0][0])
    w = [[R(0)] * E for _ in range(T)]
    for t, (S, _) in enumerate(choice):
        den = R(0)
        for e in S:
            den = den + P[t][e]
        for e in S:
            w[t][e] = P[t][e] / den if a["gate"] == "renorm" else P[t][e]
    load = [0] * E
    dropped = [[False] * E for _ in range(T)]
    for t, (S, _) in enumerate(choice):                # tokens in order (k = 1 when a capacity is set)
        for e in S:
            if a["cap"] is not None and load[e] >= a["cap"]:
                dropped[t][e] = True
            else:
                load[e] += 1
    out = []
    for t, (S, _) in enumerate(choice):
        acc = [R(0)] * Dout
        for e in S:
            if not dropped[t][e]:
                acc = [s + w[t][e] * R(v) for s, v in zip(acc, a["y"][t][e])]
        out.append(acc)
    f = [Fraction(0)] * E
    for t, (S, t1) in enumerate(choice):
        if a["auxk"] == "top1":
            f[t1] += Fraction(1, T)
        else:
            for e in S:
                if not dropped[t][e]:
                    f[e] += Fraction(1, T)
    aux = R(0)
    for e in range(E):
        Pe = R(0)
        for t in range(T):
            Pe = Pe + P[t][e]
        aux = aux + R(f[e]) * Pe / R(T)
    aux = R(a["alpha"]) * R(E) * aux
    return w, dropped, out, aux


def _plain(x):
    if isinstance(x, list):
        return [_plain(v) for v in x]
    return x.v if isinstance(x, R) else x


def _frozen(x):
    if isinstance(x, list):
        return tuple(_frozen(v) for v in x)
    if isinstance(x, R):
        if not isinstance(x.v, Fraction):
            raise NotEstablished("tied routing with inexact values: supply the implementation's selection")
        return x.v
    return x


def reference(profile, inp):
    a = parse(profile, inp)
    T, E, k, lg = a["T"], a["E"], a["k"], a["lg"]
    P = [probs(lg[t]) for t in range(T)]
    sets = [valid_sets(lg[t], k) for t in range(T)]
    topk = [frozenset(s) for s in sets]
    if a["sel"] is not None:
        choice = []
        for t, row in enumerate(a["sel"]):
            if len(row) != k or len(set(row)) != k or any(not 0 <= e < E for e in row):
                raise Invalid(f"token {t}: selection must list k distinct experts")
            if tuple(sorted(row)) not in sets[t]:
                raise Invalid(f"token {t}: {row} is not a valid top-{k} set")
            if any(_gt(lg[t][row[i + 1]], lg[t][row[i]]) for i in range(k - 1)):
                raise Invalid(f"token {t}: selection not in non-increasing router order")
            choice.append((tuple(sorted(row)), row[0]))
        w, dr, out, aux = evaluate(a, P, choice)
        return Result("value", {"topk": topk, "weights": _plain(w), "dropped": dr, "out": _plain(out),
                                "aux": _plain(aux)})
    opts = []
    for t in range(T):
        am = set(argmax_set(lg[t]))
        o = [(S, t1) for S in sets[t] for t1 in S if t1 in am]
        if a["auxk"] != "top1":                         # the top-1 expert matters only for the top-1 count
            o = [(S, S[0]) for S in sets[t]]
        opts.append(o)
    n = 1
    for o in opts:
        n *= len(o)
    if n == 1:
        w, dr, out, aux = evaluate(a, P, [o[0] for o in opts])
        return Result("value", {"topk": topk, "weights": _plain(w), "dropped": dr, "out": _plain(out),
                                "aux": _plain(aux)})
    if n > BUDGET:
        raise NotEstablished(f"{n} joint routing choices exceed the budget {BUDGET}")
    W, DR, OUT, AUX = set(), set(), set(), set()
    for choice in product(*opts):
        w, dr, out, aux = evaluate(a, P, list(choice))
        W.add(_frozen(w))
        DR.add(_frozen(dr))
        OUT.add(_frozen(out))
        AUX.add(_frozen(aux))

    def pick(s):
        if len(s) == 1:
            v = next(iter(s))
            return _unfreeze(v)
        return frozenset(s)
    return Result("value", {"topk": topk, "weights": pick(W), "dropped": pick(DR), "out": pick(OUT), "aux": pick(AUX)})


def _unfreeze(v):
    return [_unfreeze(x) for x in v] if isinstance(v, tuple) else v


# ------------------------------------------------------------------ mutants: separate implementation
def _p_indep(row):
    """softmax: exact when the finite logits are all equal, else mpmath intervals."""
    fin = [v for v in row if v != NEG]
    if all(v == fin[0] for v in fin):
        return [Fraction(0) if v == NEG else Fraction(1, len(fin)) for v in row]
    ex = [iv.mpf(0) if v == NEG else iv.exp(iv.mpf(v.numerator) / v.denominator) for v in row]
    tot = sum(ex[1:], ex[0])
    return [e / tot for e in ex]


def _indep(profile, inp, m=None):
    a = parse(profile, inp)
    T, E, k, lg = a["T"], a["E"], a["k"], a["lg"]
    gate = a["gate"]
    if m == "raw_for_renorm":
        gate = "raw"
    if m == "renorm_for_raw":
        gate = "renorm"
    auxk = a["auxk"]
    if m == "aux_dispatched_for_top1":
        auxk = "dispatched"
    if m == "aux_top1_for_dispatched":
        auxk = "top1"
    P = [_p_indep(lg[t]) for t in range(T)]
    inexact = any(not isinstance(v, Fraction) for row in P for v in row)
    conv = (lambda v: iv.mpf(v.numerator) / v.denominator if isinstance(v, Fraction) else v) if inexact else \
        (lambda v: v)
    P = [[conv(v) for v in row] for row in P]
    Y = [[[conv(v) for v in ye] for ye in yt] for yt in a["y"]]
    alpha, Ec, Tc = conv(a["alpha"]), conv(Fraction(E)), conv(Fraction(T))

    def tied_sets(t):
        order = sorted(range(E), key=lambda e: (-(lg[t][e] if lg[t][e] != NEG else -10 ** 30), e))
        kth = lg[t][order[k - 1]]
        above = [e for e in range(E) if _gt(lg[t][e], kth)]
        at = [e for e in range(E) if lg[t][e] == kth]
        return [tuple(sorted(above + list(c))) for c in combinations(at, k - len(above))]
    sets = [tied_sets(t) for t in range(T)]
    if m == "lowest_index_tiebreak":
        sets = [s[:1] for s in sets]
    if a["sel"] is not None:
        per_token = [[(tuple(sorted(r)), r[0])] for r in a["sel"]]
    else:
        per_token = []
        for t in range(T):
            mx = max(v for v in lg[t] if v != NEG)
            per_token.append([(S, t1) for S in sets[t] for t1 in S if lg[t][t1] == mx]
                             if auxk == "top1" else [(S, S[0]) for S in sets[t]])

    def run(choice, fresh_capacity=False):
        Dout = len(a["y"][0][0])
        w = [[conv(Fraction(0))] * E for _ in range(T)]
        for t, (S, _) in enumerate(choice):
            den = sum((P[t][e] for e in S), conv(Fraction(0)))
            for e in S:
                w[t][e] = P[t][e] / den if gate == "renorm" else P[t][e]
        load = [0] * E
        order = list(range(T))[::-1] if m == "capacity_keeps_last" else list(range(T))
        dropped = [[False] * E for _ in range(T)]
        for t in order:
            if fresh_capacity:
                load = [0] * E
            for e in choice[t][0]:
                if a["cap"] is not None and m != "capacity_ignored" and load[e] >= a["cap"]:
                    dropped[t][e] = True
                else:
                    load[e] += 1
        out = [[sum((w[t][e] * Y[t][e][o] for e in choice[t][0] if not dropped[t][e]), conv(Fraction(0)))
                for o in range(Dout)] for t in range(T)]
        f = [Fraction(0)] * E
        for t, (S, t1) in enumerate(choice):
            if auxk == "top1":
                f[t1] += Fraction(1, T)
            else:
                for e in S:
                    if not dropped[t][e]:
                        f[e] += Fraction(1, T)
        zero = conv(Fraction(0))
        if m == "aux_P_from_gates":
            Pm = [sum((w[t][e] for t in range(T)), zero) / Tc for e in range(E)]
        else:
            Pm = [sum((P[t][e] for t in range(T)), zero) / Tc for e in range(E)]
        aux = sum((conv(f[e]) * Pm[e] for e in range(E)), zero) * alpha * (conv(Fraction(1)) if m == "aux_without_E" else Ec)
        return w, dropped, out, aux

    topk = [frozenset(s) for s in sets]
    combos = list(product(*per_token))
    if m == "per_token_capacity":                    # v2.2: each token enumerated with its own (lost) capacity state
        results = [run(list(c), fresh_capacity=True) for c in combos]
    else:
        results = [run(list(c)) for c in combos]
    if len(results) == 1:
        w, dr, out, aux = results[0]
        return Result("value", {"topk": topk, "weights": w, "dropped": dr, "out": out, "aux": aux})

    def fz(x):
        return tuple(fz(v) for v in x) if isinstance(x, list) else x
    fields = {}
    for name, i in (("weights", 0), ("dropped", 1), ("out", 2), ("aux", 3)):
        s = {fz(r[i]) for r in results}
        fields[name] = _unfreeze(next(iter(s))) if len(s) == 1 else frozenset(s)
    return Result("value", dict(topk=topk, **fields))


def _m(kind):
    return lambda profile, inp: _indep(profile, inp, kind)


MUTANTS = {
    "MO.M1": _m("raw_for_renorm"),
    "MO.M2": _m("renorm_for_raw"),
    "MO.M3": _m("aux_dispatched_for_top1"),
    "MO.M4": _m("aux_top1_for_dispatched"),
    "MO.M5": _m("capacity_keeps_last"),
    "MO.M6": _m("capacity_ignored"),
    "MO.M7": _m("lowest_index_tiebreak"),
    "MO.M8": _m("per_token_capacity"),
    "MO.M9": _m("aux_without_E"),
    "MO.M10": _m("aux_P_from_gates"),
}
READINGS = {}
