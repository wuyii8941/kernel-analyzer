"""RQ1: the device behaviour of every primitive instance against its declared semantics (Triton 3.6.0, sm_86).

For one primitive: build its TTIR, compile, draw operands per distribution, run on the GPU, compute the declared
value with the answer machine (exact.py, refs.py) and compare.  Per distribution the record has the count of
finite comparisons, the share equal to the correctly rounded declared value, the largest and the mean signed error in
units of the result format's spacing at the exact value, flushes to zero, and special-value (nan / inf / signed zero)
disagreements; integer and predicate results are compared exactly, with results the declared semantics leaves
undefined counted apart (and the device value recorded as a pattern).

Run:  catalog.py --pid 'F1|math.exp|default|f32'      (one primitive, JSON on stdout)
      catalog.py --all --out results/catalog.json      (every primitive; one child process each, GPUs round robin)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction as F
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exact as X  # noqa: E402
import primitives as P  # noqa: E402
import refs  # noqa: E402
import ttir_build as tb  # noqa: E402

N_PER_DIST = 2048
DISTS = ["symmetric", "same-sign", "heavy-tailed", "small-integer", "near-tie", "near-format-boundary", "wide"]
FLOAT_T = ("f16", "bf16", "f32", "f64", "f8e5m2", "f8e4m3")
INT_W = {"i8": 8, "i16": 16, "i32": 32, "i64": 64, "u32": 32, "u64": 64, "i1": 1}


# ----------------------------------------------------------------------------- operand drawing
def fmt_round_array(v: np.ndarray, t: str) -> np.ndarray:
    """float64 values -> nearest values of the operand format (as float64), via torch's RNE casts."""
    import torch
    dt = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32, "f64": torch.float64,
          "f8e5m2": torch.float8_e5m2, "f8e4m3": torch.float8_e4m3fn}[t]
    with np.errstate(all="ignore"):
        x = torch.from_numpy(np.asarray(v, np.float64))
        if t in ("f8e5m2", "f8e4m3"):
            x = x.clamp(-float(X.max_finite(t)), float(X.max_finite(t)))
        return x.to(dt).to(torch.float64).numpy()


def float_dist(rng, dist: str, t: str, n: int) -> np.ndarray:
    p, emin, emax, _ = X.FORMATS[t]
    mx = float(X.max_finite(t))
    if dist == "symmetric":
        v = rng.standard_normal(n)
    elif dist == "same-sign":
        v = np.abs(rng.standard_normal(n)) + 1e-3
    elif dist == "heavy-tailed":
        v = rng.standard_normal(n) / np.maximum(np.abs(rng.standard_normal(n)), 1e-3) ** 0.5
    elif dist == "small-integer":
        v = rng.integers(-8, 9, n).astype(np.float64)
    elif dist == "near-tie":
        v = 1 + 2.0 ** -9 * (rng.random(n) - 0.5) * 1e-3
    elif dist == "near-format-boundary":
        kind = rng.integers(0, 4, n)
        base = np.where(kind == 0, 2.0 ** emin, np.where(kind == 1, 2.0 ** (emin - 1), np.where(kind == 2, mx, 2.0 ** emax)))
        v = base * (1 + (rng.random(n) - 0.5) * 2e-3) * rng.choice([-1.0, 1.0], n)
    elif dist == "wide":
        v = rng.choice([-1.0, 1.0], n) * 2.0 ** rng.uniform(emin - p + 1, emax + 1, n)
    else:
        raise ValueError(dist)
    with np.errstate(all="ignore"):
        v = np.clip(v, -mx, mx)
    return v


SPECIAL_F = [0.0, -0.0, 1.0, -1.0, 0.5, 2.0, float("inf"), float("-inf"), float("nan")]


def fold(sem: str, pos: int, v: np.ndarray) -> np.ndarray:
    """Map drawn values into the declared domain of the operand (the special block keeps the raw values)."""
    with np.errstate(all="ignore"):
        if sem in ("log", "log2", "log10", "sqrt", "rsqrt", "precise_sqrt", "y0", "y1", "fast_logf", "fast_log2f",
                   "fast_log10f", "sqrt_rn", "sqrt_rz", "sqrt_rd", "sqrt_ru", "rsqrt_rn", "log1p", "lgamma", "tgamma"):
            return np.abs(v)
        if sem in ("pow", "powf", "fast_powf") and pos == 0:
            return np.abs(v)
        if sem == "yn" and pos == 1:
            return np.abs(v)
        if sem == "acosh":
            return 1 + np.abs(v)
        if sem in ("clampf", "clampf_nan") and pos in (1, 2):
            return np.abs(v) * (-1 if pos == 1 else 1)
        if sem in ("asin", "acos", "atanh", "erfinv"):
            return v / (1 + np.abs(v))
        if sem == "erfcinv":
            return 1 + v / (1 + np.abs(v))
        if sem == "normcdfinv":
            return 1 / (1 + np.exp(-np.clip(v, -700, 700)))
    return v


def int_dist(rng, dist: str, t: str, n: int) -> list:
    w = INT_W[t]
    if t == "i1":
        return [int(b) for b in rng.integers(0, 2, n)]
    unsigned = t.startswith("u")
    lo, hi = (0, 2 ** w - 1) if unsigned else (-(2 ** (w - 1)), 2 ** (w - 1) - 1)

    def rand_bits():
        return int.from_bytes(rng.bytes(8), "little") & (2 ** w - 1)

    out = []
    for _ in range(n):
        if dist == "symmetric":
            u = rand_bits()
            x = u if unsigned else (u - 2 ** w if u >> (w - 1) else u)
        elif dist == "same-sign":
            x = rand_bits() & (2 ** (w - 1) - 1)
        elif dist in ("heavy-tailed", "wide"):
            b = int(rng.integers(0, w))
            x = int(rng.integers(0, 2 ** min(b, 62))) if b < 63 else rand_bits() & (2 ** (w - 1) - 1)
            if not unsigned and rng.random() < 0.5:
                x = -x
        elif dist == "small-integer":
            x = int(rng.integers(-8, 9))
            if unsigned:
                x = abs(x)
        elif dist == "near-tie":
            x = int(rng.choice([-2, -1, 0, 1, 2])) + (2 ** int(rng.integers(0, w - 1)))
        else:
            x = int(rng.choice([lo, lo + 1, hi, hi - 1, 0, 1, -1 if not unsigned else 2]))
        out.append(max(lo, min(hi, x)))
    return out


def shift_amounts(rng, n, w):
    return [int(x) for x in np.where(rng.random(n) < 0.9, rng.integers(0, w, n), rng.integers(w, w + 3, n))]


def draw_operands(prim: P.Prim, rng, dist: str, n: int) -> list:
    ops = []
    for pos, t in enumerate(prim.ins):
        if t in FLOAT_T:
            v = float_dist(rng, dist, t, n)
            v = fmt_round_array(fold(prim.sem, pos, v), t)
            ops.append(v)
        else:
            if prim.family == "int" and prim.sem in ("shli", "shrsi", "shrui") and pos == 1:
                ops.append(shift_amounts(rng, n, INT_W[t]))
            elif prim.sem in ("powi", "pow", "ldexp", "scalbn", "ipowi") and pos == 1:
                ops.append([int(x) for x in rng.integers(-12, 13, n)] if prim.sem != "ipowi"
                           else [int(x) for x in rng.integers(0, 9, n)])
            elif prim.sem in ("jn", "yn") and pos == 0:
                ops.append([int(x) for x in rng.integers(0, 6, n)])
            else:
                ops.append(int_dist(rng, dist, t, n))
    return ops


def special_operands(prim: P.Prim, rng) -> list:
    """Every combination of special values for float operands (other operands drawn symmetric)."""
    k = len(prim.ins)
    fl = [i for i, t in enumerate(prim.ins) if t in FLOAT_T]
    if not fl:
        return []
    combos = [[]]
    for i, t in enumerate(prim.ins):
        if i in fl and len(fl) <= 2:
            mx = float(X.max_finite(t))
            vals = SPECIAL_F + [mx, -mx, 2.0 ** X.FORMATS[t][1], 2.0 ** (X.FORMATS[t][1] - 1)]
            combos = [c + [v] for c in combos for v in vals]
        else:
            combos = [c + [None] for c in combos]
    n = len(combos)
    filler = draw_operands(prim, rng, "symmetric", n)
    ops = []
    for i, t in enumerate(prim.ins):
        col = [c[i] if c[i] is not None else filler[i][j] for j, c in enumerate(combos)]
        ops.append(fmt_round_array(np.array(col, np.float64), t) if t in FLOAT_T else col)
    return ops


# ----------------------------------------------------------------------------- module text per primitive
def module_text(prim: P.Prim) -> tuple:
    """(module text, launch kind)."""
    if prim.source == "F1" and prim.family not in ("scalar", "mx"):
        built = tb.module_for((prim.op, prim.attr, prim.sig))
        return built[0], built[3]
    if prim.source == "libdevice":
        ins = [{"u32": "i32", "u64": "i64"}.get(t, t) for t in prim.ins]
        out = {"u32": "i32", "u64": "i64"}.get(prim.outs[0], prim.outs[0])
        args = ", ".join(f"%x{i}" for i in range(len(ins)))
        types = ", ".join(tb.tensor(t) for t in ins)
        line = (f'%y = tt.extern_elementwise {args} {{libname = "", libpath = "", pure = true, symbol = "{prim.attr}"}} '
                f": ({types}) -> {tb.tensor(out)}")
        return tb.elementwise_module(ins, [line], [out]), "elementwise"
    if prim.family == "literal":
        t = prim.sig
        lit = prim.attr
        if t == "i1":
            lit = "true"
        elif t in FLOAT_T and "." not in lit.split("e")[0]:
            lit = lit.replace("e", ".0e") if "e" in lit else lit + ".0"
        return tb.elementwise_module([], [f"%y = arith.constant dense<{lit}> : {tb.tensor(t)}"], [t]), "elementwise"
    if prim.family == "scalar":
        t = prim.ins[0]
        text = ("module {\n  tt.func public @probe(%in0: !tt.ptr<" + t + ">, %out0: !tt.ptr<" + t + ">, %n: i32) {\n"
                "    %pid = tt.get_program_id x : i32\n"
                f"    %pi = tt.addptr %in0, %pid : !tt.ptr<{t}>, i32\n"
                f"    %v = tt.load %pi : !tt.ptr<{t}>\n"
                f"    %ix = {prim.op} %v : {t} to index\n"
                f"    %b = {prim.op} %ix : index to {t}\n"
                f"    %po = tt.addptr %out0, %pid : !tt.ptr<{t}>, i32\n"
                f"    tt.store %po, %b : !tt.ptr<{t}>\n"
                "    tt.return\n  }\n}\n")
        return text, "scalar"
    if prim.family == "mx":
        if prim.op == "tt.dot_scaled":
            d = {"f8e4m3": "e4m3", "f8e5m2": "e5m2", "f4e2m1": "e2m1"}[prim.sig]
            return tb.dot_scaled_module(d), ("dot_scaled" if d == "e5m2" else "compile-only")
        ins, lines, outs = tb.op_lines((prim.op, prim.attr, prim.sig))
        return tb.elementwise_module(ins, lines, outs), "compile-only"
    raise ValueError(prim.pid)


# ----------------------------------------------------------------------------- device execution
def torch_dtype(t):
    import torch
    return {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32, "f64": torch.float64,
            "f8e5m2": torch.float8_e5m2, "f8e4m3": torch.float8_e4m3fn, "i8": torch.int8, "i16": torch.int16,
            "i32": torch.int32, "i64": torch.int64, "u32": torch.int32, "u64": torch.int64, "i1": torch.int8}[t]


def to_device(col, t):
    import torch
    if t in FLOAT_T:
        return torch.from_numpy(np.asarray(col, np.float64)).to(torch_dtype(t)).cuda()
    w = INT_W[t]
    bits = [(int(x) & (2 ** w - 1)) if t != "i1" else int(x) for x in col]
    signed = [b - 2 ** w if (t != "i1" and b >> (w - 1)) else b for b in bits]
    return torch.tensor(signed, dtype=torch.int64).to(torch_dtype(t)).cuda()


def from_device(x, t) -> list:
    import torch
    if t in FLOAT_T:
        return x.to(torch.float64).cpu().numpy().tolist()
    v = x.to(torch.int64).cpu().numpy().tolist()
    if t in ("u32", "u64"):
        w = INT_W[t]
        return [int(a) & (2 ** w - 1) for a in v]
    return [int(a) for a in v]


def float_bits(x: float, t: str) -> int:
    import torch
    if t == "f64":
        return int(np.array([x]).view(np.uint64)[0])
    if t == "f32":
        return int(np.array([x], np.float32).view(np.uint32)[0])
    if t == "f16":
        return int(np.array([x], np.float16).view(np.uint16)[0])
    if t == "bf16":
        return int(torch.tensor([x], dtype=torch.float64).to(torch.bfloat16).view(torch.int16).item()) & 0xFFFF
    raise ValueError(t)


_KERNELS = {}


def compile_text(text: str, options: dict):
    import triton
    from triton.backends.compiler import GPUTarget
    key = hashlib.sha256((text + repr(sorted(options.items()))).encode()).hexdigest()[:20]
    if key in _KERNELS:
        return _KERNELS[key]
    work = HERE.parents[1] / ".cache/composition_study/ttir"
    work.mkdir(parents=True, exist_ok=True)
    path = work / f"m_{key}.ttir"
    path.write_text(text)
    k = triton.compile(str(path), target=GPUTarget("cuda", 86, 32), options=options)
    _KERNELS[key] = k
    return k


def run_elementwise(kernel, prim, ops, raw=False):
    """Device outputs per result: values (float64 / int), or with ``raw`` the storage bit patterns of float results
    (NaN payloads survive)."""
    import torch
    import triton
    n = len(ops[0]) if ops else 1024
    ins = [to_device(c, t) for c, t in zip(ops, prim.ins)]
    outs = [torch.zeros(n, dtype=torch_dtype(t), device="cuda") for t in prim.outs]
    kernel[(triton.cdiv(n, tb.BLOCK), 1, 1)](*ins, *outs, n)
    torch.cuda.synchronize()
    if raw:
        res = []
        for o, t in zip(outs, prim.outs):
            if t in FLOAT_T:
                w = o.element_size()
                b = o.view({1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[w]).to(torch.int64)
                res.append([int(v) & ((1 << (8 * w)) - 1) for v in b.cpu().numpy().tolist()])
            else:
                res.append(from_device(o, t))
        return res
    return [from_device(o, t) for o, t in zip(outs, prim.outs)]


# ----------------------------------------------------------------------------- integer and predicate semantics
def wrap(v, w, signed=True):
    v &= (1 << w) - 1
    return v - (1 << w) if (signed and v >> (w - 1)) else v


def int_declared(sem, args, t, flags="none"):
    w = INT_W[t]
    a = args[0]
    b = args[1] if len(args) > 1 else None
    ua, ub = a & ((1 << w) - 1), (b & ((1 << w) - 1)) if b is not None else None
    sa, sb = wrap(a, w), wrap(b, w) if b is not None else None
    if sem in ("addi", "subi", "muli", "shli"):
        if sem == "shli" and ub >= w:
            return refs.UB("shift >= width")
        full_s = {"addi": sa + sb, "subi": sa - sb, "muli": sa * sb, "shli": sa * (1 << ub) if sem == "shli" else 0}[sem]
        full_u = {"addi": ua + ub, "subi": ua - ub, "muli": ua * ub, "shli": ua << ub if sem == "shli" else 0}[sem]
        res = wrap(full_u, w)
        if "nsw" in flags and full_s != res:
            return refs.UB("nsw overflow")
        if "nuw" in flags and full_u != (full_u & ((1 << w) - 1)):
            return refs.UB("nuw overflow")
        return res
    if sem == "andi":
        return wrap(ua & ub, w)
    if sem == "ori":
        return wrap(ua | ub, w)
    if sem == "xori":
        return wrap(ua ^ ub, w)
    if sem in ("shrsi", "shrui"):
        if ub >= w:
            return refs.UB("shift >= width")
        return wrap(sa >> ub, w) if sem == "shrsi" else wrap(ua >> ub, w)
    if sem in ("divsi", "remsi", "ceildivsi", "floordivsi"):
        if sb == 0:
            return refs.UB("division by zero")
        if sa == -(1 << (w - 1)) and sb == -1:
            return refs.UB("signed division overflow")
        q = abs(sa) // abs(sb) * (1 if (sa >= 0) == (sb >= 0) else -1)          # toward zero
        if sem == "divsi":
            return q
        if sem == "remsi":
            return sa - q * sb
        exact_q = F(sa, sb)
        return wrap(math.ceil(exact_q) if sem == "ceildivsi" else math.floor(exact_q), w)
    if sem in ("divui", "remui", "ceildivui"):
        if ub == 0:
            return refs.UB("division by zero")
        if sem == "divui":
            return wrap(ua // ub, w)
        if sem == "remui":
            return wrap(ua % ub, w)
        return wrap(-(-ua // ub), w)
    if sem == "maxsi":
        return max(sa, sb)
    if sem == "minsi":
        return min(sa, sb)
    if sem == "maxui":
        return wrap(max(ua, ub), w)
    if sem == "minui":
        return wrap(min(ua, ub), w)
    if sem == "absi":
        return wrap(abs(sa), w)
    if sem == "ctlz":
        return w - ua.bit_length()
    if sem == "cttz":
        return w if ua == 0 else (ua & -ua).bit_length() - 1
    if sem == "ctpop":
        return bin(ua).count("1")
    if sem == "ipowi":
        if sb < 0:
            return refs.UB("negative exponent")
        return wrap(pow(sa, sb, 1 << w) if sb >= 0 else 0, w)
    if sem == "mulhiui":
        return wrap((ua * ub) >> w, w)
    raise KeyError(sem)


def int_declared_multi(sem, args, t):
    w = INT_W[t]
    a, b = args
    if sem == "mulsi_extended":
        p = wrap(a, w) * wrap(b, w)
        return [wrap(p, w), wrap(p >> w, w)]
    if sem == "mului_extended":
        p = (a & ((1 << w) - 1)) * (b & ((1 << w) - 1))
        return [wrap(p, w), wrap(p >> w, w)]
    if sem == "addui_extended":
        s = (a & ((1 << w) - 1)) + (b & ((1 << w) - 1))
        return [wrap(s, w), int(s >> w)]
    raise KeyError(sem)


def pred_declared(sem, args, t):
    if sem.startswith("cmpi."):
        w = INT_W[t]
        p = sem.split(".")[1]
        a, b = args
        sa, sb, ua, ub = wrap(a, w), wrap(b, w), a & ((1 << w) - 1), b & ((1 << w) - 1)
        return int({"eq": sa == sb, "ne": sa != sb, "slt": sa < sb, "sle": sa <= sb, "sgt": sa > sb, "sge": sa >= sb,
                    "ult": ua < ub, "ule": ua <= ub, "ugt": ua > ub, "uge": ua >= ub}[p])
    if sem.startswith("cmpf."):
        p = sem.split(".")[1]
        a, b = args
        unordered = math.isnan(a) or math.isnan(b)
        if p == "true":
            return 1
        if p == "false":
            return 0
        if p == "ord":
            return int(not unordered)
        if p == "uno":
            return int(unordered)
        rel = {"eq": lambda: a == b, "gt": lambda: a > b, "ge": lambda: a >= b, "lt": lambda: a < b,
               "le": lambda: a <= b, "ne": lambda: a != b}[p[1:]]
        if unordered:
            return int(p[0] == "u")
        return int(rel())
    a = args[0]
    p_, emin, _, _ = X.FORMATS[t]
    if sem == "isnan":
        return int(math.isnan(a))
    if sem == "isinf":
        return int(math.isinf(a))
    if sem == "isfinite":
        return int(math.isfinite(a))
    if sem == "isnormal":
        return int(math.isfinite(a) and a != 0 and abs(a) >= 2.0 ** emin)
    raise KeyError(sem)


def libint_declared(name, args, ins, out):
    w_out = INT_W.get(out, 32)
    a = args[0]
    if name in ("isnan", "isinf", "finitef", "isfinited", "signbit"):
        v = {"isnan": math.isnan(a), "isinf": math.isinf(a), "finitef": math.isfinite(a), "isfinited": math.isfinite(a),
             "signbit": math.copysign(1.0, a) < 0}[name]
        return ("truthy", int(v))
    if name == "ilogb":
        if a == 0 or math.isnan(a):
            return refs.UB("ilogb of 0 / nan")
        if math.isinf(a):
            return refs.UB("ilogb of inf")
        return X.floor_log2(abs(F(a)))
    if name in ("llrint", "llround"):
        if not math.isfinite(a):
            return refs.UB("non-finite")
        r = refs._round_half_even(F(a)) if name == "llrint" else refs._round_half_away(F(a))
        if not -(2 ** 63) <= r < 2 ** 63:
            return refs.UB("out of range")
        return int(r)
    w = INT_W[ins[0]]
    unsigned = ins[0].startswith("u")
    if name == "abs":
        return wrap(abs(wrap(a, w)), w)
    if name == "brev":
        return wrap(int(format(a & ((1 << w) - 1), f"0{w}b")[::-1], 2), w, not unsigned)
    if name == "clz":
        return w - (a & ((1 << w) - 1)).bit_length()
    if name == "popc":
        return bin(a & ((1 << w) - 1)).count("1")
    if name == "ffs":
        u = a & ((1 << w) - 1)
        return 0 if u == 0 else (u & -u).bit_length()
    b = args[1]
    if name == "mulhi":
        if unsigned:
            return wrap(((a & ((1 << w) - 1)) * (b & ((1 << w) - 1))) >> w, w, False)
        return wrap((wrap(a, w) * wrap(b, w)) >> w, w)
    if name == "mul24":
        a24, b24 = ((a & 0xFFFFFF), (b & 0xFFFFFF)) if unsigned else (wrap(a & 0xFFFFFF, 24), wrap(b & 0xFFFFFF, 24))
        return wrap(a24 * b24, 32, not unsigned)
    if name == "hadd":
        return (wrap(a, w, not unsigned) + wrap(b, w, not unsigned)) >> 1
    if name == "rhadd":
        return (wrap(a, w, not unsigned) + wrap(b, w, not unsigned) + 1) >> 1
    if name == "sad":
        c = args[2]
        return wrap(abs(wrap(a, w, not unsigned) - wrap(b, w, not unsigned)) + (c & 0xFFFFFFFF), 32,
                    not out.startswith("u"))
    if name == "byte_perm":
        s = args[2]
        src = ((b & 0xFFFFFFFF) << 32) | (a & 0xFFFFFFFF)
        r = 0
        for k in range(4):
            sel = (s >> (4 * k)) & 0x7
            r |= ((src >> (8 * sel)) & 0xFF) << (8 * k)
        return wrap(r, 32)
    raise KeyError(name)


# ----------------------------------------------------------------------------- comparison of float results
def float_compare(prim, sem, mode, out_t, args_list, K_list, stats):
    """args_list: per element a tuple of operand values (float / int); K_list: device results (float)."""
    p, emin, emax, _ = X.FORMATS[out_t]
    for args, K in zip(args_list, K_list):
        fl_args = [a for a, t in zip(args, prim.ins) if t in FLOAT_T]
        special_in = any((not math.isfinite(a)) or a == 0 for a in fl_args)
        G = None
        if not special_in:
            fargs = [F(a) if t in FLOAT_T else F(int(a)) for a, t in zip(args, prim.ins)]
            try:
                if prim.family == "real_mixed":
                    G = mixed_value(prim.extra["name"], args, fargs)
                elif sem == "convert":
                    G = fargs[0]
                elif sem == "copysign":
                    G = abs(fargs[0]) if math.copysign(1.0, args[1]) > 0 else -abs(fargs[0])
                elif sem == "nextafter":
                    G = None
                else:
                    G = refs.real_value(sem, fargs)
            except Exception as e:  # a reference failure is reported, never hidden
                stats["reference_errors"][type(e).__name__] += 1
                continue
        if G is None or (isinstance(G, float) and not math.isfinite(G)):
            special_compare(prim, sem, out_t, args, K, G, stats)
            continue
        out_of_range = magnitude_class(G, out_t)
        if out_of_range is not None:
            range_compare(out_of_range, G, K, out_t, mode, stats)
            continue
        Gf = refs.as_frac(G)
        if isinstance(Gf, float):
            special_compare(prim, sem, out_t, args, K, Gf, stats)
            continue
        R = X.round_frac(Gf, out_t, mode)
        stats["n"] += 1
        if isinstance(R, float):                          # overflow in the declared rounding
            ok = (math.isinf(K) and math.isinf(R) and (K > 0) == (R > 0)) or (math.isnan(R) and math.isnan(K))
            stats["overflow_cases"] += 1
            stats["overflow_agree"] += int(ok)
            stats["cr"] += int(ok)
            continue
        if not math.isfinite(K):
            stats["nonfinite_where_finite"] += 1
            continue
        stats["cr"] += int(F(K) == R)
        u = X.ulp_at(Gf, out_t)
        e = (F(K) - Gf) / u
        ef = float(e)
        stats["sum_ulp"] += ef
        stats["sum_ulp2"] += ef * ef
        stats["max_abs_ulp"] = max(stats["max_abs_ulp"], abs(ef))
        if Gf != 0:
            stats["sum_rel"] += float((F(K) - Gf) / abs(Gf))
        if R != 0 and K == 0:
            stats["flushed_to_zero"] += 1
        if any(a != 0 and math.isfinite(a) and X.is_subnormal(a, t) for a, t in zip(args, prim.ins) if t in FLOAT_T):
            stats["subnormal_inputs"] += 1


def magnitude_class(G, out_t):
    """'over' / 'under' when |G| is far outside the format's range (decided without building a huge rational)."""
    import mpmath as mp
    p, emin, emax, _ = X.FORMATS[out_t]
    if isinstance(G, F):
        if G == 0:
            return None
        e = G.numerator.bit_length() - G.denominator.bit_length() if G > 0 else \
            (-G).numerator.bit_length() - G.denominator.bit_length()
    elif isinstance(G, mp.mpf):
        if G == 0 or not mp.isfinite(G):
            return None
        e = int(mp.mag(G))
    else:
        return None
    if e > emax + 3:
        return "over"
    if e < emin - p - 3:
        return "under"
    return None


def range_compare(kind, G, K, out_t, mode, stats):
    """Declared result far beyond the largest finite value or far below half the smallest subnormal."""
    import mpmath as mp
    p, emin, emax, has_inf = X.FORMATS[out_t]
    neg = (G < 0)
    stats["n"] += 1
    if kind == "over":
        down = mode == "rz" or (mode == "ru" and neg) or (mode == "rd" and not neg)
        exp_val = (-1 if neg else 1) * (float(X.max_finite(out_t)) if (down or not has_inf) else math.inf)
        ok = K == exp_val or (not has_inf and math.isnan(K))
        stats["overflow_cases"] += 1
        stats["overflow_agree"] += int(ok)
        stats["cr"] += int(ok)
        return
    tiny = float(X.ulp_at(F(0), out_t))
    away = (mode == "ru" and not neg) or (mode == "rd" and neg)
    exp_val = (-tiny if neg else tiny) if away else (-0.0 if neg else 0.0)
    stats["cr"] += int(K == exp_val)
    stats["underflow_cases"] = stats.get("underflow_cases", 0) + 1
    e = float((mp.mpf(K) - G) / mp.mpf(tiny))
    stats["sum_ulp"] += e
    stats["sum_ulp2"] += e * e
    stats["max_abs_ulp"] = max(stats["max_abs_ulp"], abs(e))


def mixed_value(name, args, fargs):
    if name in ("ldexp", "scalbn"):
        return fargs[0] * F(2) ** int(args[1])
    if name == "pow":
        if fargs[0] == 0:
            return None
        return fargs[0] ** int(args[1])
    if name == "jn":
        import mpmath as mp
        return mp.besselj(int(args[0]), refs._m(fargs[1]))
    if name == "yn":
        import mpmath as mp
        if fargs[1] <= 0:
            return None
        return mp.bessely(int(args[0]), refs._m(fargs[1]))
    raise KeyError(name)


def cast_np(v: float, t: str) -> float:
    return float(fmt_round_array(np.array([v]), t)[0]) if t in FLOAT_T else v


def special_compare(prim, sem, out_t, args, K, G, stats):
    """Class agreement (nan / +-inf / signed zero / finite) with IEEE float64 behaviour of the declared function."""
    stats["special_n"] += 1
    fn = refs.NP.get(sem) if sem != "convert" else (lambda a: a)
    if prim.family == "real_mixed" or fn is None:
        stats["special_no_reference"] += 1
        return
    try:
        with np.errstate(all="ignore"):
            if sem == "convert":
                ref = args[0]
            else:
                ref = float(fn(*[np.float64(a) if t in FLOAT_T else a for a, t in zip(args, prim.ins)]))
    except Exception:
        stats["special_no_reference"] += 1
        return
    ref = cast_np(ref, out_t)

    def cls(x):
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "+inf" if x > 0 else "-inf"
        if x == 0:
            return "+0" if math.copysign(1, x) > 0 else "-0"
        return "finite"
    ck, cr = cls(K), cls(ref)
    if ck == cr and (ck != "finite" or abs(K - ref) <= 2 * float(X.ulp_at(F(ref), out_t))):
        stats["special_agree"] += 1
    else:
        key = f"{['%r' % a for a in args]} -> device {K!r}, ieee {ref!r}"
        if ck in ("+0", "-0") and cr in ("+0", "-0"):
            stats["zero_sign_disagree"] += 1
        stats["special_disagree"] += 1
        if len(stats["special_examples"]) < 6:
            stats["special_examples"].append(key)


def new_stats():
    return {"n": 0, "cr": 0, "sum_ulp": 0.0, "sum_ulp2": 0.0, "max_abs_ulp": 0.0, "sum_rel": 0.0,
            "flushed_to_zero": 0, "subnormal_inputs": 0, "overflow_cases": 0, "overflow_agree": 0,
            "nonfinite_where_finite": 0, "special_n": 0, "special_agree": 0, "special_disagree": 0,
            "zero_sign_disagree": 0, "special_no_reference": 0, "special_examples": [],
            "reference_errors": Counter()}


def finish(stats):
    n = stats["n"]
    out = {k: v for k, v in stats.items() if k not in ("sum_ulp", "sum_ulp2", "sum_rel", "reference_errors")}
    if n:
        m = stats["sum_ulp"] / n
        var = max(stats["sum_ulp2"] / n - m * m, 0.0)
        out.update(cr_share=stats["cr"] / n, mean_ulp=m, se_mean_ulp=math.sqrt(var / n) if n > 1 else None,
                   rms_ulp=math.sqrt(stats["sum_ulp2"] / n), mean_rel=stats["sum_rel"] / n)
    if stats["reference_errors"]:
        out["reference_errors"] = dict(stats["reference_errors"])
    return out


def exact_compare(prim, args_list, K_lists, stats, declared):
    """Integer / predicate / bit results: exact equality; undefined results counted apart with device patterns."""
    for j, args in enumerate(args_list):
        try:
            D = declared(args)
        except Exception as e:
            stats["reference_errors"][type(e).__name__] += 1
            continue
        Ds = D if isinstance(D, list) else [D]
        for i, d in enumerate(Ds):
            K = K_lists[i][j]
            if isinstance(d, refs.UB):
                stats["undefined"][d.why] += 1
                continue
            stats["n"] += 1
            if isinstance(d, tuple) and d[0] == "truthy":
                ok = (K != 0) == bool(d[1])
            else:
                ok = K == d
            if not ok and any(isinstance(a, float) and a != 0 and math.isfinite(a) and abs(a) < 2.0 ** -126
                              for a in args):
                try:
                    flushed = declared(tuple(0.0 if (isinstance(a, float) and abs(a) < 2.0 ** -126) else a
                                             for a in args))
                    fd = flushed[0] if isinstance(flushed, list) else flushed
                    if isinstance(fd, refs.UB) or K == (fd[1] if isinstance(fd, tuple) else fd):
                        stats["ftz_explained"] += 1
                except Exception:
                    pass
            stats["eq"] += int(ok)
            if not ok and len(stats["examples"]) < 6:
                stats["examples"].append(f"{list(args)} -> device {K}, declared {d}")


def new_exact():
    return {"n": 0, "eq": 0, "undefined": Counter(), "examples": [], "reference_errors": Counter(), "ftz_explained": 0}


def finish_exact(s):
    out = {"n": s["n"], "eq": s["eq"], "eq_share": (s["eq"] / s["n"]) if s["n"] else None, "examples": s["examples"],
           "mismatch_explained_by_input_flush": s["ftz_explained"]}
    if s["undefined"]:
        out["undefined"] = dict(s["undefined"])
    if s["reference_errors"]:
        out["reference_errors"] = dict(s["reference_errors"])
    return out


# ----------------------------------------------------------------------------- one primitive
def bits_declared(prim, args):
    a = args[0]
    name = prim.extra.get("name") or prim.op
    src, dst = prim.ins[0], prim.outs[0]
    if name in ("hiloint2double",):
        hi, lo = args[0] & 0xFFFFFFFF, args[1] & 0xFFFFFFFF
        return ("float_bits", (hi << 32) | lo)
    if name in ("double2loint", "double2hiint"):
        b = float_bits(a, "f64")
        return wrap(b & 0xFFFFFFFF if name == "double2loint" else b >> 32, 32)
    if src in FLOAT_T:
        return wrap(float_bits(a, src), INT_W[dst], not dst.startswith("u"))
    return ("float_bits", a & ((1 << INT_W[src]) - 1))


def run_prim(prim: P.Prim, seed_text: str) -> dict:
    rec = {"pid": prim.pid, "source": prim.source, "op": prim.op, "attr": prim.attr, "sig": prim.sig,
           "family": prim.family, "sem": prim.sem, "mode": prim.mode}
    try:
        text, launch = module_text(prim)
    except tb.Unbuildable as e:
        rec.update(status="no template", reason=str(e))
        return rec
    rec["ttir_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    t0 = time.time()
    try:
        k = compile_text(text, {"num_warps": 4})
    except Exception as e:
        rec.update(status="not compiled", error=str(e).splitlines()[0][:200])
        return rec
    rec["compile_s"] = round(time.time() - t0, 2)
    import compile_sweep as cs
    rec["ptx_opcodes"] = {o: n for o, n in sorted(cs.ptx_opcodes(k.asm["ptx"]).items())
                          if not o.startswith(("ld.", "st.", "mov.", "cvta", "ret", "bra"))}
    if launch == "compile-only":
        rec["status"] = "compiled (not executed)"
        return rec
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], "little"))
    if prim.family == "dot":
        rec.update(run_dot(prim, k, rng))
        rec["status"] = "measured"
        return rec
    if launch == "dot_scaled":
        rec.update(run_dot_scaled(k, rng))
        rec["status"] = "measured"
        return rec
    if prim.family == "literal":
        out = run_elementwise(k, prim, [])
        v = out[0][0]
        if prim.sig in FLOAT_T:
            R = X.round_frac(F(prim.attr), prim.sig, "rne")
            rec["literal"] = {"device": repr(v), "declared_decimal": prim.attr, "rn_of_decimal": repr(float(R)) if not isinstance(R, float) else repr(R),
                              "equal_rn": (float(R) == v), "relative_rounding": float((F(v) - F(prim.attr)) / F(prim.attr)) if v == v and math.isfinite(v) else None}
        else:
            rec["literal"] = {"device": v, "declared": prim.attr}
        rec["status"] = "measured"
        return rec
    if prim.family == "scalar":
        vals = int_dist(rng, "symmetric", prim.ins[0], 256)
        import torch
        x = to_device(vals, prim.ins[0])
        o = torch.zeros_like(x)
        k[(len(vals), 1, 1)](x, o, len(vals))
        dev = from_device(o, prim.ins[0])
        rec["roundtrip_equal"] = sum(int(a == b) for a, b in zip(vals, dev))
        rec["n"] = len(vals)
        rec["status"] = "measured"
        return rec
    per = {}
    blocks = [(d, draw_operands(prim, rng, d, N_PER_DIST)) for d in DISTS]
    sp = special_operands(prim, rng)
    if sp:
        blocks.append(("specials", sp))
    for d, ops in blocks:
        dev = run_elementwise(k, prim, ops, raw=prim.family in ("bits", "select"))
        args_list = list(zip(*ops))
        if prim.outs[0] in FLOAT_T and prim.family in ("real", "real_mixed", "convert_ff", "convert_if") \
                and not (prim.family == "convert_if"):
            st = new_stats()
            sem = "convert" if prim.family == "convert_ff" else prim.sem
            float_compare(prim, sem, prim.mode, prim.outs[0], args_list, dev[0], st)
            per[d] = finish(st)
        elif prim.family == "convert_if":
            st = new_stats()
            fake = P.Prim(prim.pid, prim.source, prim.op, prim.attr, prim.sig, ["f64"], prim.outs, "convert", "convert_ff")
            float_compare(fake, "convert", prim.mode, prim.outs[0], [(float(a[0]),) if abs(a[0]) < 2 ** 1000 else (0.0,)
                                                                     for a in args_list], dev[0], st)
            # integer inputs are exact: route through the exact rational path instead of float64
            st2 = new_stats()
            unsigned_src = prim.op == "arith.uitofp" or prim.ins[0].startswith("u")
            w_src = INT_W[prim.ins[0]]
            for a, K in zip(args_list, dev[0]):
                v = int(a[0]) & ((1 << w_src) - 1) if unsigned_src else int(a[0])   # uitofp reads the bits unsigned
                a = (v,)
                R = X.round_frac(F(int(a[0])), prim.outs[0], prim.mode)
                st2["n"] += 1
                st2["cr"] += int((not isinstance(R, float) and math.isfinite(K) and F(K) == R)
                                 or (isinstance(R, float) and K == R))
                if not isinstance(R, float) and math.isfinite(K) and a[0] != 0:
                    e = float((F(K) - F(int(a[0]))) / X.ulp_at(F(int(a[0])), prim.outs[0]))
                    st2["sum_ulp"] += e
                    st2["sum_ulp2"] += e * e
                    st2["max_abs_ulp"] = max(st2["max_abs_ulp"], abs(e))
            per[d] = finish(st2)
        elif prim.family == "convert_fi":
            st = new_exact()

            def decl(args, prim=prim):
                a = args[0]
                if not math.isfinite(a):
                    return refs.UB("non-finite input")
                fa = F(a)
                r = {"rne": refs._round_half_even, "rz": lambda x: F(math.trunc(x)), "rd": lambda x: F(math.floor(x)),
                     "ru": lambda x: F(math.ceil(x))}[prim.mode if prim.source == "libdevice" else "rz"](fa)
                t = prim.outs[0]
                w = INT_W[t]
                uns = t.startswith("u") or prim.op == "arith.fptoui" or "uint" in prim.extra.get("name", "") \
                    or "ull" in prim.extra.get("name", "")
                lo, hi = (0, 2 ** w - 1) if uns else (-(2 ** (w - 1)), 2 ** (w - 1) - 1)
                if not lo <= r <= hi:
                    return refs.UB("out of range")
                return int(r) if not uns else int(r)
            dv = [[wrap(x, INT_W[prim.outs[0]], not (prim.op == "arith.fptoui" or "uint" in prim.extra.get("name", "")
                                                    or "ull" in prim.extra.get("name", "")))
                   for x in dev[0]]]
            exact_compare(prim, args_list, dv, st, decl)
            per[d] = finish_exact(st)
        elif prim.family in ("int",):
            st = new_exact()
            t = prim.sig
            if prim.sem in ("mulsi_extended", "mului_extended", "addui_extended"):
                exact_compare(prim, args_list, dev, st, lambda a: int_declared_multi(prim.sem, a, t))
            else:
                exact_compare(prim, args_list, dev, st, lambda a: int_declared(prim.sem, a, t, prim.attr))
            per[d] = finish_exact(st)
        elif prim.family == "convert_ii":
            st = new_exact()
            a_t, b_t = prim.sig.split("->")
            wb = INT_W[b_t]

            def decl(args, op=prim.op, a_t=a_t, wb=wb):
                x = args[0]
                if op == "arith.extsi" or op == "arith.trunci":
                    return wrap(wrap(x, INT_W[a_t]), wb)
                return wrap(x & ((1 << INT_W[a_t]) - 1), wb)
            exact_compare(prim, args_list, dev, st, decl)
            per[d] = finish_exact(st)
        elif prim.family == "pred":
            st = new_exact()
            exact_compare(prim, args_list, dev, st, lambda a: pred_declared(prim.sem, a, prim.ins[0]))
            per[d] = finish_exact(st)
        elif prim.family == "select":
            st = new_exact()
            if prim.sig in FLOAT_T:
                # a NaN operand is compared as a class: its payload is not under the harness's control
                nan_bits = {"f16": (0x7C00, 0x3FF), "bf16": (0x7F80, 0x7F), "f32": (0x7F800000, 0x7FFFFF),
                            "f64": (0x7FF0000000000000, 0xFFFFFFFFFFFFF)}[prim.sig]
                dev = [[("nan",) if ((b & nan_bits[0]) == nan_bits[0] and (b & nan_bits[1])) else b for b in dev[0]]]
                exact_compare(prim, args_list, dev, st,
                              lambda a: ("nan",) if (isinstance(a[1] if a[0] else a[2], float)
                                                     and math.isnan(a[1] if a[0] else a[2]))
                              else float_bits(a[1] if a[0] else a[2], prim.sig))
            else:
                exact_compare(prim, args_list, dev, st, lambda a: wrap(a[1] if a[0] else a[2], INT_W[prim.sig])
                              if prim.sig != "i1" else (a[1] if a[0] else a[2]))
            per[d] = finish_exact(st)
        elif prim.family == "bits":
            st = new_exact()
            out_t = prim.outs[0]
            if out_t in FLOAT_T:
                dv = dev

                def decl(args):
                    v = bits_declared(prim, args)
                    return v[1] if isinstance(v, tuple) else v
            else:
                dv = dev

                def decl(args):
                    return bits_declared(prim, args)
            exact_compare(prim, args_list, dv, st, decl)
            per[d] = finish_exact(st)
        elif prim.family == "libint":
            st = new_exact()
            name = prim.extra["name"]
            exact_compare(prim, args_list, dev, st, lambda a: libint_declared(name, a, prim.ins, prim.outs[0]))
            per[d] = finish_exact(st)
        else:
            per[d] = {"skipped": f"no comparison for family {prim.family}"}
    rec["per_distribution"] = per
    rec["status"] = "measured"
    return rec


def run_dot(prim, k, rng, trials=48):
    import torch
    a_t, b_t, _ = tb.dot_attrs((prim.op, prim.attr, prim.sig))
    out = {}
    for d in ("symmetric", "same-sign", "small-integer", "wide"):
        st = new_stats()
        for _ in range(trials):
            A = fmt_round_array(float_dist(rng, d, a_t if a_t != "f32" else "f16", 256) if d == "wide"
                                else float_dist(rng, d, a_t, 256), a_t).reshape(16, 16)
            B = fmt_round_array(float_dist(rng, d, b_t if b_t != "f32" else "f16", 256) if d == "wide"
                                else float_dist(rng, d, b_t, 256), b_t).reshape(16, 16)
            C0 = np.zeros((16, 16))
            tA, tB = to_device(A.ravel(), a_t), to_device(B.ravel(), b_t)
            tC0 = torch.zeros(256, dtype=torch.float32, device="cuda")
            tC = torch.zeros(256, dtype=torch.float32, device="cuda")
            k[(1, 1, 1)](tA, tB, tC0, tC)
            torch.cuda.synchronize()
            Kv = tC.double().cpu().numpy().reshape(16, 16)
            if not (np.isfinite(A).all() and np.isfinite(B).all()):
                st["nonfinite_inputs"] = st.get("nonfinite_inputs", 0) + 1
                continue
            Af = [[F(float(x)) for x in row] for row in A]
            Bf = [[F(float(x)) for x in row] for row in B]
            for i in range(16):
                for j in range(16):
                    G = sum((Af[i][kk] * Bf[kk][j] for kk in range(16)), F(0))
                    K = float(Kv[i, j])
                    R = X.round_frac(G, "f32", "rne")
                    st["n"] += 1
                    if isinstance(R, float) or not math.isfinite(K):
                        st["nonfinite_where_finite"] += int(not isinstance(R, float))
                        continue
                    st["cr"] += int(F(K) == R)
                    e = float((F(K) - G) / X.ulp_at(G, "f32"))
                    st["sum_ulp"] += e
                    st["sum_ulp2"] += e * e
                    st["max_abs_ulp"] = max(st["max_abs_ulp"], abs(e))
                    if G != 0:
                        st["sum_rel"] += float((F(K) - G) / abs(G))
        out[d] = finish(st)
    return {"per_distribution": out}


def run_dot_scaled(k, rng, trials=16, m=32, n=32, kk=64):
    """e5m2 operands with random e8m0 scales (2^-4 .. 2^4); declared value: exact sum of scaled products."""
    import torch
    st = new_stats()
    for _ in range(trials):
        A = fmt_round_array(float_dist(rng, "symmetric", "f8e5m2", m * kk), "f8e5m2").reshape(m, kk)
        B = fmt_round_array(float_dist(rng, "symmetric", "f8e5m2", kk * n), "f8e5m2").reshape(kk, n)
        SA = rng.integers(123, 132, (m, kk // 32))
        SB = rng.integers(123, 132, (n, kk // 32))
        tA, tB = to_device(A.ravel(), "f8e5m2"), to_device(B.ravel(), "f8e5m2")
        tSA = torch.tensor((SA.ravel() - 256 * (SA.ravel() > 127)).tolist(), dtype=torch.int8).cuda()
        tSB = torch.tensor((SB.ravel() - 256 * (SB.ravel() > 127)).tolist(), dtype=torch.int8).cuda()
        tC = torch.zeros(m * n, dtype=torch.float32, device="cuda")
        k[(1, 1, 1)](tA, tB, tSA, tSB, tC)
        torch.cuda.synchronize()
        Kv = tC.double().cpu().numpy().reshape(m, n)
        for i in range(m):
            for j in range(n):
                G = sum((F(float(A[i, t])) * F(2) ** int(SA[i, t // 32] - 127) * F(float(B[t, j])) *
                         F(2) ** int(SB[j, t // 32] - 127) for t in range(kk)), F(0))
                K_ = float(Kv[i, j])
                R = X.round_frac(G, "f32", "rne")
                st["n"] += 1
                if isinstance(R, float) or not math.isfinite(K_):
                    continue
                st["cr"] += int(F(K_) == R)
                e = float((F(K_) - G) / X.ulp_at(G, "f32"))
                st["sum_ulp"] += e
                st["sum_ulp2"] += e * e
                st["max_abs_ulp"] = max(st["max_abs_ulp"], abs(e))
                if G != 0:
                    st["sum_rel"] += float((F(K_) - G) / abs(G))
    return {"per_distribution": {"symmetric": finish(st)}}


def by_pid():
    return {p.pid: p for p in P.all_prims()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--filter", default="")
    ap.add_argument("--out", default=str(HERE / "results/catalog.json"))
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--gpus", default="0,1,2,3")
    a = ap.parse_args()
    if a.pid:
        prims = by_pid()
        rec = run_prim(prims[a.pid], a.pid)
        print(json.dumps(rec, default=str))
        return
    pids = [p.pid for p in P.all_prims() if a.filter in p.pid]
    gpus = a.gpus.split(",")

    def child(i_pid):
        i, pid = i_pid
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[i % len(gpus)])
        try:
            p = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), "--pid", pid], capture_output=True,
                               text=True, timeout=1200, env=env)
        except subprocess.TimeoutExpired:
            return {"pid": pid, "status": "harness timeout"}
        lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
        if p.returncode == 0 and lines:
            rec = json.loads(lines[-1])
            if rec.get("status") == "not compiled":
                rec["diagnostics"] = [l.split(": ", 1)[-1][:200] for l in p.stderr.splitlines()
                                      if "error:" in l or "LLVM ERROR" in l][:3]
            return rec
        err = [l for l in p.stderr.splitlines() if l.strip()]
        return {"pid": pid, "status": "child failed", "returncode": p.returncode,
                "error": [l for l in err if "rror" in l][-3:] or err[-3:]}
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        recs = list(ex.map(child, list(enumerate(pids))))
    out = {"schema": "composition-study-catalog-v1", "triton": "3.6.0", "target": "cuda sm_86 (RTX A6000)",
           "options": {"num_warps": 4}, "n_per_distribution": N_PER_DIST, "distributions": DISTS,
           "primitives": len(recs), "status": dict(Counter(r.get("status") for r in recs)),
           "elapsed_s": round(time.time() - t0, 1), "records": recs}
    Path(a.out).write_text(json.dumps(out, indent=1, default=str, ensure_ascii=False))
    print(out["status"], out["elapsed_s"])


if __name__ == "__main__":
    main()
