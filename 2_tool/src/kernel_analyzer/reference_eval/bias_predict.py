"""Compositional bias prediction for a Triton launch, ex ante (no device run of the kernel).

The emulator (emulate.py) reproduces the device bits of a launch; this module predicts the *bias* of the kernel's
outputs against the exact value of the same program, from local contracts and the composition rules alone:

  E[K - G] ~ sum over rounding sites s of J_s(x) * mu_s(x)          (first order, averaged over the inputs)

where G is the exact program value, J_s the derivative of an output with respect to site s's result (forward mode
through the evaluator's tangent rules) and mu_s the site's *contract*: the conditional mean of its residual given
coarse features of its exact operands.  Every value is kept exact (float64); the tangent channel carries the interval
[delta - R, delta + R]: delta is the predicted mean error, R the first-order worst-case radius (sum of |J_s| times
the half-width of site s's residual around its mean).

Contracts (conditions -> local deviation relation; the composition study established which lowering each primitive
has on the locked build, 3.6.0 / sm_86):

* correctly rounded RN site (IEEE add / sub / mul / div / sqrt / fma, conversions, integer to float -- the catalog
  establishes that the device rounds these correctly): the local relation is the rounding function r(t) = RN(t) - t,
  evaluated at the estimate t of the exact result at the device operands (each operand: its exact value plus the
  mean error predicted so far, on its format's grid).  It carries absorption, rounding up to one ulp, crossings into
  a coarser binade, ties, short results and the asymmetric cell at each power of two (the B2 term) at the operating
  points the data visits; exact where the result is representable or the input carries no more bits than the target;
* toward zero / down / up: -sign * u / 2 (resp. -u / 2, +u / 2) times (1 - 2^-q), q the dropped significant bits;
  nearest-away: + sign * u / 2 * 2^-q (the ties);
* approximate primitives (div.full, ex2.approx, libdevice functions that are not correctly rounded, ...): the
  mean signed error per cell -- (sign, binade, significand bucket) of a unary input, (sign, binade) of a binary op's
  output -- measured on the primitive alone (contract table, 1_experiments/composition_study/contracts.py), per
  significand class of the operands (unary: uniform / 11-bit / 8-bit grid;
  binary: power of two / short / long per operand), half-width its largest error;
* software f8E5M2 conversion of sm_86 (cut toward zero to f16, then 3 bits with ties away): its function evaluated at
  the operand estimate, like an RN site;
* tensor-core dot: tf32 operands cut toward zero to 11 bits (exact where the operand carries <= 11 bits); per group
  of 4 / 8 products and the accumulator, every term with bits below 2^(E - 24) cut toward zero, the group sum cut
  toward zero to 24 bits (the rule of tensor_core.py, one contract per cut), the dropped bits averaged between the
  cut and each term's granularity (a product's lowest set bit; the accumulator's granularity tracked along the
  chain); fp32 ieee dot: an fma chain (absorption rule per step);
* sum reduction / scan: the device order of the composition study (layouts.py, scan_order.py) with the absorption
  rule per combine.

Information kept while composing: the exact value, the tangent interval (delta, R), and the significand budget of
each operand (how many significant bits its device value can carry: from its producer -- a load keeps the data's
own bits, a conversion or rounding site its target precision, layout ops pass it on).

Failure conditions (the prediction for the launch is "无法判断" when any applies):

* no contract for an operation (no catalogued lowering, no table entry, no tangent rule) -- the program aborts;
* a discontinuous threshold may flip: a compare whose margin is within R of its operands feeding a select whose
  branches differ by more than R, a float-to-int conversion or floor / ceil / round within R of a jump;
* the first-order model leaves its range: R / |G| > 2^-4 on more than 1% of an output's elements;
* subnormal or non-finite values at rounding sites (FTZ, overflow) on more than 0.1% of a site's lanes;
* a float atomic fold (the fold order is not modelled for prediction);
* an approximate primitive outside its characterized cells on more than 1% of its lanes, or evaluated at fewer than
  64 distinct operand values over at least 1024 lanes (a few fixed operating points, which a cell mean does not
  describe).
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np

from . import intervals as iv
from . import scan_order, tensor_core
from .ttir_eval import (ST_OK, TV, KernelReferenceEvaluator, NumericMode, ProgramAbort, _ftv)
from .ttir_mapping import LIBDEVICE, LIBDEVICE_ROUNDING, inline_asm_internal, recognize_combiner
from .ttir_parser import TOp

CONTRACTS_JSON = Path(__file__).resolve().parent / "data" / "bias_contracts.json.gz"
APPROX_OPS = {"arith.divf", "arith.remf", "math.exp", "math.exp2", "math.log", "math.log2", "math.log10",
              "math.log1p", "math.expm1", "math.sqrt", "math.rsqrt", "math.sin", "math.cos", "math.tan",
              "math.tanh", "math.erf", "math.powf", "math.atan", "math.sinh", "math.cosh"}
PRESERVING = {"tt.splat", "tt.broadcast", "ttg.convert_layout", "tt.expand_dims", "tt.reshape", "tt.trans",
              "arith.negf", "math.absf", "tt.join", "tt.cat", "tt.split"}
FLIP_SOURCES = {"floor", "ceil", "trunc", "round", "roundeven"}
LARGE_RELATIVE = 2.0 ** -4
# ----------------------------------------------------------------------------- local relations (pure functions)

def precision(fmt: str) -> int:
    return iv.FLOAT_FORMATS[fmt][0]


def ulp_of(x, fmt: str) -> np.ndarray:
    """Spacing of ``fmt`` at |x| (subnormal range: the smallest spacing)."""
    p, emin, _, _ = iv.FLOAT_FORMATS[fmt]
    ax = np.abs(np.asarray(x, dtype=np.float64))
    _, e = np.frexp(np.where((ax > 0) & np.isfinite(ax), ax, 1.0))
    return np.ldexp(1.0, np.maximum(e - 1, emin) - (p - 1))


def sigbits(x) -> np.ndarray:
    """Significant bits of float64 values (0 for zero)."""
    ax = np.abs(np.asarray(x, dtype=np.float64))
    m, _ = np.frexp(np.where(np.isfinite(ax), ax, 0.0))
    i = np.ldexp(m, 53).astype(np.int64)
    low = i & -i
    tz = np.where(i > 0, np.log2(np.where(low > 0, low, 1)).astype(np.int64), 53)
    return np.where(i > 0, 53 - tz, 0)


def representable(x, fmt: str) -> np.ndarray:
    r, over = iv.round_nearest_even(np.asarray(x, dtype=np.float64), fmt)
    return (r == x) & ~over


def lowbit_exp(x) -> np.ndarray:
    """Exponent of the lowest set bit of float64 values (a large number for zero)."""
    x = np.asarray(x, dtype=np.float64)
    _, e = np.frexp(np.where(x == 0, 1.0, np.abs(x)))
    return np.where(x == 0, 1 << 20, (e - 1) - sigbits(x) + 1)


def rn_contract(t, fmt, exact_inputs=False, in_bits=None):
    """(mean, half-width, variance) of a correctly rounded RN-even site to ``fmt``: the local relation is the rounding
    function itself, r(t) = RN(t) - t, evaluated at the estimate t of the exact result at the device operands.
    It carries every effect of the grid at the operating points the data visits: absorption, rounding up to one ulp,
    crossings into a coarser binade, ties, short results, and the asymmetric cell of each power of two (whose
    correlation with the value is the B2 term of the bias identity).  Exact where t is representable or the input
    carries no more significant bits than the target; variance 0 where the operands are exact (the residual is
    then determined), u^2 / 12 otherwise."""
    t = np.asarray(t, dtype=np.float64)
    u = ulp_of(t, fmt)
    exact = representable(t, fmt)
    if in_bits is not None:
        exact = exact | (np.asarray(in_bits) <= precision(fmt))
    r = iv.round_nearest_even(t, fmt)[0] - t
    mu = np.where(exact, 0.0, r)
    half = np.where(exact, 0.0, u / 2)
    var = np.where(exact | np.asarray(exact_inputs), 0.0, u * u / 12)
    return mu, half, var


def e5m2_software(x, from_f32: bool) -> np.ndarray:
    """The sm_86 software conversion to f8E5M2 (composition study): an f32 operand is first cut toward zero to f16;
    then the value is rounded to 3 significant bits with ties away from zero (subnormal spacing 2^-16).  Finite
    results only (overflow is left to the caller's special-value check)."""
    x = np.asarray(x, dtype=np.float64)
    if from_f32:
        x = iv.round_directed(x, "f16", "rtz")[0]
    _, e = np.frexp(np.where(x == 0, 1.0, np.abs(x)))
    scale = 2 - np.maximum(e - 1, -14)                    # 3 significant bits, the subnormal grid below 2^-14
    m = np.ldexp(np.abs(x), scale)
    return np.where(x == 0, x, np.sign(x) * np.ldexp(np.floor(m + 0.5), -scale))


def directed_contract(x, fmt, mode, in_bits=None):
    """Toward zero ("rtz"), down ("rd"), up ("ru"), nearest-away ("rna") of exact values x."""
    x = np.asarray(x, dtype=np.float64)
    u = ulp_of(x, fmt)
    p = precision(fmt)
    q = (np.asarray(in_bits) - p) if in_bits is not None else np.full(x.shape, 30)
    exact = representable(x, fmt) | (q <= 0)
    frac = 1.0 - np.ldexp(1.0, -np.clip(q, 1, 60))          # discrete dropped bits: mean (1 - 2^-q) / 2 of an ulp
    if mode == "rtz":
        mu = -np.sign(x) * u / 2 * frac
    elif mode == "rd":
        mu = -u / 2 * frac
    elif mode == "ru":
        mu = u / 2 * frac
    elif mode == "rna":
        mu = np.sign(x) * u / 2 * np.ldexp(1.0, -np.clip(q, 1, 60))
    else:
        raise ValueError(mode)
    return np.where(exact, 0.0, mu), np.where(exact, 0.0, u / 2), np.where(exact, 0.0, u * u / 12)


def add_contract(a, b, fmt, a_grid=True, b_grid=True, exact_inputs=False):
    """RN(a + b): the rounding relation at the device operand estimates -- an operand of the output type on the
    output grid, a fused product exact (rn_contract)."""
    a, b = np.broadcast_arrays(np.asarray(a, np.float64), np.asarray(b, np.float64))
    rn = lambda x: iv.round_nearest_even(x, fmt)[0]                    # noqa: E731
    t = np.where(a_grid, rn(a), a) + np.where(b_grid, rn(b), b)
    return rn_contract(t, fmt, exact_inputs)


def mma_contract(a, b, c, fmt, a_bits=None, b_bits=None):
    """Mean residual, half-width and variance of a tensor-core dot under the mma rule (tensor_core.py), from exact
    operands.  a (..., M, K), b (..., K, N), c (..., M, N); returns arrays of shape (..., M, N).

    Each cut drops the bits of a term (or of the group sum) below the cut; the dropped bits are taken as uniform
    between the cut and the lowest set bit the term can carry -- its granularity: the accumulator carries 24 bits at
    its own exponent, a product the lowest set bits of its two operands (the operand estimate: the exact value on its
    format's grid).  A term whose granularity is at or above the cut is not changed."""
    G = tensor_core.GROUP[fmt]
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    shape = np.broadcast_shapes(np.shape(c), a.shape[:-1] + b.shape[-1:])
    mu, half, var = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    src = "f32" if fmt == "tf32" else fmt
    a = iv.round_nearest_even(a, src)[0]
    b = iv.round_nearest_even(b, src)[0]
    if fmt == "tf32":                                # operand conversion: cut toward zero to 11 bits
        mu_a, h_a, v_a = directed_contract(a, "tf32", "rtz", a_bits)
        mu_b, h_b, v_b = directed_contract(b, "tf32", "rtz", b_bits)
        mu = mu + mu_a @ b + a @ mu_b
        half = half + h_a @ np.abs(b) + np.abs(a) @ h_b
        var = var + v_a @ (b * b) + (a * a) @ v_b
        a = np.where(h_a > 0, tensor_core.round_sig(a, 11, "rz"), a)   # the operands the tensor core sees
        b = np.where(h_b > 0, tensor_core.round_sig(b, 11, "rz"), b)
    emin = tensor_core.MIN_NORMAL_EXP[fmt]

    def expo(x):
        _, e = np.frexp(np.where(x == 0, 1.0, x))
        return e.astype(np.int64) - 1
    ea, eb = np.maximum(expo(a), emin), np.maximum(expo(b), emin)
    la, lb = lowbit_exp(a), lowbit_exp(b)
    acc = np.array(np.broadcast_to(c, shape), dtype=np.float64)
    K = a.shape[-1]
    big = 1 << 20
    # the incoming accumulator: 24 significant bits with unknown low bits (a previous dot's result) -- carried
    # through the chain as its granularity, which a sum of short products keeps coarse
    g_acc = np.where(acc == 0, big, expo(acc) - 23)
    set_acc = np.zeros(shape, dtype=bool)
    for s0 in range(0, K, G):
        P = a[..., :, s0:s0 + G, None] * b[..., None, s0:s0 + G, :]
        pe = ea[..., :, s0:s0 + G, None] + eb[..., None, s0:s0 + G, :]
        pg = la[..., :, s0:s0 + G, None] + lb[..., None, s0:s0 + G, :]
        e_acc = expo(acc)
        T = np.concatenate([acc[..., :, None, :], P], axis=-2)
        nz = T != 0
        e = np.concatenate([e_acc[..., :, None, :], pe], axis=-2)
        gran = np.concatenate([np.maximum(g_acc, e_acc - 23)[..., :, None, :], pg], axis=-2)
        gran = np.where(nz, gran, big)
        # a product's granularity is its lowest set bit (known to be 1); the accumulator's is its tracked
        # granularity (its bit there known only when it came from a single product): the mean of d dropped bits is
        # 2^(cut-1) for a known bit, 2^(cut-1) (1 - 2^-d) otherwise
        known = np.ones(T.shape, dtype=bool)
        known[..., 0, :] = set_acc
        E = np.where(nz, e, -100000).max(axis=-2)
        q = (E - 23 - tensor_core.EXTRA_BITS)[..., None, :]
        kept = gran >= q
        below = np.abs(T) < np.ldexp(1.0, q)
        k = np.clip(q - gran, 1, 60)
        frac = np.where(known, 1.0, 1.0 - np.ldexp(1.0, -k))
        term_mu = np.where(kept, 0.0, np.where(below, -T, -np.sign(T) * np.ldexp(0.5, q) * frac))
        term_h = np.where(kept | below, 0.0, np.ldexp(0.5, q))
        term_v = np.where(kept | below, 0.0, np.ldexp(1.0, 2 * q) / 12)
        g_after = np.where(kept, gran, np.where(below, big, q))
        known_after = kept & known
        S = T.sum(axis=-2) + term_mu.sum(axis=-2)                      # the cut group sum (mean estimate)
        g_S = g_after.min(axis=-2)
        at_min = g_after == g_S[..., None, :]
        set_S = (at_min.sum(axis=-2) == 1) & np.any(at_min & known_after, axis=-2)
        u_e = expo(S) - 23
        d = np.where((S == 0) | (g_S >= big), 0, u_e - g_S)
        fin_frac = np.where(set_S, 1.0, 1.0 - np.ldexp(1.0, -np.clip(d, 1, 60)))
        fin_mu = np.where(d >= 1, -np.sign(S) * np.ldexp(0.5, u_e) * fin_frac, 0.0)
        fin_h = np.where(d >= 1, np.ldexp(0.5, u_e), 0.0)
        fin_v = np.where(d >= 1, np.ldexp(1.0, 2 * u_e) / 12, 0.0)
        any_nz = nz.any(axis=-2)
        mu = mu + np.where(any_nz, term_mu.sum(axis=-2) + fin_mu, 0.0)
        half = half + np.where(any_nz, term_h.sum(axis=-2) + fin_h, 0.0)
        var = var + np.where(any_nz, term_v.sum(axis=-2) + fin_v, 0.0)
        g_acc = np.where(any_nz, np.where(d >= 1, u_e, g_S), g_acc)
        set_acc = np.where(any_nz, np.where(d >= 1, False, set_S), set_acc)
        acc = acc + P.sum(axis=-2)
    return mu, half, var


def fma_chain_contract(a, b, c):
    """fp32 ieee dot: acc = RN(a_k * b_k + acc) in k order; absorption of a small product into the accumulator."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    acc = np.array(c, dtype=np.float64)
    mu, half, var = np.zeros(acc.shape), np.zeros(acc.shape), np.zeros(acc.shape)
    for k in range(a.shape[-1]):
        prod = a[..., :, k][..., :, None] * b[..., k, :][..., None, :]
        m, h, v = add_contract(acc, prod, "f32", a_grid=True, b_grid=False)
        mu, half, var = mu + m, half + h, var + v
        acc = acc + prod
    return mu, half, var


def sum_tree_contract(vals, fmt, order_fn):
    """Mean residual and half-width of a sum evaluated by ``order_fn(values, combine)`` (the device order) with the
    absorption rule per combine.  vals (..., n) exact values; returns (mu, half) of shape (...) or of the order's
    output shape."""
    def comb_mean(x, y):
        # the operands as the device holds them: exact partial plus the mean error predicted so far (a long
        # low-precision sum drifts away from the exact partials before it meets the next binade boundary)
        c = add_contract(x.real + x.imag, y.real + y.imag, fmt)[0]
        return (x.real + y.real) + 1j * (x.imag + y.imag + c)

    def comb(i):
        def f(x, y):
            c = add_contract(x.real, y.real, fmt)[i]
            return (x.real + y.real) + 1j * (x.imag + y.imag + c)
        return f
    v = np.asarray(vals, dtype=np.complex128)
    return (np.imag(order_fn(v, comb_mean)),) + tuple(np.imag(order_fn(v, comb(i))) for i in (1, 2))


# ----------------------------------------------------------------------------- contract table (approximations)

class ContractTable:
    """Measured local relations of approximate primitives: per key, the mean signed error and the largest error in
    output ulps per (sign, binade) cell of the input (unary) or output (otherwise)."""

    def __init__(self, path: Optional[Path] = None):
        path = Path(path) if path else CONTRACTS_JSON
        if path.exists():
            import gzip
            raw = path.read_bytes()
            self.data = json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)
        else:
            self.data = {"approx": {}, "correctly_rounded": []}
        self.cr = set(self.data.get("correctly_rounded", []))

    @staticmethod
    def key(op: TOp, out_elem: str) -> str:
        if op.name == "tt.extern_elementwise":
            return f"libdevice|{op.attrs.get('symbol', '').strip(chr(34))}|{out_elem}"
        if op.name == "tt.elementwise_inline_asm":
            from .ttir_mapping import _normalize_asm
            return f"asm|{_normalize_asm(op.attrs.get('asm', ''))}|{out_elem}"
        return f"{op.name}|{out_elem}"

    def lookup(self, key: str):
        return self.data["approx"].get(key)

    @staticmethod
    def _code(name: str, sub: int = 1) -> int:
        if name == "z":
            return -1
        if name == "n":
            return -2
        head, _, bucket = name.partition("/")
        return (2 * (int(head[1:]) + 4096) + (head[0] == "-")) * sub + (int(bucket) if bucket else 0)

    @staticmethod
    def codes(x, sub: int = 1) -> np.ndarray:
        """Integer cell codes of float64 values: (2 * (binade + 4096) + negative) * sub + significand bucket;
        -1 zero, -2 non-finite."""
        x = np.asarray(x, dtype=np.float64)
        fin = np.isfinite(x)
        m, e = np.frexp(np.where((x != 0) & fin, np.abs(x), 1.0))
        c = (2 * ((e - 1) + 4096) + (x < 0)) * sub + np.minimum(((2 * m - 1) * sub).astype(np.int64), sub - 1)
        return np.where(~fin, -2, np.where(x == 0, -1, c))

    def _tables(self, key, cls=None, coarse=False):
        """(sorted codes, [mean, worst, rms]) of a key's cells with enough samples, for one significand class
        (None: the main table); ``coarse``: the (sign, binade) cells merged over their significand buckets (the
        fallback where a bucket has too few samples) (cached)."""
        cache = self.__dict__.setdefault("_cache", {})
        if (key, cls, coarse) not in cache:
            entry = self.data["approx"][key]
            sub = entry.get("sub", 1)
            min_n = entry.get("min_n", self.data.get("min_n", 64))
            cells = entry["cells"] if cls is None else entry.get("classes", {}).get(cls, {})
            if coarse:
                merged = {}
                for name, cell in cells.items():
                    head = name.partition("/")[0]
                    n, m, w = cell[0], cell[1], cell[2]
                    r = cell[3] if len(cell) > 3 else w
                    t = merged.setdefault(head, [0, 0.0, 0.0, 0.0])
                    t[0] += n
                    t[1] += n * m
                    t[2] = max(t[2], w)
                    t[3] += n * r * r
                cells = {h: [t[0], t[1] / t[0], t[2], (t[3] / t[0]) ** 0.5] for h, t in merged.items() if t[0]}
                sub = 1
            rows = sorted((self._code(name, sub), cell) for name, cell in cells.items() if cell[0] >= min_n)
            codes = np.array([c for c, _ in rows], dtype=np.int64)
            stat = np.array([[cell[1], cell[2], cell[3] if len(cell) > 3 else cell[2]] for _, cell in rows])
            cache[(key, cls, coarse)] = (codes, stat.reshape(-1, 3))
        return cache[(key, cls, coarse)]

    @staticmethod
    def operand_classes(args) -> np.ndarray:
        """Per element: the significand class of each operand (unary: 's8' / 's11' / 'l'; otherwise 'p' / 's' / 'l'
        per operand, concatenated)."""
        sb = [sigbits(a) for a in args]
        if len(args) == 1:
            return np.where(sb[0] <= 8, "s8", np.where(sb[0] <= 11, "s11", "l"))
        parts = [np.where(b <= 1, "p", np.where(b <= 11, "s", "l")) for b in sb]
        out = parts[0]
        for q in parts[1:]:
            out = np.char.add(out, q)
        return out

    def approx(self, key, x_in, y_out, out_elem, args_est=None):
        """(mu, half, variance, outside-mask) for an approximate primitive; ``args_est`` (the device operand
        estimates) select the significand class table of each element."""
        entry = self.lookup(key)
        if entry is None:
            return None
        cond = x_in if entry.get("cond", "input") == "input" else y_out
        c = self.codes(cond, entry.get("sub", 1))
        g = entry["global"]
        glob = np.array([g[1], g[2], g[3] if len(g) > 3 else g[2]])
        shape = np.shape(y_out)
        chosen = np.broadcast_to(glob, shape + (3,)).copy()
        inside = np.zeros(shape, dtype=bool)
        main = "l" if args_est is None or len(args_est) == 1 else "l" * len(args_est)
        classes = np.full(shape, main) if args_est is None else np.broadcast_to(self.operand_classes(args_est), shape)
        c_coarse = self.codes(cond, 1)
        for cls in np.unique(classes):
            sel = classes == cls
            for coarse, cc in ((True, c_coarse), (False, c)):        # the bucket where it has samples, else binade
                codes, stat = self._tables(key, None if cls == main else str(cls), coarse)
                if not len(codes):
                    continue
                pos = np.clip(np.searchsorted(codes, cc), 0, len(codes) - 1)
                hit = sel & (codes[pos] == cc)
                chosen = np.where(hit[..., None], stat[pos], chosen)
                inside |= hit
        mean, worst, rms = chosen[..., 0], chosen[..., 1], chosen[..., 2]
        u = ulp_of(y_out, out_elem)
        mu = mean * u
        half = (worst + np.abs(mean)) * u
        var = np.maximum(rms * rms - mean * mean, 0.0) * u * u
        return mu, half, var, ~inside


# ----------------------------------------------------------------------------- the predictor

class BiasPredictor(KernelReferenceEvaluator):
    """Exact evaluation of a launch with each rounding site's contract injected into the tangent channel."""

    def __init__(self, module, ttgir: str, contracts: Optional[ContractTable] = None, enable_fp_fusion: bool = True,
                 enable_reflect_ftz: bool = True, ptx: Optional[str] = None, func_name: Optional[str] = None,
                 noise: bool = False, seed: int = 0):
        from .emulate import contraction_plan, ptx_product_choices
        from .layouts import reduce_layouts, scan_layouts
        super().__init__(module, mode=NumericMode.NUMERICAL_DIFFERENCE, func_name=func_name, ttgir=ttgir)
        self.contracts = contracts or ContractTable()
        self.noise = noise
        self._rng = np.random.default_rng(seed)
        self.layouts = reduce_layouts(module, ttgir)
        self.scan_layouts = scan_layouts(module, ttgir)
        self.reflect_ftz = enable_reflect_ftz
        plan = contraction_plan(module) if enable_fp_fusion else {}
        choices = ptx_product_choices(module, ptx, plan) if (ptx and enable_fp_fusion) else {}
        self.fusion = {n: (alt if (choices.get(n) == "swap" and alt is not None) else (i, m))
                       for n, (i, m, alt) in plan.items()}
        self._def = {}
        for fn in module.funcs.values():
            for o in fn.walk():
                for r in o.results:
                    self._def[r] = o
        self.sites = collections.Counter()       # contract kind -> lanes
        self.flags = collections.Counter()       # failure condition -> lanes
        self.lanes = collections.Counter()       # node -> lanes evaluated (for the per-site fractions)
        self._env = None
        self._cmp = {}                           # compare result name -> (risk mask, margin)

    # the tangent channel is always on: an op without a tangent rule aborts instead of dropping the error
    @property
    def _derivative(self):
        return True

    @_derivative.setter
    def _derivative(self, value):
        pass

    def _exec(self, op, env, state):
        saved = self._env
        self._env = env
        try:
            return super()._exec(op, env, state)
        finally:
            self._env = saved

    def _maybe_round(self, v, op, fmt, declared):
        return v

    # -- helpers ------------------------------------------------------------------

    @staticmethod
    def _point(v: TV) -> TV:
        if v.kind != "f" or v.hi is None:
            return v
        mid = np.where(v.st == ST_OK, 0.5 * (v.lo + v.hi), 0.0)
        out = _ftv(v.elem, mid, mid.copy(), v.st, v.cond, v.reasons)
        out.d = v.d
        return out

    def _site(self, out: TV, op, contract, kind: str) -> TV:
        """Inject a site's contract (mean, half-width, variance) into the tangent channel of its (exact, point)
        result.  Mean pass: the interval [delta - R, delta + R] gains [mu - half, mu + half].  Noise pass: a point
        tangent gains a random +-sqrt(variance) (Rademacher), so the output tangent's square estimates the output
        variance sum_s J_s^2 var_s."""
        mu, half, var = contract
        out = self._point(out)
        ok = out.st == ST_OK
        t_lo, t_hi = out.tangent()
        t_lo, t_hi = np.broadcast_to(t_lo, out.shape), np.broadcast_to(t_hi, out.shape)
        mid, rad = 0.5 * (t_lo + t_hi), 0.5 * (t_hi - t_lo)
        if self.noise:
            sd = np.where(ok, np.sqrt(np.broadcast_to(var, out.shape)), 0.0)
            mid = mid + sd * self._rng.choice((-1.0, 1.0), size=out.shape)
        else:
            mid = mid + np.where(ok, np.broadcast_to(mu, out.shape), 0.0)
            rad = rad + np.where(ok, np.broadcast_to(half, out.shape), 0.0)
        out.d = (mid - rad, mid + rad)
        n = int(ok.sum())
        self.sites[kind] += n
        self.lanes[op.node_id] += n
        if out.elem in iv.FLOAT_FORMATS:
            _, emin, _, _ = iv.FLOAT_FORMATS[out.elem]
            g = out.lo
            bad = ok & (g != 0) & (np.abs(g) < np.ldexp(1.0, emin))
            bad |= ~np.isfinite(g)
            if bad.any():
                self.flags[f"subnormal_or_nonfinite@{op.node_id}"] += int(bad.sum())
        return out

    def _dev(self, v: TV, shape):
        """Estimate of the device value of an operand: the exact value plus the mean error predicted so far (mean
        pass); the exact value in the noise pass."""
        g = np.broadcast_to(v.lo, shape)
        if self.noise or v.d is None:
            return g
        return g + np.broadcast_to(0.5 * (v.d[0] + v.d[1]), shape)

    def _grid(self, v: TV, shape):
        """The device value estimate of an operand on its own format's grid."""
        x = self._dev(v, shape)
        if v.elem in iv.FLOAT_FORMATS and v.elem != "f64":
            x = iv.round_nearest_even(x, v.elem)[0]
        return x

    def _radius(self, v: TV):
        if v.d is None:
            return np.zeros(v.shape), np.zeros(v.shape)
        lo, hi = v.d
        return 0.5 * (lo + hi), 0.5 * (hi - lo)

    def _bits(self, name: str, v: TV, depth: int = 0):
        """Significand budget of an operand's device value (see module doc)."""
        full = precision(v.elem) if v.elem in iv.FLOAT_FORMATS else 64
        op = self._def.get(name)
        if op is None or depth > 32:
            return full
        if op.name in PRESERVING and op.operands:
            return self._bits(op.operands[0], v, depth + 1)
        if op.name == "tt.load":
            return np.minimum(sigbits(v.lo), full)
        if op.name == "tt.elementwise_inline_asm" and \
                inline_asm_internal(op.attrs.get("asm", "")) == "tf32_round":
            return 11
        if op.name in ("arith.extf", "tt.fp_to_fp") and op.operand_types:
            src = getattr(op.operand_types[0], "elem", None)
            if src in iv.FLOAT_FORMATS and precision(src) < full:
                inner = self._bits(op.operands[0], v, depth + 1)
                return np.minimum(inner, precision(src))
        return full

    # -- elementwise ----------------------------------------------------------------

    def _float_op(self, name, op, args, out_elem):
        node = op.node_id
        if node in self.fusion:
            idx, mul = self.fusion[node]
            a, b = (self._env[v] for v in mul.operands)
            c = args[1 - idx]
            if op.name == "arith.subf":
                if idx == 0:
                    c = TV("f", c.elem, -c.hi, -c.lo, None, c.st, c.cond, c.reasons,
                           None if c.d is None else (-c.d[1], -c.d[0]))
                else:
                    a = TV("f", a.elem, -a.hi, -a.lo, None, a.st, a.cond, a.reasons,
                           None if a.d is None else (-a.d[1], -a.d[0]))
            fused = TOp("math.fma", op.results, op.result_types, [mul.operands[0], mul.operands[1], "%c"], op.attrs,
                        [], op.line, op.text, op.operand_types)
            out = super()._float_op("fma", fused, [a, b, c], out_elem)
            prod = self._dev(a, out.shape) * self._dev(b, out.shape)
            c3 = add_contract(self._dev(c, out.shape), prod, out_elem, a_grid=True, b_grid=False)
            return self._site(out, op, c3, "rn_fma_contracted")
        if op.name in APPROX_OPS or op.name in ("tt.extern_elementwise", "tt.elementwise_inline_asm"):
            key = ContractTable.key(op, out_elem)
            if op.name == "tt.extern_elementwise" and op.attrs.get("symbol", "").strip('"') in LIBDEVICE_ROUNDING:
                pass                                   # rounding-suffixed libdevice: an IEEE site (below)
            elif key in self.contracts.cr:
                out = self._point(super()._float_op(name, op, args, out_elem))
                return self._site(out, op, rn_contract(out.lo, out_elem), "rn_correctly_rounded_library")
            else:
                return self._approx(name, op, args, out_elem, key)
        out = self._point(super()._float_op(name, op, args, out_elem))
        mode = op.attrs.get("rounding", "rtne")
        mode = {"rtne": "rtne", "rn": "rtne", "rtz": "rtz", "rz": "rtz", "rd": "rd", "rm": "rd", "ru": "ru",
                "rp": "ru"}.get(str(mode).strip(), "rtne")
        if out_elem not in iv.FLOAT_FORMATS:
            raise ProgramAbort(f"{node}: bias contract: no format model for {out_elem}")
        in_bits = None
        if name == "cast" and args:
            in_bits = self._bits(op.operands[0], args[0])
            if op.name == "tt.fp_to_fp" and out_elem == "f8E5M2":
                return self._e5m2(out, op, args[0], in_bits)
        if name in FLIP_SOURCES:
            self._flip_check(op, args[0].lo, self._radius(args[0]), "integer boundary", 0.5 if name in
                             ("round", "roundeven") else 0.0)
        if name in ("add", "sub") and mode == "rtne" and len(args) == 2:
            a, b = self._dev(args[0], out.shape), self._dev(args[1], out.shape)
            exact_in = (np.broadcast_to(self._radius(args[0])[1] == 0, out.shape)
                        & np.broadcast_to(self._radius(args[1])[1] == 0, out.shape))
            c3 = add_contract(a, -b if name == "sub" else b, out_elem, exact_inputs=exact_in)
            kind = "rn_add"
        elif mode == "rtne" and name in ("mul", "fma", "cast") and all(a.kind == "f" for a in args):
            est = [self._grid(a, out.shape) for a in args]
            t = est[0] * est[1] if name == "mul" else (est[0] * est[1] + est[2] if name == "fma" else est[0])
            exact_in = np.all([np.broadcast_to(self._radius(a)[1] == 0, out.shape) for a in args], axis=0)
            c3 = rn_contract(t, out_elem, exact_in)
            if name == "cast":                      # an operand carrying no more bits than the target: exact
                fits = np.broadcast_to(np.asarray(in_bits) <= precision(out_elem), out.shape)
                c3 = tuple(np.where(fits, 0.0, x) for x in c3)
            kind = "rn_conversion" if name == "cast" else "rn"
        elif mode == "rtne":                       # div, sqrt, ...: the estimate of the exact result at the device
            t = out.lo + (0.0 if self.noise or out.d is None else 0.5 * (out.d[0] + out.d[1]))   # operands
            c3 = rn_contract(t, out_elem, False, in_bits)
            kind = "rn"
        else:
            c3 = directed_contract(out.lo, out_elem, mode, in_bits)
            kind = f"{mode}"
        ftz = self.reflect_ftz and out_elem == "f32" and (
            (op.name in ("math.floor", "math.ceil")) or
            (op.name == "tt.extern_elementwise" and op.attrs.get("symbol", "").strip('"') in LIBDEVICE_ROUNDING))
        if ftz:
            tiny = (out.lo != 0) & (np.abs(out.lo) < np.ldexp(1.0, -126))
            c3 = (np.where(tiny, -out.lo, c3[0]), c3[1], c3[2])
        return self._site(out, op, c3, kind)

    def _e5m2(self, out, op, x, in_bits):
        """sm_86 software conversion to f8E5M2: its local relation (e5m2_software) evaluated at the estimate of the
        device operand, like a correctly rounded site."""
        t = self._grid(x, out.shape)
        r = e5m2_software(t, x.elem == "f32") - t
        u = ulp_of(t, "f8E5M2")
        exact = r == 0
        exact_in = np.broadcast_to(self._radius(x)[1] == 0, out.shape)
        c3 = (r, np.where(exact, 0.0, u), np.where(exact | exact_in, 0.0, u * u / 12))
        return self._site(out, op, c3, "e5m2_software")

    def _approx(self, name, op, args, out_elem, key):
        out = self._point(super()._float_op(name, op, args, out_elem))
        est = [self._grid(a, out.shape) for a in args]
        x = est[0] if len(args) == 1 else out.lo
        got = self.contracts.approx(key, x, out.lo, out_elem, est)
        if got is None:
            raise ProgramAbort(f"{op.node_id}: bias contract: no contract for {key}")
        mu, half, var, outside = got
        ok = out.st == ST_OK
        if (outside & ok).any():
            self.flags[f"approx_outside_cells@{op.node_id}"] += int((outside & ok).sum())
        # a few fixed operating points (e.g. after floor): a cell mean does not speak for them
        if ok.sum() >= 1024 and len(np.unique(np.stack([e[ok] for e in est]), axis=1).T) < 64:
            self.flags[f"approx_discrete_inputs@{op.node_id}"] += int(ok.sum())
        return self._site(out, op, (mu, half, var), f"approx:{key}")

    def _op_extern(self, op, args, env, state):
        symbol = op.attrs.get("symbol", "").strip('"')
        if LIBDEVICE.get(symbol) is None:
            raise ProgramAbort(f"{op.node_id}: bias contract: libdevice {symbol} has no declared semantics")
        return super()._op_extern(op, args, env, state)

    def _op_inline_asm(self, op, args, env, state):
        internal = inline_asm_internal(op.attrs.get("asm", ""))
        out_elem = op.result_types[0].elem
        if internal == "tf32_round":
            out = self._point(args[0])
            bits = self._bits(op.operands[0], args[0])
            return self._site(out, op, directed_contract(out.lo, "tf32", "rna", bits), "tf32_rna")
        if internal == "approx_div" or (internal is not None and re.search(r"\.approx", op.attrs.get("asm", ""))):
            return self._approx("div" if internal == "approx_div" else internal, op, args, out_elem,
                                ContractTable.key(op, out_elem))
        if internal is None:
            raise ProgramAbort(f"{op.node_id}: bias contract: inline asm without a contract")
        return super()._op_inline_asm(op, args, env, state)

    def _op_approx_div(self, op, args, env, state):
        return self._approx("div", op, args, op.result_types[0].elem, ContractTable.key(op, op.result_types[0].elem))

    def _int_to_float(self, name, op, x, out_elem):
        out = self._point(super()._int_to_float(name, op, x, out_elem))
        return self._site(out, op, rn_contract(out.lo, out_elem, True), "rn_int_to_float")

    # -- thresholds -------------------------------------------------------------------

    def _flip_check(self, op, g, mr, what, offset=0.0):
        """Discontinuity within R of the exact value: flag the lanes (integer boundaries, offset 0.5 for round)."""
        mid, rad = mr
        g = np.asarray(g, dtype=np.float64)
        if not np.any(rad > 0) and not np.any(mid != 0):
            return
        t = g - offset
        nearest = np.rint(t)
        dist = np.abs(t - nearest)
        risk = (dist <= np.abs(mid) + rad) & np.isfinite(g)
        if risk.any():
            self.flags[f"flip_risk_discontinuous:{what}@{op.node_id}"] += int(risk.sum())

    def _elementwise(self, name, op, args):
        if name == "cmpf":
            out = super()._elementwise(name, op, args)
            a, b = args
            (ma, ra), (mb, rb) = self._radius(a), self._radius(b)
            margin = np.broadcast_to(a.lo, out.shape) - np.broadcast_to(b.lo, out.shape)
            err = np.broadcast_to(np.abs(ma - mb) + ra + rb, out.shape)
            risk = (np.abs(margin) <= err) & (err > 0)
            self._cmp[op.results[0]] = risk
            return out
        if name == "select":
            out = super()._elementwise(name, op, args)
            risk = self._cond_risk(op.operands[0], out.shape)
            if risk is not None and risk.any() and out.kind == "f":
                t, f = args[1], args[2]
                (mt, rt), (mf, rf) = self._radius(t), self._radius(f)
                jump = np.abs(np.broadcast_to(t.lo, out.shape) - np.broadcast_to(f.lo, out.shape))
                scale = np.broadcast_to(np.abs(mt) + rt + np.abs(mf) + rf, out.shape)
                disc = risk & (jump > scale)
                if disc.any():
                    self.flags[f"flip_risk_discontinuous:select@{op.node_id}"] += int(disc.sum())
                else:
                    self.sites["continuous_select_risk_lanes"] += int(risk.sum())
            return out
        if name in ("fptosi", "fptoui"):
            self._flip_check(op, args[0].lo, self._radius(args[0]), "float to int")
        return super()._elementwise(name, op, args)

    def _cond_risk(self, name, shape, depth=0):
        if name in self._cmp:
            r = self._cmp[name]
            try:
                return np.broadcast_to(r, shape)
            except ValueError:
                return np.broadcast_to(r.any(), shape)
        op = self._def.get(name)
        if op is None or depth > 32:
            return None
        if op.name in PRESERVING or op.name in ("arith.andi", "arith.ori", "arith.xori"):
            risks = [self._cond_risk(v, shape, depth + 1) for v in op.operands]
            risks = [r for r in risks if r is not None]
            if risks:
                return np.logical_or.reduce(risks)
        return None

    def _op_if(self, op, args, env, state):
        risk = self._cond_risk(op.operands[0], ())
        if risk is not None and np.any(risk):
            self.flags[f"flip_risk_discontinuous:branch@{op.node_id}"] += 1
        return super()._op_if(op, args, env, state)

    # -- reductions, scans, dot, atomics ----------------------------------------------

    def _op_reduce(self, op, args, env, state):
        out = super()._op_reduce(op, args, env, state)
        outs = out if isinstance(out, list) else [out]
        combiner = recognize_combiner(op)
        if combiner != "sum" or len(args) != 1 or args[0].kind != "f":
            return [self._point(o) if isinstance(o, TV) else o for o in outs] if isinstance(out, list) \
                else self._point(out)
        x = args[0]
        axis = int(op.attrs["axis"].split(":")[0])
        vals = np.moveaxis(np.where(x.st == ST_OK, x.lo, 0.0), axis, -1)
        lay = self.layouts.get(op.node_id)
        if lay is not None and "perm" in lay:
            vals = vals[..., lay["perm"]]
        if lay is None:
            self.flags[f"reduce_order_unknown@{op.node_id}"] += 1
            lay = {"spt": vals.shape[-1], "tpw": 1, "wpc": 1}
        from .emulate import GpuEmulator

        def order(v, combine):
            res, _ = GpuEmulator._reduce_tree(None, v, lay, combine)
            return res
        return self._site(outs[0], op, sum_tree_contract(vals, x.elem, order), "rn_reduce_sum")

    def _op_scan(self, op, args, env, state):
        out = super()._op_scan(op, args, env, state)
        outs = out if isinstance(out, list) else [out]
        combiner = recognize_combiner(op)
        x = args[0]
        if len(args) != 1 or x.kind != "f" or combiner in ("max_num", "min_num", "max_nan", "min_nan"):
            return self._point(outs[0]) if not isinstance(out, list) else [self._point(o) for o in outs]
        info = self.scan_layouts.get(op.node_id)
        if combiner != "sum" or info is None or len(x.shape) != 1 or info["reverse"] or info["axis"] != 0 \
                or info["tpw"] != 32 or x.shape[0] % (info["spt"] * 32 * info["wpc"]):
            raise ProgramAbort(f"{op.node_id}: bias contract: scan outside the composition rule")
        vals = np.where(x.st == ST_OK, x.lo, 0.0)
        c3 = sum_tree_contract(vals, x.elem, lambda v, comb: scan_order.scan_1d(v, info["spt"], info["wpc"], comb))
        return self._site(outs[0], op, c3, "rn_scan_sum")

    def _op_dot(self, op, args, env, state):
        a, b, c = args
        out = self._point(super()._op_dot(op, args, env, state))
        if out.kind != "f":
            return out
        precision_attr = op.attrs.get("inputPrecision", "ieee").strip()
        tc = tensor_core.mma_format(a.elem, b.elem, c.elem, precision_attr)
        A = np.where(a.st == ST_OK, a.lo if a.hi is None else 0.5 * (a.lo + a.hi), 0.0)
        B = np.where(b.st == ST_OK, b.lo if b.hi is None else 0.5 * (b.lo + b.hi), 0.0)
        C = np.where(c.st == ST_OK, c.lo if c.hi is None else 0.5 * (c.lo + c.hi), 0.0)
        if not self.noise and c.d is not None:          # the incoming accumulator as the device holds it
            C = C + np.where(c.st == ST_OK, 0.5 * (c.d[0] + c.d[1]), 0.0)
        if tc is not None:
            a_bits = np.broadcast_to(self._bits(op.operands[0], a), A.shape)
            b_bits = np.broadcast_to(self._bits(op.operands[1], b), B.shape)
            return self._site(out, op, mma_contract(A, B, C, tc, a_bits, b_bits), f"mma_{tc}")
        if a.elem == b.elem == c.elem == "f32" and precision_attr.startswith("ieee"):
            return self._site(out, op, fma_chain_contract(A, B, C), "fma_chain")
        raise ProgramAbort(f"{op.node_id}: bias contract: dot ({precision_attr}, {a.elem}) outside the contracts")

    def _op_atomic_rmw(self, op, args, env, state, kind_override=None):
        value = args[-1] if args else None
        if any(isinstance(a, TV) and a.kind == "f" for a in args):
            self.flags[f"float_atomic@{op.node_id}"] += 1
            raise ProgramAbort(f"{op.node_id}: bias contract: float atomic fold order not modelled")
        return super()._op_atomic_rmw(op, args, env, state, kind_override)


# ----------------------------------------------------------------------------- driver

_ERF = np.frompyfunc(__import__("math").erf, 1, 1)
_ERFC = np.frompyfunc(__import__("math").erfc, 1, 1)
_NP_FUNCS = {"exp": np.exp, "exp2": np.exp2, "exp10": lambda x: np.power(10.0, x), "expm1": np.expm1, "log": np.log,
             "log2": np.log2, "log10": np.log10, "log1p": np.log1p, "sqrt": np.sqrt, "rsqrt": lambda x: 1 / np.sqrt(x),
             "tanh": np.tanh, "erf": lambda x: _ERF(x).astype(np.float64),
             "erfc": lambda x: _ERFC(x).astype(np.float64), "atan": np.arctan, "asinh": np.arcsinh,
             "sinh": np.sinh, "cbrt": np.cbrt, "asin": np.arcsin, "acos": np.arccos, "atanh": np.arctanh,
             "acosh": np.arccosh, "sin": np.sin, "cos": np.cos, "tan": np.tan, "cosh": np.cosh}


def _float64_elementary(original):
    """float64 values of the elementary functions at point arguments (the predictor needs the exact value to far below
    an output ulp, not an enclosure; relative error ~1e-16); interval arguments use the rigorous enclosure."""
    def bounds(name, lo, hi):
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        fn = _NP_FUNCS.get(name)
        if fn is None or not np.array_equal(lo, hi):
            return original(name, lo, hi)
        with np.errstate(all="ignore"):
            v = np.asarray(fn(lo), dtype=np.float64)
        ok = np.isfinite(v)
        dom = iv._MONOTONE.get(name)
        if dom is not None and dom[2] is not None:
            ok &= (lo > dom[2]) if dom[3] else (lo >= dom[2])
        up = iv._UPPER_DOMAIN.get(name)
        if up is not None:
            ok &= (lo < up[0]) if up[1] else (lo <= up[0])
        v = np.where(ok, v, 0.0)
        return v, v.copy(), ok
    return bounds


def _run(launch, contracts, programs, noise, seed):
    from .ttir_parser import parse_ttir
    module = parse_ttir(launch.asm["ttir"])
    module.source_text = launch.asm["ttir"]
    pred = BiasPredictor(module, launch.asm["ttgir"], contracts,
                         enable_fp_fusion=bool(launch.metadata.get("enable_fp_fusion", True)),
                         enable_reflect_ftz=bool(launch.metadata.get("enable_reflect_ftz", True)),
                         ptx=launch.asm.get("ptx"), noise=noise, seed=seed)
    original = iv.elementary_bounds
    iv.elementary_bounds = _float64_elementary(original)
    try:
        return pred, pred.evaluate(launch, programs=programs)
    finally:
        iv.elementary_bounds = original


def predict(launch, contracts: Optional[ContractTable] = None, programs: Optional[list] = None, seed: int = 0) -> dict:
    """Predicted bias of every float output of a captured launch (the device outputs are not read).

    Two passes: the mean pass gives delta (predicted mean error) and R (first-order worst-case radius); the noise pass
    gives one draw of the first-order random error per element, whose square estimates the error variance."""
    contracts = contracts or ContractTable()
    pred, result = _run(launch, contracts, programs, False, seed)
    noise_pred, noise_result = _run(launch, contracts, programs, True, seed)
    outputs = {}
    for ptr, buf in result.buffers.items():
        if buf.kind != "f" or not buf.written.any():
            continue
        ok = buf.written & (buf.st == ST_OK)
        g = np.where(ok, 0.5 * (buf.lo + (buf.hi if buf.hi is not None else buf.lo)), 0.0)
        mid = rad = np.zeros(g.shape)
        if buf.d is not None:
            mid, rad = 0.5 * (buf.d[0] + buf.d[1]), 0.5 * (buf.d[1] - buf.d[0])
        nb = noise_result.buffers.get(ptr)
        draw = np.zeros(g.shape) if nb is None or nb.d is None else 0.5 * (nb.d[0] + nb.d[1])
        outputs[buf.name] = {"elem": buf.elem, "written": buf.written.copy(), "ok": ok, "exact": g,
                             "delta": np.where(ok, mid, 0.0), "radius": np.where(ok, rad, 0.0),
                             "noise_draw": np.where(ok, draw, 0.0)}
    return {"outputs": outputs, "sites": dict(pred.sites), "flags": dict(pred.flags), "lanes": dict(pred.lanes),
            "aborted_programs": len(result.aborted), "abort_reasons": sorted(set(result.aborted.values()))[:5]}


def summarize(pred: dict, max_relative=LARGE_RELATIVE) -> dict:
    """Per output: predicted alignment <delta, G> / <G, G> and its noise (the standard error a measurement of the same
    elements would have under the first-order model), the mean delta in output ulps, the typical relative error; and
    the failure conditions that apply (status "无法判断" when any does)."""
    out = {}
    for name, o in pred["outputs"].items():
        ok = o["ok"]
        g, dl, nd = o["exact"][ok], o["delta"][ok], o["noise_draw"][ok]
        gg = float((g * g).sum())
        u = ulp_of(g, o["elem"])
        nz = g != 0
        rel_sd = np.abs(nd[nz]) / np.abs(g[nz])
        out[name] = {"elements": int(ok.sum()), "written": int(o["written"].sum()),
                     "alignment": float((dl * g).sum() / gg) if gg > 0 else 0.0,
                     "alignment_noise_se": float(np.sqrt(((nd * g) ** 2).sum()) / gg) if gg > 0 else 0.0,
                     "mean_ulp": float(np.mean(dl / u)) if dl.size else 0.0,
                     "rms_relative_error": float(np.sqrt(np.mean(rel_sd ** 2))) if nz.any() else 0.0,
                     "share_large_relative": float(np.mean(rel_sd > max_relative)) if nz.any() else 0.0}
    failures = []
    if pred["aborted_programs"]:
        failures.append("aborted: " + "; ".join(pred["abort_reasons"]))
    total = max(1, sum(pred["lanes"].values()))
    for k, v in pred["flags"].items():
        node = k.split("@")[-1]
        lanes = max(1, pred["lanes"].get(node, total))
        if k.startswith("flip_risk_discontinuous") or k.startswith("float_atomic") or k.startswith("reduce_order"):
            failures.append(f"{k} ({v} lanes)")
        elif k.startswith("subnormal_or_nonfinite") and v / lanes > 1e-3:
            failures.append(f"{k} ({v} of {lanes} lanes)")
        elif k.startswith("approx_outside_cells") and v / lanes > 1e-2:
            failures.append(f"{k} ({v} of {lanes} lanes)")
        elif k.startswith("approx_discrete_inputs"):
            failures.append(f"{k} ({v} lanes)")
    for name, s in out.items():
        if s["share_large_relative"] > 0.01:
            failures.append(f"large relative error in {name} ({s['share_large_relative']:.3f} of elements)")
        if s["elements"] < s["written"]:
            failures.append(f"{name}: {s['written'] - s['elements']} written elements without an exact value")
    return {"outputs": out, "failures": failures, "status": "无法判断" if failures else "predicted",
            "sites": pred["sites"]}
