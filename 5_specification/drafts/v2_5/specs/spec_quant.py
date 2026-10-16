"""Class F (quantization formats, quantize / dequantize) and K11 (training scaling state).

Sources and identity
  OCP 8-bit Floating Point (OFP8) and OCP Microscaling Formats (MX) v1.0 ............ standard
  NVIDIA Transformer Engine user guide (NVFP4 two-level scaling; delayed scaling) .... standard (vendor documentation)
  TensorRT 'MX-compliant dynamic quantization' (round_up_to_e8m0) .................... vendor documentation (reading)
  GPTQ / AWQ reference implementations (packing order, zero-point form) .............. engine reference
Dimensions: D2 (eps, margin), D3 (packing order), D5 (granularity, scale rounding, element rounding, saturation,
double rounding, zero-point form), D7 (amax history), D8 (target of the comparison), D9 (stochastic rounding given u),
D11 (zero blocks, zero scale)."""
from _base import *

# element formats: explicit mantissa bits, min normal exponent, max finite value, infinity available, NaN available
FMT = {
    'e4m3':  dict(m=3, emin=-6,   maxv=F(448),               inf=False, nan=True),   # OFP8 E4M3 (fn)
    'e5m2':  dict(m=2, emin=-14,  maxv=F(57344),             inf=True,  nan=True),   # OFP8 E5M2
    'e2m3':  dict(m=3, emin=0,    maxv=F(15, 2),             inf=False, nan=False),  # MX FP6
    'e3m2':  dict(m=2, emin=-2,   maxv=F(28),                inf=False, nan=False),  # MX FP6
    'e2m1':  dict(m=1, emin=0,    maxv=F(6),                 inf=False, nan=False),  # MX FP4 / NVFP4
    'bf16':  dict(m=7, emin=-126, maxv=(2 - F(1, 2**7)) * F(2)**127,  inf=True, nan=True),
    'fp16':  dict(m=10, emin=-14, maxv=F(65504),             inf=True,  nan=True),
    'fp32':  dict(m=23, emin=-126, maxv=(2 - F(1, 2**23)) * F(2)**127, inf=True, nan=True),
}
def emax_elem(fmt): return floor_log2(FMT[fmt]['maxv'])

def round_fmt(x, fmt, mode='rne', saturate=False, overflow_test='after_rounding', u=None):
    """exact rounding of a rational to an element format.
    mode: rne (ties to even) | rna (ties away from zero) | rtz (toward zero) | stochastic (needs u in [0,1): round up iff u < fraction)
    overflow: saturate -> +-max; otherwise inf if the format has it, else NaN; formats with neither (FP6, FP4) always saturate.
    overflow_test: 'after_rounding' (IEEE-like) or 'before_rounding' (|x| > max decides, ONNX Cast table read literally)."""
    p = FMT[fmt]; x = Q(x)
    if x == 0: return F(0)
    s = -1 if x < 0 else 1; a = abs(x)
    sat = saturate or not (p['inf'] or p['nan'])
    if overflow_test == 'before_rounding' and a > p['maxv']:
        return s * p['maxv'] if sat else ('inf' if p['inf'] else 'nan') if s > 0 else ('-inf' if p['inf'] else 'nan')
    e = max(floor_log2(a), p['emin']); q = F(2) ** (e - p['m']); n = a / q
    lo = n.numerator // n.denominator; fr = n - lo
    if mode == 'rne':   up = fr > F(1, 2) or (fr == F(1, 2) and lo % 2 == 1)
    elif mode == 'rna': up = fr >= F(1, 2)
    elif mode == 'rtz': up = False
    elif mode == 'stochastic':
        if u is None: raise SpecInputError('stochastic rounding needs the uniform u used by the kernel (D9)')
        up = fr > 0 and Q(u) < fr
    else: raise SpecInputError(mode)
    r = (lo + (1 if up else 0)) * q
    if r > p['maxv']:
        if sat: r = p['maxv']
        else: return ('inf' if p['inf'] else 'nan') if s > 0 else ('-inf' if p['inf'] else 'nan')
    return s * r

# ---------------------------------------------------------------- MX (OCP v1.0): 32-element blocks, E8M0 shared scale
def mx_quantize(block, fmt, scale_rule='ocp_floor', mode='rne', u=None):
    """returns (X, elements). ocp_floor: X = 2^(floor(log2 amax) - emax_elem) clamped to [2^-127, 2^127] (OCP v1.0 6.3);
    ceil_pow2: X = 2^ceil(log2(amax / max_elem)) (TensorRT 'round_up_to_e8m0', registered reading).
    Elements: round(v / X) with saturation to the element max (OCP)."""
    block = [Q(v) for v in block]
    if len(block) > 32: raise SpecInputError('MX block has at most 32 elements')
    amax = max(abs(v) for v in block)
    if amax == 0: return None, [F(0)] * len(block)                  # scale unconstrained (allowed set), elements 0
    if scale_rule == 'ocp_floor': se = floor_log2(amax) - emax_elem(fmt)
    elif scale_rule == 'ceil_pow2': se = ceil_log2(amax / FMT[fmt]['maxv'])
    else: raise SpecInputError(scale_rule)
    se = max(-127, min(127, se)); X = F(2) ** se
    return X, [round_fmt(v / X, fmt, mode=mode, saturate=True, u=None if u is None else u[i]) for i, v in enumerate(block)]

def mx_dequantize(X, elems): return [F(0) if X is None else e * X for e in elems]

# ---------------------------------------------------------------- NVFP4: 16-element blocks, E4M3 block scale, FP32 global scale
def nvfp4_quantize(tensor_blocks, global_amax=None, s_global_rounding='fp32', mode='rne', u=None):
    """x = x_e2m1 * s_block * s_global;  s_global = global_amax / (448 * 6);  s_block = (block_amax / 6) / s_global stored in E4M3
    (Transformer Engine NVFP4 guide). Returns s_global and per block (s_block_e4m3, elements)."""
    blocks = [[Q(v) for v in b] for b in tensor_blocks]
    if any(len(b) > 16 for b in blocks): raise SpecInputError('NVFP4 block has at most 16 elements')
    gmax = Q(global_amax) if global_amax is not None else max(abs(v) for b in blocks for v in b)
    if gmax == 0: return None, [(F(0), [F(0)] * len(b)) for b in blocks]
    sg = gmax / (448 * 6)
    if s_global_rounding == 'fp32': sg = round_fmt(sg, 'fp32')
    out = []
    for bi, b in enumerate(blocks):
        bmax = max(abs(v) for v in b)
        sb = round_fmt((bmax / 6) / sg, 'e4m3', saturate=True)
        if sb == 0: out.append((F(0), [F(0)] * len(b))); continue
        out.append((sb, [round_fmt(v / (sb * sg), 'e2m1', mode=mode, saturate=True, u=None if u is None else u[bi][i]) for i, v in enumerate(b)]))
    return sg, out

def nvfp4_dequantize(sg, blocks): return [[F(0) if sg is None else e * sb * sg for e in el] for sb, el in blocks]

# ---------------------------------------------------------------- scaled FP8 / integer quantization with explicit granularity
def absmax_scale(group, qmax, eps=F(0), eps_rule='clamp_min'):
    amax = max(abs(Q(v)) for v in group)
    amax = max(amax, Q(eps)) if eps_rule == 'clamp_min' else amax + Q(eps)
    if amax == 0: raise SpecNotEstablished('all-zero group with eps = 0: scale undefined (D11)')
    return amax / qmax

def quant_fp8_groups(x, group, fmt='e4m3', eps=F(0), scale_store=None):
    """per-group (group = row length for per-token, 128 for 1x128) scaled FP8: scale = amax / fmt_max, q = round(v / scale) saturated.
    scale_store: None (exact) or 'fp32' (scale rounded to fp32 before use)."""
    x = [Q(v) for v in x]; out = []
    for g0 in range(0, len(x), group):
        g = x[g0:g0 + group]; sc = absmax_scale(g, FMT[fmt]['maxv'], eps)
        if scale_store == 'fp32': sc = round_fmt(sc, 'fp32')
        out.append((sc, [round_fmt(v / sc, fmt, saturate=True) for v in g]))
    return out

def quant_int_sym(x, bits=8, group=None, eps=F(0), mode='rne', qmin_rule='symmetric'):
    """symmetric integer quantization: scale = amax / (2^(b-1) - 1); q = clamp(round(v / scale)); qmin = -qmax or -qmax-1."""
    x = [Q(v) for v in x]; group = group or len(x); qmax = 2 ** (bits - 1) - 1
    qmin = -qmax if qmin_rule == 'symmetric' else -qmax - 1; out = []
    for g0 in range(0, len(x), group):
        g = x[g0:g0 + group]; sc = absmax_scale(g, qmax, eps)
        out.append((sc, [max(qmin, min(qmax, _round_int(v / sc, mode))) for v in g]))
    return out

def quant_int_asym(x, bits=4, group=None, zero_form='z'):
    """asymmetric (zero-point) quantization: scale = (max - min) / (2^b - 1); z = round(-min / scale) clamped;
    q = clamp(round(v / scale) + z). zero_form 'z' stores z; 'z_minus_1' stores z - 1 and the dequantizer adds 1 back."""
    x = [Q(v) for v in x]; group = group or len(x); top = 2 ** bits - 1; out = []
    for g0 in range(0, len(x), group):
        g = x[g0:g0 + group]; lo, hi = min(min(g), 0), max(max(g), 0)
        if hi == lo: raise SpecNotEstablished('constant group: scale undefined (D11)')
        sc = (hi - lo) / top; z = max(0, min(top, _round_int(-lo / sc, 'rne')))
        q = [max(0, min(top, _round_int(v / sc, 'rne') + z)) for v in g]
        out.append((sc, z if zero_form == 'z' else z - 1, q))
    return out

def dequant_int_asym(groups, zero_form='z'):
    return [(qq - (z if zero_form == 'z' else z + 1)) * sc for sc, z, q in groups for qq in q]

def _round_int(v, mode):
    v = Q(v); lo = v.numerator // v.denominator; fr = v - lo
    if mode == 'rne': return lo + (1 if fr > F(1, 2) or (fr == F(1, 2) and lo % 2 == 1) else 0)
    if mode == 'rna': return lo + (1 if fr > F(1, 2) or (fr == F(1, 2) and v > 0) else 0) if v >= 0 else -_round_int(-v, 'rna')
    if mode == 'rtz': return lo if v >= 0 else -((-v).numerator // (-v).denominator)
    raise SpecInputError(mode)

# ---------------------------------------------------------------- 4-bit packing into 32-bit words (D3)
AWQ_POS = [0, 4, 1, 5, 2, 6, 3, 7]          # logical element i is stored in nibble AWQ_POS[i] (AWQ unpack order)
def pack_int4(vals, order='gptq'):
    if len(vals) != 8 or any(not 0 <= v <= 15 for v in vals): raise SpecInputError('8 unsigned 4-bit values')
    pos = list(range(8)) if order == 'gptq' else AWQ_POS
    return sum(v << (4 * pos[i]) for i, v in enumerate(vals))
def unpack_int4(word, order='gptq'):
    pos = list(range(8)) if order == 'gptq' else AWQ_POS
    return [(word >> (4 * pos[i])) & 0xF for i in range(8)]

# ---------------------------------------------------------------- K11: delayed scaling state (Transformer Engine)
class DelayedScaling:
    """scale used at step t comes from the amax history up to step t-1;
    new scale = (FP8_MAX / amax) / 2^margin with amax = max(history) or the most recent value (TE recipe)."""
    def __init__(self, fp8_max=F(448), history_len=4, margin=0, algo='max', init_scale=F(1)):
        self.fmax, self.L, self.margin, self.algo = Q(fp8_max), history_len, margin, algo
        self.hist, self.scale = [], Q(init_scale)
    def step(self, x, fmt='e4m3'):
        x = [Q(v) for v in x]; used = self.scale
        q = [round_fmt(v * used, fmt, saturate=True) for v in x]
        self.hist = ([max(abs(v) for v in x)] + self.hist)[:self.L]
        amax = max(self.hist) if self.algo == 'max' else self.hist[0]
        if amax > 0: self.scale = (self.fmax / amax) / F(2) ** self.margin
        return used, q
