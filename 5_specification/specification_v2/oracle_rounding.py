"""Independent rounding oracle: pure-rational round-to-nearest-even (and directed modes) into IEEE-style binary formats.
Written from the IEEE 754 definitions; imports neither numpy nor any rule/tool code. Used to generate answers and
constructed-residual inputs; cross-checked against hardware (numpy) in test_oracle.py."""
from fractions import Fraction as F

FORMATS = {  # precision p (incl. hidden bit), emin, emax, max finite, has_inf
    'f16': (11, -14, 15, F(65504), True), 'bf16': (8, -126, 127, (2 - F(1, 128)) * F(2) ** 127, True),
    'f32': (24, -126, 127, (2 - F(1, 2 ** 23)) * F(2) ** 127, True), 'f64': (53, -1022, 1023, (2 - F(1, 2 ** 52)) * F(2) ** 1023, True),
    'f8e4m3fn': (4, -6, 8, F(448), False),    # OCP/NVIDIA E4M3FN: no infinities, 0x7F is NaN, max finite 448
    'f8e5m2': (3, -14, 15, F(57344), True),
}
FORMATS['f8e4m3'] = FORMATS['f8e4m3fn']
def _floor_log2(x):
    e = x.numerator.bit_length() - x.denominator.bit_length()
    while F(2) ** e > x: e -= 1
    while F(2) ** (e + 1) <= x: e += 1
    return e

def round_to(x, fmt, mode='rne', saturate=False):
    """x: Fraction. Returns the exact rational value of the rounded result; on overflow: rne -> 'inf'/'-inf' (or 'nan' for a
    format without infinities), directed modes -> the largest finite value of the proper sign per IEEE 754; saturate=True
    forces the largest finite value in every mode (the TransformerEngine-style FP8 conversion). Subnormals exact."""
    p, emin, emax, maxv, has_inf = FORMATS[fmt]
    if x == 0: return F(0)
    s = -1 if x < 0 else 1; a = abs(x)
    e = max(_floor_log2(a), emin)
    ulp = F(2) ** (e - p + 1)
    q, r = divmod(a, ulp)                                         # a = q*ulp + r, 0 <= r < ulp
    if mode == 'rne':
        if r * 2 > ulp or (r * 2 == ulp and q % 2 == 1): q += 1
    elif mode == 'rz': pass
    elif mode == 'ru': q += (1 if (r > 0 and s > 0) else 0)
    elif mode == 'rd': q += (1 if (r > 0 and s < 0) else 0)
    else: raise ValueError(mode)
    v = q * ulp
    if v > maxv:
        if saturate: return s * maxv
        if mode == 'rz': return s * maxv
        if mode == 'ru': return (s * maxv) if s < 0 else ('inf' if has_inf else 'nan')
        if mode == 'rd': return (s * maxv) if s > 0 else ('-inf' if has_inf else 'nan')
        if not has_inf: return 'nan'
        return '-inf' if s < 0 else 'inf'
    return s * v

def residual(exact, fmt, mode='rne'):
    r = round_to(exact, fmt, mode)
    if isinstance(r, str): return r
    return r - exact
