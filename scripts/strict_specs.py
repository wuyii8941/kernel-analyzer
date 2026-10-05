"""Strict enclosures of the specification f for the registered findings (plan WP2: "严格包围", as opposed to the
screening-level f64_point_spec with a declared 2^-40 budget).

- rational_enclosure: an exact rational value -> the float64 interval [down, up] that contains it (exact when the
  value is representable).
- B016 (scatter-mean of one segment): exact rational arithmetic on the float32 inputs.
- B012 (one RAdam step from a given state): MPFR at 256 bits (gmpy2), each result rounded outward to float64 with one
  extra ulp; the rectification branch rho_t > 5 must be decided with a margin far above the 256-bit error.
"""
from fractions import Fraction

import numpy as np


def rational_enclosure(q: Fraction):
    v = float(q)
    fv = Fraction(v)
    if fv == q:
        return v, v
    if fv < q:
        return v, float(np.nextafter(v, np.inf))
    return float(np.nextafter(v, -np.inf)), v


def _enclose_array(values):
    lo = np.empty(len(values))
    hi = np.empty(len(values))
    for i, q in enumerate(values):
        lo[i], hi[i] = rational_enclosure(q)
    return lo, hi


def scatter_mean_one_segment(x):
    """out[0, j] = sum_i x[i, j] / N exactly (one segment, N rows)."""
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    cols = [sum((Fraction(float(v)) for v in x[:, j]), Fraction(0)) / n for j in range(x.shape[1])]
    lo, hi = _enclose_array(cols)
    return lo.reshape(1, -1), hi.reshape(1, -1)


def scaled_count(x):
    """sum(x) / count with count = number of elements, exactly."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    q = sum((Fraction(float(v)) for v in x), Fraction(0)) / len(x)
    lo, hi = rational_enclosure(q)
    return np.array([lo]), np.array([hi])


def radam_step(p, grad, exp_avg, exp_avg_sq, step, lr, beta1, beta2, eps, weight_decay=0.0, prec=256):
    """One torch.optim.RAdam step (decoupled_weight_decay=False, maximize=False) in MPFR; returns float64 enclosures
    of the new param, exp_avg and exp_avg_sq.  `step` is the step count after the increment."""
    import gmpy2
    from gmpy2 import mpfr

    ctx = gmpy2.get_context()
    ctx.precision = prec
    lr, b1, b2, eps_, wd = (mpfr(float(v)) for v in (lr, beta1, beta2, eps, weight_decay))
    t = int(step)
    bc1 = 1 - b1 ** t
    bc2 = 1 - b2 ** t
    rho_inf = 2 / (1 - b2) - 1
    rho_t = rho_inf - 2 * t * b2 ** t / bc2
    if abs(rho_t - 5) < mpfr(2) ** (-(prec // 2)):
        raise ValueError("rho_t too close to 5 to decide the branch at this precision")
    rectified = rho_t > 5
    rect = gmpy2.sqrt((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t)) if rectified else None
    shape = np.shape(p)
    outs = {"param": [], "exp_avg": [], "exp_avg_sq": []}
    for pi, gi, mi, vi in zip(*(np.asarray(a, dtype=np.float64).reshape(-1) for a in (p, grad, exp_avg, exp_avg_sq))):
        P, G, M, V = mpfr(pi), mpfr(gi), mpfr(mi), mpfr(vi)
        if weight_decay:
            G = G + wd * P
        M = M + (1 - b1) * (G - M)            # lerp_(grad, 1 - beta1)
        V = V * b2 + (1 - b2) * G * G         # mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        mhat = M / bc1
        if rectified:
            P = P - mhat * lr * (gmpy2.sqrt(bc2) / (gmpy2.sqrt(V) + eps_)) * rect
        else:
            P = P - mhat * lr
        for k, val in (("param", P), ("exp_avg", M), ("exp_avg_sq", V)):
            outs[k].append(val)
    res = {}
    for k, vals in outs.items():
        # 256-bit values: rounding to float64 down / up and one more ulp outward encloses the exact result
        lo = np.array([float(np.nextafter(float(v), -np.inf)) for v in vals])
        hi = np.array([float(np.nextafter(float(v), np.inf)) for v in vals])
        res[k] = (lo.reshape(shape), hi.reshape(shape))
    return res, bool(rectified)
