"""Parameter-update layer: a kernel output and its reference interval propagated through an optimizer step.

The output y (flattened) is taken as the gradient g of a parameter block theta.  The step is evaluated in the
optimizer's declared real semantics with interval arithmetic (directed rounding) twice: for the actual
output K (a point) and for the reference K_R (an interval), on the same state (theta, m, v, t).  The update
difference u = step(K) - step(K_R) is then an enclosure of what the kernel alone changes in the update;
the optimizer's own floating-point rounding is not part of this measurement point (it would be a separate
implementation under test).  The reference update r = step(K_R) - theta (midpoint) defines the
reference-dependent directions at this layer.

Optimizer constants (lr, betas, eps, weight decay) are taken as the exact decimal or dyadic values given in
the declaration and enclosed with directed rounding; the state (theta, m, v) is given as float64 values and
treated as exact.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from . import intervals as iv


def _const(value, shape=()):
    """Enclosure of an exact constant (a Fraction, an int or a decimal string such as "0.999")."""

    q = Fraction(value) if not isinstance(value, float) else Fraction(repr(value))
    lo, hi = iv.rational_down(q), iv.rational_up(q)
    return np.full(shape, lo), np.full(shape, hi)


def _point(x):
    x = np.asarray(x, dtype=np.float64)
    return x, x


def sgd_delta(g_lo, g_hi, lr):
    """Delta theta = -lr * g (interval)."""

    c = _const(lr, np.shape(g_lo))
    lo, hi = iv.imul(*c, g_lo, g_hi)
    return -hi, -lo


def adamw_delta(g_lo, g_hi, m, v, t, lr, beta1, beta2, eps, weight_decay=0, theta=None):
    """Delta theta of one AdamW step (decoupled weight decay) in real semantics, as an interval:
    m' = b1 m + (1 - b1) g;  v' = b2 v + (1 - b2) g^2;  mhat = m' / (1 - b1^t);  vhat = v' / (1 - b2^t);
    Delta = -lr (mhat / (sqrt(vhat) + eps) + wd theta)."""

    shape = np.shape(g_lo)
    b1, b2 = Fraction(beta1), Fraction(beta2)
    m_lo, m_hi = _point(m)
    v_lo, v_hi = _point(v)
    mp = iv.iadd(*iv.imul(*_const(b1, shape), m_lo, m_hi), *iv.imul(*_const(1 - b1, shape), g_lo, g_hi))
    vp = iv.iadd(*iv.imul(*_const(b2, shape), v_lo, v_hi), *iv.imul(*_const(1 - b2, shape), *iv.isquare(g_lo, g_hi)))
    mhat = iv.idiv(*mp, *_const(1 - b1 ** int(t), shape))
    vhat = iv.idiv(*vp, *_const(1 - b2 ** int(t), shape))
    den = iv.iadd(*iv.isqrt(np.maximum(vhat[0], 0.0), vhat[1]), *_const(Fraction(eps), shape))
    ratio = iv.idiv(*mhat, *den)  # den >= eps > 0
    if weight_decay and theta is not None:
        ratio = iv.iadd(*ratio, *iv.imul(*_const(weight_decay, shape), *_point(theta)))
    lo, hi = iv.imul(*_const(lr, shape), *ratio)
    return -hi, -lo


def update_difference(k, kr_lo, kr_hi, delta):
    """u = step(K) - step(K_R) and the reference update r = step(K_R) - theta (midpoint).

    ``delta(g_lo, g_hi)`` returns the interval of Delta theta; theta cancels in u."""

    dk = delta(*_point(k))
    dr = delta(np.asarray(kr_lo, dtype=np.float64), np.asarray(kr_hi, dtype=np.float64))
    u_lo, u_hi = iv.isub(*dk, *dr)
    return u_lo, u_hi, 0.5 * (dr[0] + dr[1])


def ema_state(history, beta1, beta2):
    """(m, v) after the gradients in ``history`` (oldest first), from zero state, in float64."""

    b1, b2 = float(Fraction(beta1)), float(Fraction(beta2))
    m = np.zeros_like(np.asarray(history[0], dtype=np.float64))
    v = np.zeros_like(m)
    for g in history:
        g = np.asarray(g, dtype=np.float64)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
    return m, v
