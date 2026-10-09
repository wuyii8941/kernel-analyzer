"""Independent specification f for LayerNorm, RMSNorm, GroupNorm and BatchNorm.  (v0.1)

Documentation (fetched 2026-10-08):
  LayerNorm : https://docs.pytorch.org/docs/2.10/generated/torch.nn.LayerNorm.html
  RMSNorm   : https://docs.pytorch.org/docs/2.10/generated/torch.nn.RMSNorm.html
  GroupNorm : https://docs.pytorch.org/docs/2.10/generated/torch.nn.GroupNorm.html
  BatchNorm : https://docs.pytorch.org/docs/2.10/generated/torch.nn.BatchNorm2d.html

Documented semantics used:
  N-D1 LayerNorm: y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta, statistics over the last normalized_shape dims,
       "the standard-deviation is calculated via the biased estimator" (Var with correction 0); eps inside the sqrt.
  N-D2 RMSNorm: y = x / sqrt(eps + mean(x^2)) * gamma over normalized_shape; eps=None means torch.finfo(x.dtype).eps
       (the caller passes the resolved eps; which dtype's eps is part of the condition).
  N-D3 GroupNorm: channels split into num_groups groups; mean and biased variance over (C/G, *spatial) per sample and
       group; eps inside the sqrt; per-channel gamma, beta.
  N-D4 BatchNorm: training normalises with the biased batch variance (torch.var(input, correction=0)) over (N, *spatial)
       per channel, "however, the value stored in the moving average ... is calculated via the unbiased estimator"
       (correction=1). running <- (1 - momentum) * running + momentum * batch_stat; momentum=None means the cumulative
       moving average (simple average) over the batches seen; eval normalises with running statistics.

Ambiguities (ambiguities_phase2.md):
  NORM-A1 RMSNorm eps=None: finfo eps of WHICH dtype (input dtype, or the compute dtype after autocast/upcast)? The spec
          takes eps as given; the condition records which was used.
  NORM-A2 BatchNorm unbiased running variance when the per-channel count n = 1 (division by n - 1 = 0): undefined;
          spec declines. Also when N*spatial == 1 the biased variance is 0 and the normalised output is 0 * gamma + beta.
  NORM-A3 the "(1 + gamma)" convention of some RMSNorm variants (Gemma) is a different declared formula, not a reading of
          torch.nn.RMSNorm; candidates declaring it are compared against their own declared formula.
Numerics: exact rationals for means/variances; sqrt as a rigorous Interval. Inputs are nested lists.
"""
from fractions import Fraction as F
from rigorous import Interval, SpecInputError, SpecNotEstablished


def _flat(t):
    return [t] if not isinstance(t, list) else [v for e in t for v in _flat(e)]


def _mean(vals):
    return sum((F(v) for v in vals), F(0)) / len(vals)


def _var(vals, correction=0):
    m = _mean(vals)
    n = len(vals)
    if n - correction <= 0:
        raise SpecNotEstablished("variance with zero degrees of freedom is undefined")
    return sum(((F(v) - m) ** 2 for v in vals), F(0)) / (n - correction)


def layer_norm(x_row, gamma, beta, eps):
    """one normalisation row (the last normalized_shape dims flattened). Returns list of Intervals."""
    vals = [F(v) for v in x_row]
    if gamma is not None and len(gamma) != len(vals):
        raise SpecInputError("gamma length must equal the normalised size")
    m = _mean(vals); var = _var(vals, 0)
    denom = Interval.exact(var + F(eps)).sqrt()
    out = []
    for i, v in enumerate(vals):
        y = Interval.exact(v - m) / denom
        if gamma is not None:
            y = y * F(gamma[i])
        if beta is not None:
            y = y + F(beta[i])
        out.append(y)
    return out


def rms_norm(x_row, gamma, eps):
    vals = [F(v) for v in x_row]
    ms = sum((v * v for v in vals), F(0)) / len(vals)
    denom = Interval.exact(F(eps) + ms).sqrt()
    out = []
    for i, v in enumerate(vals):
        y = Interval.exact(v) / denom
        if gamma is not None:
            y = y * F(gamma[i])
        out.append(y)
    return out


def group_norm(x, num_groups, gamma, beta, eps):
    """x: [C][*spatial] for ONE sample (caller loops over N). Returns nested list of Intervals with x's shape."""
    C = len(x)
    if C % num_groups != 0:
        raise SpecInputError("num_channels must be divisible by num_groups")
    per = C // num_groups
    out = [None] * C
    for g in range(num_groups):
        chans = range(g * per, (g + 1) * per)
        vals = [F(v) for c in chans for v in _flat(x[c])]
        m = _mean(vals); var = _var(vals, 0)
        denom = Interval.exact(var + F(eps)).sqrt()
        for c in chans:
            def norm(t):
                if isinstance(t, list):
                    return [norm(e) for e in t]
                y = Interval.exact(F(t) - m) / denom
                if gamma is not None:
                    y = y * F(gamma[c])
                if beta is not None:
                    y = y + F(beta[c])
                return y
            out[c] = norm(x[c])
    return out


class BNState:
    def __init__(self, C):
        self.running_mean = [F(0)] * C; self.running_var = [F(1)] * C; self.num_batches_tracked = 0


def batch_norm(x, state, gamma, beta, eps, momentum=F(1, 10), training=True, track_running_stats=True):
    """x: [N][C][*spatial]. Training: normalise with biased batch statistics and update running statistics with the
    unbiased variance (N-D4). Eval: normalise with running statistics. Returns (output nested list of Intervals)."""
    N = len(x); C = len(x[0])
    out = [[None] * C for _ in range(N)]
    for c in range(C):
        vals = [F(v) for n in range(N) for v in _flat(x[n][c])]
        if training or not track_running_stats:
            m = _mean(vals); var_b = _var(vals, 0)
            if training and track_running_stats:
                var_u = _var(vals, 1)                        # NORM-A2: declines when count == 1
                state.num_batches_tracked += 1
                mom = F(1, state.num_batches_tracked) if momentum is None else F(momentum)
                state.running_mean[c] = (1 - mom) * state.running_mean[c] + mom * m
                state.running_var[c] = (1 - mom) * state.running_var[c] + mom * var_u
        else:
            m = state.running_mean[c]; var_b = state.running_var[c]
        denom = Interval.exact(var_b + F(eps)).sqrt()
        for n in range(N):
            def norm(t):
                if isinstance(t, list):
                    return [norm(e) for e in t]
                y = Interval.exact(F(t) - m) / denom
                if gamma is not None:
                    y = y * F(gamma[c])
                if beta is not None:
                    y = y + F(beta[c])
                return y
            out[n][c] = norm(x[n][c])
    return out


# ---------------------------------------------------------------- reference-free properties
def prop_shift_scale_invariance(fn, x_row, a, b, **kw):
    """Precondition: a != 0 and eps == 0 for exact invariance (with eps > 0 the identity holds only approximately).
    LayerNorm(a*x + b) == sign(a) * LayerNorm(x); RMSNorm(a*x) == sign(a) * RMSNorm(x)."""
    xs = [F(a) * F(v) + F(b) for v in x_row]
    return fn(xs, **kw), fn(x_row, **kw)


def prop_normalized_moments(y_row):
    """Precondition: gamma = None, beta = None, eps = 0. The output has mean 0 and biased variance 1 (as intervals)."""
    n = len(y_row)
    mean = sum(y_row[1:], y_row[0]) / n
    return mean
