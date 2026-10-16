"""Class R (randomness): checks conditional on the kernel's own random numbers, plus a separate statistical test of the
random stream. Sources: torch.nn.Dropout documentation (inverted dropout, scale 1 / (1 - p)) ... torch_doc;
inverse-CDF sampling given a uniform u .... mathematical definition (D1: strict vs non-strict comparison, D2: temperature floor).
Dimensions: D9 (conditioning on the kernel's mask / uniforms), D12 (dropout backward reuses the forward mask)."""
from _base import *
from scipy.stats import beta as _beta

def dropout_fwd(x, keep, p):
    p = Q(p); return [Q(v) * k / (1 - p) for v, k in zip(x, keep)]

def dropout_bwd(dy, keep, p):
    p = Q(p); return [Q(g) * k / (1 - p) for g, k in zip(dy, keep)]

def keep_rate_interval(n_keep, n, alpha=0.05):
    """two-sided Clopper-Pearson interval for the keep probability; the stream passes if 1 - p lies inside."""
    lo = 0.0 if n_keep == 0 else _beta.ppf(alpha / 2, n_keep, n - n_keep + 1)
    hi = 1.0 if n_keep == n else _beta.ppf(1 - alpha / 2, n_keep + 1, n - n_keep)
    return lo, hi

def sample_inverse_cdf(probs, u, rule='gt'):
    """smallest i with cumulative sum > u ('gt') or >= u ('ge'); exact for rational probabilities."""
    c = F(0); u = Q(u)
    for i, p in enumerate(probs):
        c += Q(p)
        if (c > u) if rule == 'gt' else (c >= u): return i
    return len(probs) - 1
