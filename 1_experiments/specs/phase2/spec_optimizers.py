"""Independent specification f for one step and short sequences of torch.optim.{AdamW, Adam, SGD, RMSprop}.  (v0.1)

Written ONLY from the official documentation algorithm boxes (no PyTorch source consulted for semantics):
  Adam / AdamW : https://docs.pytorch.org/docs/2.10/generated/torch.optim.Adam.html, .../torch.optim.AdamW.html
  SGD          : https://docs.pytorch.org/docs/2.10/generated/torch.optim.SGD.html
  RMSprop      : https://docs.pytorch.org/docs/2.10/generated/torch.optim.RMSprop.html
  zero_grad    : https://docs.pytorch.org/docs/2.10/generated/torch.optim.Optimizer.zero_grad.html
  (fetched 2026-10-08; the algorithm boxes are quoted in ambiguities_phase2.md with the version they come from)

Documented semantics used:
  O-D1 (Adam/AdamW) g_t <- grad; if maximize: g_t <- -g_t.
       Adam (L2):        if lambda != 0: g_t <- g_t + lambda * theta_{t-1}
       AdamW (decoupled): theta_t <- theta_{t-1} - gamma * lambda * theta_{t-1}  (before the adaptive update)
       m_t <- beta1 m_{t-1} + (1-beta1) g_t ;  v_t <- beta2 v_{t-1} + (1-beta2) g_t^2
       m^_t <- m_t / (1 - beta1^t)
       if amsgrad: v_t^max <- max(v_{t-1}^max, v_t); v^_t <- v_t^max / (1 - beta2^t)   [recent docs, see OPT-A1]
       else:       v^_t <- v_t / (1 - beta2^t)
       theta_t <- theta_t - gamma * m^_t / (sqrt(v^_t) + eps)
  O-D2 (SGD) g_t <- grad; if lambda != 0: g_t <- g_t + lambda theta_{t-1};
       if mu != 0: b_t <- mu b_{t-1} + (1 - tau) g_t for t > 1, b_1 <- g_1 (no dampening on the first step);
       if nesterov: g_t <- g_t + mu b_t else g_t <- b_t ;  theta_t <- theta_{t-1} - gamma g_t (+ if maximize).
  O-D3 (RMSprop) g_t <- grad (+ lambda theta_{t-1} if lambda != 0); v_t <- alpha v_{t-1} + (1-alpha) g_t^2;
       if centered: gave_t <- alpha gave_{t-1} + (1-alpha) g_t ; v~_t <- v_t - gave_t^2  else v~_t <- v_t;
       if mu > 0: b_t <- mu b_{t-1} + g_t / (sqrt(v~_t) + eps); theta_t <- theta_{t-1} - gamma b_t
       else: theta_t <- theta_{t-1} - gamma g_t / (sqrt(v~_t) + eps).   maximize negates g_t first.
  O-D4 (zero_grad) "optimizers have a different behavior if the gradient is 0 or None (in one case it does the step with
       a gradient of 0 and in the other it skips the step altogether)": grad is None -> the parameter is skipped entirely
       (no state update, no weight decay, step counter not advanced for that parameter); grad == 0 -> a full step.

Ambiguities / version dependence (ambiguities_phase2.md):
  OPT-A1 amsgrad: recent documentation takes the max over the UNCORRECTED second moment v_t and bias-corrects the max
         (main reading); documentation of older releases wrote v^_t^max <- max(v^_t^max, v^_t) over the bias-corrected
         value (reading R_old). The two differ once t > 1. Classified by the locked version's documentation.
  OPT-A2 weight decay with maximize: the boxes apply maximize to g_t first and weight decay unchanged; recorded as read.
  OPT-A3 SGD first step: b_1 <- g_1 regardless of dampening (documented); an implementation that applies (1-tau) at t=1
         is a reading difference under declared interpretation only if its documentation says so, otherwise a violation.
Numerics: all quantities exact rationals except sqrt, which is a rigorous Interval; states are kept exact between steps
so that a short sequence is exact as well. Non-finite gradients are refused here (a candidate's skip logic under AMP is
specified in the AMP/GradScaler spec, not here).
"""
from fractions import Fraction as F
from rigorous import Interval, SpecInputError, imax


def _iv(v):
    return Interval.exact(v)


def _num(v):
    """exact rational, or an Interval carried over from an earlier step of a sequence."""
    return v if isinstance(v, Interval) else F(v)


def _mx(a, b):
    if isinstance(a, Interval) or isinstance(b, Interval):
        return imax(_iv(a), _iv(b))
    return max(a, b)


def _check_hp(**hp):
    for k, v in hp.items():
        if v is None:
            continue
        if isinstance(v, bool):
            continue
        if not isinstance(v, (int, F, float)):
            raise SpecInputError(f"hyperparameter {k} must be a number, got {type(v).__name__}")


class AdamState:
    def __init__(self, n):
        self.m = [F(0)] * n; self.v = [F(0)] * n; self.vmax = [F(0)] * n; self.t = 0


def adam_step(theta, grad, state, lr, beta1=F(9, 10), beta2=F(999, 1000), eps=F(1, 10 ** 8), weight_decay=F(0),
              amsgrad=False, maximize=False, decoupled=True, amsgrad_reading="main"):
    """One AdamW (decoupled=True) or Adam (decoupled=False) step on a flat parameter list.
    grad None -> skipped (O-D4); returns (new theta as list of Interval/Fraction, state updated in place)."""
    _check_hp(lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay)
    lr, b1, b2, eps, wd = F(lr), F(beta1), F(beta2), F(eps), F(weight_decay)
    if grad is None:
        return [_num(x) for x in theta]                   # whole parameter skipped, state untouched
    if len(grad) != len(theta):
        raise SpecInputError("grad and theta length differ")
    state.t += 1
    t = state.t
    out = []
    for i, (x, g) in enumerate(zip(theta, grad)):
        x, g = _num(x), _num(g)
        if maximize:
            g = -g
        if not decoupled and wd != 0:
            g = g + wd * x
        x_pre = x * (1 - lr * wd) if (decoupled and wd != 0) else x
        state.m[i] = b1 * state.m[i] + (1 - b1) * g
        state.v[i] = b2 * state.v[i] + (1 - b2) * g * g
        m_hat = state.m[i] / (1 - b1 ** t)
        if amsgrad:
            if amsgrad_reading == "main":
                state.vmax[i] = _mx(state.vmax[i], state.v[i])
                v_hat = state.vmax[i] / (1 - b2 ** t)
            else:                                            # R_old: max over bias-corrected values
                v_hat_now = state.v[i] / (1 - b2 ** t)
                state.vmax[i] = _mx(state.vmax[i], v_hat_now)
                v_hat = state.vmax[i]
        else:
            v_hat = state.v[i] / (1 - b2 ** t)
        denom = _iv(v_hat).sqrt() + eps
        out.append(_iv(x_pre) - _iv(lr * m_hat) / denom)
    return out


class SGDState:
    def __init__(self, n):
        self.b = [None] * n                                 # None = momentum buffer not yet initialised


def sgd_step(theta, grad, state, lr, momentum=F(0), dampening=F(0), weight_decay=F(0), nesterov=False, maximize=False):
    _check_hp(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay)
    lr, mu, tau, wd = F(lr), F(momentum), F(dampening), F(weight_decay)
    if nesterov and (mu == 0 or tau != 0):
        raise SpecInputError("nesterov requires momentum > 0 and dampening == 0 (documented precondition)")
    if grad is None:
        return [_num(x) for x in theta]
    out = []
    for i, (x, g) in enumerate(zip(theta, grad)):
        x, g = _num(x), _num(g)
        if wd != 0:
            g = g + wd * x
        if mu != 0:
            if state.b[i] is None:
                state.b[i] = g                               # O-D2: first step, no dampening
            else:
                state.b[i] = mu * state.b[i] + (1 - tau) * g
            g = g + mu * state.b[i] if nesterov else state.b[i]
        out.append(x + lr * g if maximize else x - lr * g)
    return out


class RMSpropState:
    def __init__(self, n):
        self.v = [F(0)] * n; self.gave = [F(0)] * n; self.b = [F(0)] * n


def rmsprop_step(theta, grad, state, lr, alpha=F(99, 100), eps=F(1, 10 ** 8), weight_decay=F(0), momentum=F(0),
                 centered=False, maximize=False):
    _check_hp(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum)
    lr, a, eps, wd, mu = F(lr), F(alpha), F(eps), F(weight_decay), F(momentum)
    if grad is None:
        return [_num(x) for x in theta]
    out = []
    for i, (x, g) in enumerate(zip(theta, grad)):
        x, g = _num(x), _num(g)
        if maximize:
            g = -g
        if wd != 0:
            g = g + wd * x
        state.v[i] = a * state.v[i] + (1 - a) * g * g
        if centered:
            state.gave[i] = a * state.gave[i] + (1 - a) * g
            vt = state.v[i] - state.gave[i] ** 2
        else:
            vt = state.v[i]
        denom = _iv(vt).sqrt() + eps
        if mu > 0:
            # b_t is an interval once division by the interval denominator enters; keep exact midpoint bookkeeping
            # impossible, so the buffer becomes an Interval (sequence stays rigorous).
            step = _iv(g) / denom
            prev = state.b[i] if isinstance(state.b[i], Interval) else _iv(state.b[i])
            state.b[i] = prev * mu + step
            out.append(_iv(x) - state.b[i] * lr)
        else:
            out.append(_iv(x) - (_iv(g) / denom) * lr)
    return out


def run_sequence(step_fn, theta, grads, state, **hp):
    """Apply step_fn over a list of gradient vectors (None entries = skipped steps per O-D4). Parameters that become
    Intervals stay Intervals; the function returns the trajectory as a list of parameter lists."""
    traj = []
    cur = list(theta)
    for g in grads:
        cur = step_fn(cur, g, state, **hp)
        traj.append(cur)
    return traj


# ---------------------------------------------------------------- reference-free properties
def prop_none_is_skip(step_fn, theta, state_factory, **hp):
    """Precondition: none. grad=None must leave parameters AND state untouched (O-D4)."""
    st = state_factory(len(theta))
    before = repr(vars(st))
    out = step_fn(theta, None, st, **hp)
    return [_num(x) for x in theta] == out and repr(vars(st)) == before


def prop_maximize_mirror(step_fn, theta, grad, state_factory, **hp):
    """Precondition: weight_decay == 0. With maximize=True and gradient g the step equals the step with maximize=False
    and gradient -g (both for the parameter and for the stateful moments)."""
    if F(hp.get("weight_decay", 0)) != 0:
        raise SpecInputError("mirror property needs weight_decay == 0")
    s1, s2 = state_factory(len(theta)), state_factory(len(theta))
    a = step_fn(theta, grad, s1, maximize=True, **hp)
    b = step_fn(theta, [-F(g) for g in grad], s2, maximize=False, **hp)
    return a, b
