"""Parameter averaging (torch.optim.swa_utils.AveragedModel). Source: PyTorch documentation of AveragedModel,
get_ema_avg_fn and get_swa_avg_fn. Declared semantics in exact rationals:
  update 0 (n_averaged == 0): averaged := copy of current parameters
  update n >= 1:  EMA  averaged := decay * averaged + (1 - decay) * current
                  SWA  averaged := averaged + (current - averaged) / (n + 1)
  n_averaged += 1 after every update (it is part of the state and must survive save/load)."""
from fractions import Fraction as F

class AveragedState:
    def __init__(self): self.n_averaged = 0; self.params = None

def _check(params, state):
    if state.params is not None and len(params) != len(state.params):
        raise ValueError(f'parameter length mismatch: {len(params)} vs {len(state.params)}')

def ema_update(state, params, decay):
    decay = F(decay)
    if not (0 <= decay <= 1): raise ValueError('decay must lie in [0, 1]')
    _check(params, state)
    if state.n_averaged == 0: state.params = [F(p) for p in params]
    else: state.params = [decay * a + (1 - decay) * F(p) for a, p in zip(state.params, params)]
    state.n_averaged += 1
    return state.params

def swa_update(state, params):
    _check(params, state)
    if state.n_averaged == 0: state.params = [F(p) for p in params]
    else:
        n = state.n_averaged
        state.params = [a + (F(p) - a) / (n + 1) for a, p in zip(state.params, params)]
    state.n_averaged += 1
    return state.params

# ---- properties derived from the recursions
def prop_swa_is_mean(history):
    """After updates with p_1..p_n, SWA equals the arithmetic mean (induction on n)."""
    st = AveragedState()
    for p in history: swa_update(st, p)
    mean = [sum((F(h[i]) for h in history), F(0)) / len(history) for i in range(len(history[0]))]
    return st.params, mean

def prop_ema_closed_form(history, decay):
    """EMA after the first copy: a_n = d^(n-1) p_1 + sum_{k=2..n} (1-d) d^(n-k) p_k (unrolling the recursion)."""
    st = AveragedState()
    for p in history: ema_update(st, p, decay)
    d = F(decay); n = len(history)
    closed = [d ** (n - 1) * F(history[0][i]) + sum(((1 - d) * d ** (n - k) * F(history[k - 1][i]) for k in range(2, n + 1)), F(0))
              for i in range(len(history[0]))]
    return st.params, closed
