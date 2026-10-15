from common import *
import spec_optimizers as so
def reference(inp):
    st = so.AdamState(1); theta = [q(inp['theta'])]
    for g in inp['g']:
        theta = so.adam_step(theta, [q(g)], st, q(inp['lr']), q(inp['beta1']), q(inp['beta2']), q(inp['eps']), q(inp['wd']), decoupled=True)
    return bounds(theta[0])
def _adam(inp, decay='before', coupled=False, bias=True, eps_inside=False):
    th = mpq(inp['theta']); lr, b1, b2, eps, wd = (mpq(inp[k]) for k in ('lr', 'beta1', 'beta2', 'eps', 'wd')); m = v = mp.mpf(0)
    for t, g in enumerate(inp['g'], 1):
        g = mpq(g) + (wd * th if coupled else 0)
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        mh, vh = (m / (1 - b1 ** t), v / (1 - b2 ** t)) if bias else (m, v)
        step = lr * mh / (mp.sqrt(vh + eps) if eps_inside else mp.sqrt(vh) + eps)
        if coupled: th = th - step
        elif decay == 'before': th = th * (1 - lr * wd) - step
        else: th = (th - step) * (1 - lr * wd)
    return th
mutants = dict(M1=lambda i: _adam(i, decay='after'), M2=lambda i: _adam(i, coupled=True), M3=lambda i: _adam(i, bias=False), M4=lambda i: _adam(i, eps_inside=True))
