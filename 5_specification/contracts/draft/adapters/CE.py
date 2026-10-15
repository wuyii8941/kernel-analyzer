from common import *
import spec_cross_entropy as ce
def _call(inp, reading='R_A'):
    r = ce.cross_entropy([[q(v) for v in row] for row in inp['logits']], inp['target'],
                         weights=None if 'weight' not in inp else [q(v) for v in inp['weight']], ignore_index=-100,
                         reduction=inp['reduction'], label_smoothing=q(inp.get('label_smoothing', 0)), reading=reading)
    if not r.get('established', True): return 'not established'
    return UNDEF if r['nan'] else bounds(r['loss'])
def reference(inp): return _call(inp)
def reading(name, inp): return _call(inp, reading=name)
# ---- misreadings, written from their one-line descriptions (no spec code used)
def _rows(inp, smoothing='uniform'):
    X, Y = inp['logits'], inp['target']; C = len(X[0]); eps = mpq(inp.get('label_smoothing', 0))
    w = [mpq(v) for v in inp['weight']] if 'weight' in inp else [mp.mpf(1)] * C
    out = []
    for x, y in zip(X, Y):
        if y == -100: out.append(None); continue
        x = [mpq(v) for v in x]; lse = mp.log(mp.fsum(mp.e ** v for v in x)); logp = [v - lse for v in x]
        if smoothing == 'uniform': qv = [(1 - eps) * (c == y) + eps / C for c in range(C)]
        else: qv = [(1 - eps) if c == y else eps / (C - 1) for c in range(C)]
        out.append(-mp.fsum(w[c] * qv[c] * logp[c] for c in range(C)) if eps else -w[y] * logp[y])
    return out, w
def _reduce(inp, rows, denom):
    s = mp.fsum(r for r in rows if r is not None)
    if inp['reduction'] == 'sum': return s
    return UNDEF if denom == 0 else s / denom
def M1(inp): rows, w = _rows(inp); return _reduce(inp, rows, len(rows))
def M2(inp): rows, w = _rows(inp); return _reduce(inp, rows, sum(1 for r in rows if r is not None))
def M3(inp): rows, w = _rows(inp, smoothing='other'); Y = inp['target']; return _reduce(inp, rows, mp.fsum(w[y] for y in Y if y != -100))
def M4(inp): rows, w = _rows(inp); Y = inp['target']; return _reduce(inp, rows, mp.fsum(w[y] for y in Y if y != -100) + sum(1 for y in Y if y == -100))
mutants = dict(M1=M1, M2=M2, M3=M3, M4=M4)
