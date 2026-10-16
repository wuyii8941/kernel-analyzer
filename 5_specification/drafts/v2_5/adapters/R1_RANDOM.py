from _c import *
import spec_random as R
def reference(call, a):
    if call == 'fwd': return tuple(R.dropout_fwd(QL(a['x']), a['keep'], Q(a['p'])))
    if call == 'bwd': return tuple(R.dropout_bwd(QL(a['dy']), a['keep'], Q(a['p'])))
    if call == 'rate':
        lo, hi = R.keep_rate_interval(a['n_keep'], a['n']); return 'pass' if lo <= 1 - float(Q(a['p'])) <= hi else 'fail'
    return R.sample_inverse_cdf(QL(a['probs']), Q(a['u']))
def reading(name, call, a): return R.sample_inverse_cdf(QL(a['probs']), Q(a['u']), rule='ge')
def M1(call, a):        # backward multiplies by (1 - p) (trident, survey)
    return tuple(Q(g) * k * (1 - Q(a['p'])) for g, k in zip(a['dy'], a['keep'])) if call == 'bwd' else None
def M2(call, a):        # backward rebuilds the mask from y > 0 (trident, survey)
    if call != 'bwd' or 'x' not in a: return None
    y = R.dropout_fwd(QL(a['x']), a['keep'], Q(a['p'])); m = [1 if v > 0 else 0 for v in y]
    return tuple(Q(g) * k / (1 - Q(a['p'])) for g, k in zip(a['dy'], m))
def M3(call, a):
    return ('pass' if abs(a['n_keep'] / a['n'] - (1 - float(Q(a['p'])))) < 0.01 else 'fail') if call == 'rate' else None
MUTANTS = dict(M1=M1, M2=M2, M3=M3)
