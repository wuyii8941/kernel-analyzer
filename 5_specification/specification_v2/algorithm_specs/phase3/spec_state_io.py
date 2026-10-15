"""Save/load path equivalence. Source: the documented contract of state_dict()/load_state_dict(): loading a saved state
and continuing must equal continuing without saving. The declared serialization is exact (every stored number is a
rational and round-trips bit-exactly); every field of the state, including counters (Adam step t, AveragedModel
n_averaged, BatchNorm num_batches_tracked), is part of the state."""
import json
from fractions import Fraction as F

def save(state):
    def enc(x):
        if isinstance(x, list): return [enc(v) for v in x]
        if isinstance(x, F): return {'q': f'{x.numerator}/{x.denominator}'}
        return x
    return json.dumps({k: enc(v) for k, v in vars(state).items()})

def load(blob, cls, drop=()):
    def dec(x):
        if isinstance(x, list): return [dec(v) for v in x]
        if isinstance(x, dict) and 'q' in x: n, d = x['q'].split('/'); return F(int(n), int(d))
        return x
    obj = cls.__new__(cls); d = json.loads(blob)
    for k, v in d.items():
        if k not in drop: setattr(obj, k, dec(v))
    for k in drop:                       # a loader that forgets a field leaves the class default (mutation for testing)
        fresh = cls(len(d['m'])) if 'm' in d else cls()
        setattr(obj, k, getattr(fresh, k))
    return obj
