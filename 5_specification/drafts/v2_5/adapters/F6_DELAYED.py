from _c import *
import spec_quant as S
def reference(call, a):
    st = S.DelayedScaling(history_len=a.get('L', 4), margin=a.get('margin', 0), algo=a.get('algo', 'max'))
    out = [st.step(QL(x)) for x in a['steps']]
    return tuple((used, tuple(q)) for used, q in out) if call == 'full' else tuple(used for used, _ in out)
def M1(call, a):        # current scaling: the scale of step t uses the amax of step t
    out = []
    for x in a['steps']:
        x = QL(x); sc = (F(448) / max(abs(v) for v in x)) / F(2) ** a.get('margin', 0)
        out.append((sc, tuple(S.round_fmt(v * sc, 'e4m3', saturate=True) for v in x)))
    return tuple(out) if call == 'full' else tuple(u for u, _ in out)
def M2(call, a):        # margin multiplies instead of divides
    st = S.DelayedScaling(history_len=a.get('L', 4), margin=-a.get('margin', 0), algo=a.get('algo', 'max'))
    out = [st.step(QL(x)) for x in a['steps']]
    return tuple((u, tuple(q)) for u, q in out) if call == 'full' else tuple(u for u, _ in out)
MUTANTS = dict(M1=M1, M2=M2)
