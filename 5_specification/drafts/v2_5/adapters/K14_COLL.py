from _c import *
import spec_collectives as K
def reference(call, a):
    if call == 'allreduce': return tuple(K.all_reduce(QL(a['xs']), a['op'], a.get('weights')))
    if call == 'rs': return tuple(tuple(s) for s in K.reduce_scatter(QL(a['xs'])))
    if call == 'gemm_rs': return tuple(tuple(tuple(r) for r in s) for s in K.gemm_rs(QL(a['A']), QL(a['B'])))
    if call == 'merge': return K.merge_partials([(Q(m), Q(l), QL(o)) for m, l, o in a['parts']])[0][0]
    return tuple(r[0] for r in K.ep_dispatch_combine(QL(a['x']), [[(e, Q(w)) for e, w in r] for r in a['routes']],
                                                     a['placement'], lambda e, x: [(e + 1) * v for v in x]))
def M1(call, a): return tuple(K.all_reduce(QL(a['xs']), 'avg')) if call == 'allreduce' and a['op'] == 'weighted' else None
def M2(call, a): return tuple(tuple(s) for s in reversed(K.reduce_scatter(QL(a['xs'])))) if call == 'rs' else None
def M3(call, a): return sum(Q(o[0]) for _, _, o in a['parts']) / len(a['parts']) if call == 'merge' else None
def M4(call, a):
    if call != 'ep': return None
    return tuple(sum((e + 1) * x[0] for e, _ in r) for x, r in zip(QL(a['x']), a['routes']))
MUTANTS = dict(M1=M1, M2=M2, M3=M3, M4=M4)
