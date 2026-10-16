from _c import *
import spec_moe_routing as M
def reference(call, a):
    if call == 'align':
        r = M.moe_align(a['ids'], a['E'], a['B'])
        return (tuple(r['padded'][e] for e in range(a['E'])), r['total'], tuple(r['expert_of_block']), r['pad_id'],
                tuple(tuple(sorted(r['sets'][e])) for e in range(a['E'])))
    return M.fused_moe(QL(a['x']), a['ids'], QL(a['w']), QL(a['Wg']), QL(a['Wu']), QL(a['Wd']))[0]
def M1(call, a):        # pad id -1
    if call != 'align': return None
    r = reference(call, a); return r[:3] + (-1,) + r[4:]
def M2(call, a):        # routing weight applied to the input of the first GEMM
    if call != 'moe': return None
    x = [Q(a['w'][0]) * v for v in QL(a['x'])]
    return M.fused_moe(x, a['ids'], [F(1)], QL(a['Wg']), QL(a['Wu']), QL(a['Wd']))[0]
def M3(call, a):        # an expert with no tokens still gets one block
    if call != 'align': return None
    r = M.moe_align(a['ids'], a['E'], a['B']); pad = tuple(max(a['B'], r['padded'][e]) for e in range(a['E']))
    return (pad, sum(pad), tuple(e for e in range(a['E']) for _ in range(pad[e] // a['B'])), r['pad_id'],
            tuple(tuple(sorted(r['sets'][e])) for e in range(a['E'])))
MUTANTS = dict(M1=M1, M2=M2, M3=M3)
