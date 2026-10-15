"""Checks each contract against its executable reference. Outcomes per example: established / contradiction / not established.
Each misreading (mutant) must be rejected by at least one example whose reference result is established."""
import sys, os, glob, importlib, yaml, mpmath as mp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(HERE, 'adapters')]
from common import UNDEF
mp.mp.dps = 60
ENV = dict(log=mp.log, exp=mp.exp, sqrt=mp.sqrt, e=mp.e)
def expect(s): return UNDEF if s == 'undefined' else mp.mpf(eval(str(s).replace('/', '*mp.mpf(1)/'), dict(ENV, mp=mp)))
def judge(ref, exp_):
    if exp_ == UNDEF: return 'established' if ref == UNDEF else 'contradiction'
    if ref in (UNDEF, 'not established'): return 'contradiction' if ref == UNDEF else 'not established'
    lo, hi = ref; d = mp.mpf(10) ** -50 * max(1, abs(exp_))
    if exp_ < lo - d or exp_ > hi + d: return 'contradiction'
    return 'established' if hi - lo <= mp.mpf(10) ** -30 * max(1, abs(exp_)) else 'not established'
report, summary = [], []
for path in sorted(p for p in glob.glob(os.path.join(HERE, '*.yaml')) if not p.endswith('TEMPLATE.yaml')):
    c = yaml.safe_load(open(path)); A = importlib.import_module(c['id'])
    problems = []
    for cid, cl in c['clauses'].items():
        if not cl.get('basis'): problems.append(f'{cid}: no basis')
        if cl.get('guarantee') not in ('exact', 'rigorous_enclosure', 'allowed_set', 'declared_approximation'): problems.append(f'{cid}: guarantee label missing')
        if not any(cid in e['clauses'] for e in c['examples'].values()): problems.append(f'{cid}: no example')
    ex_out, refs = {}, {}
    for eid, e in c['examples'].items():
        ref = A.reference(e['input']); refs[eid] = ref; ex_out[eid] = judge(ref, expect(e['expected']))
        for rd, rexp in (e.get('readings') or {}).items():
            ex_out[f'{eid}[{rd}]'] = judge(A.reading(rd, e['input']), expect(rexp))
    killed = {}
    for mid, m in c['mutants'].items():
        if mid not in A.mutants: problems.append(f'{mid}: no implementation'); continue
        killers = []
        for eid, e in c['examples'].items():
            if e['expected'] == 'undefined' or ex_out[eid] != 'established': continue
            mv = A.mutants[mid](e['input']); lo, hi = refs[eid]; d = mp.mpf(10) ** -40 * max(1, abs(lo))
            if mv == UNDEF or mv < lo - d or mv > hi + d: killers.append(eid)
        killed[mid] = killers
    for cid, cl in c['clauses'].items():
        exs = [k for k, e in c['examples'].items() if cid in e['clauses']]
        outs = [ex_out[k] for k in exs]
        ms = [k for k, m in c['mutants'].items() if cid in m['clauses']]
        alive = [k for k in ms if not killed.get(k)]
        st = 'contradiction' if 'contradiction' in outs else 'open' if 'not established' in outs else 'weak (misreading not rejected)' if alive else 'established'
        report.append((c['id'], cid, cl['basis'], cl['guarantee'], ', '.join(f'{k}:{ex_out[k]}' for k in exs),
                       ', '.join(f"{k}->{'/'.join(killed[k]) or 'ALIVE'}" for k in ms) or '-', st))
    summary.append((c['id'], len(c['clauses']), len(c['examples']), len(c['mutants']), sum(1 for k in killed if killed[k]), problems,
                    {k: v for k, v in ex_out.items() if '[' in k}))
print('| contract | clause | basis | guarantee | examples | misreadings (rejected by) | status |\n|---|---|---|---|---|---|---|')
for r in report: print('| ' + ' | '.join(str(x) for x in r) + ' |')
print()
for cid, nc, ne, nm, nk, pr, rd in summary:
    print(f'{cid}: clauses={nc} examples={ne} misreadings rejected {nk}/{nm}; schema problems={pr or "none"}; reading checks={rd or "-"}')
