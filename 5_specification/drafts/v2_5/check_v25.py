"""v2.5 contract checker. For every contract: schema (sources with identity, 12-dimension checklist, clause bases),
examples (established / contradiction / not established), readings, misreadings (each must be rejected by an example),
clause status, and the conclusion a clause supports given the identity of its sources."""
import os, sys, glob, importlib, yaml, mpmath as mp
from fractions import Fraction as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(HERE, 'adapters')]
from _c import iv, SpecNotEstablished
mp.mp.dps = 60
IDENT = {'standard', 'paper', 'torch_doc', 'model_impl', 'engine_ref', 'math'}
DIMS = [f'D{i}' for i in range(1, 13)]
IVT = type(iv.mpf(0))
ENV = dict(e=mp.e, exp=mp.exp, log=mp.log, sqrt=mp.sqrt, tanh=mp.tanh, mp=mp)

def canon(x):
    if isinstance(x, dict) and 'set' in x: return frozenset(canon(e) for e in x['set'])
    if isinstance(x, (frozenset, set)): return frozenset(canon(e) for e in x)
    if isinstance(x, (list, tuple)): return tuple(canon(e) for e in x)
    if isinstance(x, bool): return x
    if isinstance(x, (int, F)): return F(x)
    if x is None: return 'none'
    if isinstance(x, str):
        s = x.strip()
        if s in ('nan', 'inf', '-inf', 'none', 'undefined'): return s
        try: return F(s)
        except ValueError: return s
    return x

def as_iv(x):
    if isinstance(x, IVT): return mp.mpf(x.a), mp.mpf(x.b)
    if isinstance(x, F): v = mp.mpf(x.numerator) / x.denominator; return v, v
    v = mp.mpf(x); return v, v

def judge(ref, exp):
    if isinstance(ref, IVT):
        e = mp.mpf(eval(str(exp).replace('/', '*mp.mpf(1)/'), dict(ENV)))
        lo, hi = as_iv(ref); d = mp.mpf(10) ** -40 * max(1, abs(e))
        if e < lo - d or e > hi + d: return 'contradiction'
        return 'established' if hi - lo <= mp.mpf(10) ** -30 * max(1, abs(e)) else 'not established'
    return 'established' if canon(ref) == canon(exp) else 'contradiction'

def killed(ref, m):
    if m is None: return None
    if isinstance(ref, IVT):
        if isinstance(m, str): return True
        lo, hi = as_iv(ref); mlo, mhi = as_iv(m); d = mp.mpf(10) ** -40 * max(1, abs(lo))
        return mhi < lo - d or mlo > hi + d
    if isinstance(ref, frozenset): return canon(m) not in canon(ref)
    return canon(m) != canon(ref)

rows, summary, total = [], [], dict(clauses=0, established=0, mutants=0, rejected=0, examples=0)
for path in sorted(glob.glob(os.path.join(HERE, 'contracts', '*.yaml'))):
    c = yaml.safe_load(open(path)); A = importlib.import_module(c['id']); probs = []
    for sid, s in c['sources'].items():
        if s.get('identity') not in IDENT: probs.append(f'{sid}: identity missing')
    for d in DIMS:
        st = c['dimensions'].get(d, {})
        if st.get('status') not in ('defined', 'n/a', 'open'): probs.append(f'{d}: status missing')
        if st.get('status') == 'defined' and not all(x in c['clauses'] for x in st.get('clauses', [])): probs.append(f'{d}: unknown clause')
        if st.get('status') == 'open' and not st.get('reason'): probs.append(f'{d}: open without reason')
    out, refs = {}, {}
    for eid, e in c['examples'].items():
        try: ref = A.reference(e['call'], e['input'])
        except SpecNotEstablished: ref = 'undefined'
        except Exception as ex: ref = 'not established'; out[eid] = f'error: {ex}'; continue
        refs[eid] = ref; out[eid] = judge(ref, e['expected'])
        for rd, rexp in (e.get('readings') or {}).items():
            try: out[f'{eid}[{rd}]'] = judge(A.reading(rd, e['call'], e['input']), rexp)
            except Exception as ex: out[f'{eid}[{rd}]'] = f'error: {ex}'
    kills = {}
    for mid in c['mutants']:
        if mid not in A.MUTANTS: probs.append(f'{mid}: no implementation'); continue
        def mval(e):
            try: return A.MUTANTS[mid](e['call'], e['input'])
            except Exception: return None                      # a misreading that cannot evaluate this input is not applicable
        kills[mid] = [eid for eid, e in c['examples'].items()
                      if out.get(eid) == 'established' and killed(refs[eid], mval(e))]
    for cid, cl in c['clauses'].items():
        exs = [k for k, e in c['examples'].items() if cid in e['clauses']]
        if not exs: probs.append(f'{cid}: no example')
        res = [out[k] for k in exs]
        ms = [m for m, d in c['mutants'].items() if cid in d['clauses']]; alive = [m for m in ms if not kills.get(m)]
        st = ('open (no example)' if not exs else 'contradiction' if any(r == 'contradiction' or str(r).startswith('error') for r in res) else
              'open' if 'not established' in res else 'weak' if alive else 'established')
        ids = {c['sources'][b]['identity'] for b in cl['basis']}
        concl = 'agrees with a reference program' if ids <= {'model_impl', 'engine_ref'} else 'meets the task definition'
        rows.append((c['id'], cid, '/'.join(sorted(ids)), cl['guarantee'], ', '.join(f'{k}:{out[k]}' for k in exs),
                     ', '.join(f"{m}->{'/'.join(kills[m]) or 'ALIVE'}" for m in ms) or '-', st, concl))
        total['clauses'] += 1; total['established'] += st == 'established'
    total['mutants'] += len(c['mutants']); total['rejected'] += sum(1 for m in kills if kills[m]); total['examples'] += len(c['examples'])
    dimc = {s: sum(1 for d in DIMS if c['dimensions'][d]['status'] == s) for s in ('defined', 'n/a', 'open')}
    opens = [f"{d}: {c['dimensions'][d]['reason']}" for d in DIMS if c['dimensions'][d]['status'] == 'open']
    reads = {k: v for k, v in out.items() if '[' in k}
    summary.append((c['id'], c['kind'], c.get('survey_class'), dimc, opens, probs, reads))
print('| contract | clause | source identity | guarantee | examples | misreadings (rejected by) | status | conclusion supported |')
print('|---|---|---|---|---|---|---|---|')
for r in rows: print('| ' + ' | '.join(str(x) for x in r) + ' |')
print('\n| contract | kind | survey class | dimensions defined / n.a. / open | readings | schema problems |\n|---|---|---|---|---|---|')
for cid, k, sc, dc, op, pr, rd in summary:
    print(f"| {cid} | {k} | {sc} | {dc['defined']} / {dc['n/a']} / {dc['open']} | {', '.join(f'{a}={b}' for a, b in rd.items()) or '-'} | {'; '.join(pr) or 'none'} |")
print('\nOpen dimensions (reported as "not established" by the tool):')
for cid, k, sc, dc, op, pr, rd in summary:
    for o in op: print(f'- {cid} {o}')
print(f"\nTOTAL: {len(summary)} contracts, {total['clauses']} clauses ({total['established']} established), "
      f"{total['examples']} examples, misreadings rejected {total['rejected']}/{total['mutants']}")
