#!/usr/bin/env python3
"""Review supplement: exact-integer boundary oracle and declaration binding.

Default executes source transcriptions in small fixtures, NOT the whole product.
--repo verifies executable AST against the baseline checkout, then executes the
same fixtures. --native executes the checkout's _int_op and expand directly;
its runtime/import dependencies must be installed. Nothing launches a GPU.
--expect fixed checks independent expectations after repair, with --native.
Outputs are outside the repository; no frozen source or results are modified.
"""
from __future__ import annotations
import argparse
import ast
import collections
import copy
import hashlib
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = '1aee15e9df7a45f434699b7016056e6650fdae99'
sys.path.insert(0, str(ROOT/'rerun/repro'))
import run_probes as old


def source_environment(repo: Path | None, native: bool) -> tuple[dict, dict]:
    env, _ = old.load_functions()
    env.update(copy=copy, hashlib=hashlib, json=json,
               missing_items=lambda d: [], MissingDeclaration=ValueError,
               DEFAULT_UNITS={'development':32,'confirmation':64,'seed_offset':0},
               DEFAULT_RULE_CLASSES={'fixed_mean':['R1','R5'],'aligned':['R2','R3']},
               ALLOWED_REFERENCE_CLASSES=['complete_composed'],
               DEFAULT_RESOLUTION={'ulp_fraction':0.125},
               _factor_levels=lambda *args: [{}], _versions=lambda: {'test_fixture':True})
    p=Path(__file__).with_name('new_excerpts.py')
    src=p.read_text(); tree=ast.parse(src)
    record={'native':native,'source_ast_check':{},'checkout_commit':None,
            'fixture_dependencies':[] if native else ['old audit TV/status helpers','measure validation/factor/version fixtures']}
    paths={'_int_op':'src/kernel_analyzer/reference_eval/ttir_eval.py',
           'expand':'src/kernel_analyzer/measure.py',
           '_execution_status':'src/kernel_analyzer/check.py'}
    if repo is not None:
        r=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True)
        record['checkout_commit']=r.stdout.strip() if r.returncode==0 else 'not a git checkout'
    if repo is not None and not native:
        for n in tree.body:
            if isinstance(n,ast.FunctionDef) and n.name in paths:
                actual=ast.parse((repo/paths[n.name]).read_text())
                other=next(x for x in ast.walk(actual) if isinstance(x,ast.FunctionDef) and x.name==n.name)
                equal=old.normalize(n)==old.normalize(other)
                record['source_ast_check'][n.name]=equal
                if not equal: raise ValueError(f'{n.name}: baseline AST mismatch; use native mode for repaired code')
    exec(compile(src,str(p),'exec'),env)
    if native:
        if repo is None: raise ValueError('--native requires --repo')
        sys.path.insert(0,str(repo/'src'))
        tt=importlib.import_module('kernel_analyzer.reference_eval.ttir_eval')
        measure=importlib.import_module('kernel_analyzer.measure')
        for module in (tt,measure):
            Path(module.__file__).resolve().relative_to((repo/'src').resolve())
        env['_int_op']=tt._int_op;env['TV']=tt.TV;env['expand']=measure.expand
        check=importlib.import_module('kernel_analyzer.check')
        env['_execution_status']=check._execution_status
    return env,record


def wrap(x: int, width: int) -> int:
    return ((x+(1<<(width-1)))%(1<<width))-(1<<(width-1))


def integer_cases(env: dict) -> list[dict]:
    rows=[]
    def test(width: int, kind: str, a: int, b: int, expected: int) -> None:
        op=NS(result_types=[NS(elem=f'i{width}')],node_id='review_integer')
        args=[env['TV']('i',f'i{width}',np.array([wrap(x,width)],dtype=np.int64)) for x in (a,b)]
        result=env['_int_op'](kind,op,args)
        observed=int(result.lo[0]) % (1<<width)
        expected_bits=expected % (1<<width)
        status=int(result.st[0])
        rows.append({'id':f'{width}:{kind}:{a}:{b}', 'width':width,'op':kind,
                     'a':str(a),'b':str(b),'expected_bits':str(expected_bits),
                     'observed_bits':str(observed),'status':status,
                     'matches':status==0 and observed==expected_bits})
    for w in (8,16,32,64):
        H=1<<(w-1); M=(1<<w)-1
        values=[0,1,2,H-1,H,H+1,M-1,M]
        for a in values:
            for b in (1,2,3,H-1,H,M):
                test(w,'divui',a,b,a//b)
                test(w,'remui',a,b,a%b)
                test(w,'ceildivui',a,b,(a+b-1)//b)
                test(w,'maxui',a,b,max(a,b))
                test(w,'minui',a,b,min(a,b))
            for shift in (0,1,w-2,w-1):
                test(w,'shrui',a,shift,a>>shift)
        for a in (-H,-H+1,-3,-1,0,1,H-1):
            for b in (-H,-3,-2,-1,1,2,3,H-1):
                if a==-H and b==-1: continue  # outside the defined signed division domain
                q=(abs(a)//abs(b))*(1 if (a>=0)==(b>=0) else -1)
                test(w,'divsi',a,b,q)
                test(w,'remsi',a,b,a-q*b)
                test(w,'ceildivsi',a,b,-((-a)//b))
    return rows


def declaration_cases(env: dict) -> list[dict]:
    base={'call':'unused:function','inputs':{'x':{'const':1}},
          'compare':{'mode':'A','measure':['out']},
          'budget':{'cpu_seconds':1,'gpu_seconds':1,'case_timeout':1,'max_units':96}}
    first=env['expand'](base)
    cases={'alpha':0.01,'units':{'development':16,'confirmation':80,'seed_offset':100},
           'resolution':{'ulp_fraction':0.0625},
           'magnitude_bound':{'elementwise':100.,'basis':'test change; not a proved bound'}}
    out=[]
    for key,value in cases.items():
        changed=copy.deepcopy(base);changed[key]=value
        expanded=env['expand'](changed)
        out.append({'field_changed':key,'expanded_field_changed':first.get(key)!=expanded.get(key),
                    'original_digest':first['declaration_sha256'],
                    'changed_digest':expanded['declaration_sha256'],
                    'digest_binds_change':first['declaration_sha256']!=expanded['declaration_sha256']})
    return out


def execution_cases(env: dict) -> list[dict]:
    out=[]
    for case,reason,k2,atomics in [
        ('unknown_same','not_established:execution validity not established@probe',1.,False),
        ('unknown_differ_with_atomic','not_established:execution validity not established@probe',2.,True),
        ('known_race','not_established:execution race@probe',1.,False),
        ('ordinary_stable',None,1.,False)]:
        row={'written':np.array([True]),'k':np.array([1.]),'k_reps':[np.array([k2])],
             'seed':0,'repeat_inputs_differ':False,'reasons':{reason:1} if reason else {}}
        result=env['_execution_status']([row],[{'float_atomics':atomics}],2)
        expected='per launch' if case=='ordinary_stable' else 'withheld'
        out.append({'case':case,'expected_statistics':expected,'observed':result,
                    'matches':result['statistics']==expected})
    return out


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path);ap.add_argument('--native',action='store_true')
    ap.add_argument('--expect',choices=['baseline','fixed','report'],default='baseline')
    ap.add_argument('--out',type=Path,default=ROOT/'recheck/results/additional_checks.json')
    args=ap.parse_args()
    env,source=source_environment(args.repo,args.native)
    integers=integer_cases(env); decl=declaration_cases(env); execution=execution_cases(env)
    bad=[x for x in integers if not x['matches']]
    groups=collections.Counter(f"i{x['width']}/{x['op']}" for x in bad)
    data={'schema':'audit-recheck-v2','baseline_commit':BASE,
          'scope':'isolated source functions, no GPU or full interpreter run',
          'python':platform.python_version(),'numpy':np.__version__,'source':source,
          'integer_case_count':len(integers),'integer_failures':len(bad),
          'integer_failures_by_op':dict(sorted(groups.items())),
          'integer_cases':integers,'declaration_checks':decl,'execution_checks':execution}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in data.items() if k!='integer_cases'},ensure_ascii=False,indent=2))
    if args.expect=='baseline':
        assert bad, 'No baseline integer failures; investigate version/source difference'
        assert all(not x['digest_binds_change'] for x in decl)
        assert [x['matches'] for x in execution]==[False,False,True,True]
        print('Known baseline defects reproduced. Exit 0 does NOT mean product correctness.')
    elif args.expect=='fixed':
        assert args.native,'Fixed-code checking requires --native --repo'
        assert not bad,f'{len(bad)} integer cases fail independent exact expectations'
        assert all(x['digest_binds_change'] for x in decl),'Declaration hash still omits measured settings'
        assert all(x['matches'] for x in execution),'Execution-validity gate still admits unknowns'
        print('This small rule-level regression passes; other findings/E2E are not certified.')
    return 0
if __name__=='__main__':
    raise SystemExit(main())
