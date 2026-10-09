#!/usr/bin/env python3
"""CPU provenance regression on a source excerpt and simulated capture metadata.
Not a GPU capture. Optional --repo checks the executable AST against check.py.
"""
from __future__ import annotations
import argparse, ast, copy, json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]

def normalized(n):
    n=copy.deepcopy(n)
    if isinstance(n.body[0],ast.Expr) and isinstance(n.body[0].value,ast.Constant) and isinstance(n.body[0].value.value,str):
        n.body=n.body[1:]
    return ast.dump(n,include_attributes=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo');args=ap.parse_args()
    src=(ROOT/'repro/check_excerpts.py').read_text();node=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef))
    verified=None
    if args.repo:
        actual=Path(args.repo)/'src/kernel_analyzer/check.py'
        other=next(n for n in ast.parse(actual.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==node.name)
        verified=normalized(node)==normalized(other)
        assert verified,'Source executable AST differs'
    env={'np':np,'torch':torch};exec(src,env)
    x=torch.tensor([1.,2.,3.],dtype=torch.float32)
    zs=[('rounded_compute',(x[:1]+2.**-25).repeat(4)),('honest_copy',x[:1].repeat(4)),
        ('ordinary_compute',(x[:1]+0.125).repeat(4))]
    rows=[]
    for name,z in zs:
        raw=z.numpy().copy().view(np.uint8)
        inp=NS(kind='tensor',storage_ptr=101,name='upstream',dtype='float32',before=raw,after=raw.copy())
        out=NS(kind='tensor',storage_ptr=202,name='out',dtype='float32',before=np.zeros_like(raw),after=raw.copy())
        launch=NS(args=[inp,out],kernel_name='identity_kernel')
        ref=NS(loaded={101},loaded_any={101},stored={202})
        deps=env['torch_intermediates']([launch],NS(launches=[ref]),{'x':x})
        mixed=deps.get(202,[])
        backfilled=[m for m in mixed if not m.endswith('[copy of inputs]')]
        rows.append({'case':name,'z':z.tolist(),'upstream_tags':mixed,
                     'classification_from_current_measure_rule':'kernel-level' if backfilled else 'call-level',
                     'true_upstream_is_arithmetic':name!='honest_copy'})
    # check.run indexes the missing field before evaluate_sequence can choose a frontend.
    try:
        l=NS(asm={'ttgir':'module {}'})
        _=l.asm['ttir']
        gluon='unexpectedly succeeded'
    except KeyError as e:
        gluon=f'KeyError: {e}'
    data={'scope':'CPU excerpt + simulated metadata; not GPU launch',
          'source_ast_matches':verified,'torch':torch.__version__,'provenance':rows,
          'gluon_header_lookup':gluon}
    (ROOT/'results/binding_probes.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(data,ensure_ascii=False,indent=2))
    assert rows[0]['classification_from_current_measure_rule']=='call-level'
    assert rows[1]['classification_from_current_measure_rule']=='call-level'
    assert rows[2]['classification_from_current_measure_rule']=='kernel-level'
if __name__=='__main__':main()
