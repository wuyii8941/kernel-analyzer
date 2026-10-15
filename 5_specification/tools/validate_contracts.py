#!/usr/bin/env python3
"""Validate review-package structure only. This does not certify an oracle."""
from __future__ import annotations
import argparse, ast, json, sys
from pathlib import Path
from typing import Any
import yaml

ROOT=Path(__file__).resolve().parents[1]
OUTCOMES={'value','domain_undefined','invalid_input','unsupported','not_established'}
GUARANTEES={'exact','rigorous_enclosure','allowed_set','declared_approximation'}

def check_expression(x:Any)->None:
    if isinstance(x,bool):raise ValueError('Boolean is not a numeric expected value')
    if isinstance(x,int):return
    if isinstance(x,list):
        for y in x:check_expression(y)
        return
    if not isinstance(x,str):raise ValueError('Expected values must be integers, rational/formula strings or lists')
    tree=ast.parse(x,mode='eval')
    def walk(n:ast.AST)->None:
        if isinstance(n,ast.Expression):walk(n.body)
        elif isinstance(n,ast.Constant) and type(n.value) is int:pass
        elif isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):walk(n.operand)
        elif isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow)):
            walk(n.left);walk(n.right)
        elif isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in {'exp','log','sqrt'} and len(n.args)==1 and not n.keywords:walk(n.args[0])
        else:raise ValueError(f'Unsupported expected-expression node: {ast.dump(n)}')
    walk(tree)

def validate_contract(c:dict,sources:dict)->list[str]:
    errors=[]
    def need(ok:bool,msg:str):
        if not ok:errors.append(msg)
    need(c.get('schema_version')=='1.1','schema_version must be 1.1')
    for field in ('id','role','title','review','activation','profiles','outputs','signature','domain','clauses','examples','mutation_requirements'):
        need(bool(c.get(field)),f'missing or empty {field}')
    if errors:return errors
    need(c['review'].get('status')=='reviewed_for_implementation','contract is not reviewed_for_implementation')
    need(c['activation'].get('implementation_status') in {'pending','validated'},'invalid implementation_status')
    need(c['activation'].get('runtime_binding_status') in {'pending','validated'},'invalid runtime_binding_status')
    for field,info in c['outputs'].items():need(info.get('guarantee') in GUARANTEES,f'output {field}: invalid guarantee')
    ids=[cl.get('id') for cl in c['clauses']]
    need(len(ids)==len(set(ids)),'duplicate clause IDs')
    dids=[d.get('id') for d in c.get('decisions',[])]
    need(len(dids)==len(set(dids)),'duplicate decision IDs')
    known={f'{sid}.{item}' for sid,s in sources.items() for item in s.get('items',{})}|set(dids)
    for node in [*c['clauses'],*c.get('decisions',[])]:
        refs=node.get('basis',[])
        need(isinstance(refs,list) and len(refs)>0,f"{node.get('id')}: empty/non-list basis")
        for ref in refs:need(ref in known,f"{node.get('id')}: unresolved basis {ref}")
    for cl in c['clauses']:
        need(set(cl.get('observes',[]))<=set(c['outputs']),f"{cl['id']}: unknown observed field")
        need(bool(cl.get('observes')),f"{cl['id']}: no observed field")
    eids=[e.get('id') for e in c['examples']]
    need(len(eids)==len(set(eids)),'duplicate example IDs')
    observed=set()
    for e in c['examples']:
        prefix=e.get('id','example')
        need(e.get('profile') in c['profiles'],f'{prefix}: unknown profile')
        need(set(e.get('clauses',[]))<=set(ids),f'{prefix}: unknown clause')
        observed.update(e.get('clauses',[]))
        expected=e.get('expected',{})
        need(expected.get('outcome') in OUTCOMES,f'{prefix}: invalid outcome')
        if expected.get('outcome')=='value':
            fields=expected.get('fields',{})
            need(set(fields)==set(c['outputs']),f'{prefix}: output keys do not exactly match required outputs')
            for v in fields.values():
                try:check_expression(v)
                except (ValueError,SyntaxError) as err:errors.append(f'{prefix}: {err}')
        else:need('fields' not in expected,f'{prefix}: non-value result must not invent value fields')
    need(set(ids)<=observed,'clauses without any example')
    mids=[m.get('id') for m in c['mutation_requirements']]
    need(len(mids)==len(set(mids)),'duplicate mutation IDs')
    for m in c['mutation_requirements']:
        ws=m.get('witnesses',[])
        need(bool(ws) and set(ws)<=set(eids),f"{m.get('id')}: missing or invalid witness")
    return errors

def validate_all(root:Path)->dict:
    try:
        sources=json.loads((root/'SOURCES.json').read_text())
        paths=sorted((root/'contracts').glob('*.yaml'))
        if {p.stem for p in paths}!={'CE','ADAMW','RANKMEAN'}:
            raise ValueError('Expected exactly three named pilot contracts; empty/partial runs are not success')
        contracts=[yaml.safe_load(p.read_text()) for p in paths]
        result={c['id']:validate_contract(c,sources) for c in contracts}
        counts={'contracts':len(contracts),'clauses':sum(len(c['clauses']) for c in contracts),'examples':sum(len(c['examples']) for c in contracts),'mutation_requirements':sum(len(c['mutation_requirements']) for c in contracts)}
        return {'kind':'contract_structure_validation_only','errors':result,'counts':counts,'runtime_ready':False,'runtime_note':'Reference adapters, numerical evidence and runtime bindings must still be implemented and validated by Codex.'}
    except (OSError,ValueError,KeyError,TypeError,yaml.YAMLError) as err:
        return {'kind':'contract_structure_validation_only','errors':{'package':[str(err)]},'runtime_ready':False}

def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--require-runtime-ready',action='store_true')
    a=p.parse_args();r=validate_all(a.root)
    print(json.dumps(r,ensure_ascii=False,indent=2))
    if any(r['errors'].values()):return 1
    if a.require_runtime_ready:return 2
    return 0
if __name__=='__main__':sys.exit(main())
