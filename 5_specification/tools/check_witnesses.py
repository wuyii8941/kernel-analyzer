#!/usr/bin/env python3
"""Cross-check reviewed positive witnesses against existing specs; not a runtime adapter or whole-domain proof."""
from __future__ import annotations
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal
import argparse,ast,json,sys,yaml
import sympy as s
R=Path(__file__).resolve().parents[1]

def expr(text):
 if type(text) is int:return s.Integer(text)
 node=ast.parse(str(text),mode='eval').body
 def go(n):
  if isinstance(n,ast.Constant) and type(n.value) is int:return s.Integer(n.value)
  if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):
   q=go(n.operand);return q if isinstance(n.op,ast.UAdd) else -q
  if isinstance(n,ast.BinOp):
   a,b=go(n.left),go(n.right)
   if isinstance(n.op,ast.Add):return a+b
   if isinstance(n.op,ast.Sub):return a-b
   if isinstance(n.op,ast.Mult):return a*b
   if isinstance(n.op,ast.Div):return a/b
   if isinstance(n.op,ast.Pow):return a**b
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and len(n.args)==1 and not n.keywords:
   fs={'exp':s.exp,'log':s.log,'sqrt':s.sqrt}
   if n.func.id in fs:return fs[n.func.id](go(n.args[0]))
  raise ValueError(f'Unsupported expression {ast.dump(n)}')
 return go(node)

def endpoints(v):
 if isinstance(v,F):return v,v
 if type(v) is int:return F(v),F(v)
 if hasattr(v,'lo') and hasattr(v,'hi'):return F(v.lo),F(v.hi)
 raise TypeError(f'Unsupported reference result type {type(v)}')

def compare_tree(actual,expected,path=''):
 if isinstance(expected,list):
  if not isinstance(actual,list) or len(actual)!=len(expected):raise AssertionError(f'{path}: shape mismatch')
  return [r for i,(a,b) in enumerate(zip(actual,expected)) for r in compare_tree(a,b,f'{path}[{i}]')]
 lo,hi=endpoints(actual)
 if lo>hi:raise AssertionError(f'{path}: reversed interval')
 target=expr(expected)
 if target.is_Rational:
  point=F(int(target.p),int(target.q));kind='exact_rational_membership'
 else:
  point=F(Decimal(str(s.N(target,130))));kind='high_precision_point_crosscheck_not_certificate'
 if not lo<=point<=hi:raise AssertionError(f'{path}: expected {target} not inside actual enclosure')
 return [{'field':path,'check':kind,'width':str(hi-lo)}]

def run(source_root:Path):
 specs=source_root/'specification_v2/algorithm_specs'
 if not (specs/'phase1/spec_cross_entropy.py').is_file():raise ValueError('source-root must be extracted uploaded package root')
 sys.path[:0]=[str(specs/'phase1'),str(specs/'phase2')]
 import spec_cross_entropy as ce
 import spec_optimizers as aw
 import spec_training_program as tp
 result=[];skipped=[]
 for file in sorted((R/'contracts').glob('*.yaml')):
  c=yaml.safe_load(file.read_text())
  if c['id'] not in ('CE','ADAMW','RANKMEAN'):continue  # the reviewer's witnesses cover the three pilot contracts only
  for e in c['examples']:
   kind=e['expected']['outcome'];i=e['input'];cid=c['id']
   if kind in {'invalid_input','unsupported'}:
    skipped.append({'id':e['id'],'reason':'Adapter guard is a Codex implementation requirement, not yet implemented in the original package.'});continue
   actual=None;outcome='value'
   try:
    if cid=='ADAMW':
     state=i['state'];st=aw.AdamState(len(state['theta']));st.m=[F(x) for x in state['m']];st.v=[F(x) for x in state['v']];st.t=state['t']
     theta=aw.adam_step([F(x) for x in state['theta']],None if i['gradient'] is None else [F(x) for x in i['gradient']],st,F(i['lr']),F(i['beta1']),F(i['beta2']),F(i['eps']),F(i['weight_decay']),amsgrad=False,maximize=False,decoupled=True)
     actual={'theta':theta,'m':st.m,'v':st.v,'t':st.t}
    elif cid=='CE':
     prob=e['profile']=='probability_unsmoothed'
     y=[[F(x) for x in row] for row in i['target']] if prob else i['target']
     z=ce.cross_entropy([[F(x) for x in row] for row in i['logits']],y,weights=None if i['weights'] is None else [F(x) for x in i['weights']],ignore_index=i['ignore_index'],reduction=i['reduction'],label_smoothing=F(i['eps']),reading='R_A',prob_target=prob)
     if not z.get('established',False):outcome='not_established'
     elif z['nan']:outcome='domain_undefined'
     else:
      u=i['upstream'];grad=[[v*F(u[n] if isinstance(u,list) else u) for v in row] for n,row in enumerate(z['grad'])]
      actual={'loss':z['loss'],'gradient':grad}
    else:
     actual={'loss':tp.cross_rank_mean([F(x) for x in i['S']],i['n'])}
   except ZeroDivisionError:
    if cid=='ADAMW' and kind=='domain_undefined':outcome='domain_undefined'
    else:raise
   except tp.SpecNotEstablished:
    if cid=='RANKMEAN' and sum(i['n'])==0 and kind=='domain_undefined':outcome='domain_undefined'
    else:raise
   if kind!=outcome:raise AssertionError(f"{e['id']}: expected {kind}, received {outcome}")
   checks=[]
   if kind=='value':
    wanted=e['expected']['fields']
    if set(actual)!=set(wanted):raise AssertionError(f"{e['id']}: missing output fields")
    for key,expected in wanted.items():checks.extend(compare_tree(actual[key],expected,key))
   result.append({'id':e['id'],'outcome':outcome,'checks':checks})
 return {'scope':'independent explicit examples, directly calling original specs; no kernel/GPU; no production adapter certification','completed_cases':len(result),'implementation_pending_cases':skipped,'scalar_field_checks':sum(len(r['checks']) for r in result),'results':result}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args()
 try:
  d=run(a.source_root)
 except Exception as err:
  print(f'WITNESS CHECK FAILED: {type(err).__name__}: {err}',file=sys.stderr);return 1
 text=json.dumps(d,ensure_ascii=False,indent=2)
 if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text+'\n')
 print(json.dumps({k:v for k,v in d.items() if k!='results'},ensure_ascii=False,indent=2));return 0
if __name__=='__main__':sys.exit(main())
