#!/usr/bin/env python3
"""Freeze generated indexed-accumulation calls without reading effect results."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from scripts.run_numerical_coverage import read,sha


def check_call(expression):
    call=ast.parse(expression,mode='eval').body
    if (not isinstance(call,ast.Call) or ast.unparse(call.func)!='aten.index_put_'
            or len(call.args)!=4 or call.keywords
            or not isinstance(call.args[0],ast.Name) or not isinstance(call.args[2],ast.Name)
            or not isinstance(call.args[1],ast.List) or len(call.args[1].elts)!=1
            or not isinstance(call.args[1].elts[0],ast.Name)
            or not isinstance(call.args[3],ast.Constant) or call.args[3].value is not True):
        raise ValueError('Expected one-index, accumulate=True generated call')
    return dict(initial_name=call.args[0].id,index_name=call.args[1].elts[0].id,
                values_name=call.args[2].id,call_ast_sha256=hashlib.sha256(ast.dump(call).encode()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bound-plan',type=Path,required=True)
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('Use a new output under /data1/tzh')
    inventory_path=a.release/'inventory.json.gz'
    task_path=a.release/'same_dtype_tasks.json.gz'
    tasks={r['task_id']:r for r in read(task_path)['rows']}
    rows=read(inventory_path)['runtime_call_audit']['rows']
    contracts=[]
    for case in read(a.bound_plan)['cases']:
        task=tasks[case['task_id']]
        matches=[r for r in rows if r.get('compute_region_id')==task['candidate_region_id']
                 and r.get('implementation_kind_or_helper_role')=='DIRECT_ATEN']
        if len(matches)!=1: raise ValueError('Ambiguous generated call')
        row=matches[0]
        source=(a.release/'trace'/row['source_path']).resolve()
        if not source.is_relative_to((a.release/'trace').resolve()): raise ValueError('Source outside declared trace')
        text=source.read_text(); line=text.splitlines()[row['source_line']-1].strip()
        if hashlib.sha256(line.encode()).hexdigest()!=row['source_line_sha256']:
            raise ValueError('Recorded call line changed')
        contract=check_call(line)
        if contract!=check_call(row['call_expression']) or contract['initial_name']!=task['tensor_variable']:
            raise ValueError('Call and task output disagree')
        contracts.append(dict(contract,case_id=case['case_id'],task_id=case['task_id'],
            source_path=str(source),source_sha256=sha(source),source_line=row['source_line'],
            source_line_sha256=row['source_line_sha256'],
            runtime_identity_status='MUST_VALIDATE_LIVE_GENERATED_MODULE'))
    result=dict(schema='indexed-call-source-contracts-v1',contracts=contracts,
        source_sha256={str(p.resolve()):sha(p) for p in (a.bound_plan,inventory_path,task_path,Path(__file__))},
        numerical_results_read=False,runtime_measurement_complete=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(checked_calls=len(contracts))))


if __name__=='__main__': main()
