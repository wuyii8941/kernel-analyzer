"""Bind checked internal GELU outputs to an explicitly declared parameter scope.

The parameter comes from the existing trainability protocol, not an inferred
dataflow path. Numerical reach remains a runtime question.
"""
import argparse
from pathlib import Path
from scripts.run_numerical_coverage import read, save, sha


def bind(scan, tasks, scope):
    parameters=scope.get('trainable_parameters',[])
    if len(parameters)!=1 or not isinstance(parameters[0],str):
        raise ValueError('Explicit existing single-parameter training scope required')
    contracts={}
    for source in scan['sources']:
        for row in source['rows']:
            if row['status']!='SOURCE_CHECKED': continue
            if row['symbol'] in contracts: raise ValueError('Ambiguous source symbol')
            contracts[row['symbol']]=row['contract']
    cases=[]
    seen=set()
    for task in tasks['rows']:
        if task['task_id'] in seen: raise ValueError('Duplicate task')
        seen.add(task['task_id'])
        contract=contracts.get(task.get('symbol'))
        if contract is None or task.get('formal_pointer')!=contract['output_pointer']: continue
        if task.get('phase')!='BACKWARD' or task.get('implementation_kind')!='TRITON':
            raise ValueError('Declared GELU task implementation differs')
        cases.append(dict(case_id='gelu_'+task['task_id'].replace(':','_'),
            task_id=task['task_id'],carrier=parameters[0],
            reference_method='GELU_PRODUCT_COMMON_INPUT', expected_symbol=task['symbol'],
            reference_output_pointer=contract['output_pointer'],implementation_kind='TRITON',
            exact_aot_endpoint_id=task.get('exact_aot_endpoint_id'),
            historical_boundary_status=task.get('status'),
            downstream_closure_tasks=task.get('closed_by_semantic_endpoint_tasks',[]),
            parameter_scope='DECLARED_SINGLE_TRAINABLE_PARAMETER',
            parameter_selection='HISTORICAL_TRAINABILITY_PROTOCOL_NOT_INFERRED_GRAPH_REACH',
            runtime_parameter_reach='NOT_YET_MEASURED'))
    return dict(schema='gelu-product-task-plan-v1',cases=cases,contracts=contracts,
        unmatched_checked_symbols=sorted(set(contracts)-{c['expected_symbol'] for c in cases}),
        trainable_parameters=parameters, runtime_measurement_complete=False,
        scope='Internal output replacements under original trainability; no fabricated AOT mapping')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('scan','release','trainability','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('New output under /data1/tzh required')
    scan=read(a.scan)
    for path,digest in scan['source_sha256'].items():
        if sha(Path(path))!=digest: raise ValueError('Scan dependency changed')
    for source in scan['sources']:
        path=Path(source['source'])
        if sha(path)!=source['source_sha256']: raise ValueError('Scanned source changed')
        expected=a.release/'trace'/path.parent.name/'output_code.py'
        if not expected.exists() or sha(expected)!=source['source_sha256']:
            raise ValueError('Scan does not belong to selected release')
    tasks=a.release/'same_dtype_tasks.json.gz'
    result=bind(scan,read(tasks),read(a.trainability))
    dependencies=[Path(__file__),a.scan,a.trainability,tasks,a.release/'capture.json',a.release/'inventory.json.gz']
    result['source_sha256']={str(path.resolve()):sha(path) for path in dependencies}
    save(a.output,result)
    print(dict(bound_tasks=len(result['cases']),unmatched_symbols=len(result['unmatched_checked_symbols'])))


if __name__=='__main__': main()
