#!/usr/bin/env python3
"""Select indexed accumulation by exact saved task/parameter evidence, not effects."""
import argparse
import json
from pathlib import Path
from scripts.run_numerical_coverage import read,sha
from scripts.build_reference_reach_inventory import merge_carriers


def bind(tasks,cases):
    checked={}
    merge_carriers(checked,cases,tasks)
    mapping={c['task_id']:c for c in cases}
    bound=[]; unresolved=[]
    for task in tasks:
        if task.get('implementation_kind')!='DIRECT_ATEN' or task.get('symbol')!='index_put_': continue
        case=mapping.get(task['task_id'])
        if (not case or task.get('status')!='EXACT_CANDIDATE_BUFFER_TO_AOT_SEMANTIC_ENDPOINT'
                or task.get('formal_pointer')!='mutated_output_0'):
            unresolved.append(dict(task_id=task['task_id'],reason='EXACT_OUTPUT_PARAMETER_BINDING_REQUIRED'))
            continue
        bound.append(dict(case,case_id=case['case_id']+'-indexed-input-order',
             source_case_id=case['case_id'],reference_method='INDEXED_INPUT_ORDER_FP32',
             runtime_parameter_reach='NOT_YET_MEASURED',
             reference_scope='Pre-call FP32 rows, single index, accumulate=True; input-order variant, not exact truth',
             runtime_semantics_status='MUST_VALIDATE_ARGUMENTS_AND_EXECUTING_SOURCE'))
    return dict(cases=bound,unresolved=unresolved,numerical_results_read=False,
                training_support_established=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks',type=Path,required=True)
    p.add_argument('--mapping',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('Use new output under /data1/tzh')
    result=bind(read(a.tasks)['rows'],read(a.mapping)['cases'])
    result['source_sha256']={str(p.resolve()):sha(p) for p in (a.tasks,a.mapping,Path(__file__))}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(bound=len(result['cases']),unresolved=len(result['unresolved']))))


if __name__=='__main__':main()
