#!/usr/bin/env python3
"""Recheck the full four-model/three-length mapping grid, not only successes."""
import argparse
import json
from pathlib import Path

from scripts.run_numerical_coverage import read, sha


def validate(mapping, tasks):
    by_task={t['task_id']:t for t in tasks['rows']}
    cases=mapping['cases']
    if len(by_task)!=len(tasks['rows']) or len({c['task_id'] for c in cases})!=len(cases):
        raise ValueError('Duplicate task identity')
    for case in cases:
        task=by_task[case['task_id']]
        if (task['status']!='EXACT_CANDIDATE_BUFFER_TO_AOT_SEMANTIC_ENDPOINT'
                or case['exact_aot_endpoint_id']!=task['exact_aot_endpoint_id']
                or case['expected_symbol']!=task.get('symbol')
                or case['carrier']!=case['mapping_evidence']['name']):
            raise ValueError('Mapping disagrees with declared task or parameter: '+case['task_id'])
    missing=mapping['unresolved_backward_outputs']
    if len(cases)+len(missing)!=mapping['exact_backward_endpoints']:
        raise ValueError('Backward mapping denominator differs')
    return dict(mapped_backward_positions=len(cases), unresolved_backward_positions=len(missing),
                release_positions=len(tasks['rows']),
                status='STATIC_TASK_RECORDS_CONSISTENT_NOT_RUNTIME_VERIFIED')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mapping-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('Choose a new output under /data1/tzh')
    rows=[]
    for model in ('deepseek','qwen','phi','mamba'):
        for length in (64,128,256):
            path=a.mapping_dir/f'{model}{length}.json'
            row=dict(model=model,sequence_length=length,mapping=str(path.resolve()))
            if not path.exists():
                rows.append(dict(**row,status='MISSING_MAPPING')); continue
            mapping=read(path)
            sources=[Path(s) for s in mapping['source_sha256'] if Path(s).name=='same_dtype_tasks.json.gz']
            if len(sources)!=1: raise ValueError('Ambiguous source task file')
            source=sources[0]
            if sha(source)!=mapping['source_sha256'][str(source)]: raise ValueError('Task source changed')
            rows.append(dict(**row,**validate(mapping,read(source)),mapping_sha256=sha(path),
                             tasks_sha256=sha(source),unresolved=mapping['unresolved_backward_outputs']))
    result=dict(configurations=rows,expected_configurations=12,
        records_checked=sum(r['status']=='STATIC_TASK_RECORDS_CONSISTENT_NOT_RUNTIME_VERIFIED' for r in rows),
        mapped_backward_positions=sum(r.get('mapped_backward_positions',0) for r in rows),
        unresolved_backward_positions=sum(r.get('unresolved_backward_positions',0) for r in rows),
        release_positions=sum(r.get('release_positions',0) for r in rows),
        source_sha256=sha(Path(__file__)),all_kernel_support_established=False,
        scope='Static declared task/parameter records; excludes runtime reach, internal-buffer expansion and forward mapping')
    with a.output.open('x') as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ('configurations','source_sha256')}))


if __name__=='__main__': main()
