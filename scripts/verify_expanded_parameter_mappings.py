#!/usr/bin/env python3
"""Compare every overlapping old binding, without treating agreement as execution."""
import argparse
import json
from pathlib import Path
from scripts.run_numerical_coverage import sha, save


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mapping',type=Path,required=True)
    p.add_argument('--registry',type=Path,required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--sequence-length',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('Use a new output under /data1/tzh')
    mapping=json.loads(a.mapping.read_text()); registry=json.loads(a.registry.read_text())
    by_task={r['task_id']:r for r in mapping['cases']}
    if len(by_task)!=len(mapping['cases']):
        raise ValueError('Duplicate new task identities')
    rows=[]
    for cell in registry['cells']:
        if cell.get('model')!=a.model or cell.get('sequence_length')!=a.sequence_length:
            continue
        rep=cell['representative']; old=cell.get('nearest_carrier')
        if not rep.get('exact_endpoint_executable') or not old:
            continue
        new=by_task.get(rep['task_id'])
        rows.append({'task_id':rep['task_id'],'old_carrier':old['name'],
                     'new_carrier':None if new is None else new['carrier'],
                     'match':bool(new and new['carrier']==old['name'] and new['exact_aot_endpoint_id']==rep['exact_aot_endpoint_id']
                                  and new['expected_symbol']==rep['region_symbol'])})
    save(a.output,{'status':'MAPPINGS_AGREE_NOT_RUNTIME_VALIDATED' if rows and all(r['match'] for r in rows) else 'MAPPING_DISAGREEMENT_OR_NO_OVERLAP',
                   'expanded_count':len(by_task),'old_count_checked':len(rows),'rows':rows,
                   'source_sha256':{str(path.resolve()):sha(path) for path in (a.mapping,a.registry)}})
    print(json.dumps({'checked':len(rows),'mismatches':sum(not r['match'] for r in rows)}))


if __name__=='__main__':main()
