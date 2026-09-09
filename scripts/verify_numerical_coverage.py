#!/usr/bin/env python3
"""Verify completed queue outputs against raw data and declared parameters.

This checks recorded evidence and recomputation, not independent execution of
the CUDA kernels. Unfinished and failed attempts stay in the report.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from kernel_analyzer.numerical_campaign import digest
from kernel_analyzer.training_numerical_analysis import analyze_artifact
from scripts.run_numerical_coverage import read, save, sha


def verify(root):
    protocol=read(root/'protocol.json'); coverage=read(root/'coverage.json')
    if digest(coverage)!=protocol['coverage_sha256']:
        raise ValueError('Coverage changed')
    snapshots=root/'source_snapshot.json'
    snapshot_verified=False
    if snapshots.exists():
        sources=read(snapshots)
        required={path for path in protocol['source_sha256'] if Path(path).suffix=='.py'}
        snapshot_verified=set(sources)==required and bool(sources) and all(hashlib.sha256(text.encode()).hexdigest()==protocol['source_sha256'][path]
                                               for path,text in sources.items())
        if not snapshot_verified:
            raise ValueError('Source snapshot differs from frozen digest')
    records=[]
    for row in coverage['rows']:
        run=root/'runs'/hashlib.sha256(row['task_id'].encode()).hexdigest()[:20]
        if not (run/'status.json').exists():
            continue
        status=read(run/'status.json')
        if status.get('status')!='VALID':
            records.append({'task_id':row['task_id'],'status':status['status'],'verified_measurement':False})
            continue
        case=row['case']; raw_path=run/'raw'/(case['case_id']+'.json')
        raw=read(raw_path); report=read(run/'analysis.json')
        checks={'raw_digest':sha(raw_path)==report['provenance']['raw_sha256'],
                'case_identity':raw['case_id']==case['case_id'],
                'parameter_scope':raw.get('carrier')==case['carrier'],
                'boundary':raw.get('runtime_boundary',{}).get('task_id')==row['task_id'],
                'exact_repeat':raw.get('determinism',{}).get('all_exact') is True,
                'three_stage_original_statistics':all(len(raw.get('original_coordinate_statistics',{}).get(stage,[]))==32
                                                       for stage in ('LOCAL','PARAMETER_GRADIENT','PARAMETER_WRITE')),
                'complete_state_set':len(raw.get('state_ids',[]))==32 and len(set(raw.get('state_ids',[])))==32}
        expected=analyze_artifact(raw,protocol); observed=copy.deepcopy(report)
        observed['provenance'].pop('raw_sha256',None)
        checks['recomputation']=observed==expected
        records.append({'task_id':row['task_id'],'case_id':case['case_id'],'checks':checks,
                        'verified_measurement':all(checks.values()),'status':'VERIFIED' if all(checks.values()) else 'VERIFICATION_FAILED',
                        'raw':str(raw_path.resolve()),'raw_sha256':sha(raw_path),
                        'result':report['bias_analysis'],'equivalence_decision':report['equivalence_decision']})
    return {'schema':'numerical-coverage-verification-v1','source_snapshot_verified':snapshot_verified,
            'verified_measurements':sum(r['verified_measurement'] for r in records),
            'records':records,'scope':'Recorded identity, parameter scope, original statistics and recomputation; not independent kernel execution',
            'verifies_full_plan':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('Use a new output under /data1/tzh')
    result=verify(a.root);save(a.output,result)
    print(json.dumps({'verified_measurements':result['verified_measurements'],'records':len(result['records'])}))
    if any(r['status']=='VERIFICATION_FAILED' for r in result['records']):
        raise SystemExit(1)


if __name__=='__main__':main()
