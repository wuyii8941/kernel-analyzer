#!/usr/bin/env python3
"""Choose bounded family representatives solely from frozen coverage metadata."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--coverage',type=Path,required=True)
    p.add_argument('--family',action='append',required=True)
    p.add_argument('--per-family',type=int,default=1)
    p.add_argument('--reference-method', choices=['AOT_REPLAY', 'EXTERNAL_FP32_RECOMPUTE'], default='AOT_REPLAY')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.per_family<1 or a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('Use a positive count and new output under /data1/tzh')
    data=a.coverage.read_bytes(); coverage=json.loads(data); selected=[]; missing=[]
    for family in dict.fromkeys(a.family):
        rows=[r for r in coverage['rows'] if r['status']=='READY_FOR_CAPTURE' and r['case'].get('family')==family]
        if a.reference_method == 'EXTERNAL_FP32_RECOMPUTE':
            rows = [r for r in rows if r.get('implementation_kind') == 'EXTERN'
                    and str(r.get('symbol')).removeprefix('extern_kernels.') in {'mm', 'bmm', 'addmm'}]
        # Keep the release's execution order, not a ranking of observed effects.
        if len(rows)<a.per_family:
            missing.append({'family':family,'requested':a.per_family,'available':len(rows)})
        for row in rows[:a.per_family]:
            case = dict(row['case'])
            if case.get('reference_method') != a.reference_method:
                case['source_case_id'] = case['case_id']
                case['case_id'] += '-common-input-fp32'
            case['reference_method'] = a.reference_method
            selected.append(case)
    result={'schema':'numerical-family-plan-v1','cases':selected,'missing':missing,
            'source_sha256':hashlib.sha256(data).hexdigest(),'source':str(a.coverage.resolve()),
            'selection_rule':'FIRST_IN_RELEASE_ORDER_PER_PREDECLARED_FAMILY',
            'selection_uses_numerical_results':False,'families':a.family}
    result['reference_method'] = a.reference_method
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'selected':len(selected),'missing':missing}))


if __name__=='__main__':main()
