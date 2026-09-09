#!/usr/bin/env python3
"""Update static reference eligibility without promoting runtime measurements."""
import argparse
from collections import Counter
from pathlib import Path
from scripts.run_numerical_coverage import read, save, sha
from scripts.run_decayed_recurrence_capture import plan_method, select_contracts
from scripts.build_reference_reach_inventory import merge_carriers
from scripts.join_coverage_measurements import validate_retained_measurements


def join(inventory, release, plan, plan_path):
    forward = plan.get('schema') == 'residual-rms-forward-bound-plan-v1'
    if forward:
        from scripts.run_residual_rms_forward_capture import select
        select(plan, plan['cases'])
        method, family = 'RESIDUAL_RMS_FORWARD_COMMON_INPUT', 'RESIDUAL_RMS_FORWARD'
    else:
        method, family = plan_method(plan)
        select_contracts(plan, plan['cases'])
    plan_digest = sha(plan_path)
    tasks_path = release/'same_dtype_tasks.json.gz'
    task_digest = plan.get('tasks_sha256', plan['source_sha256'].get(str(tasks_path.resolve())))
    if task_digest != sha(tasks_path):
        raise ValueError('Plan belongs to another release')
    for name, digest in plan['source_sha256'].items():
        if sha(Path(name)) != digest:
            raise ValueError('Bound source changed: '+name)
    tasks = read(tasks_path)['rows']
    merge_carriers({}, plan['cases'], tasks)
    rows = [dict(r) for r in inventory['records']]
    index = {(str(Path(r['release']).resolve()), r['task_id']): r for r in rows}
    if len(index) != len(rows): raise ValueError('Duplicate inventory position')
    for case in plan['cases']:
        key = (str(release.resolve()), case['task_id'])
        if key not in index: raise ValueError('Bound position absent from inventory')
        row = index[key]
        if (row['symbol'] != case['expected_symbol']
                or row['formal_pointer'] != case['reference_output_pointer']
                or row['implementation_kind'] != 'TRITON'
                or row.get('carrier') not in (None, case['carrier'])):
            raise ValueError('Inventory disagrees with bound position')
        row['carrier'] = case['carrier']
        row['eligibility'] = 'REFERENCE_AND_STATIC_PARAMETER_BINDING_PRESENT'
        candidate = dict(family=family, reference_method=method,
                         output_pointer=case['reference_output_pointer'],
                         function_ast_sha256=(plan['contracts'][case['expected_symbol']] if forward
                                              else plan['contract'])['function_ast_sha256'],
                         bound_plan=str(plan_path.resolve()), bound_plan_sha256=plan_digest)
        row['reference_candidates'] = [*row.get('reference_candidates', []), candidate]
        # No changes to runtime_measurement_status or measurement_evidence.
    return dict(inventory, records=rows,
                counts=dict(Counter(r['eligibility'] for r in rows)),
                all_kernel_support_established=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--plans',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('Choose new output under /data1/tzh')
    result=read(a.inventory)
    retained=validate_retained_measurements(result)
    seen=set()
    for path in a.plans:
        plan=read(path)
        ids={c['task_id'] for c in plan['cases']}
        if seen & ids: raise ValueError('Overlapping plans require explicit comparison, not duplicate coverage')
        seen |= ids
        result=join(result,a.release,plan,path)
    result['binding_update']=dict(positions=len(seen),retained_verified_measurements=retained,
        runtime_measurement_status_unchanged=True,
        input_sha256={str(x.resolve()):sha(x) for x in [a.inventory,*a.plans,Path(__file__)]})
    save(a.output,result)
    print(dict(updated_bindings=len(seen),retained_verified_measurements=retained,positions=len(result['records'])))


if __name__=='__main__': main()
