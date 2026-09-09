"""Bind source-checked convolution outputs through declared downstream paths.

The bias endpoint supplies a possible parameter path, not the replacement
boundary. Only the external convolution output will be replaced.
"""
import argparse
from pathlib import Path
from scripts.run_numerical_coverage import read, save, sha


def bind(pairs, mapping):
    paths = {}
    for row in mapping['rows']:
        if row['endpoint'] in paths: raise ValueError('Duplicate mapped endpoint')
        paths[row['endpoint']] = row
    cases, unresolved, seen = [], [], set()
    for pair in pairs['records']:
        task = pair['convolution_task_id']
        if task in seen: raise ValueError('Duplicate convolution task')
        seen.add(task)
        endpoint = pair['exact_aot_endpoint_id']
        parameters = paths.get(endpoint, {}).get('parameters', [])
        if not parameters:
            unresolved.append(dict(task_id=task, reason='NO_OBSERVED_DOWNSTREAM_PARAMETER_PATH'))
            continue
        selected = min(parameters, key=lambda r:(r['aot_distance'], r['name']))
        cases.append(dict(case_id='mapped_'+task.replace(':','_')+'-depthwise-convolution',
            task_id=task, carrier=selected['name'], parameter_scope='SELECTED_PARAMETER_ONLY',
            mapping_evidence=selected, downstream_closure_endpoint=endpoint,
            downstream_closure_task_id=pair['closed_task_id'],
            replacement_boundary='EXTERNAL_CONVOLUTION_OUTPUT_BEFORE_SEPARATE_BIAS',
            exact_aot_endpoint_id=None,
            reference_method='DEPTHWISE_CONV1D_COMMON_INPUT',
            expected_symbol='convolution', implementation_kind='EXTERN',
            source_contract=pair, runtime_parameter_reach='NOT_YET_MEASURED'))
    return dict(cases=cases, unresolved=unresolved, runtime_measurement_complete=False,
                numerical_results_read=False,
                scope='Downstream graph path only; not runtime nonzero reach or complete model gradient coverage')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('pairs','mapping','output'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('New output under /data1/tzh required')
    pairs, mapping=read(a.pairs),read(a.mapping)
    for record in (pairs,mapping):
        for path,digest in record['source_sha256'].items():
            if sha(Path(path))!=digest: raise ValueError('Dependency changed: '+path)
    result=bind(pairs,mapping)
    result.update(schema='depthwise-convolution-bound-plan-v1',source_sha256={
        str(p.resolve()):sha(p) for p in (a.pairs,a.mapping,Path(__file__))})
    save(a.output,result)
    print(dict(bound=len(result['cases']),unresolved=len(result['unresolved'])))


if __name__=='__main__': main()
