#!/usr/bin/env python3
"""Extend existing AOT parameter tracing from representatives to all backward outputs.

This does not infer parameter ownership from kernel names, inspect residuals,
or certify runtime reach. Forward-only and unresolved outputs remain explicit.
"""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.bind_backward_rescreen_carriers import load, bind_forward_parameters, endpoint_reachability


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not a.output.resolve().is_relative_to(Path('/data1/tzh')) or a.output.exists():
        p.error('Use a new output under /data1/tzh')
    tasks_path = a.release / 'same_dtype_tasks.json.gz'
    tasks = load(tasks_path)['rows']
    raw = load(a.capture)
    capture = raw.get('capture', raw)
    forward = next(g for g in capture['graphs'] if g['phase'] == 'FORWARD')
    bindings, by_primal = bind_forward_parameters(forward, a.model)
    eligible = [t for t in tasks if t.get('status') == 'EXACT_CANDIDATE_BUFFER_TO_AOT_SEMANTIC_ENDPOINT'
                and str(t.get('exact_aot_endpoint_id', '')).startswith('backward:')]
    endpoints = {t['exact_aot_endpoint_id'] for t in eligible}
    reached = endpoint_reachability(capture, by_primal, endpoints)
    cases, missing = [], []
    for t in eligible:
        reach = reached[t['exact_aot_endpoint_id']]
        parameters = sorted(reach['parameters'], key=lambda x: (x['aot_distance'], x['name']))
        if not parameters:
            missing.append({'task_id': t['task_id'], 'reason': reach['status']})
            continue
        cases.append({'case_id': 'mapped_' + t['task_id'].replace(':', '_'),
                      'task_id': t['task_id'], 'carrier': parameters[0]['name'],
                      'reference_method': 'AOT_REPLAY', 'implementation_kind': t.get('implementation_kind'),
                      'expected_symbol': t.get('symbol'), 'exact_aot_endpoint_id': t['exact_aot_endpoint_id'],
                      'mapping_evidence': parameters[0], 'runtime_parameter_reach': 'NOT_YET_MEASURED'})
    result = {'schema': 'release-parameter-mappings-v1', 'cases': cases,
              'model_path': str(a.model.resolve()), 'release_path': str(a.release.resolve()),
              'unresolved_backward_outputs': missing, 'parameter_binding_records': bindings,
              'release_endpoint_count': len(tasks), 'exact_backward_endpoints': len(eligible),
              'selection_uses_numerical_results': False,
              'source_sha256': {str(path.resolve()): sha(path) for path in (
                  tasks_path, a.capture, a.model/'config.json', Path(__file__),
                  Path(__file__).with_name('bind_backward_rescreen_carriers.py'))},
              'claim_scope': 'STATIC_AOT_REACH_NOT_RUNTIME_MEASUREMENT'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({'endpoints': len(tasks), 'mapped_backward': len(cases), 'unresolved_backward': len(missing)}))


if __name__ == '__main__':
    main()
