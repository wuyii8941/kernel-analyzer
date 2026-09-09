#!/usr/bin/env python3
"""Reuse recorded saved-value identities and existing backward parameter tracing.

This builds possible dataflow paths, not proof of nonzero numerical influence.
Multi-segment captures require explicit parameter-output pairing and are rejected
here rather than merged by placeholder order.
"""
import argparse
import re
from pathlib import Path
from scripts.run_numerical_coverage import read, save, sha
from scripts.bind_backward_rescreen_carriers import bind_forward_parameters, endpoint_reachability
from kernel_analyzer import forward_runtime_edges


def map_endpoints(capture, by_primal, endpoints):
    graphs = capture['graphs']
    if (len(graphs) != 2 or sorted(g['phase'] for g in graphs) != ['BACKWARD', 'FORWARD']
            or any(g.get('graph_index', 0) != 0 for g in graphs)):
        raise ValueError('Explicit single graph pair required for parameter-output binding')
    parsed = {}
    for endpoint in sorted(endpoints):
        match = re.fullmatch(r'forward:graph(\d+):(.+)', endpoint)
        if not match:
            raise ValueError('Explicit forward graph endpoint required')
        parsed[endpoint] = (int(match[1]), match[2])
    paths = forward_runtime_edges.forward_backward_entries(capture, list(parsed.values()))
    backward_ids = {f"backward:graph{e['graph']}:{e['placeholder']}"
                    for row in paths['rows'] for e in row['entries']}
    reached = endpoint_reachability(capture, by_primal, backward_ids)
    rows = []
    for endpoint, path in zip(parsed, paths['rows']):
        parameters = {}
        for entry in path['entries']:
            identifier = f"backward:graph{entry['graph']}:{entry['placeholder']}"
            for parameter in reached[identifier]['parameters']:
                distance = entry['distance'] + parameter['aot_distance']
                candidate = dict(parameter, aot_distance=distance,
                                 backward_entry=identifier)
                previous = parameters.get(parameter['name'])
                if previous is None or distance < previous['aot_distance']:
                    parameters[parameter['name']] = candidate
        rows.append(dict(endpoint=endpoint, parameters=sorted(parameters.values(),
            key=lambda p: (p['aot_distance'], p['name'])),
            status='OBSERVED_SAVED_VALUE_PARAMETER_PATH' if parameters else 'UNRESOLVED_PARAMETER_PATH'))
    return dict(rows=rows, unresolved_identity=paths['unresolved_identity'],
                runtime_measurement_complete=False,
                claim_scope='POSSIBLE_DATAFLOW_NOT_NONZERO_INFLUENCE_OR_COMPLETE_DERIVATIVE')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('capture', 'tasks', 'model', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path('/data1/tzh')):
        parser.error('Choose a new output under /data1/tzh')
    outer = read(args.capture)
    capture = outer.get('capture', outer)
    tasks = [t for t in read(args.tasks)['rows']
             if t.get('status') == 'EXACT_CANDIDATE_BUFFER_TO_AOT_SEMANTIC_ENDPOINT'
             and str(t.get('exact_aot_endpoint_id', '')).startswith('forward:')]
    forward = next(g for g in capture['graphs'] if g['phase'] == 'FORWARD')
    bindings, by_primal = bind_forward_parameters(forward, args.model)
    result = map_endpoints(capture, by_primal, {t['exact_aot_endpoint_id'] for t in tasks})
    result.update(schema='forward-parameter-paths-v1', tasks=tasks,
                  parameter_binding_records=bindings, numerical_results_read=False)
    dependencies = [args.capture, args.tasks, args.model / 'config.json', Path(__file__),
                    Path(forward_runtime_edges.__file__),
                    Path(__file__).with_name('bind_backward_rescreen_carriers.py')]
    result['source_sha256'] = {str(p.resolve()): sha(p) for p in dependencies}
    save(args.output, result)
    print(dict(endpoints=len(result['rows']),
        mapped=sum(bool(r['parameters']) for r in result['rows'])))


if __name__ == '__main__':
    main()
