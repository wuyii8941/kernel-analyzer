"""Reuse the existing outcome-blind forward parameter path selection."""
import argparse
from pathlib import Path
from scripts import bind_residual_rms_forward as path_binding
from scripts import scan_grouped_causal_softmax_sources as source_scan
from scripts.run_numerical_coverage import read, save, sha
from kernel_analyzer import grouped_causal_softmax_source, grouped_causal_softmax_reference


def bind(tasks, contracts, mapping):
    # The reused function selects only by recorded distance and parameter name.
    # Replace its family-specific dispatch labels, not its selection algorithm.
    result = path_binding.bind(tasks, contracts, mapping)
    for case in result['cases']:
        suffix = '-residual-rms-forward'
        if not case['case_id'].endswith(suffix):
            raise ValueError('Reused binding interface changed')
        case['case_id'] = case['case_id'][:-len(suffix)] + '-grouped-causal-softmax-forward'
        case['reference_method'] = 'GROUPED_CAUSAL_SOFTMAX_FORWARD_COMMON_INPUT'
    result['schema'] = 'grouped-causal-softmax-forward-bound-plan-v1'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'tasks', 'mapping', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path('/data1/tzh')):
        parser.error('Choose a new output under /data1/tzh')
    mapping = read(args.mapping)
    if mapping.get('schema') != 'forward-parameter-paths-v1':
        raise ValueError('Audited forward path mapping required')
    if mapping.get('source_sha256', {}).get(str(args.tasks.resolve())) != sha(args.tasks):
        raise ValueError('Task identity differs')
    for name, digest in mapping['source_sha256'].items():
        if sha(Path(name)) != digest:
            raise ValueError('Mapping dependency changed: ' + name)
    rows = source_scan.scan(args.source.read_text())
    contracts = {r['symbol']: r['contract'] for r in rows if r['status'] == 'SOURCE_CHECKED'}
    result = bind(read(args.tasks)['rows'], contracts, mapping)
    result['source_scan'] = rows
    dependencies = [args.source, args.tasks, args.mapping, Path(__file__),
        Path(path_binding.__file__), Path(source_scan.__file__),
        Path(source_scan.scan_recurrence_sources.__file__),
        Path(grouped_causal_softmax_source.__file__), Path(grouped_causal_softmax_reference.__file__)]
    result['source_sha256'] = {str(p.resolve()): sha(p) for p in dependencies}
    save(args.output, result)
    print(dict(bound=len(result['cases']), unresolved=len(result['unresolved']),
               runtime_measurement_complete=False))


if __name__ == '__main__':
    main()
