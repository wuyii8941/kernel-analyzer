"""Map the frozen convolution closures using the existing graph algorithm.

This selected-family mapping does not replace the all-endpoint mapping task.
"""
import argparse
from pathlib import Path
from scripts.run_numerical_coverage import read, save, sha
from scripts.build_forward_parameter_mappings import map_endpoints
from scripts.bind_backward_rescreen_carriers import bind_forward_parameters


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('pairs', 'capture', 'model', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('New output under /data1/tzh required')
    pairs = read(a.pairs)
    for path, digest in pairs['source_sha256'].items():
        if sha(Path(path)) != digest: raise ValueError('Source contract dependency changed')
    capture = read(a.capture)
    capture = capture.get('capture', capture)
    forward = next(g for g in capture['graphs'] if g['phase'] == 'FORWARD')
    bindings, by_primal = bind_forward_parameters(forward, a.model)
    print(dict(phase='PARAMETERS_BOUND', count=len(by_primal)), flush=True)
    endpoints = {r['exact_aot_endpoint_id'] for r in pairs['records']}
    result = map_endpoints(capture, by_primal, endpoints)
    base = Path(__file__).resolve().parents[1]
    dependencies = [a.pairs, a.capture, a.model/'config.json', Path(__file__),
        base/'scripts/build_forward_parameter_mappings.py',
        base/'scripts/bind_backward_rescreen_carriers.py',
        base/'src/kernel_analyzer/forward_runtime_edges.py']
    result.update(schema='convolution-forward-parameter-paths-v1',
        parameter_binding_records=bindings, all_endpoint_mapping_replaced=False,
        source_sha256={str(p.resolve()):sha(p) for p in dependencies})
    save(a.output, result)
    print(dict(endpoints=len(result['rows']), mapped=sum(bool(r['parameters']) for r in result['rows'])))


if __name__ == '__main__': main()
