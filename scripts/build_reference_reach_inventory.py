#!/usr/bin/env python3
"""Join saved reference checks to every observed output position, including ATen.

This eligibility inventory does not import numerical verdicts or claim runtime
completion. Internal buffers require their own explicit binding, never a guess
from a matching symbol or a nearby parameter.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from scripts.run_numerical_coverage import read, sha


def validate_additional_manifest(manifest):
    """A historical source match alone cannot establish a usable new adapter."""
    from scripts.run_declared_reference_family import declaration, validate_capture
    recorded = manifest.get('additional_reference_declaration')
    if not isinstance(recorded, dict):
        raise ValueError('Additional manifest lacks a frozen reference declaration')
    expected = declaration(recorded['family'], recorded['module'], recorded['case_suffix'])
    validate_capture(manifest, expected)


def classify(task, references, carrier):
    matches = [r for r in references if r['output_pointer'] == task.get('formal_pointer')]
    if task.get('implementation_kind') != 'TRITON':
        return 'OTHER_IMPLEMENTATION_REQUIRES_REFERENCE_AUDIT', []
    if not matches:
        return 'NO_CHECKED_REFERENCE_FOR_THIS_OUTPUT', []
    if len({r['function_ast_sha256'] for r in matches}) != 1:
        return 'AMBIGUOUS_SOURCE_DEFINITION', matches
    if not carrier:
        return 'REFERENCE_PRESENT_PARAMETER_BINDING_REQUIRED', matches
    if task.get('status') != 'EXACT_CANDIDATE_BUFFER_TO_AOT_SEMANTIC_ENDPOINT':
        return 'EXPLICIT_INTERNAL_BOUNDARY_BINDING_REQUIRED', matches
    return 'REFERENCE_AND_STATIC_PARAMETER_BINDING_PRESENT', matches


def merge_carriers(target, cases, tasks):
    """Reject conflicting mappings instead of letting file order pick a weight."""
    by_id = {t['task_id']: t for t in tasks}
    if len(by_id) != len(tasks):
        raise ValueError('Duplicate task identity')
    for case in cases:
        task = by_id.get(case['task_id'])
        if (task is None or not case.get('carrier')
                or case.get('expected_symbol') != task.get('symbol')
                or case.get('exact_aot_endpoint_id') != task.get('exact_aot_endpoint_id')
                or task.get('status') != 'EXACT_CANDIDATE_BUFFER_TO_AOT_SEMANTIC_ENDPOINT'):
            raise ValueError('Mapping disagrees with source task: ' + case['task_id'])
        previous = target.setdefault(case['task_id'], case['carrier'])
        if previous != case['carrier']:
            raise ValueError('Conflicting parameter binding: ' + case['task_id'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scans', type=Path, required=True)
    p.add_argument('--additional-manifests', type=Path, nargs='*', default=[],
                   help='Explicit new-family source manifests; original release layout is required.')
    p.add_argument('--mapping-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('Use new output under /data1/tzh')
    index = defaultdict(list)
    releases = set()
    inputs = {}
    for file in sorted(set(a.scans.glob('*.json')) | set(a.additional_manifests)):
        manifest = read(file)
        if 'reference_family' not in manifest:
            if file in a.additional_manifests:
                raise ValueError('Additional input is not a reference manifest: '+str(file))
            continue
        if file in a.additional_manifests:
            validate_additional_manifest(manifest)
        inputs[str(file.resolve())] = sha(file)
        for source in manifest['sources']:
            path = Path(source['source'])
            if sha(path) != source['source_sha256']:
                raise ValueError('Source changed: '+str(path))
            release = path.parents[2]
            if release.parent.name != 'runtime_releases':
                raise ValueError('Unrecognized release layout')
            releases.add(release)
            phase = 'BACKWARD' if '_backward_' in path.parent.name else 'FORWARD' if '_forward_' in path.parent.name else None
            if phase is None:
                raise ValueError('Unrecognized trace phase')
            for row in source['rows']:
                if row['status'] == 'SOURCE_CHECKED':
                    c = row['contract']
                    index[(str(release),phase,row['symbol'])].append(dict(
                        family=manifest['reference_family'], source=str(path),
                        output_pointer=c['output_pointer'], function_ast_sha256=c['function_ast_sha256']))
    carriers = defaultdict(dict)
    non_mapping_files = []
    for file in sorted(a.mapping_dir.glob('*.json')):
        mapping = read(file)
        if 'cases' not in mapping:
            non_mapping_files.append(str(file.resolve()))
            continue
        for task_source,digest in mapping.get('source_sha256',{}).items():
            path = Path(task_source)
            if path.name == 'same_dtype_tasks.json.gz':
                if sha(path) != digest:
                    raise ValueError('Task file changed')
                inputs[str(file.resolve())] = sha(file)
                merge_carriers(carriers[str(path.parent)], mapping['cases'], read(path)['rows'])
    rows = []
    for release in sorted(releases):
        file = release/'same_dtype_tasks.json.gz'
        inputs[str(file)] = sha(file)
        tasks = read(file)['rows']
        if len({r['task_id'] for r in tasks}) != len(tasks):
            raise ValueError('Duplicate task identity')
        for task in tasks:
            refs = index.get((str(release),task.get('phase'),task.get('symbol')),[])
            carrier = carriers[str(release)].get(task['task_id'])
            status,matches = classify(task,refs,carrier)
            rows.append(dict(release=str(release),task_id=task['task_id'],
                             implementation_kind=task.get('implementation_kind'),phase=task.get('phase'),
                             symbol=task.get('symbol'),formal_pointer=task.get('formal_pointer'),
                             carrier=carrier,eligibility=status,reference_candidates=matches,
                             runtime_measurement_status='NOT_ASSESSED_BY_THIS_INVENTORY'))
    result = dict(schema='reference-reach-inventory-v1',records=rows,positions=len(rows),
                  releases=len(releases),source_sha256=inputs,numerical_results_read=False,
                  non_mapping_files=non_mapping_files,
                  counts=dict(Counter(r['eligibility'] for r in rows)),
                  implementation_counts=dict(Counter(r['implementation_kind'] for r in rows)),
                  scope='Every output task in the saved releases, not all possible LLM kernels',
                  all_kernel_support_established=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('positions','releases','counts','implementation_counts')}))


if __name__ == '__main__':
    main()
