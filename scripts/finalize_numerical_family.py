#!/usr/bin/env python3
"""Audit every declared family position and reuse the shared analysis.

Missing/invalid records remain in the denominator. This verifies saved evidence,
not independent CUDA execution, and never changes the capture's frozen policy.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import types
from collections import Counter
from pathlib import Path

from kernel_analyzer.training_numerical_analysis import analyze_artifact

ROOT = Path(__file__).resolve().parents[1]
STAGES = ('LOCAL', 'PARAMETER_GRADIENT', 'PARAMETER_WRITE')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def argument(arguments, name):
    if arguments.count(name) != 1:
        raise ValueError('Expected exactly one capture argument: ' + name)
    return arguments[arguments.index(name) + 1]


def audit_record(raw, case, protocol, count, expected_ids=None, analysis_function=analyze_artifact):
    ids = raw.get('state_ids', [])
    scope = raw.get('reference_comparison_scope', {})
    checks = {
        'case': raw.get('case_id') == case['case_id'],
        'parameter': raw.get('carrier') == case['carrier'],
        'boundary': raw.get('runtime_boundary', {}).get('task_id') == case['task_id'],
        'complete': raw.get('status') == 'COMPLETE',
        'states': len(ids) == count and len(set(ids)) == count,
        'three_stage_statistics': all(len(raw.get('original_coordinate_statistics', {}).get(s, [])) == count for s in STAGES),
        'determinism': raw.get('determinism', {}).get('all_exact') is True,
        'reference': scope.get('comparison') == 'COMMON_OPERAND_SOURCE_CHECKED_' + protocol['reference_family']
                     and scope.get('same_local_operands') is True
                     and scope.get('includes_possible_upstream_differences') is False
                     and scope.get('reference_variant') == protocol['variant'],
    }
    if expected_ids is not None:
        checks['frozen_state_order'] = ids == expected_ids
    # No verdict is issued for an unverified identity or incomplete measurement.
    result = analysis_function(raw, protocol) if all(checks.values()) else None
    status = 'VERIFIED' if result and result['measurement_status'] == 'VALID' else 'INVALID_OR_INCOMPLETE'
    return {'case_id': case['case_id'], 'task_id': case['task_id'], 'checks': checks,
            'status': status, 'analysis': result}


def _analysis_from_snapshot(snapshot, frozen):
    """Load protocol-pinned analysis modules without replacing working files."""
    module_paths = {
        "kernel_analyzer.training_equivalence": ROOT / "src/kernel_analyzer/training_equivalence.py",
        "kernel_analyzer.training_numerical_analysis": ROOT / "src/kernel_analyzer/training_numerical_analysis.py",
    }
    for path in module_paths.values():
        name = str(path.resolve())
        if name not in snapshot or hashlib.sha256(snapshot[name].encode()).hexdigest() != frozen.get(name):
            raise ValueError("Frozen analysis module missing from content-addressed snapshot: " + name)
    previous = {name: sys.modules.get(name) for name in module_paths}
    loaded = {}
    try:
        for name in (
            "kernel_analyzer.training_equivalence",
            "kernel_analyzer.training_numerical_analysis",
        ):
            module = types.ModuleType(name)
            module.__file__ = str(module_paths[name])
            module.__package__ = "kernel_analyzer"
            sys.modules[name] = module
            exec(compile(snapshot[str(module_paths[name].resolve())], module.__file__, "exec"), module.__dict__)
            loaded[name] = module
        return loaded["kernel_analyzer.training_numerical_analysis"].analyze_artifact
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def finalize(root, *, allow_content_addressed_snapshot=False):
    root = root.resolve()
    protocol_path = root / 'raw/family_execution_protocol.json'
    protocol = read(protocol_path)
    args = protocol['capture_arguments']
    plan_path = Path(argument(args, '--case-plan'))
    if not plan_path.is_absolute():
        plan_path = ROOT / plan_path
    plan_path = plan_path.resolve()
    frozen = protocol['source_sha256']
    if sha(plan_path) != frozen.get(str(plan_path)):
        raise ValueError('Case plan differs from frozen capture')
    # Do not silently use a newly edited estimator to certify an old capture.
    changed_analysis_sources = []
    for name in ('training_numerical_analysis.py', 'training_equivalence.py'):
        source = ROOT / 'src/kernel_analyzer' / name
        if sha(source) != frozen.get(str(source)):
            changed_analysis_sources.append(name)
    snapshot_path = root / 'source_snapshot.json'
    snapshot = read(snapshot_path)
    expected = {p for p in frozen if Path(p).suffix == '.py'}
    if set(snapshot) != expected or any(hashlib.sha256(snapshot[p].encode()).hexdigest() != frozen[p] for p in expected):
        raise ValueError('Frozen Python source snapshot is incomplete or changed')
    # An exact protocol-pinned snapshot is the historical analyzer, not a
    # retrospective use of current code. Load it automatically and record that
    # fact. The legacy flag remains accepted for existing recovery commands.
    analysis_function = (
        _analysis_from_snapshot(snapshot, frozen) if changed_analysis_sources else analyze_artifact
    )
    count = int(argument(args, '--states'))
    if count < 2:
        raise ValueError('Invalid declared state count')
    bank_path = Path(argument(args, '--state-bank' if '--state-bank' in args else '--input-bank'))
    if not bank_path.is_absolute():
        bank_path = ROOT / bank_path
    bank_path = bank_path.resolve()
    if sha(bank_path) != frozen.get(str(bank_path)):
        raise ValueError('State bank differs from frozen capture')
    bank = read(bank_path)
    warmup = int(argument(args, '--warmup-steps')) if '--warmup-steps' in args else 0
    states = bank.get('states', bank.get('records', []))[warmup:warmup + count]
    expected_ids = [str(s.get('state_id', s.get('sequence_id', i))) for i, s in enumerate(states)]
    if warmup < 0 or len(expected_ids) != count or len(set(expected_ids)) != count:
        raise ValueError('Incomplete or duplicate declared states')
    cases = read(plan_path)['cases']
    if not cases or len({c['case_id'] for c in cases}) != len(cases) or len({c['task_id'] for c in cases}) != len(cases):
        raise ValueError('Empty plan or duplicate case/task identities')
    records = []
    for case in cases:
        if Path(case['case_id']).name != case['case_id']:
            raise ValueError('Unsafe case filename')
        path = root / 'raw' / (case['case_id'] + '.json')
        if not path.exists():
            records.append({'case_id': case['case_id'], 'task_id': case['task_id'], 'status': 'NOT_CAPTURED'})
            continue
        before = sha(path)
        try:
            record = audit_record(
                read(path), case, protocol, count, expected_ids,
                analysis_function=analysis_function,
            )
        except (ValueError, TypeError, KeyError) as error:
            record = {'case_id': case['case_id'], 'task_id': case['task_id'],
                      'status': 'INVALID_OR_INCOMPLETE', 'reason': str(error)}
        if sha(path) != before:
            raise ValueError('Raw artifact changed during reading: ' + str(path))
        record.update(raw_artifact=str(path), raw_sha256=before)
        records.append(record)
    counts = dict(Counter(r['status'] for r in records))
    cost_path = root / 'raw/capture_cost.json'
    cost = ({'status': 'RECORDED_EXECUTION_COST', 'artifact': str(cost_path),
             'sha256': sha(cost_path), 'measurements': read(cost_path)} if cost_path.exists()
            else {'status': 'NOT_RECORDED', 'reason': 'Do not reconstruct historical runtime from result existence'})
    return {'schema': 'numerical-family-completion-v1', 'reference_family': protocol['reference_family'],
            'protocol_sha256': sha(protocol_path), 'source_snapshot_verified': True,
            'analysis_execution': (
                'PROTOCOL_PINNED_CONTENT_ADDRESSED_SNAPSHOT'
                if changed_analysis_sources else 'CURRENT_SOURCE_MATCHES_PROTOCOL'
            ),
            'current_analysis_sources_changed': changed_analysis_sources,
            'declared_positions': len(cases), 'counts': counts,
            'measurement_complete': counts.get('VERIFIED', 0) == len(cases),
            'verifies_full_research_plan': False,
            'capture_cost': cost,
            'scope': 'Recorded common-input identities and shared fixed-suite analysis; not independent execution, independent runs, or training quality proof',
            'records': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baselines-dir', type=Path,
                        help='After the complete batch verifies, generate three-stage tables and SVG using the existing baseline program.')
    parser.add_argument('--allow-content-addressed-snapshot', action='store_true',
                        help='Run the exact protocol-pinned analyzer from a verified saved source snapshot.')
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path('/data1/tzh')):
        parser.error('Choose a new output under /data1/tzh')
    if args.baselines_dir and (args.baselines_dir.exists() or not args.baselines_dir.resolve().is_relative_to(Path('/data1/tzh'))):
        parser.error('Choose a new baseline directory under /data1/tzh')
    report = finalize(args.root, allow_content_addressed_snapshot=args.allow_content_addressed_snapshot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps({k: report[k] for k in ('declared_positions', 'counts', 'measurement_complete')}))
    if not report['measurement_complete']:
        raise SystemExit(2)
    if args.baselines_dir:
        local_path = args.root / 'raw/local_tolerance_comparisons.json'
        extra = ['--local-comparisons', str(local_path)] if local_path.exists() else []
        subprocess.run([sys.executable, str(ROOT / 'scripts/build_same_data_baselines.py'),
                        '--raw', *[row['raw_artifact'] for row in report['records']],
                        '--output', str(args.baselines_dir), *extra], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
