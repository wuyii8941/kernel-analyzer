#!/usr/bin/env python3
"""Freeze coverage, execute declared comparisons, and reuse the v2 analyzer.

No backend whitelist or case-name dispatch. The existing AOT capture adapter
continues to verify actual execution and parameter reach at runtime.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from kernel_analyzer.numerical_campaign import plan_release, digest
from kernel_analyzer.numerical_batching import capture_batches
from kernel_analyzer.training_numerical_analysis import analyze_artifact

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path)) as f:
        return json.load(f)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def save_idempotent(path, value):
    """Finish an interrupted metadata freeze without changing its contents.

    A process may stop after writing the source snapshot but before writing the
    protocol.  Exact reruns are safe; a changed value remains fail-closed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise ValueError('Existing partial freeze differs: ' + str(path))
        return
    temporary = path.with_name(path.name + '.tmp')
    if temporary.exists():
        temporary.unlink()
    with temporary.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['freeze', 'run', 'report'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--release', type=Path)
    p.add_argument('--plans', type=Path, nargs='*', default=[])
    p.add_argument('--carrier-registry', type=Path)
    p.add_argument('--registry-model')
    p.add_argument('--sequence-length', type=int)
    p.add_argument('--architecture')
    p.add_argument('--model', type=Path)
    p.add_argument('--input-bank', type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--limit', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=1,
                   help='Freeze shared capture size; each endpoint is still replaced and analyzed separately')
    p.add_argument('--allow-graph-breaks', action='store_true',
                   help='Freeze segmented compilation explicitly; runtime identity checks still apply')
    p.add_argument('--parallel-measurement', action='store_true',
                   help='Freeze the tested bitwise-preserving parallel sketch adapter')
    a = p.parse_args()
    out = a.output.resolve()
    if not out.is_relative_to(Path('/data1/tzh')):
        p.error('Output must stay under /data1/tzh')
    if a.command == 'freeze':
        if a.batch_size < 1:
            p.error('batch-size must be positive')
        if not all([a.release, a.architecture, a.model, a.input_bank]):
            p.error('freeze requires release, architecture, model and input-bank')
        try:
            release_metadata = read(a.release / 'capture.json')
            if not release_metadata.get('modules') or not release_metadata.get('input', {}).get('token_ids_sha256'):
                p.error('Release capture metadata lacks declared modules/input')
        except (ValueError, UnicodeError, OSError) as error:
            p.error('Release capture metadata is unreadable; preserve it and repair the package explicitly: ' + str(error))
        declared = [c for path in a.plans for c in read(path)['cases']]
        tasks = read(a.release / 'same_dtype_tasks.json.gz')
        task_map = {t['task_id']: t for t in tasks['rows']}
        if a.carrier_registry:
            if not a.registry_model or not a.sequence_length:
                p.error('registry requires registry-model and sequence-length')
            for cell in read(a.carrier_registry)['cells']:
                rep = cell.get('representative', {})
                carrier = cell.get('nearest_carrier')
                if (cell.get('model') == a.registry_model and cell.get('sequence_length') == a.sequence_length
                        and rep.get('exact_endpoint_executable') and carrier and carrier.get('name')):
                    live = task_map.get(rep['task_id'], {})
                    # Task ordinals alone are not implementation identity.
                    if (live.get('exact_aot_endpoint_id') != rep.get('exact_aot_endpoint_id')
                            or live.get('symbol') != rep.get('region_symbol')):
                        continue
                    declared.append({'case_id': cell['cell_id'], 'task_id': rep['task_id'],
                                     'carrier': carrier['name'], 'reference_method': 'AOT_REPLAY',
                                     'family': cell['family'], 'source': str(a.carrier_registry.resolve())})
        coverage = plan_release(tasks, declared)
        sources = [a.release / 'same_dtype_tasks.json.gz', a.release / 'capture.json',
                   a.release / 'campaign.json.gz', a.release / 'inventory.json.gz',
                   a.model / 'config.json', a.input_bank, *a.plans,
                   ROOT / 'src/kernel_analyzer/training_numerical_analysis.py',
                   ROOT / 'src/kernel_analyzer/update_write.py',
                   ROOT / 'scripts/capture_bound_endpoint_bias_formation_v21.py',
                   Path(__file__), ROOT / 'src/kernel_analyzer/numerical_campaign.py']
        sources += [ROOT / 'scripts/run_training_bias_profile_v2_empirical.py',
                    ROOT / 'src/kernel_analyzer/numerical_batching.py',
                    ROOT / 'scripts/same_dtype_semantic_observer.py',
                    ROOT / 'scripts/generated_nontriton_fp32_observer.py',
                    ROOT / 'scripts/generated_fp32_observer.py',
                    ROOT / 'src/kernel_analyzer/training_equivalence.py',
                    ROOT / 'src/kernel_analyzer/training_bias_profile.py',
                    ROOT / 'src/kernel_analyzer/analysis_result.py']
        if a.parallel_measurement:
            sources += [ROOT / 'scripts/run_parallel_bound_capture.py',
                        ROOT / 'src/kernel_analyzer/parallel_measurement.py']
        if a.carrier_registry:
            sources.append(a.carrier_registry)
        if (a.release / 'redeclaration_provenance.json').exists():
            sources.append(a.release / 'redeclaration_provenance.json')
        protocol = {'schema': 'training-numerical-coverage-v1',
                    'primary_stage': 'PARAMETER_WRITE', 'fixed_suite_margins': {'full_update_rms': .01},
                    'claim_scope': 'FIXED_SUITE_UPDATE',
                    'data_use': 'EXISTING_METADATA_NEW_CAPTURE_NOT_BLIND_IMPLEMENTATION_SELECTION',
                    'architecture': a.architecture, 'model': str(a.model.resolve()),
                    'capture_batch_size': a.batch_size,
                    'release_metadata_schema': release_metadata.get('schema', 'UNVERSIONED'),
                    'allow_graph_breaks': a.allow_graph_breaks,
                    'capture_entrypoint': 'scripts/run_parallel_bound_capture.py' if a.parallel_measurement else 'scripts/capture_bound_endpoint_bias_formation_v21.py',
                    'input_bank': str(a.input_bank.resolve()), 'release': str(a.release.resolve()),
                    'coverage_sha256': digest(coverage),
                    'source_sha256': {str(s.resolve()): sha(s) for s in sources}}
        save_idempotent(out / 'source_snapshot.json', {str(s.resolve()): s.read_text() for s in sources if s.suffix=='.py'})
        save_idempotent(out / 'protocol.json', protocol)
        save_idempotent(out / 'coverage.json', coverage)
        print(json.dumps(coverage['counts']))
        return
    protocol = read(out / 'protocol.json')
    coverage = read(out / 'coverage.json')
    if digest(coverage) != protocol['coverage_sha256']:
        raise SystemExit('Frozen coverage changed')
    if a.command == 'run':
        if a.limit < 1:
            p.error('limit must be positive')
        for path, expected in protocol['source_sha256'].items():
            if sha(path) != expected:
                raise SystemExit('Frozen source changed: ' + path)
        import torch
        if not torch.cuda.is_available():
            raise SystemExit('RUNTIME_UNAVAILABLE: CUDA initialization failed; no cases launched')
        def case_run(row):
            return out / 'runs' / hashlib.sha256(row['task_id'].encode()).hexdigest()[:20]
        pending = [row for row in coverage['rows']
                   if row['status'] == 'READY_FOR_CAPTURE' and not case_run(row).exists()][:a.limit]
        task_rows = read(Path(protocol['release']) / 'same_dtype_tasks.json.gz')['rows']
        groups = capture_batches(pending, {r['task_id']: r for r in task_rows}, protocol.get('capture_batch_size', 1))
        for group in groups:
            for path, expected in protocol['source_sha256'].items():
                if sha(path) != expected:
                    raise SystemExit('Frozen source changed before next task: ' + path)
            run = case_run(group[0]) if len(group) == 1 else out / 'batches' / digest([r['task_id'] for r in group])[:20]
            save(run / 'case_plan.json', {'cases': [row['case'] for row in group]})
            cmd = [sys.executable, str(ROOT / protocol.get('capture_entrypoint','scripts/capture_bound_endpoint_bias_formation_v21.py')),
                   '--architecture', protocol['architecture'], '--model', protocol['model'],
                   '--input-bank', protocol['input_bank'], '--release-dir', protocol['release'],
                   '--case-plan', str(run / 'case_plan.json'), '--output-dir', str(run / 'legacy'),
                   '--spool-dir', str(run / 'spool'), '--states', '32', '--device', a.device,
                   '--training-bias-profile-v2-output-dir', str(run / 'raw')]
            if protocol.get('allow_graph_breaks'):
                cmd.append('--allow-graph-breaks')
            save(run / 'started.json', {'task_ids': [r['task_id'] for r in group], 'command': cmd,
                                       'simultaneous_endpoint_replacement': False})
            if len(group) > 1:
                for row in group:
                    target = case_run(row)
                    save(target / 'case_plan.json', {'cases': [row['case']]})
                    save(target / 'started.json', {'task_id': row['task_id'], 'shared_capture': str(run), 'command': cmd})
                    (target / 'raw').symlink_to((run / 'raw').resolve(), target_is_directory=True)
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=f'{ROOT}/src:{ROOT}',
                       HF_HOME='/data1/tzh/cache/huggingface', XDG_CACHE_HOME='/data1/tzh/cache/xdg',
                       TRITON_CACHE_DIR='/data1/tzh/cache/triton', TORCHINDUCTOR_CACHE_DIR='/data1/tzh/cache/torchinductor',
                       OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
            with (run / 'capture.log').open('x') as f:
                completed = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
            for row in group:
                target = case_run(row)
                status = {'returncode': completed.returncode, 'status': 'EXECUTION_FAILED'}
                if completed.returncode == 0:
                    raw = target / 'raw' / (row['case']['case_id'] + '.json')
                    if raw.exists():
                        report = analyze_artifact(read(raw), protocol)
                        report['provenance']['raw_sha256'] = sha(raw)
                        save(target / 'analysis.json', report)
                        status['status'] = report['measurement_status']
                    else:
                        status['status'] = 'RAW_OUTPUT_MISSING'
                save(target / 'status.json', status)
                print(json.dumps({'task_id': row['task_id'], **status}), flush=True)
    rows = []
    for row in coverage['rows']:
        run = out / 'runs' / hashlib.sha256(row['task_id'].encode()).hexdigest()[:20]
        status = read(run / 'status.json')['status'] if (run / 'status.json').exists() else (
            'INCOMPLETE_ATTEMPT' if run.exists() else row['status'])
        rows.append({'task_id': row['task_id'], 'status': status})
    from collections import Counter
    summary = {'endpoint_count': len(rows), 'counts': dict(Counter(r['status'] for r in rows)),
               'complete_all_measurements': bool(rows) and all(r['status'] == 'VALID' for r in rows),
               'rows': rows}
    from datetime import datetime, timezone
    save(out / 'reports' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f') + '.json'), summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
