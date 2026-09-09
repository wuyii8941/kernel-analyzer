"""Join forward record audits without upgrading them to independent verification."""
import argparse
from pathlib import Path
from scripts.finalize_residual_rms_forward import finalize
from scripts.join_coverage_measurements import join, validate_retained_measurements
from scripts.run_numerical_coverage import read, save, sha


def apply(inventory, release, root):
    protocol_path = root / 'raw/family_execution_protocol.json'
    protocol = read(protocol_path)
    task_path = (release / 'same_dtype_tasks.json.gz').resolve()
    hashes = {str(Path(k).resolve()): v for k, v in protocol['source_sha256'].items()}
    if hashes.get(str(task_path)) != sha(task_path):
        raise ValueError('Capture is not bound to this release task inventory')
    retained = validate_retained_measurements(inventory)
    audit = finalize(root)
    records = []
    for original in audit['records']:
        record = dict(original)
        if 'raw_path' in record:
            record['raw_artifact'] = record.pop('raw_path')
        records.append(record)
    result = join(inventory, release, dict(records=records, errors=[]))
    result['retained_measurement_integrity'] = dict(checked=retained)
    result['forward_audit_scope'] = 'SAVED_RECORD_CHECK_NOT_INDEPENDENT_EXECUTION_VERIFICATION'
    result['forward_protocol_sha256'] = sha(protocol_path)
    return result


def queue_roots(queue):
    """Only terminal-success captures are eligible; this does not infer liveness."""
    protocol = read(queue/'queue_protocol.json')
    if protocol.get('schema') != 'family-plan-queue-v2':
        raise ValueError('Unsupported queue schema')
    roots, statuses, seen, seen_cases = [], [], set(), set()
    for job in protocol['jobs']:
        root = Path(job['root']).resolve()
        if root.parent != queue.resolve() or root in seen:
            raise ValueError('Duplicate or out-of-queue capture root')
        seen.add(root)
        plan = Path(job['plan'])
        if sha(plan) != job['plan_sha256']:
            raise ValueError('Frozen queue plan changed')
        cases = [row['case_id'] for row in read(plan)['cases']]
        if (not cases or len(cases) != len(set(cases))
                or sorted(cases) != sorted(job['case_ids']) or seen_cases.intersection(cases)):
            raise ValueError('Queue case declarations conflict or repeat')
        seen_cases.update(cases)
        status = 'NO_TERMINAL_RECORD_NOT_A_LIVENESS_CLAIM'
        if (root/'exit.json').exists():
            if read(root/'exit.json')['exit_code'] == 0:
                captured = read(root/'raw/family_execution_protocol.json')
                if captured.get('schema') != 'residual-rms-forward-capture-v1':
                    raise ValueError('Queue output uses a different capture interface')
                if captured['source_sha256'].get(str(plan.resolve())) != job['plan_sha256']:
                    raise ValueError('Successful capture was not bound to the queued partition')
                status = 'TERMINAL_SUCCESS_REQUIRES_FRESH_AUDIT'
                roots.append(root)
            else:
                status = 'CAPTURE_FAILED'
        statuses.append(dict(root=str(root), status=status, case_ids=job['case_ids']))
    return roots, statuses


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--release', type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--root', type=Path)
    source.add_argument('--queue', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path('/data1/tzh')):
        parser.error('Choose a new output under /data1/tzh')
    result = read(args.inventory)
    roots, statuses = queue_roots(args.queue) if args.queue else ([args.root], [])
    for root in roots:
        result = apply(result, args.release, root)
    if args.queue:
        result['queue_snapshot'] = dict(jobs=statuses,
            protocol_sha256=sha(args.queue/'queue_protocol.json'),
            audited_terminal_captures=len(roots), whole_queue_complete=False)
    result['input_inventory_sha256'] = sha(args.inventory)
    save(args.output, result)
    print(result['runtime_counts'])
