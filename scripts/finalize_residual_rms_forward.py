#!/usr/bin/env python3
"""Verify forward capture provenance and every declared output record."""
import argparse
import hashlib
from collections import Counter
from pathlib import Path
from scripts.run_numerical_coverage import read, save, sha, ROOT
from scripts.finalize_numerical_family import argument
from scripts.run_residual_rms_forward_capture import select
from kernel_analyzer.forward_capture_audit import audit_record


def verify_translation(original, translated):
    cases = []
    for row in translated['cases']:
        case = dict(row)
        if case.pop('reference_method', None) != 'PARTIAL_REDUCTION_FROM_BOUND_INPUT':
            raise ValueError('Unexpected reference dispatch')
        if case.pop('declared_reference_method', None) != 'RESIDUAL_RMS_FORWARD_COMMON_INPUT':
            raise ValueError('Original reference method missing')
        case['reference_method'] = 'RESIDUAL_RMS_FORWARD_COMMON_INPUT'
        cases.append(case)
    select(original, cases)
    return cases


def finalize(root):
    root = root.resolve()
    protocol_path = root / 'raw/family_execution_protocol.json'
    protocol = read(protocol_path)
    if protocol.get('schema') != 'residual-rms-forward-capture-v1':
        raise ValueError('Unexpected capture protocol')
    frozen = protocol['source_sha256']
    snapshot_path = root / 'source_snapshot.json'
    # New family-first campaigns keep all capture artifacts below ``raw``;
    # older jobs placed this snapshot at the campaign root.  Accept both
    # layouts while retaining the content-addressed verification.
    if not snapshot_path.exists():
        snapshot_path = root / 'raw' / 'source_snapshot.json'
    if not snapshot_path.exists():
        # The family-first launcher writes the protocol before invoking the
        # shared capture, while older queue jobs supplied this snapshot from
        # an outer wrapper.  Recreate it only after checking every frozen
        # source digest; this is not a retrospective source substitution.
        snapshot = {}
        for name, expected in frozen.items():
            path = Path(name)
            if path.suffix != '.py':
                continue
            if sha(path) != expected:
                raise ValueError('Frozen source changed before snapshot creation: ' + name)
            snapshot[name] = path.read_text()
        save(snapshot_path, snapshot)
    snapshot = read(snapshot_path)
    python_paths = {p for p in frozen if Path(p).suffix == '.py'}
    if set(snapshot) != python_paths or any(hashlib.sha256(snapshot[p].encode()).hexdigest() != frozen[p] for p in python_paths):
        raise ValueError('Frozen source snapshot differs')
    originals = []
    for name, digest in frozen.items():
        path = Path(name)
        if path.suffix != '.json': continue
        if sha(path) != digest: raise ValueError('Frozen input changed: ' + name)
        data = read(path)
        if isinstance(data, dict) and data.get('schema') == 'residual-rms-forward-bound-plan-v1':
            originals.append(data)
    if len(originals) != 1: raise ValueError('One original family plan required')
    def frozen_argument(name):
        path = Path(argument(protocol['capture_arguments'], name))
        path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
        if sha(path) != frozen.get(str(path)): raise ValueError('Unfrozen argument: ' + name)
        return path
    cases = verify_translation(originals[0], read(frozen_argument('--case-plan')))
    bank = read(frozen_argument('--input-bank'))
    count = (int(argument(protocol['capture_arguments'], '--states'))
             if '--states' in protocol['capture_arguments'] else 32)
    warmup = int(argument(protocol['capture_arguments'], '--warmup-steps')) if '--warmup-steps' in protocol['capture_arguments'] else 0
    states = bank.get('states', bank.get('records', []))[warmup:warmup + count]
    ids = [str(s.get('state_id', s.get('sequence_id', i))) for i, s in enumerate(states)]
    if count < 2 or warmup < 0 or len(ids) != count or len(set(ids)) != count:
        raise ValueError('Invalid frozen state selection')
    records = []
    for case in cases:
        if Path(case['case_id']).name != case['case_id']: raise ValueError('Unsafe case filename')
        path = root / 'raw' / (case['case_id'] + '.json')
        record = dict(status='NOT_CAPTURED')
        if path.exists():
            before = sha(path)
            record = audit_record(read(path), case, protocol, ids)
            if before != sha(path): raise ValueError('Record changed while reading')
            record.update(raw_path=str(path), raw_sha256=before)
        records.append(dict(record, case_id=case['case_id'], task_id=case['task_id']))
    counts = dict(Counter(r['status'] for r in records))
    return dict(schema='forward-capture-record-audit-v1', records=records, counts=counts,
        declared_positions=len(cases), protocol_sha256=sha(protocol_path),
        recorded_measurement_complete=counts.get('RECORDED_MEASUREMENT_CHECKED', 0) == len(cases),
        full_research_goal_complete=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path('/data1/tzh')):
        parser.error('Choose a new output under /data1/tzh')
    report = finalize(args.root)
    save(args.output, report)
    print(report['counts'])
    if not report['recorded_measurement_complete']: raise SystemExit(2)


if __name__ == '__main__': main()
