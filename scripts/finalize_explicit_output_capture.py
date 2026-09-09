"""Verify saved convolution/GELU measurements without supplying new margins."""
import argparse
import hashlib
from collections import Counter
from pathlib import Path
from scripts.run_numerical_coverage import read,save,sha,ROOT
from scripts.finalize_numerical_family import argument,_analysis_from_snapshot
from scripts.snapshot_frozen_sources import preserve
from kernel_analyzer.explicit_output_capture_audit import audit_record,CONTRACTS


def finalize(root):
    root=root.resolve()
    protocol_path=root/'raw/family_execution_protocol.json'
    protocol=read(protocol_path)
    if protocol.get('schema') not in CONTRACTS:raise ValueError('Unknown explicit capture protocol')
    frozen=protocol['source_sha256']
    snapshot=read(root/'source_snapshot.json')
    expected={p for p in frozen if Path(p).suffix=='.py'}
    if set(snapshot)!=expected or any(hashlib.sha256(snapshot[p].encode()).hexdigest()!=frozen[p] for p in expected):
        raise ValueError('Source snapshot differs')
    changed=[]
    pinned=[]
    for name in ('training_numerical_analysis.py','training_equivalence.py'):
        path=ROOT/'src/kernel_analyzer'/name
        if str(path) in frozen:
            pinned.append(name)
            if frozen[str(path)]!=sha(path):changed.append(name)
    analysis_function=_analysis_from_snapshot(snapshot,frozen) if changed else None
    def input_path(flag):
        path=Path(argument(protocol['capture_arguments'],flag))
        path=(ROOT/path).resolve() if not path.is_absolute() else path.resolve()
        if frozen.get(str(path))!=sha(path):raise ValueError('Frozen input differs: '+flag)
        return path
    cases=read(input_path('--case-plan'))['cases']
    if not cases or len({c['task_id'] for c in cases})!=len(cases) or len({c['case_id'] for c in cases})!=len(cases):
        raise ValueError('Nonempty unique case plan required')
    bank=read(input_path('--state-bank' if '--state-bank' in protocol['capture_arguments'] else '--input-bank'))
    count=int(argument(protocol['capture_arguments'],'--states'))
    warm=int(argument(protocol['capture_arguments'],'--warmup-steps')) if '--warmup-steps' in protocol['capture_arguments'] else 0
    states=bank.get('states',bank.get('records',[]))[warm:warm+count]
    ids=[str(s.get('state_id',s.get('sequence_id',i))) for i,s in enumerate(states)]
    if count<2 or warm<0 or len(ids)!=count or len(set(ids))!=count:
        raise ValueError('Invalid complete state selection')
    records=[]
    for case in cases:
        if Path(case['case_id']).name!=case['case_id']:raise ValueError('Unsafe case path')
        path=root/'raw'/(case['case_id']+'.json')
        record=dict(status='NOT_CAPTURED')
        if path.exists():
            before=sha(path)
            record=audit_record(
                read(path),case,protocol,ids,
                **({'analysis_function':analysis_function} if analysis_function is not None else {}),
            )
            if sha(path)!=before:raise ValueError('Record changed during verification')
            record.update(raw_path=str(path),raw_sha256=before)
        records.append(dict(record,case_id=case['case_id'],task_id=case['task_id']))
    counts=dict(Counter(r['status'] for r in records))
    return dict(schema='explicit-output-capture-verification-v1',records=records,counts=counts,
        declared_positions=len(cases),recorded_measurement_complete=counts.get('RECORDED_MEASUREMENT_CHECKED',0)==len(cases),
        protocol_sha256=sha(protocol_path),full_research_goal_complete=False,
        verifier_sha256=sha(Path(__file__)),
        analysis_execution=(
            'PROTOCOL_PINNED_CONTENT_ADDRESSED_SNAPSHOT' if changed else
            'CURRENT_SOURCE_MATCHES_PROTOCOL' if len(pinned)==2 else
            'CURRENT_SOURCE_NOT_FULLY_PINNED_BY_LEGACY_PROTOCOL'
        ),
        pinned_analysis_sources=pinned,
        current_analysis_sources_changed=changed,
        scope='Saved record consistency, not independently reproduced GPU execution or population guarantee')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preserve-matching-sources',action='store_true')
    a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('New output under /data1/tzh required')
    snapshot=a.root/'source_snapshot.json'
    if a.preserve_matching_sources and not snapshot.exists():
        preserve(a.root/'raw/family_execution_protocol.json',snapshot)
    report=finalize(a.root);save(a.output,report)
    print(report['counts'])
    if not report['recorded_measurement_complete']:raise SystemExit(2)


if __name__=='__main__':main()
