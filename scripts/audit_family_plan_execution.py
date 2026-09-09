#!/usr/bin/env python3
"""Audit partition coverage against the full source plan, not a completed subset."""
import argparse
import json
from collections import Counter
from pathlib import Path
from scripts.finalize_numerical_family import finalize, read, sha, argument, ROOT


def check_partition(full_cases, partition_cases):
    full={c['case_id']:c for c in full_cases}
    if len(full)!=len(full_cases): raise ValueError('Duplicate source case')
    ids=[c['case_id'] for c in partition_cases]
    if not ids or len(set(ids))!=len(ids): raise ValueError('Empty or duplicate partition cases')
    for c in partition_cases:
        if c != full.get(c['case_id']): raise ValueError('Partition changes the original case: '+c['case_id'])
    return ids


def check_reference_translation(full_cases, translated_cases):
    """Verify the narrow internal adapter used by source-checked families.

    Family runners translate their descriptive reference name to the shared
    capture callback name.  No case identity, parameter, or other field may
    change.  Return the original case records so the ordinary partition check
    remains the final authority.
    """
    full={c['case_id']:c for c in full_cases}
    originals=[]
    for translated in translated_cases:
        original=full.get(translated.get('case_id'))
        if original is None:
            raise ValueError('Translated case is absent from source plan')
        expected={**original,
                  'reference_method':'PARTIAL_REDUCTION_FROM_BOUND_INPUT',
                  'declared_reference_method':original['reference_method']}
        if translated!=expected:
            raise ValueError('Translated case changes more than reference adapter: '
                             +str(translated.get('case_id')))
        originals.append(original)
    return originals


def finalize_capture(root, protocol):
    """Use the verifier declared for the recorded protocol, without fallback."""
    schema=protocol.get('schema')
    if schema=='selected-nll-capture-v1':
        from scripts.finalize_selected_nll_capture import verify
        report=verify(root)
    else:
        from kernel_analyzer.explicit_output_capture_audit import CONTRACTS
        if schema in CONTRACTS:
            from scripts.finalize_explicit_output_capture import finalize as explicit_finalize
            report=explicit_finalize(root)
        else:
            report=finalize(root)
    normalized=[]
    for row in report['records']:
        if row.get('status')=='RECORDED_MEASUREMENT_CHECKED':
            row=dict(row,status='VERIFIED',raw_artifact=row.get('raw_path'))
        normalized.append(row)
    return dict(report,records=normalized)


def audit(plan_path, roots, queues):
    source=read(plan_path); cases=source['cases']; entries=[]; inputs={str(plan_path.resolve()):sha(plan_path)}
    for root in roots:
        proto_path=root/'raw/family_execution_protocol.json'
        proto=read(proto_path)
        part=Path(argument(proto['capture_arguments'],'--case-plan'))
        if not part.is_absolute(): part=ROOT/part
        recurrence=proto.get('schema')=='decayed-recurrence-capture-v1'
        parent_digest=proto.get('source_sha256',{}).get(str(plan_path.resolve()))
        if recurrence and parent_digest!=sha(plan_path):
            raise ValueError('Recurrence capture not bound to original source plan')
        entries.append((root,part,proto['source_sha256'].get(str(part.resolve())),
                        recurrence,parent_digest))
    for queue in queues:
        qp=queue/'queue_protocol.json'; metadata=read(qp); inputs[str(qp.resolve())]=sha(qp)
        for job in metadata['jobs']:
            entries.append((Path(job['root']),Path(job['plan']),job['plan_sha256'],False,None))
    scheduled={}; records={}; errors=[]
    for root,part,expected,translated,parent_digest in entries:
        if sha(part)!=expected: raise ValueError('Partition digest changed: '+str(part))
        part_data=read(part)
        if translated:
            from scripts.finalize_decayed_recurrence import verify_translation
            part_cases=verify_translation(source,part_data)
        elif part.resolve()!=plan_path.resolve():
            declared_parent=part_data.get('source_plan_sha256')
            # Early family runners pinned both the complete source plan and
            # their translated case plan in the immutable execution protocol,
            # but did not duplicate the parent digest inside the translation.
            # Accept that explicit protocol link; exact case equality is still
            # checked immediately below.
            if declared_parent==sha(plan_path):
                part_cases=part_data['cases']
            elif parent_digest==sha(plan_path):
                part_cases=check_reference_translation(cases,part_data['cases'])
            else:
                raise ValueError('Partition is not bound to the source plan')
        else:
            part_cases=part_data['cases']
        ids=check_partition(cases,part_cases)
        inputs[str(part.resolve())]=sha(part)
        for cid in ids:
            if cid in scheduled: raise ValueError('Case scheduled more than once: '+cid)
            scheduled[cid]=str(root.resolve())
        if not (root/'raw/family_execution_protocol.json').exists() or not (root/'source_snapshot.json').exists():
            for cid in ids: records[cid]=dict(case_id=cid,status='NO_VERIFIABLE_CAPTURE_YET')
            continue
        try:
            capture_protocol=read(root/'raw/family_execution_protocol.json')
            if capture_protocol.get('schema')=='decayed-recurrence-capture-v1':
                from scripts.finalize_decayed_recurrence import verify_plan
                verify_plan(root)
            report=finalize_capture(root,capture_protocol)
            if {r['case_id'] for r in report['records']}!=set(ids):
                raise ValueError('Capture case set differs from scheduled partition')
            for r in report['records']: records[r['case_id']]=r
        except (ValueError,KeyError,TypeError,json.JSONDecodeError) as exc:
            errors.append(dict(root=str(root),reason=str(exc)))
            for cid in ids: records[cid]=dict(case_id=cid,status='CAPTURE_NOT_VERIFIABLE',reason=str(exc))
    rows=[{**records.get(c['case_id'],dict(case_id=c['case_id'],status='NOT_SCHEDULED')),
           'task_id':c['task_id'],'scheduled_root':scheduled.get(c['case_id'])} for c in cases]
    counts=dict(Counter(r['status'] for r in rows))
    return dict(schema='full-family-plan-execution-audit-v1',eligible_positions=len(cases),
        source_unresolved_positions=len(source.get('unresolved',[])),scheduled_positions=len(scheduled),
        counts=counts,eligible_measurement_complete=bool(cases) and counts.get('VERIFIED',0)==len(cases),
        all_source_positions_supported=False if source.get('unresolved') else 'NOT_ESTABLISHED_BY_THIS_AUDIT',
        whole_research_plan_complete=False,process_liveness='NOT_INFERRED_FROM_FILES',
        records=rows,errors=errors,input_sha256=inputs,
        scope='Fixed source plan; missing or incomplete cases retained; no population bias or loss claim')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--root',type=Path,nargs='*',default=[])
    p.add_argument('--queue',type=Path,nargs='*',default=[])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('Choose a new output under /data1/tzh')
    result=audit(a.plan,a.root,a.queue)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('eligible_positions','scheduled_positions','counts','eligible_measurement_complete')}))


if __name__=='__main__':main()
