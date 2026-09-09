#!/usr/bin/env python3
"""Check recurrence plan translation, then reuse the shared numerical verifier."""
import argparse
import json
from pathlib import Path
from scripts.finalize_numerical_family import finalize, read, sha, argument, ROOT
from scripts.run_decayed_recurrence_capture import select_contracts,PLAN_METHODS
from scripts.snapshot_frozen_sources import preserve


def verify_translation(original, translated):
    schema=original.get('schema','decayed-recurrence-bound-plan-v1')
    if schema not in PLAN_METHODS: raise ValueError('Unknown original recurrence schema')
    method=PLAN_METHODS[schema][0]
    restored=[]
    for case in translated['cases']:
        case=dict(case)
        if case.pop('reference_method', None)!='PARTIAL_REDUCTION_FROM_BOUND_INPUT':
            raise ValueError('Unexpected capture reference dispatch')
        if case.pop('declared_reference_method', None)!=method:
            raise ValueError('Missing recurrence reference declaration')
        case['reference_method']=method
        restored.append(case)
    select_contracts(original, restored)
    return restored


def verify_plan(root):
    protocol=read(root/'raw/family_execution_protocol.json')
    if protocol.get('schema')!='decayed-recurrence-capture-v1':
        raise ValueError('Not a declared recurrence capture')
    frozen=protocol['source_sha256']
    sources=[]
    for name,digest in frozen.items():
        path=Path(name)
        if path.suffix!='.json': continue
        if sha(path)!=digest: raise ValueError('Frozen JSON changed: '+name)
        value=read(path)
        if isinstance(value,dict) and value.get('schema') in PLAN_METHODS:
            sources.append((path,value))
    if len(sources)!=1: raise ValueError('Require one frozen original recurrence plan')
    source_path,source=sources[0]
    part=Path(argument(protocol['capture_arguments'],'--case-plan'))
    if not part.is_absolute(): part=ROOT/part
    if sha(part)!=frozen.get(str(part.resolve())):
        raise ValueError('Translated plan is not frozen')
    cases=verify_translation(source,read(part))
    return dict(status='ORIGINAL_RECURRENCE_CASES_PRESERVED',positions=len(cases),
                original_plan=str(source_path),original_plan_sha256=sha(source_path),
                translated_plan=str(part),translated_plan_sha256=sha(part))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-content-addressed-snapshot',action='store_true')
    a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('Choose new output under /data1/tzh')
    binding=verify_plan(a.root)
    snapshot=a.root/'source_snapshot.json'
    if not snapshot.exists():
        preserve(a.root/'raw/family_execution_protocol.json',snapshot)
    report=finalize(a.root,allow_content_addressed_snapshot=a.allow_content_addressed_snapshot)
    report['recurrence_plan_verification']=binding
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f: json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({k:report[k] for k in ('declared_positions','counts','measurement_complete')}))
    if not report['measurement_complete']: raise SystemExit(2)


if __name__=='__main__': main()
