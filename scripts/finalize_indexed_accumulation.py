#!/usr/bin/env python3
"""Recheck indexed runtime identity, then reuse the unchanged family verifier."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from scripts.finalize_numerical_family import finalize,read,sha
from scripts.indexed_runtime_binding import bind_live_call


def verify_runtime(root):
    protocol=read(root/'raw/family_execution_protocol.json')
    evidence=root/'raw/indexed_runtime_bindings.json'
    observed=read(evidence)['observations']
    contracts=protocol['contracts']
    if not observed or not contracts: raise ValueError('Missing indexed runtime evidence')
    checked={}
    for group in observed:
        if set(group)!=set(contracts): raise ValueError('Runtime case set differs')
        for task,record in group.items():
            key=(task,record['executing_filename'],record['executing_source_sha256'])
            if key in checked:
                if checked[key]!=record: raise ValueError('Inconsistent repeated binding')
                continue
            path=Path(record['executing_filename'])
            if sha(path)!=record['executing_source_sha256']: raise ValueError('Runtime source changed')
            actual=bind_live_call(contracts[task],[SimpleNamespace(__file__=str(path))])
            if actual!=record: raise ValueError('Recorded and recomputed runtime bindings differ')
            checked[key]=actual
    return dict(status='SELECTED_INDEXED_CALLS_REVERIFIED',observations=len(observed),
                unique_bindings=len(checked),artifact=str(evidence),sha256=sha(evidence))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('Choose new output under /data1/tzh')
    runtime=verify_runtime(a.root)
    report=finalize(a.root)
    report['indexed_runtime_verification']=runtime
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f: json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({k:report[k] for k in ('declared_positions','counts','measurement_complete')}))
    if not report['measurement_complete']: raise SystemExit(2)


if __name__=='__main__':main()
