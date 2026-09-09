#!/usr/bin/env python3
"""Add full-coordinate means to one family capture without changing its result.

The extension has its own frozen provenance. Existing Q analysis and old running
queues are unchanged; this is explicitly a same-data post-discovery recapture.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from kernel_analyzer.streaming_mean_profile import StreamingMeanProfile


ROOT=Path(__file__).resolve().parents[1]
STAGES=('LOCAL','PARAMETER_GRADIENT','PARAMETER_WRITE')
def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f: json.dump(value,f,indent=2,allow_nan=False)


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--original-raw',type=Path,required=True)
    p.add_argument('--repair-energy-floor',type=float,default=1e-30)
    a,remaining=p.parse_known_args()
    options=argparse.ArgumentParser(add_help=False)
    options.add_argument('--case-plan',type=Path,required=True)
    options.add_argument('--states',type=int,required=True)
    options.add_argument('--training-bias-profile-v2-output-dir',type=Path,required=True)
    o,_=options.parse_known_args(remaining)
    root=o.training_bias_profile_v2_output_dir.resolve().parent
    if root.exists() or not root.is_relative_to(Path('/data1/tzh')):
        raise ValueError('Use a new result root under /data1/tzh')
    cases=read(o.case_plan)['cases']; original=read(a.original_raw)
    if o.states!=32 or len(cases)!=1 or original['case_id']!=cases[0]['case_id']:
        raise ValueError('One declared existing case and 32 states required')
    collectors={s:StreamingMeanProfile(16,16,a.repair_energy_floor) for s in STAGES}
    sources=[Path(__file__),ROOT/'src/kernel_analyzer/streaming_mean_profile.py']
    extension=dict(schema='full-coordinate-mean-extension-v1',
        original_raw=str(a.original_raw.resolve()),original_sha256=sha(a.original_raw),
        capture_arguments=remaining,repair_energy_floor=a.repair_energy_floor,
        data_use='POST_DISCOVERY_SAME_STATE_RECAPTURE',statistical_decision_changed=False,
        source_sha256={str(p.resolve()):sha(p) for p in sources})
    save(root/'mean_extension_protocol.json',extension)
    save(root/'mean_extension_source_snapshot.json',{str(p.resolve()):p.read_text() for p in sources})
    from scripts import capture_bound_endpoint_bias_formation_v21 as capture
    from scripts import run_row_reduction_capture as family
    previous=capture.append_v2_contrast
    def append(store,stage,effect,repair,original_statistics=None):
        # Only reads the vectors; the original measurement call is retained.
        if stage in collectors: collectors[stage].append(effect,repair)
        return previous(store,stage,effect,repair,original_statistics)
    capture.append_v2_contrast=append
    sys.argv=[sys.argv[0],*remaining]
    family.main()
    raw_path=o.training_bias_profile_v2_output_dir/(cases[0]['case_id']+'.json')
    raw=read(raw_path)
    if sha(raw_path)!=extension['original_sha256']:
        raise ValueError('Raw capture changed; retain results but do not certify identical recapture')
    stages={s:collectors[s].finish() for s in STAGES}
    for stage,result in stages.items():
        for calculated,recorded in zip(result['rows'],raw['original_coordinate_statistics'][stage]):
            for key in ('effect_energy','repair_energy','effect_repair_inner_product'):
                x,y=calculated[key],recorded[key]
                if abs(x-y)>1e-10*max(abs(x),abs(y),1e-300):
                    raise ValueError('Full-coordinate accumulation differs from primary statistics')
    for name,digest in extension['source_sha256'].items():
        if sha(Path(name))!=digest: raise ValueError('Extension changed during execution')
    protocol_path=root/'raw/family_execution_protocol.json'
    protocol=read(protocol_path); frozen={}
    for name,digest in protocol['source_sha256'].items():
        path=Path(name)
        if sha(path)!=digest: raise ValueError('Capture source changed during execution')
        if path.suffix=='.py': frozen[name]=path.read_text()
    save(root/'source_snapshot.json',frozen)
    save(root/'full_coordinate_means.json',dict(schema='full-coordinate-means-v1',
        status='VERIFIED_IDENTICAL_RECAPTURE',case_id=raw['case_id'],
        state_ids=raw['state_ids'],calibration_state_ids=raw['calibration_state_ids'],
        confirmation_state_ids=raw['confirmation_state_ids'],
        raw_sha256=sha(raw_path),extension_protocol_sha256=sha(root/'mean_extension_protocol.json'),
        stages=stages,claim_scope='FIXED_SUITE_DESCRIPTIVE_NO_POPULATION_OR_PERSISTENCE_GUARANTEE'))
    print(json.dumps(dict(event='FULL_COORDINATE_MEANS_VERIFIED',case_id=raw['case_id'])),flush=True)


if __name__=='__main__':main()
