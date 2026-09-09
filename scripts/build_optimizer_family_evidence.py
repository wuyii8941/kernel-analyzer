"""Verify and expose one optimizer-implementation measurement to family reports."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root):
    protocol_path=root/'protocol.json'; raw_path=root/'raw.json'
    analysis_path=root/'analysis.json'; recomputed_path=root/'recomputed_analysis.json'
    protocol=json.loads(protocol_path.read_text()); raw=json.loads(raw_path.read_text())
    analysis=json.loads(analysis_path.read_text()); recomputed=json.loads(recomputed_path.read_text())
    if analysis!=recomputed:
        raise ValueError('Stored and independently recomputed analyses differ')
    if raw.get('status')!='COMPLETE' or raw.get('determinism',{}).get('all_exact') is not True:
        raise ValueError('Optimizer capture is incomplete or nondeterministic')
    if raw.get('protocol_sha256')!=sha(protocol_path):
        raise ValueError('Raw result does not bind the protocol')
    if analysis.get('measurement_status')!='VALID':
        raise ValueError('Unified analysis is not valid')
    sources=raw['runtime_boundary']['generated_sources']['files']
    triton_sources=0
    for row in sources:
        path=root/row['copy']
        if sha(path)!=row['sha256']:
            raise ValueError('Generated optimizer source changed')
        triton_sources += '@triton.jit' in path.read_text()
    if triton_sources<1:
        raise ValueError('No generated Triton definition was retained')
    stages={}
    for stage, rows in raw['original_coordinate_statistics'].items():
        if len(rows)!=32:
            raise ValueError('Expected 32 rows per optimizer stage')
        confirmation=rows[16:]
        effect=math.fsum(row['effect_energy'] for row in confirmation)
        repair=math.fsum(row['repair_energy'] for row in confirmation)
        stages[stage]={
            'confirmation_total_rms': math.sqrt(effect/repair) if repair>0 else None,
            'confirmation_nonzero_coordinate_range': [
                min(row['nonzero_effect_coordinates'] for row in confirmation),
                max(row['nonzero_effect_coordinates'] for row in confirmation),
            ],
        }
    return {'schema':'operator-family-additional-evidence-v1','records':[{
        'family':'OPTIMIZER_UPDATE',
        'artifact_path':str(analysis_path),
        'artifact_sha256':sha(analysis_path),
        'evidence_kind':'REAL_LLM_GRADIENT_OPTIMIZER_IMPLEMENTATION_FIXED_SUITE',
        'backend':'TORCHAO_TORCH_COMPILE_GENERATED_TRITON_VS_TORCH_ADAMW',
        'state_count':32,
        'claim_scope':analysis['claim_scope'],
        'measurement_status':'VALID_FIXED_SUITE_ENGINEERING_MEASUREMENT',
        'equivalence_decision':analysis['equivalence_decision'],
        'stage_effects':stages,
        'generated_source_file_count':len(sources),
        'generated_files_containing_triton_jit':triton_sources,
        'training_quality_claim':False,
        'position_inventory_count':0,
        'scope':('New optimizer-update family measured with real Mamba gradients. '
                 'No random-population or training-quality conclusion.'),
    }]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path('/data1/tzh')):
        parser.error('New output under /data1/tzh required')
    result=build(args.root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle: json.dump(result,handle,indent=2,allow_nan=False)
    print(json.dumps(result['records'][0],indent=2))


if __name__=='__main__': main()
