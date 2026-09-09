"""Verify one forced attention-backend fixed-suite result for family reporting."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root):
    protocol=json.loads((root/'protocol.json').read_text())
    raw=json.loads((root/'raw.json').read_text())
    analysis=json.loads((root/'analysis.json').read_text())
    recomputed=json.loads((root/'recomputed_analysis.json').read_text())
    if analysis!=recomputed: raise ValueError('Stored and recomputed analyses differ')
    if raw.get('status')!='COMPLETE' or raw.get('determinism',{}).get('all_exact') is not True:
        raise ValueError('Attention capture is incomplete or nondeterministic')
    if raw.get('protocol_sha256')!=sha(root/'protocol.json'):
        raise ValueError('Raw result does not bind protocol')
    identity=raw['runtime_boundary']['identity']
    if len(identity['candidate_calls'])!=28 or len(identity['reference_calls'])!=28:
        raise ValueError('Unexpected attention call count')
    if 'FLASH_ATTENTION' not in identity['candidate_calls'][0]['backend']:
        raise ValueError('Target candidate call was not forced to Flash Attention')
    if any('MATH' not in row['backend'] for row in identity['candidate_calls'][1:]):
        raise ValueError('A later candidate layer differs from common math backend')
    if any('MATH' not in row['backend'] for row in identity['reference_calls']):
        raise ValueError('Reference did not use the forced math backend')
    stages={}
    for stage,rows in raw['original_coordinate_statistics'].items():
        if len(rows)!=32: raise ValueError('Expected 32 rows per stage')
        confirmation=rows[16:]
        effect=math.fsum(row['effect_energy'] for row in confirmation)
        repair=math.fsum(row['repair_energy'] for row in confirmation)
        stages[stage]={'confirmation_total_rms':math.sqrt(effect/repair),
                       'confirmation_nonzero_coordinate_range':[
                           min(row['nonzero_effect_coordinates'] for row in confirmation),
                           max(row['nonzero_effect_coordinates'] for row in confirmation)]}
    loss=[row['candidate_loss']-row['reference_loss'] for row in raw['determinism']['rows']]
    return {'schema':'operator-family-additional-evidence-v1','records':[{
        'family':'FUSED_ATTENTION','artifact_path':str(root/'analysis.json'),
        'artifact_sha256':sha(root/'analysis.json'),
        'evidence_kind':'FORCED_FLASH_VS_MATH_CAUSAL_ATTENTION_FIXED_SUITE',
        'backend':'PYTORCH_SDPA_FLASH_CUDA_VS_MATH','state_count':32,
        'claim_scope':analysis['claim_scope'],
        'measurement_status':'VALID_FIXED_SUITE_ENGINEERING_MEASUREMENT',
        'equivalence_decision':analysis['equivalence_decision'],'stage_effects':stages,
        'loss_difference_descriptive':{'minimum':min(loss),'maximum':max(loss),
                                       'mean':math.fsum(loss)/len(loss)},
        'training_quality_claim':False,'position_inventory_count':0,
        'scope':('One Qwen layer-0 causal attention backend substitution. Fixed suite and '
                 'cold AdamW only; per-state loss differences are not a training outcome.'),
    }]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path('/data1/tzh')):
        parser.error('New output under /data1/tzh required')
    result=build(args.root); args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle: json.dump(result,handle,indent=2,allow_nan=False)
    print(json.dumps(result['records'][0],indent=2))


if __name__=='__main__': main()
