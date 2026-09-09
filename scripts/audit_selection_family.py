"""Recover actual selection calls from saved dispatcher traces, not symbol names."""
import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def audit(data):
    if data.get('schema')!='kernel-analyzer-full-architecture-invocation-inventory-v1':
        raise ValueError('Unsupported invocation trace')
    events=data['trace']['events']
    ids=[e['invocation_id'] for e in events]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate invocation identity')
    consumers=defaultdict(list)
    for event in events:
        for tensor in event['input_tensors']:
            consumers[(tensor.get('source_ordinal'),tensor['tensor_id'])].append(event['invocation_id'])
    rows=[]
    for event in events:
        if event.get('overload') not in ('aten.topk.default','aten.sort.default'):continue
        bindings={b['name']:b for b in event['argument_bindings']}
        if len(bindings)!=len(event['argument_bindings']):raise ValueError('Duplicate argument binding')
        rows.append(dict(invocation_id=event['invocation_id'],overload=event['overload'],
            phase=event['phase'],module_context=event['module_context'],
            dispatcher_schema=event['dispatcher_schema'],argument_bindings=event['argument_bindings'],
            inputs=event['input_tensors'],outputs=event['output_tensors'],
            output_consumers=[consumers.get((t.get('source_ordinal'),t['tensor_id']),[])
                              for t in event['output_tensors']],
            forward_autograd_nodes=event.get('forward_output_autograd_nodes'),
            reference_status='NOT_YET_REVIEWED',three_stage_measurement=False))
    return dict(schema='selection-family-source-audit-v1',model=data['model'],
        implementation=data['implementation'],records=rows,
        counts=dict(Counter(r['overload'] for r in rows)),
        proposed_reporting_family='SELECTION_TOPK_SORT',
        scope='Historical dispatcher observations and immediate tensor consumers only; '
              'not new GPU execution, Triton attribution, parameter reach proof or bias evidence',
        new_bias_confirmed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('New output under /data1/tzh required')
    raw=a.inventory.read_bytes()
    result=audit(json.loads(gzip.decompress(raw) if a.inventory.suffix=='.gz' else raw))
    result['input_sha256']=hashlib.sha256(raw).hexdigest()
    result['input_path']=str(a.inventory.resolve())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(result['counts'])


if __name__=='__main__':main()
